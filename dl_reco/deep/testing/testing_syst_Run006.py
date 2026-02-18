import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import argparse

import copy
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time
from scipy.signal import savgol_filter


import torch
from torch.utils.data import DataLoader
import jammy_flows.helper_fns as helper_fns

import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.channelGenericNoiseAdder
import logging
logger = logging.getLogger('test_raytracing')
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioMC.simulation.time_logger
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
from NuRadioMC.simulation import simulation as sim
from NuRadioReco.detector import detector
from NuRadioReco.framework.base_trace import BaseTrace
from scipy.signal import correlate

import dll_calc

import importlib.util
import sys
import shutil
sys.path.insert(1, '..')


#---------parser arguments to specify the run and the epoch under investigation
parser = argparse.ArgumentParser()
parser.add_argument("Run_number", type=str, default='RunXXX', help="Name a new Run to be saved")
parser.add_argument("epoch", type=str, default='RunXXX', help="Name a new Run to be saved")
parser.add_argument("systematic", type=str, default='RunXXX', help="Name a new Run to be saved")
parser.add_argument("--test_batchSize", type=int, default=10, help="test batch size")

args = parser.parse_args()

#---------loading the correct checkpoint
model_dir = '~/gen2_deep_reco/reconstruction/paper_prep/combined/results/' + args.Run_number + '/model/'
model_file = 'epoch=' + args.epoch + '_model_checkpoint.ckpt' 

#---------loading the correct run files
spec = importlib.util.spec_from_file_location('hyperparameters', '~/gen2_deep_reco/reconstruction/paper_prep/combined/results/' + args.Run_number + '/hyperparameters_' + args.Run_number + '.py')
hyperparameters = importlib.util.module_from_spec(spec)
sys.modules['hyperparameters'] = hyperparameters
spec.loader.exec_module(hyperparameters)

spec = importlib.util.spec_from_file_location('data_gen', '~/gen2_deep_reco/reconstruction/paper_prep/combined/results/' + args.Run_number + '/data_' + args.Run_number + '.py')
data_gen = importlib.util.module_from_spec(spec)
sys.modules['data_gen'] = data_gen
spec.loader.exec_module(data_gen)

spec = importlib.util.spec_from_file_location('models', '~/gen2_deep_reco/reconstruction/paper_prep/combined/results/' + args.Run_number + '/models_' + args.Run_number + '.py')
models = importlib.util.module_from_spec(spec)
sys.modules['models'] = models
spec.loader.exec_module(models)

#---------save this test file to the run
src_data = 'test_syst.py'
dst_data = 'results/' + args.Run_number + '/testing_syst_' + args.Run_number + '.py'
shutil.copy(src_data, dst_data)

#---------save the resulting dictionary here
save_path = '~/gen2_deep_reco/reconstruction/paper_prep/combined/results/' + args.Run_number + '/full_y_pred/systematics/'
if(not os.path.exists(save_path)):
    os.makedirs(save_path)

syst_dataset = data_gen.Prepare_Dataset_systematics(syst_name = args.systematic)

test_loader = DataLoader(syst_dataset, batch_size=args.test_batchSize, shuffle=False, num_workers=0)
device = torch.device(f"cuda:0")

#---------initialize the model
model = models.Split(loss_function = hyperparameters.loss_function)
model.float()
model.load_state_dict(torch.load(model_dir + model_file)['state_dict'])
model.pdf_energy_vertex.double()
model.pdf_direction.double()
model.eval()

#---------create the directory to save the data
samplesize = 50000
samplesize_goodness = 10000

results_dict = {}
def convert_lists_to_arrays(data):
    for key, value in data.items():
        if isinstance(value, list): 
            data[key] = np.array(value)
        elif isinstance(value, dict):  
            convert_lists_to_arrays(value)
    return data

results_dict['true_zenith'] = []
results_dict['true_azimuth'] = []

results_dict['pred_mean_zenith'] = []
results_dict['pred_mean_azimuth'] = []
results_dict['pred_max_zenith'] = []
results_dict['pred_max_azimuth'] = []

results_dict['true_energy'] = []
results_dict['true_xx'] = []
results_dict['true_yy'] = []
results_dict['true_zz'] = []

results_dict['true_trigger_time'] = []
results_dict['true_signal_time'] = []

results_dict['pred_energy'] = []
results_dict['pred_energy_max'] = []
results_dict['pred_energy_var'] = []
results_dict['pred_xx'] = []
results_dict['pred_xx_var'] = []
results_dict['pred_yy'] = []
results_dict['pred_yy_var'] = []
results_dict['pred_zz'] = []
results_dict['pred_zz_var'] = []
results_dict['pred_EV_cov_mat'] = []

results_dict['pred_energy_mode'] = []
results_dict['pred_xx_mode'] = []
results_dict['pred_yy_mode'] = []
results_dict['pred_zz_mode'] = []

results_dict['true_flavor'] = []
results_dict['pred_flavor'] = []

results_dict['pred_area_68'] = []
results_dict['pred_area_50'] = []

#save samples
results_dict['samples_zenith'] = []
results_dict['samples_azimuth'] = []
results_dict['samples_energy'] = []
results_dict['samples_xx'] = []
results_dict['samples_yy'] = []
results_dict['samples_zz'] = []

#---------create the directories to do the coverage correction
results_dict['pred_area_percentage'] = {}
results_dict['pred_energy_percentage'] = {}
for perc in np.arange(1, 101, 1):
    results_dict['pred_area_percentage'][perc] = []
    results_dict['pred_energy_percentage'][perc] = []

results_dict['pred_energy_68'] = []
results_dict['cov_energy_exact_sampled'] = []
results_dict['pred_energy_std'] = []

results_dict['cov_energy_approx'] = []
results_dict['cov_energy_exact'] = []
results_dict['cov_direction_approx'] = []
results_dict['cov_direction_exact'] = []

results_dict['pred_direction_entropy'] = []
results_dict['pred_kl_diff'] = []
results_dict['pred_kl_diff_inv'] =[]

results_dict['channel_0_max'] = []
results_dict['channel_1_max'] = []
results_dict['channel_2_max'] = []
results_dict['channel_3_max'] = []
results_dict['channel_4_max'] = []
results_dict['channel_5_max'] = []
results_dict['channel_6_max'] = []
results_dict['channel_7_max'] = []
results_dict['channel_8_max'] = []
results_dict['channel_9_max'] = []
results_dict['channel_10_max'] = []
results_dict['channel_11_max'] = []
results_dict['channel_12_max'] = []
results_dict['channel_13_max'] = []
results_dict['channel_14_max'] = []
results_dict['channel_15_max'] = []

results_dict['channel_0_max_manual'] = []
results_dict['channel_1_max_manual'] = []
results_dict['channel_2_max_manual'] = []
results_dict['channel_3_max_manual'] = []
results_dict['channel_4_max_manual'] = []
results_dict['channel_5_max_manual'] = []
results_dict['channel_6_max_manual'] = []
results_dict['channel_7_max_manual'] = []
results_dict['channel_8_max_manual'] = []
results_dict['channel_9_max_manual'] = []
results_dict['channel_10_max_manual'] = []
results_dict['channel_11_max_manual'] = []
results_dict['channel_12_max_manual'] = []
results_dict['channel_13_max_manual'] = []
results_dict['channel_14_max_manual'] = []
results_dict['channel_15_max_manual'] = []



#---------looping through the data set
lll = 0
with torch.no_grad():
    for count, test_quant in enumerate(tqdm(test_loader)):

        #---------get data and labels from the data loader 
        x_test, true_direction, true_flavor, true_energy_vertex, true_snr, true_times = test_quant

        true_flavor = true_flavor.squeeze()
        true_energy = true_energy_vertex[:, 0].squeeze()
        true_xx = true_energy_vertex[:, 1].squeeze()
        true_yy = true_energy_vertex[:, 2].squeeze()
        true_zz = true_energy_vertex[:, 3].squeeze()
        true_zenith = true_direction[:, 0].squeeze()
        true_azimuth = true_direction[:, 1].squeeze()
        true_trigger_time = true_times[:, 0].squeeze()
        true_signal_time = true_times[:, 1].squeeze()

        batch_length = len(true_flavor)
        

        #---------write true values to output dictionary
        results_dict['true_flavor'].extend(true_flavor.detach().cpu().numpy().squeeze())
        results_dict['true_energy'].extend(true_energy_vertex[:, 0].detach().cpu().numpy().squeeze())
        results_dict['true_xx'].extend(true_energy_vertex[:, 1].detach().cpu().numpy().squeeze())
        results_dict['true_yy'].extend(true_energy_vertex[:, 2].detach().cpu().numpy().squeeze())
        results_dict['true_zz'].extend(true_energy_vertex[:, 3].detach().cpu().numpy().squeeze())
        results_dict['true_zenith'].extend(true_direction[:, 0].detach().cpu().numpy().squeeze())
        results_dict['true_azimuth'].extend(true_direction[:, 1].detach().cpu().numpy().squeeze())
        results_dict['true_trigger_time'].extend(true_times[:, 0].detach().cpu().numpy().squeeze())
        results_dict['true_signal_time'].extend(true_times[:, 1].detach().cpu().numpy().squeeze())

        results_dict['channel_0_max'].extend(true_snr[:, 0].detach().cpu().numpy().squeeze())
        results_dict['channel_1_max'].extend(true_snr[:, 1].detach().cpu().numpy().squeeze())
        results_dict['channel_2_max'].extend(true_snr[:, 2].detach().cpu().numpy().squeeze())
        results_dict['channel_3_max'].extend(true_snr[:, 3].detach().cpu().numpy().squeeze())
        results_dict['channel_4_max'].extend(true_snr[:, 4].detach().cpu().numpy().squeeze())
        results_dict['channel_5_max'].extend(true_snr[:, 5].detach().cpu().numpy().squeeze())
        results_dict['channel_6_max'].extend(true_snr[:, 6].detach().cpu().numpy().squeeze())
        results_dict['channel_7_max'].extend(true_snr[:, 7].detach().cpu().numpy().squeeze())
        results_dict['channel_8_max'].extend(true_snr[:, 8].detach().cpu().numpy().squeeze())
        results_dict['channel_9_max'].extend(true_snr[:, 9].detach().cpu().numpy().squeeze())
        results_dict['channel_10_max'].extend(true_snr[:, 10].detach().cpu().numpy().squeeze())
        results_dict['channel_11_max'].extend(true_snr[:, 11].detach().cpu().numpy().squeeze())
        results_dict['channel_12_max'].extend(true_snr[:, 12].detach().cpu().numpy().squeeze())
        results_dict['channel_13_max'].extend(true_snr[:, 13].detach().cpu().numpy().squeeze())
        results_dict['channel_14_max'].extend(true_snr[:, 14].detach().cpu().numpy().squeeze())
        results_dict['channel_15_max'].extend(true_snr[:, 15].detach().cpu().numpy().squeeze())


        x_max = x_test.abs().max(dim=-1).values 
        x_max = x_max.detach().cpu().squeeze(1)
        result_snr = x_max.transpose(0, 1)#.tolist()


        results_dict['channel_0_max_manual'].extend(result_snr[0].detach().cpu().numpy().squeeze())
        results_dict['channel_1_max_manual'].extend(result_snr[1].detach().cpu().numpy().squeeze())
        results_dict['channel_2_max_manual'].extend(result_snr[2].detach().cpu().numpy().squeeze())
        results_dict['channel_3_max_manual'].extend(result_snr[3].detach().cpu().numpy().squeeze())
        results_dict['channel_4_max_manual'].extend(result_snr[4].detach().cpu().numpy().squeeze())
        results_dict['channel_5_max_manual'].extend(result_snr[5].detach().cpu().numpy().squeeze())
        results_dict['channel_6_max_manual'].extend(result_snr[6].detach().cpu().numpy().squeeze())
        results_dict['channel_7_max_manual'].extend(result_snr[7].detach().cpu().numpy().squeeze())
        results_dict['channel_8_max_manual'].extend(result_snr[8].detach().cpu().numpy().squeeze())
        results_dict['channel_9_max_manual'].extend(result_snr[9].detach().cpu().numpy().squeeze())
        results_dict['channel_10_max_manual'].extend(result_snr[10].detach().cpu().numpy().squeeze())
        results_dict['channel_11_max_manual'].extend(result_snr[11].detach().cpu().numpy().squeeze())
        results_dict['channel_12_max_manual'].extend(result_snr[12].detach().cpu().numpy().squeeze())
        results_dict['channel_13_max_manual'].extend(result_snr[13].detach().cpu().numpy().squeeze())
        results_dict['channel_14_max_manual'].extend(result_snr[14].detach().cpu().numpy().squeeze())
        results_dict['channel_15_max_manual'].extend(result_snr[15].detach().cpu().numpy().squeeze())


        #---------apply the model to the data
        conv_out = model.forward(x_test)
        conv_out_repeated = torch.repeat_interleave(conv_out, samplesize, dim=0)

        #---------flavor prediction
        flavor_pred = model.classifier(conv_out).squeeze()
        results_dict['pred_flavor'].extend(flavor_pred.detach().cpu().numpy().squeeze())

        #---------energy and vertex prediction
        moments_dict_energy_vertex = model.pdf_energy_vertex.marginal_moments(conditional_input=conv_out.to(torch.double).to(device), calc_kl_diff_and_entropic_quantities=True)
        coverage_dict_energy = model.pdf_energy_vertex.coverage_and_or_pdf_scan(labels = true_energy_vertex.to(torch.double).to(device).squeeze(), conditional_input=conv_out.to(torch.double).to(device))

        samples_e_v, _, sample_log_pdf, _ = model.pdf_energy_vertex.sample(conditional_input=conv_out_repeated.to(torch.double).to(device), device=device, force_embedding_coordinates=True)
        samples_e_v_reshaped = torch.reshape(samples_e_v, (batch_length, samplesize, 4))
        sample_log_pdf_reshape = torch.reshape(sample_log_pdf, (batch_length, samplesize))

        top_indices = torch.argmax(sample_log_pdf_reshape, dim=1)
        top_ev_values = samples_e_v_reshaped[torch.arange(samples_e_v_reshaped.shape[0]), top_indices]

        results_dict['pred_energy_mode'].extend(top_ev_values[:, 0].cpu().tolist())
        results_dict['pred_xx_mode'].extend(top_ev_values[:, 1].cpu().tolist())
        results_dict['pred_yy_mode'].extend(top_ev_values[:, 2].cpu().tolist())
        results_dict['pred_zz_mode'].extend(top_ev_values[:, 3].cpu().tolist())


        results_dict['samples_energy'].extend(samples_e_v_reshaped[:, :samplesize_goodness, 0].cpu())
        results_dict['samples_xx'].extend(samples_e_v_reshaped[:, :samplesize_goodness, 1].cpu())
        results_dict['samples_yy'].extend(samples_e_v_reshaped[:, :samplesize_goodness, 2].cpu())
        results_dict['samples_zz'].extend(samples_e_v_reshaped[:, :samplesize_goodness, 3].cpu())
        
        pred_energy = moments_dict_energy_vertex['mean_0'][:, 0].squeeze()
        pred_energy_std = moments_dict_energy_vertex['varlike_0'][:, 0, 0].squeeze()
        pred_xx = moments_dict_energy_vertex['mean_0'][:, 1].squeeze()
        pred_xx_std = moments_dict_energy_vertex['varlike_0'][:, 1, 1].squeeze()
        pred_yy = moments_dict_energy_vertex['mean_0'][:, 2].squeeze()
        pred_yy_std = moments_dict_energy_vertex['varlike_0'][:, 2, 2].squeeze()
        pred_zz = moments_dict_energy_vertex['mean_0'][:, 3].squeeze()
        pred_zz_std = moments_dict_energy_vertex['varlike_0'][:, 3, 3].squeeze()
        cov_energy_approx = coverage_dict_energy['approx_cov_values'].squeeze()

        results_dict['pred_energy'].extend(moments_dict_energy_vertex['mean_0'][:, 0].squeeze())
        results_dict['pred_energy_var'].extend(moments_dict_energy_vertex['varlike_0'][:, 0, 0].squeeze())

        results_dict['pred_xx'].extend(moments_dict_energy_vertex['mean_0'][:, 1].squeeze())
        results_dict['pred_xx_var'].extend(moments_dict_energy_vertex['varlike_0'][:, 1, 1].squeeze())

        results_dict['pred_yy'].extend(moments_dict_energy_vertex['mean_0'][:, 2].squeeze())
        results_dict['pred_yy_var'].extend(moments_dict_energy_vertex['varlike_0'][:, 2, 2].squeeze())

        results_dict['pred_zz'].extend(moments_dict_energy_vertex['mean_0'][:, 3].squeeze())
        results_dict['pred_zz_var'].extend(moments_dict_energy_vertex['varlike_0'][:, 3, 3].squeeze())

        results_dict['pred_EV_cov_mat'].extend(moments_dict_energy_vertex['varlike_0'].squeeze())

        results_dict['cov_energy_approx'].extend(coverage_dict_energy['approx_cov_values'].squeeze())

        #---------direction prediction
        moments_dict_direction = model.pdf_direction.marginal_moments(conditional_input=conv_out.to(torch.double).to(device), calc_kl_diff_and_entropic_quantities=True)
        coverage_dict_direction = model.pdf_direction.coverage_and_or_pdf_scan(labels = true_direction.to(torch.double).to(device), conditional_input=conv_out.to(torch.double).to(device), exact_coverage_calculation=True, save_pdf_scan=True, calculate_MAP=True)

        samples_d, _, _, _ = model.pdf_direction.sample(conditional_input=conv_out_repeated.to(torch.double).to(device), device=device, force_embedding_coordinates=False)
        samples_d_reshaped = torch.reshape(samples_d, (batch_length, samplesize, 2))

        results_dict['samples_zenith'].extend(samples_d_reshaped[:, :samplesize_goodness, 0].cpu())
        results_dict['samples_azimuth'].extend(samples_d_reshaped[:, :samplesize_goodness, 1].cpu())

        pred_mean_zenith = moments_dict_direction['mean_0_angles'][:, 0].squeeze()
        pred_mean_azimuth = moments_dict_direction['mean_0_angles'][:, 1].squeeze()

        #---------looping through every event for specific calculations
        areas_50_list = []
        areas_68_list = []
        goodness_MC_list = []
        goodness_mean_list = []
        goodness_score_list = []  
        goodness_MC_score_list = []  
        energy_exact_coverage_list = []
        energy_68_list = []
        pred_energy_max_list = []

        area_percentages = {}
        energy_percentages = {}
        for perc in np.arange(1, 101, 1):
            area_percentages[perc] = []
            energy_percentages[perc] = []

        for indx, event in enumerate(coverage_dict_direction['pdf_scan_positions']):

            #---------energy size & coverage calculation
            event_samples_energy = samples_e_v_reshaped[indx, :, :]
            event_samples_direction = samples_d_reshaped[indx, :, :]
            only_energy_samples = event_samples_energy[:,0]

            event_energy = true_energy_vertex.to(torch.double).squeeze().cpu().numpy()[indx, 0]

            _, density_bounds,_=helper_fns.grid_functions.obtain_bins_and_visualization_regions(event_samples_energy, model.pdf_energy_vertex, percentiles=[0.5,99.5])
            pdf_eval, positions = np.histogram(only_energy_samples.cpu(), bins=np.linspace(density_bounds[0][0], density_bounds[0][1], 1000), density=True)

            size = positions[1] - positions[0]
            positions = positions[:-1]+size/2

            smoothed_data = savgol_filter(pdf_eval, window_length=50, polyorder=3)
            pred_energy_max = positions[smoothed_data == max(smoothed_data)]
            pred_energy_max_list.append(pred_energy_max[0])

            large_to_small_probs_mask=np.argsort(pdf_eval)[::-1]
            sorted_pos = positions[large_to_small_probs_mask]

            diffs_sorted = event_energy-sorted_pos
            min_index = np.argmin(np.abs(diffs_sorted))

            sorted_evals = size * pdf_eval[large_to_small_probs_mask]
            cumsum_energy = np.cumsum(sorted_evals)
            real_cov_value = cumsum_energy[min_index]

            energy_exact_coverage_list.append(real_cov_value)

            energy_pos_perc = np.argmin(abs(0.6827 - cumsum_energy))
            energy_68_list.append(size * energy_pos_perc)


            for perc in np.arange(1, 101, 1):

                energy_pos_perc = np.argmin(abs(perc/100 - cumsum_energy))
                energy_percentages[perc].append(size * energy_pos_perc)

            #---------direction size  calculation
            batch_positions = coverage_dict_direction['pdf_scan_positions'][indx]
            batch_sizes = coverage_dict_direction['pdf_scan_volume_sizes'][indx]
            batch_evals = coverage_dict_direction['pdf_scan_log_evals'][indx]

            xyz_positions_pred, _ = model.pdf_direction.layer_list[0][0].spherical_to_eucl_embedding(torch.tensor(batch_positions), 0.0)
            xyz_positions_pred = xyz_positions_pred.numpy()

            large_to_small_probs_mask_direction = np.argsort(batch_evals)[::-1]
            sorted_pos_direction = xyz_positions_pred[large_to_small_probs_mask_direction]
            sorted_areas = batch_sizes[large_to_small_probs_mask_direction]
            sorted_evals_direction = sorted_areas * np.exp(batch_evals[large_to_small_probs_mask_direction])

            cumsum_direction = np.cumsum(sorted_evals_direction)

            pixel_68 = np.argmin(abs(0.6827 - cumsum_direction))
            area_68 = np.sum(sorted_areas[:pixel_68])
            areas_68_list.append(area_68)

            pixel_50 = np.argmin(abs(0.5 - cumsum_direction))
            area_50 = np.sum(sorted_areas[:pixel_50])
            areas_50_list.append(area_50)

            for perc in np.arange(1, 101, 1):

                pixel_perc = np.argmin(abs(perc/100 - cumsum_direction))
                area_perc = np.sum(sorted_areas[:pixel_perc])
                area_percentages[perc].append(area_perc)


        #---------writing events to the results dictionary
        results_dict['pred_mean_zenith'].extend(moments_dict_direction['mean_0_angles'][:, 0].squeeze())
        results_dict['pred_mean_azimuth'].extend(moments_dict_direction['mean_0_angles'][:, 1].squeeze())
        results_dict['pred_max_zenith'].extend(moments_dict_direction['argmax_0_angles'][:, 0].squeeze())
        results_dict['pred_max_azimuth'].extend(moments_dict_direction['argmax_0_angles'][:, 1].squeeze())
        results_dict['pred_area_68'].extend(areas_68_list)
        results_dict['pred_area_50'].extend(areas_50_list)
        results_dict['pred_energy_max'].extend(pred_energy_max_list)
        results_dict['pred_energy_68'].extend(energy_68_list)
        results_dict['cov_energy_exact_sampled'].extend(energy_exact_coverage_list)
        results_dict['cov_direction_approx'].extend(coverage_dict_direction['approx_cov_values'].squeeze())
        results_dict['cov_direction_exact'].extend(coverage_dict_direction['real_cov_values'].squeeze())
        results_dict['pred_direction_entropy'].extend(moments_dict_direction['approx_entropy_0'].squeeze())
        results_dict['pred_kl_diff'].extend(moments_dict_direction['kl_diff_exact_approx_0'].squeeze())
        results_dict['pred_kl_diff_inv'].extend(moments_dict_direction['kl_diff_approx_exact_0'].squeeze())


        for perc in np.arange(1, 101, 1):
            results_dict['pred_area_percentage'][perc].extend(area_percentages[perc])
            results_dict['pred_energy_percentage'][perc].extend(energy_percentages[perc])

        lll += 1

        if lll == 100:
            np.save(save_path + args.systematic + 'pre_results_' + args.epoch + '.npy', results_dict)

        
results_dict = convert_lists_to_arrays(results_dict)
np.save(save_path + args.systematic + '_results_' + args.epoch + '.npy', results_dict)

