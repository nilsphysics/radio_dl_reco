import numpy as np
import argparse
import os
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
parser.add_argument("--data", type=str, default='test', help="test batch size")
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
src_data = 'test_model_pre.py'
dst_data = 'results/' + args.Run_number + '/testing_' + args.Run_number + '.py'
shutil.copy(src_data, dst_data)

#---------save the resulting dictionary here
save_path = '~/gen2_deep_reco/reconstruction/paper_prep/combined/results/' + args.Run_number + '/full_y_pred/'
if(not os.path.exists(save_path)):
    os.makedirs(save_path)

#---------initialize the data loader with test or validation data
if args.data == 'test':
    list_of_file_ids_test = np.arange(hyperparameters.n_files_test, dtype=int)
    test_dataset = data_gen.Prepare_Dataset_direction(file_ids=list_of_file_ids_test,points = hyperparameters.n_events_per_file, data_type = '_testing', worker_num=hyperparameters.worker_num, shuffle=False)

elif args.data == 'val':
    validation_files = 5
    list_of_file_ids_val = np.arange(validation_files, dtype=int)
    test_dataset = data_gen.Prepare_Dataset_direction(file_ids=list_of_file_ids_val,points = hyperparameters.n_events_per_file, data_type = '_validating', worker_num=hyperparameters.worker_num, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=args.test_batchSize, shuffle=False, num_workers=0)
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


for file in range(hyperparameters.n_files_test):
    results_dict[file] = {}

    results_dict[file]['true_zenith'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['true_azimuth'] = np.zeros([hyperparameters.n_events_per_file])

    results_dict[file]['pred_mean_zenith'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_mean_azimuth'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_max_zenith'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_max_azimuth'] = np.zeros([hyperparameters.n_events_per_file])

    results_dict[file]['true_energy'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['true_xx'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['true_yy'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['true_zz'] = np.zeros([hyperparameters.n_events_per_file])

    results_dict[file]['true_trigger_time'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['true_signal_time'] = np.zeros([hyperparameters.n_events_per_file])

    results_dict[file]['pred_energy'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_energy_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_energy_mode'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_energy_var'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_xx'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_xx_mode'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_xx_var'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_yy'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_yy_mode'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_yy_var'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_zz'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_zz_mode'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_zz_var'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_EV_cov_mat'] = np.zeros(([hyperparameters.n_events_per_file][0], 4, 4))

    results_dict[file]['true_flavor'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_flavor'] = np.zeros([hyperparameters.n_events_per_file])

    results_dict[file]['pred_area_68'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_area_50'] = np.zeros([hyperparameters.n_events_per_file])

    #save samples
    results_dict[file]['samples_zenith'] = np.zeros([hyperparameters.n_events_per_file, samplesize_goodness])
    results_dict[file]['samples_azimuth'] = np.zeros([hyperparameters.n_events_per_file, samplesize_goodness])
    results_dict[file]['samples_energy'] = np.zeros([hyperparameters.n_events_per_file, samplesize_goodness])
    results_dict[file]['samples_xx'] = np.zeros([hyperparameters.n_events_per_file, samplesize_goodness])
    results_dict[file]['samples_yy'] = np.zeros([hyperparameters.n_events_per_file, samplesize_goodness])
    results_dict[file]['samples_zz'] = np.zeros([hyperparameters.n_events_per_file, samplesize_goodness])

    #---------create the directories to do the coverage correction
    results_dict[file]['pred_area_percentage'] = {}
    results_dict[file]['pred_energy_percentage'] = {}
    for perc in np.arange(1, 101, 1):
        results_dict[file]['pred_area_percentage'][perc] = np.zeros([hyperparameters.n_events_per_file])
        results_dict[file]['pred_energy_percentage'][perc] = np.zeros([hyperparameters.n_events_per_file])

    results_dict[file]['pred_energy_68'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['cov_energy_exact_sampled'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_energy_std'] = np.zeros([hyperparameters.n_events_per_file])

    results_dict[file]['cov_energy_approx'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['cov_energy_exact'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['cov_direction_approx'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['cov_direction_exact'] = np.zeros([hyperparameters.n_events_per_file])

    results_dict[file]['pred_direction_entropy'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_kl_diff'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['pred_kl_diff_inv'] = np.zeros([hyperparameters.n_events_per_file])


    results_dict[file]['channel_0_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_1_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_2_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_3_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_4_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_5_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_6_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_7_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_8_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_9_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_10_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_11_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_12_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_13_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_14_max_manual'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_15_max_manual'] = np.zeros([hyperparameters.n_events_per_file])


    results_dict[file]['channel_0_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_1_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_2_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_3_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_4_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_5_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_6_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_7_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_8_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_9_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_10_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_11_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_12_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_13_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_14_max'] = np.zeros([hyperparameters.n_events_per_file])
    results_dict[file]['channel_15_max'] = np.zeros([hyperparameters.n_events_per_file])


#---------looping through the data set
lll = 0
with torch.no_grad():
    for count, test_quant in enumerate(tqdm(test_loader)):

        file_number = (count * args.test_batchSize) // hyperparameters.n_events_per_file

        #---------define list index for batch item
        first_item = (count * args.test_batchSize) % hyperparameters.n_events_per_file
        last_item = (count * args.test_batchSize) % hyperparameters.n_events_per_file  + args.test_batchSize 

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

        #---------write true values to output dictionary
        results_dict[file_number]['true_flavor'][first_item : last_item] = true_flavor.detach().cpu().numpy().squeeze()
        results_dict[file_number]['true_energy'][first_item : last_item] = true_energy_vertex[:, 0].detach().cpu().numpy().squeeze()
        results_dict[file_number]['true_xx'][first_item : last_item] = true_energy_vertex[:, 1].detach().cpu().numpy().squeeze()
        results_dict[file_number]['true_yy'][first_item : last_item] = true_energy_vertex[:, 2].detach().cpu().numpy().squeeze()
        results_dict[file_number]['true_zz'][first_item : last_item] = true_energy_vertex[:, 3].detach().cpu().numpy().squeeze()
        results_dict[file_number]['true_zenith'][first_item : last_item] = true_direction[:, 0].detach().cpu().numpy().squeeze()
        results_dict[file_number]['true_azimuth'][first_item : last_item] = true_direction[:, 1].detach().cpu().numpy().squeeze()
        results_dict[file_number]['true_trigger_time'][first_item : last_item] = true_times[:, 0].detach().cpu().numpy().squeeze()
        results_dict[file_number]['true_signal_time'][first_item : last_item] = true_times[:, 1].detach().cpu().numpy().squeeze()


        x_max = x_test.abs().max(dim=-1).values 
        x_max = x_max.detach().cpu().squeeze(1)
        result_snr = x_max.transpose(0, 1)#.tolist()

        results_dict[file_number]['channel_0_max_manual'][first_item : last_item] = result_snr[0].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_1_max_manual'][first_item : last_item] = result_snr[1].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_2_max_manual'][first_item : last_item] = result_snr[2].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_3_max_manual'][first_item : last_item] = result_snr[3].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_4_max_manual'][first_item : last_item] = result_snr[4].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_5_max_manual'][first_item : last_item] = result_snr[5].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_6_max_manual'][first_item : last_item] = result_snr[6].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_7_max_manual'][first_item : last_item] = result_snr[7].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_8_max_manual'][first_item : last_item] = result_snr[8].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_9_max_manual'][first_item : last_item] = result_snr[9].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_10_max_manual'][first_item : last_item] = result_snr[10].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_11_max_manual'][first_item : last_item] = result_snr[11].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_12_max_manual'][first_item : last_item] = result_snr[12].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_13_max_manual'][first_item : last_item] = result_snr[13].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_14_max_manual'][first_item : last_item] = result_snr[14].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_15_max_manual'][first_item : last_item] = result_snr[15].detach().cpu().numpy().squeeze()


        results_dict[file_number]['channel_0_max'][first_item : last_item] = true_snr[:, 0].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_1_max'][first_item : last_item] = true_snr[:, 1].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_2_max'][first_item : last_item] = true_snr[:, 2].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_3_max'][first_item : last_item] = true_snr[:, 3].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_4_max'][first_item : last_item] = true_snr[:, 4].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_5_max'][first_item : last_item] = true_snr[:, 5].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_6_max'][first_item : last_item] = true_snr[:, 6].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_7_max'][first_item : last_item] = true_snr[:, 7].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_8_max'][first_item : last_item] = true_snr[:, 8].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_9_max'][first_item : last_item] = true_snr[:, 9].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_10_max'][first_item : last_item] = true_snr[:, 10].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_11_max'][first_item : last_item] = true_snr[:, 11].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_12_max'][first_item : last_item] = true_snr[:, 12].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_13_max'][first_item : last_item] = true_snr[:, 13].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_14_max'][first_item : last_item] = true_snr[:, 14].detach().cpu().numpy().squeeze()
        results_dict[file_number]['channel_15_max'][first_item : last_item] = true_snr[:, 15].detach().cpu().numpy().squeeze()


        #---------apply the model to the data
        conv_out = model.forward(x_test)
        conv_out_repeated = torch.repeat_interleave(conv_out, samplesize, dim=0)

        #---------flavor prediction
        flavor_pred = model.classifier(conv_out).squeeze()
        results_dict[file_number]['pred_flavor'][first_item : last_item] = flavor_pred.detach().cpu().numpy().squeeze()

        #---------energy and vertex prediction
        moments_dict_energy_vertex = model.pdf_energy_vertex.marginal_moments(conditional_input=conv_out.to(torch.double).to(device), calc_kl_diff_and_entropic_quantities=True)
        coverage_dict_energy = model.pdf_energy_vertex.coverage_and_or_pdf_scan(labels = true_energy_vertex.to(torch.double).to(device).squeeze(), conditional_input=conv_out.to(torch.double).to(device))

        samples_e_v, _, sample_log_pdf, _ = model.pdf_energy_vertex.sample(conditional_input=conv_out_repeated.to(torch.double).to(device), device=device, force_embedding_coordinates=True)
        samples_e_v_reshaped = torch.reshape(samples_e_v, (args.test_batchSize, samplesize, 4))
        sample_log_pdf_reshape = torch.reshape(sample_log_pdf, (args.test_batchSize, samplesize))


        #plt.scatter(samples_e_v_reshaped[0,:,0].cpu(), sample_log_pdf_reshape[0].cpu(), s=1)
        #plt.xlim(16.5, 18.5)
        #plt.ylim(5, 10)
        #plt.savefig('check_event_energy_spectrum.png')
        #quit()


        top_indices = torch.argmax(sample_log_pdf_reshape, dim=1)
        top_ev_values = samples_e_v_reshaped[torch.arange(samples_e_v_reshaped.shape[0]), top_indices]

        results_dict[file_number]['pred_energy_mode'][first_item : last_item] = top_ev_values[:, 0].cpu()
        results_dict[file_number]['pred_xx_mode'][first_item : last_item] = top_ev_values[:, 1].cpu()
        results_dict[file_number]['pred_yy_mode'][first_item : last_item] = top_ev_values[:, 2].cpu()
        results_dict[file_number]['pred_zz_mode'][first_item : last_item] = top_ev_values[:, 3].cpu()


        results_dict[file_number]['samples_energy'][first_item : last_item, :] = samples_e_v_reshaped[:, :samplesize_goodness, 0].cpu()
        results_dict[file_number]['samples_xx'][first_item : last_item, :] = samples_e_v_reshaped[:, :samplesize_goodness, 1].cpu()
        results_dict[file_number]['samples_yy'][first_item : last_item, :] = samples_e_v_reshaped[:, :samplesize_goodness, 2].cpu()
        results_dict[file_number]['samples_zz'][first_item : last_item, :] = samples_e_v_reshaped[:, :samplesize_goodness, 3].cpu()
        
        pred_energy = moments_dict_energy_vertex['mean_0'][:, 0].squeeze()
        pred_energy_std = moments_dict_energy_vertex['varlike_0'][:, 0, 0].squeeze()
        pred_xx = moments_dict_energy_vertex['mean_0'][:, 1].squeeze()
        pred_xx_std = moments_dict_energy_vertex['varlike_0'][:, 1, 1].squeeze()
        pred_yy = moments_dict_energy_vertex['mean_0'][:, 2].squeeze()
        pred_yy_std = moments_dict_energy_vertex['varlike_0'][:, 2, 2].squeeze()
        pred_zz = moments_dict_energy_vertex['mean_0'][:, 3].squeeze()
        pred_zz_std = moments_dict_energy_vertex['varlike_0'][:, 3, 3].squeeze()
        cov_energy_approx = coverage_dict_energy['approx_cov_values'].squeeze()

        results_dict[file_number]['pred_energy'][first_item : last_item] = moments_dict_energy_vertex['mean_0'][:, 0].squeeze()
        results_dict[file_number]['pred_energy_var'][first_item : last_item] = moments_dict_energy_vertex['varlike_0'][:, 0, 0].squeeze()

        results_dict[file_number]['pred_xx'][first_item : last_item] = moments_dict_energy_vertex['mean_0'][:, 1].squeeze()
        results_dict[file_number]['pred_xx_var'][first_item : last_item] = moments_dict_energy_vertex['varlike_0'][:, 1, 1].squeeze()

        results_dict[file_number]['pred_yy'][first_item : last_item] = moments_dict_energy_vertex['mean_0'][:, 2].squeeze()
        results_dict[file_number]['pred_yy_var'][first_item : last_item] = moments_dict_energy_vertex['varlike_0'][:, 2, 2].squeeze()

        results_dict[file_number]['pred_zz'][first_item : last_item] = moments_dict_energy_vertex['mean_0'][:, 3].squeeze()
        results_dict[file_number]['pred_zz_var'][first_item : last_item] = moments_dict_energy_vertex['varlike_0'][:, 3, 3].squeeze()

        results_dict[file_number]['pred_EV_cov_mat'][first_item : last_item] = moments_dict_energy_vertex['varlike_0'].squeeze()

        results_dict[file_number]['cov_energy_approx'][first_item : last_item] = coverage_dict_energy['approx_cov_values'].squeeze()

        #---------direction prediction
        moments_dict_direction = model.pdf_direction.marginal_moments(conditional_input=conv_out.to(torch.double).to(device), calc_kl_diff_and_entropic_quantities=True)
        coverage_dict_direction = model.pdf_direction.coverage_and_or_pdf_scan(labels = true_direction.to(torch.double).to(device), conditional_input=conv_out.to(torch.double).to(device), exact_coverage_calculation=True, save_pdf_scan=True, calculate_MAP=True)

        samples_d, _, _, _ = model.pdf_direction.sample(conditional_input=conv_out_repeated.to(torch.double).to(device), device=device, force_embedding_coordinates=False)
        samples_d_reshaped = torch.reshape(samples_d, (args.test_batchSize, samplesize, 2))

        results_dict[file_number]['samples_zenith'][first_item : last_item, :] = samples_d_reshaped[:, :samplesize_goodness, 0].cpu()
        results_dict[file_number]['samples_azimuth'][first_item : last_item, :] = samples_d_reshaped[:, :samplesize_goodness, 1].cpu()

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
        results_dict[file_number]['pred_mean_zenith'][first_item : last_item] = moments_dict_direction['mean_0_angles'][:, 0].squeeze()
        results_dict[file_number]['pred_mean_azimuth'][first_item : last_item] = moments_dict_direction['mean_0_angles'][:, 1].squeeze()
        results_dict[file_number]['pred_max_zenith'][first_item : last_item] = moments_dict_direction['argmax_0_angles'][:, 0].squeeze()
        results_dict[file_number]['pred_max_azimuth'][first_item : last_item] = moments_dict_direction['argmax_0_angles'][:, 1].squeeze()
        results_dict[file_number]['pred_area_68'][first_item : last_item] = areas_68_list
        results_dict[file_number]['pred_area_50'][first_item : last_item] = areas_50_list
        results_dict[file_number]['pred_energy_68'][first_item : last_item] = energy_68_list
        results_dict[file_number]['pred_energy_max'][first_item : last_item] = pred_energy_max_list
        results_dict[file_number]['cov_energy_exact_sampled'][first_item : last_item] = energy_exact_coverage_list
        results_dict[file_number]['cov_direction_approx'][first_item : last_item] = coverage_dict_direction['approx_cov_values'].squeeze()
        results_dict[file_number]['cov_direction_exact'][first_item : last_item] = coverage_dict_direction['real_cov_values'].squeeze()
        results_dict[file_number]['pred_direction_entropy'][first_item : last_item] = moments_dict_direction['approx_entropy_0'].squeeze()
        results_dict[file_number]['pred_kl_diff'][first_item : last_item] = moments_dict_direction['kl_diff_exact_approx_0'].squeeze()
        results_dict[file_number]['pred_kl_diff_inv'][first_item : last_item] = moments_dict_direction['kl_diff_approx_exact_0'].squeeze()


        for perc in np.arange(1, 101, 1):
            results_dict[file_number]['pred_area_percentage'][perc][first_item : last_item] = area_percentages[perc]
            results_dict[file_number]['pred_energy_percentage'][perc][first_item : last_item] = energy_percentages[perc]

        lll += 1
        
        if lll == 600:

            for file in range(hyperparameters.n_files_test):
                np.save(save_path + args.data + '_pre_results_' + args.epoch + '_' +str(file) + '.npy', results_dict[file])

for file in range(hyperparameters.n_files_test):
    np.save(save_path + args.data + '_results_' + args.epoch + '_' +str(file) + '.npy', results_dict[file])

