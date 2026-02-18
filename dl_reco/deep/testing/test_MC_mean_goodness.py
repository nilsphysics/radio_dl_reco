import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy.stats import chi2
from multiprocessing import Pool
import time
from scipy import constants

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
from NuRadioReco.utilities import signal_processing
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
parser.add_argument("epoch", type=str, default='0', help="Name a new Run to be saved")
parser.add_argument("file", type=int, default=0, help="Name a new Run to be saved")
parser.add_argument("--sample", type=int, default=0, help="Name a new Run to be saved")
parser.add_argument("--file_size", type=int, default=10000, help="Name a new Run to be saved")
parser.add_argument("--number_of_samples", type=int, default=100, help="Name a new Run to be saved")
parser.add_argument("--plot_event", type=int, default=0, help="Name a new Run to be saved")

args = parser.parse_args()


#---------loading the correct checkpoint
result_dir = '~/gen2_deep_reco/reconstruction/paper_prep/combined/results/' + args.Run_number + '/full_y_pred/'
model_file = 'test_results_' + args.epoch + '_' + str(args.file) + '.npy' 
data_dir = '~/data/gen2_deep_snr/nc_cc/testing'


data = np.load(os.path.join(data_dir, f"{'deep_snr_nccc_testing_10k_'}{args.file}_.npy"), allow_pickle=True)
labels_tmp = np.load(result_dir + model_file, allow_pickle=True)

result_dict = labels_tmp.item()

goodness_dict = {}

if args.plot_event:
    plot_energy = 19.76087188720703
    plot_zenith = 1.1867822408676147
plot_id = 5

def get_goodness_of_fit(data_trace, n_channels, n_samples, sampling_rate, shower_list, station_id, detector_file, propagator_dicr, ice_model, config_file, time_logger_numc):
    """
    ---------function to calculate the goodness of fit for a specific event
    input:
        data_trace, 
        n_channels, 
        n_samples, 
        sampling_rate, 
        shower_list, 
        station_id, 
        detector_file, 
        propagator_dicr, 
        ice_model, 
        config_file, 
        time_logger_numc

    output:
        compare_signal_final    - corrected resimulated event
        dll_comp                - dll between the data and the template
        corr_score              - correlation between the data and the template
    """

    array_data = np.zeros((n_channels, n_samples))
    array_raw = np.zeros((n_channels, n_samples))

    trace_raw = np.zeros((n_channels, 2, n_samples))
    time_raw = np.zeros((n_channels, 2, n_samples))

    trace_raw_added = {}
    time_raw_added = {}

    signal_channels = []


    if (args.plot_event == 1) &  (shower_list[0][shp.charge_excess_profile_id] == plot_id):

        fig, axes = plt.subplots(16, 1, sharex=True, figsize=(5, 10))  # 5 rows, 1 column
        fig.subplots_adjust(hspace=0.0)

        for channel in range(n_channels):

            sim_station_raw = sim.calculate_sim_efield(shower_list, station_id, channel,
                                    detector_file, propagator_dicr, ice_model, config_file,
                                    time_logger=time_logger_numc)

            if len(sim_station_raw.get_electric_fields()) == 0:
                continue

            sim.apply_det_response_sim(sim_station_raw, detector_file, config_file, _detector_simulation_filter_amp)

            for i, ray_solution in enumerate(sim_station_raw.iter_channels()):
                ray_solution.resample(sampling_rate)
                axes[channel].plot(ray_solution.get_times(), ray_solution.get_trace()*1000000)
        axes[2].set_ylabel(r'Voltage [$\mu$V]')
        axes[4].set_xlabel('relative time [ns]')
        plt.savefig('test_event_split.png')
        print('saved split fig')


    for channel in range(n_channels):
        #---------for every channel get the two ray tracing solutions

        #---------simulate e-fields
        sim_station_raw = sim.calculate_sim_efield(shower_list, station_id, channel,
                                detector_file, propagator_dicr, ice_model, config_file,
                                time_logger=time_logger_numc)

        if len(sim_station_raw.get_electric_fields()) == 0:
            continue

        sim.apply_det_response_sim(sim_station_raw, detector_file, config_file, _detector_simulation_filter_amp)

        signal_channels.append(channel)

        channel_trace_0 = BaseTrace()
        channel_trace_1 = BaseTrace()
        
        for i, ray_solution in enumerate(sim_station_raw.iter_channels()):
            #---------for ray tracing solution get the trace and the starting time

            ray_solution.resample(sampling_rate)
            trace_raw[channel, i, :] = ray_solution.get_trace()
            time_raw[channel, i, :] = ray_solution.get_times()

    for channel in signal_channels:
        #---------add the two ray tracing solutions with the correct timing
        channel_trace_0.set_trace_start_time(time_raw[channel, 0, 0])
        channel_trace_1.set_trace_start_time(time_raw[channel, 1, 0])

        channel_trace_0.set_trace(trace_raw[channel, 0, :], sampling_rate=sampling_rate)
        channel_trace_1.set_trace(trace_raw[channel, 1, :], sampling_rate=sampling_rate)

        trace_base = channel_trace_0.__add__(channel_trace_1)

        trace_raw_added[channel] = trace_base.get_trace()
        time_raw_added[channel] = trace_base.get_times()

    #---------create one large time window for all channels to fit in
    max_value_raw = max(i for v in time_raw_added.values() for i in v)
    min_value_raw = min(i for v in time_raw_added.values() for i in v)

    number_of_samples = math.ceil((max_value_raw - min_value_raw) * sampling_rate)

    if number_of_samples % 2 == 0:
        number_of_samples = number_of_samples
    else:
        number_of_samples += 1 # Odd

    new_max_raw = min_value_raw + number_of_samples / sampling_rate
    time_window_raw = np.linspace(min_value_raw, new_max_raw, number_of_samples)
    shifted_trace_raw = np.zeros([n_channels, number_of_samples])
    
    #---------shifting traces accorting to their respective start time
    for channel in signal_channels:
        shifted_channel = BaseTrace()

        trace_to_shift = trace_raw_added[channel]
        times_to_shift = time_raw_added[channel]

        trace_to_shift = np.pad(trace_to_shift, (0, number_of_samples - len(trace_to_shift)))

        shifted_channel.set_trace(trace_to_shift, sampling_rate)
        shifted_channel.set_trace_start_time(min_value_raw)

        shifted_channel.apply_time_shift(min(times_to_shift) - min_value_raw, silent=True)
        shifted_trace_raw[channel, :] = shifted_channel.get_trace()

    #---------find correlation score between template and data trace
    corr_score, start_index, end_index, lag = cross_correlate_rough(data_trace, shifted_trace_raw, number_of_channels)

    #---------pad templates that are cut too short and cut template according to the correlation
    if end_index > shifted_trace_raw.shape[1]:
        compare_signal = np.concatenate((shifted_trace_raw[:, start_index:end_index], np.zeros((number_of_channels, end_index - shifted_trace_raw.shape[1]))), axis=1)
    elif lag < 0:
        compare_signal = np.concatenate((np.zeros((number_of_channels, abs(lag))), shifted_trace_raw[:, start_index:end_index]), axis=1)
    else:
        compare_signal = shifted_trace_raw[:, start_index:end_index]

    #---------find small time shift correction due to binning
    check_times = np.linspace(0, 0.5, 21) - 0.25 # 0.5 is slightly larger than 1/sampling rate so we are scanning the time between bins
    total_time_shift, corr_score, output_per_event = cross_correlate_fine(data_trace, compare_signal, check_times)

    #---------apply small time shift correction due to binning
    compare_signal_final = np.zeros([n_channels, n_samples])

    for channel in range(n_channels):

        final_pulse = BaseTrace()

        final_pulse.set_trace(compare_signal[channel], sampling_rate)
        final_pulse.apply_time_shift(total_time_shift, silent=True)
        final_trace = final_pulse.get_trace()

        compare_signal_final[channel] = final_trace

    #---------calculate delta log likelihood between template and data
    dll_comp = nois_class.calculate_minus_two_delta_llh(data_trace, signal=compare_signal_final, frequency_domain=False)

    return compare_signal_final, dll_comp, corr_score

#---------define simulation properties for NuRadioMC
time_logger = NuRadioMC.simulation.time_logger.timeLogger(logger)
ice = medium.get_ice_model('southpole_2015')
cfg = sim.get_config("~/data/sim_files/deep/config.yaml")
kwargs = dict(json_filename="~/data/sim_files/deep/detector_samples.json", antenna_by_depth=False)
det = detector.Detector(**kwargs)
det.update(datetime.now())

prop_config = {}
prop_config['propagation'] = {}
prop_config['propagation']['attenuate_ice'] = True
prop_config['propagation']['focusing_limit'] = 2
prop_config['propagation']['focusing'] = True
prop_config['propagation']['birefringence'] = False
propagator = propagation.get_propagation_module("analytic")(ice,
            attenuation_model="SP1",
            n_frequencies_integration=25, 
            n_reflections=0,
            config=prop_config,
            detector=det,)

sid = 1001
sampling_rate_global = 2.4 
samples_per_channel = 2046
number_of_channels = 16
profile_numbers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#profile_ids = np.array([0])
#samplesize = 10000 # samples for the goodness of fit test
samplesize = 100 # samples for the goodness of fit test

#---------define filter response for all antennas
passband_low = {}
passband_low_trigger = {}
passband_high = {}
filter_type = {}
order_low = {}
order_high = {}

PA_4ch_channels = [0, 1, 2, 3]
reco_channels = range(4, 16)

passband_low = {}
passband_low_trigger = {}
passband_high = {}
filter_type = {}
order_low = {}
order_high = {}
for channel_id in PA_4ch_channels:
    passband_low[channel_id] = [0 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [0 * units.MHz, 220 * units.MHz]
    passband_high[channel_id] = [96 * units.MHz, 100 * units.GHz]
    filter_type[channel_id] = 'cheby1'
    order_low[channel_id] = 7
    order_high[channel_id] = 4
    
for channel_id in reco_channels:
    passband_low[channel_id] = [0 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [0 * units.MHz, 220 * units.MHz]
    passband_high[channel_id] = [96 * units.MHz, 100 * units.GHz]
    filter_type[channel_id] = 'cheby1'
    order_low[channel_id] = 7
    order_high[channel_id] = 4

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

def _detector_simulation_filter_amp(evt, station, det):

    channelBandPassFilter.run(evt, station, det,
                                passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
    channelBandPassFilter.run(evt, station, det,
                                passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)
    channelBandPassFilter.run(evt, station, det, passband=[1 * units.MHz, 500 * units.MHz],
                                filter_type='butter', order=10)

test_freq = np.fft.rfftfreq(samples_per_channel, d=(1. / sampling_rate_global))



threshold_amp = 0.1
nois_class = dll_calc.NoiseModel(number_of_channels, samples_per_channel, sampling_rate_global, threshold_amplitude=threshold_amp)
noise_traces = np.load('~/data/gen2_deep_snr/syst_files/deep_syst_pure_noise.npy')

window = signal_processing.half_hann_window(samples_per_channel, half_percent=0.05)
window_integral = np.sum(np.ones(samples_per_channel) - window)
print('window_integral', window_integral)

filter_new = channelBandPassFilter.get_filter(test_freq, 0, 0, det, passband=[1 * units.MHz, 500 * units.MHz],
                                filter_type='butter', order=10)

plt.plot(test_freq, np.abs(filter_new))
plt.savefig('filter.png')

noise_traces = np.fft.irfft(np.fft.rfft(noise_traces*window, axis=-1) * filter_new, axis=-1)
nois_class.initialize_with_data(noise_traces, method="using_spectra")
#dll_noise = nois_class.calculate_minus_two_delta_llh(noise_traces[:1000, :, :])


matrix_rank = 0
for channel_i in range(number_of_channels):
    matrix_rank += np.linalg.matrix_rank(nois_class.cov_inv[channel_i])

print('matrix_rank', matrix_rank)
#quit()

"""
std_dev = np.sqrt(2 * matrix_rank)
x = np.linspace(matrix_rank - 5 * std_dev, matrix_rank + 5 * std_dev, 1000)
pdf1 = chi2.pdf(x, matrix_rank)

plt.hist(dll_noise, bins=50, alpha= 0.5, facecolor='tab:blue', edgecolor= 'k', density=True, label='DLL')
#plt.hist(data_goodness_cut[en_mask], bins=30, alpha= 0.5, facecolor='tab:blue', edgecolor= 'k', density=True, label='DLL')
plt.axvline(matrix_rank, color='k', linestyle='--', linewidth='2', label='rank')
plt.axvline(np.median(dll_noise), color='r', linestyle='--', linewidth='2', label='median')
plt.plot(x, pdf1, label=f'Chi-Squared (rank = ' + str(matrix_rank) + ')', color='tab:orange')
plt.xlabel('dll')
plt.ylabel('Events')
plt.savefig('dll_noise.png')

quit()
"""


def cross_correlate_rough(data, template, num_antennas):
    """
    --------- function that calculates the correlation between all bins of the template and the data
    input:
        data, 
        template
        num_antennas

    output:
        np.max(corr) - maximum correlation 
        start_idx - start index to cut
        end_idx - end index to cut
        max_lag
    """
    cross_corr_scores = np.zeros(data.shape[1] + template.shape[1] - 1)
    max_lags = []

    for i in range(num_antennas):
        corr = correlate(template[i], data[i], mode='full', method='fft')
        cross_corr_scores += corr 

    max_lag = np.argmax(cross_corr_scores) - (data.shape[1] - 1)

    if max_lag >= 0:
        start_idx = max_lag
        end_idx = start_idx + data.shape[1]
    else:
        start_idx = 0
        end_idx = data.shape[1] + max_lag

    return np.max(corr), start_idx, end_idx, max_lag

def cross_correlate_fine(data, template, t_array):
    """
    --------- function that calculates the correlation for small time shifts between template and data
    input:
        data, 
        template
        t_array

    output:
        t_array[np.argmax(output)], 
        np.max(output), 
        output
    """
    shifting_pulse = BaseTrace()
    shifting_pulse.set_trace(template, sampling_rate_global)

    output = np.zeros(len(t_array))
    for i in range(len(t_array)):
        shifting_pulse.set_trace(template, sampling_rate_global)
        shifting_pulse.apply_time_shift(t_array[i], silent=True)
        shifted_trace = shifting_pulse.get_trace()
        output[i] = nois_class.calculate_minus_two_delta_llh(data, signal=shifted_trace, frequency_domain=False)
        #output[i] = np.correlate(data.flatten(), shifted_trace.flatten())

    #return t_array[np.argmax(output)], np.max(output), output
    return t_array[np.argmin(output)], np.min(output), output

"""
def matched_filter(data, template, noise_power_spectral_density, t_array, frequencies, n_antennas):
    #---------currently not used, theoretically the better option to the correlation score
    s = np.fft.rfft(data, axis=1).flatten()
    h = np.fft.rfft(template, axis=1).flatten()
    noise_psd = noise_power_spectral_density.flatten()
    frequencies_flattened = np.tile(frequencies, n_antennas)
    n_t = len(t_array)
    output = np.zeros(n_t)

    for i in range(n_t):
        integrand = s * h.conj() / noise_psd * np.exp(1j*2*np.pi*frequencies_flattened*t_array[i])
        output[i] = 4 * np.real(np.sum(integrand[noise_psd>np.max(noise_psd)/10000] * (frequencies_flattened[1] - frequencies_flattened[0])))

    return t_array[np.argmax(output)], np.max(output), output, np.argmax(output)
"""

def goodness_with_showers(data, profile_ids, energy, xx, yy, zz, zenith, azimuth, with_plot=0):

    goodness_of_fit_list = []
    compare_signal_dict = {}

    #---------loop through shower realizations
    for profile_id in profile_ids:

        #---------create true shower list with MC truth
        showers = []
        shower = NuRadioReco.framework.radio_shower.RadioShower(0)
        shower[shp.zenith] = zenith * units.rad
        shower[shp.azimuth] = azimuth * units.rad
        shower[shp.energy] = 10**energy * units.eV
        shower[shp.vertex] = np.array([xx*units.m, yy*units.m, zz*units.m])
        shower[shp.type] = 'had'
        shower[shp.charge_excess_profile_id] = profile_id
        showers.append(shower)

        #_, goodness_of_fit, _ = get_goodness_of_fit(data, number_of_channels, samples_per_channel, sampling_rate_global, showers, sid, det, propagator, ice, cfg, time_logger)

        try:
            compare_signal_id, goodness_of_fit, _ = get_goodness_of_fit(data, number_of_channels, samples_per_channel, sampling_rate_global, showers, sid, det, propagator, ice, cfg, time_logger)
            goodness_of_fit_list.append(goodness_of_fit[0])
            compare_signal_dict[int(profile_id)] = compare_signal_id

        except Exception as e:
            #print('couldnt find solution')
            print(f"An error occurred: {e}")
            goodness_of_fit_list.append(-2)
            compare_signal_dict[int(profile_id)] = np.zeros((number_of_channels, samples_per_channel))

    mask_dll = np.array(goodness_of_fit_list) == min(np.array(goodness_of_fit_list))
    max_prof_id = profile_ids[mask_dll][0]

    compare_signal_final = compare_signal_dict[int(max_prof_id)]
    goodness_of_fit_final = goodness_of_fit_list[max_prof_id]

    if with_plot:
        return goodness_of_fit_final, compare_signal_final, max_prof_id

    else:
        return goodness_of_fit_final


#---------looping through the data set
"""
#for idx, x in tqdm(enumerate(inputs), desc="Processing", ncols=80, total=len(inputs)):
for indx, event_data in tqdm(enumerate(data), desc="Processing", ncols=80, total=len(data)):

    true_flavor = result_dict['true_flavor'][indx]

    #---------only use NC events
    if true_flavor == 1:
        goodness_dict['goodness_MC'][indx] = -1
        goodness_dict['goodness_mean'][indx] = -1
        continue

    true_energy = result_dict['true_energy'][indx]
    true_xx = result_dict['true_xx'][indx]
    true_yy = result_dict['true_yy'][indx]
    true_zz = result_dict['true_zz'][indx]
    true_azimuth = result_dict['true_azimuth'][indx]
    true_zenith = result_dict['true_zenith'][indx]

    pred_energy = result_dict['pred_energy'][indx]
    pred_xx = result_dict['pred_xx'][indx]
    pred_yy = result_dict['pred_yy'][indx]
    pred_zz = result_dict['pred_zz'][indx]
    pred_azimuth = result_dict['pred_mean_azimuth'][indx]
    pred_zenith = result_dict['pred_mean_zenith'][indx]


    goodness_MC = goodness_with_showers(event_data, profile_numbers, true_energy, true_xx, true_yy, true_zz, true_zenith, true_azimuth)
    goodness_mean = goodness_with_showers(event_data, profile_numbers, pred_energy, pred_xx, pred_yy, pred_zz, pred_zenith, pred_azimuth)

    goodness_dict['goodness_MC'][indx] = goodness_MC
    goodness_dict['goodness_mean'][indx] = goodness_mean


np.save(save_path + args.data + '_goodness_MC_' + args.epoch + '_' + str(args.file) + '.npy', goodness_dict)
"""

def task_function(indx):

    true_flavor = result_dict['true_flavor'][indx]
    true_energy = result_dict['true_energy'][indx]
    true_zenith = result_dict['true_zenith'][indx]

    event_data = data[indx]
    event_data *= window
    event_data = np.fft.irfft(np.fft.rfft(event_data, axis=-1) * filter_new, axis=-1)

    #---------only use NC events
    if true_flavor == 1:
        goodness_MC = -1
        goodness_mean = -1
        goodness_mode = -1

    elif args.plot_event:

        if plot_zenith == true_zenith:

            mode_energy = result_dict['pred_energy_mode'][indx]
            mode_xx = result_dict['pred_xx_mode'][indx]
            mode_yy = result_dict['pred_yy_mode'][indx]
            mode_zz = result_dict['pred_zz_mode'][indx]
            mode_azimuth = result_dict['pred_max_azimuth'][indx]
            mode_zenith = result_dict['pred_max_zenith'][indx]

            print(true_zenith)
            print(true_energy)
            goodness_mode, signal_trace, max_id = goodness_with_showers(event_data, profile_numbers, mode_energy, mode_xx, mode_yy, mode_zz, mode_zenith, mode_azimuth, with_plot=1)

            save_dir = {}
            save_dir['deep'] = {}
            save_dir['deep']['signal'] = signal_trace
            save_dir['deep']['data'] = event_data
            save_dir['deep']['goodness_mode'] = goodness_mode
            np.save('/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/paper_prep/paper_plots/example_event_data/deep_example.npy', save_dir)


            fig, axes = plt.subplots(len(signal_trace), 1, sharex=True, figsize=(5, 10))  # 5 rows, 1 column
            fig.subplots_adjust(hspace=0.0)
            axes[2].set_ylabel(r'Voltage [$\mu$V]')
            axes[4].set_xlabel('relative time [ns]')
            for i in range(len(signal_trace)):
                print(i)

                
                axes[i].plot(event_data[i], color='tab:orange')
                axes[i].plot(signal_trace[i], color='tab:blue')

            #axes[0].set_title(str(np.round(goodness_mode, 0)))
            #plt.tight_layout()
            plt.savefig('test_event.png')

        else:

            goodness_MC = -1
            goodness_mean = -1
            goodness_mode = -1

    else:

        true_energy = result_dict['true_energy'][indx]
        true_xx = result_dict['true_xx'][indx]
        true_yy = result_dict['true_yy'][indx]
        true_zz = result_dict['true_zz'][indx]
        true_azimuth = result_dict['true_azimuth'][indx]
        true_zenith = result_dict['true_zenith'][indx]

        mean_energy = result_dict['pred_energy'][indx]
        mean_xx = result_dict['pred_xx'][indx]
        mean_yy = result_dict['pred_yy'][indx]
        mean_zz = result_dict['pred_zz'][indx]
        mean_azimuth = result_dict['pred_mean_azimuth'][indx]
        mean_zenith = result_dict['pred_mean_zenith'][indx]

        mode_energy = result_dict['pred_energy_mode'][indx]
        mode_xx = result_dict['pred_xx_mode'][indx]
        mode_yy = result_dict['pred_yy_mode'][indx]
        mode_zz = result_dict['pred_zz_mode'][indx]
        mode_azimuth = result_dict['pred_max_azimuth'][indx]
        mode_zenith = result_dict['pred_max_zenith'][indx]

        goodness_MC = goodness_with_showers(event_data, profile_numbers, true_energy, true_xx, true_yy, true_zz, true_zenith, true_azimuth)
        goodness_mean = goodness_with_showers(event_data, profile_numbers, mean_energy, mean_xx, mean_yy, mean_zz, mean_zenith, mean_azimuth)
        goodness_mode = goodness_with_showers(event_data, profile_numbers, mode_energy, mode_xx, mode_yy, mode_zz, mode_zenith, mode_azimuth)

        print('true_energy', true_energy)
        print('goodness_MC', goodness_MC)
        print('goodness_mean', goodness_mean)
        print('goodness_mode', goodness_mode)

    return indx, goodness_MC, goodness_mean, goodness_mode




def main_multiprocessing():
    #inputs = range(10)  # Example input values for the loop

    MC_array = np.zeros([args.file_size])  # Initialize the shared numpy array
    mean_array = np.zeros([args.file_size])
    mode_array = np.zeros([args.file_size])

    # Specify the number of CPUs (e.g., use 4 CPUs)
    num_cpus = 20  # You can change this number or use os.cpu_count() for all available CPUs

    # Create a pool of worker processes
    with Pool(processes=num_cpus) as pool:
        # Initialize tqdm progress bar
        with tqdm(total=2*len(data)) as pbar:
            # Define a callback function to update progress
            def update_progress(_):
                pbar.update()

            # Distribute tasks across worker processes
            results = [
                pool.apply_async(
                    task_function, 
                    args=(idx, ), 
                    callback=update_progress
                )
                for idx in range(len(data))
            ]

            collected_results = []
            for r in results:
                collected_results.append(r.get())  # Gather results from workers
                pbar.update()  # Update the progress bar after each task completes
                r.wait()


    for indx, goodness_MC, goodness_mean, goodness_mode in collected_results:
        MC_array[indx] = goodness_MC
        mean_array[indx] = goodness_mean
        mode_array[indx] = goodness_mode

        
    # Convert the numpy array to a regular list for display
    goodness_dict['goodness_MC'] = MC_array
    goodness_dict['goodness_mean'] = mean_array
    goodness_dict['goodness_mode'] = mode_array
    goodness_dict['matrix_rank'] = matrix_rank

    if args.plot_event ==0:
        np.save(result_dir + 'MC_goodness_' + args.epoch + '_' + str(args.file) + '.npy' , goodness_dict)

        print("\nResults saved")
    print('done')

if __name__ == "__main__":
    main_multiprocessing()