from NuRadioReco.utilities import units, noise
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.detector import detector
from datetime import datetime
import numpy as np
from matplotlib import pylab as plt
from scipy import constants
import logging
import argparse


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')


#parser.add_argument("--SNR", type=float, default=5.0, help="SNR threshold for trigger")
parser.add_argument("--theshold", type=float, default= 30.68, help=" ")
parser.add_argument("--number_of_events", type=int, default=1000, help=" ")
args = parser.parse_args()

args.theshold

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

kwargs = dict(json_filename="Gen2_deep.json", antenna_by_depth=False)
det = detector.Detector(**kwargs)
det.update(datetime.now())

number_of_samples = 2046
sampling_rate = 2.4 * units.GHz

sampling_rate_intern = 4.8 * units.GHz
     
number_of_samples_intern = 4096
test_freq = np.fft.rfftfreq(number_of_samples, d=(1. / sampling_rate))
test_freq_intern = np.fft.rfftfreq(number_of_samples_intern, d=(1. / sampling_rate_intern))

#print(test_freq_intern)
#print(test_freq_intern.shape)
#quit()

filter_trig_low_intern = channelBandPassFilter.get_filter(test_freq_intern, 0, 0, det, passband=[0 * units.MHz, 220 * units.MHz],
                                filter_type='cheby1', order=7, rp=0.1)

filter_high_intern = channelBandPassFilter.get_filter(test_freq_intern, 0, 0, det, passband=[96 * units.MHz, 100 * units.GHz],
                                filter_type='cheby1', order=4, rp=0.1)

filter_low_intern = channelBandPassFilter.get_filter(test_freq_intern, 0, 0, det, passband=[0 * units.MHz, 1000 * units.MHz],
                                filter_type='cheby1', order=7, rp=0.1)

filter_total_intern = filter_high_intern * filter_low_intern
filter_trig_total_intern = filter_high_intern * filter_low_intern * filter_trig_low_intern

#filter_trig_low_output = channelBandPassFilter.get_filter(test_freq, 0, 0, det, passband=[0 * units.MHz, 220 * units.MHz],
#                                filter_type='cheby1', order=7, rp=0.1)

filter_high_output = channelBandPassFilter.get_filter(test_freq, 0, 0, det, passband=[96 * units.MHz, 100 * units.GHz],
                                filter_type='cheby1', order=4, rp=0.1)

filter_low_output = channelBandPassFilter.get_filter(test_freq, 0, 0, det, passband=[0 * units.MHz, 1000 * units.MHz],
                                filter_type='cheby1', order=7, rp=0.1)

filter_total_output = filter_high_output * filter_low_output
#filter_trig_total_output = filter_high_output * filter_low_output * filter_trig_low_output



# add thermal noise of fixed noise temperature

#vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5

#bandwidth = np.trapz(np.abs(filter_trig_total) ** 2, test_freq)
#vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5

Tnoise = 300  
bandwidth_trig = np.trapz(np.abs(filter_trig_total_intern) ** 2, test_freq_intern)
vrms_full_trig = (Tnoise * 50 * constants.k * bandwidth_trig / units.Hz) ** 0.5


#vrms = 0.00001382 * units.volt

#threshold = 5 * vrms         


#filt = filter_trig_total
noise_type="rayleigh"

detector_filename = "Gen2_deep.json"
station_id = 1001
triggered_channels = np.array([0, 1, 2, 3])

bandwidth = np.trapz(np.abs(filter_trig_total_intern) ** 2, test_freq_intern)
vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5
threshold = args.theshold * np.power(vrms, 2.0) 
#threshold = 9.71333762327373e-10
#threshold = 2.5e-9
ref_index = 1.75
noise_type="rayleigh"
log_level=logging.NOTSET
pre_trigger_time = 200 * units.ns
trace_length = 852.5 * units.ns


noise_class = noise.thermalNoiseGeneratorPhasedArray(detector_filename, 
                                                station_id, 
                                                triggered_channels,
                                                vrms, 
                                                threshold, 
                                                ref_index,
                                                noise_type="rayleigh", 
                                                log_level=logging.NOTSET,
                                                pre_trigger_time=pre_trigger_time, 
                                                trace_length=trace_length, 
                                                filt=filter_trig_total_intern,
                                                upsampling=2, 
                                                window_length=16 * units.ns, 
                                                step_size=8 * units.ns,
                                                main_low_angle=np.deg2rad(-59.54968597864437),
                                                main_high_angle=np.deg2rad(59.54968597864437),
                                                n_beams=11, 
                                                quantize=True, 
                                                keep_full_band=True)





full_data = np.zeros((args.number_of_events, 16, 2046))
event_counter = 0

pure_noise_data = np.load('~/single_station_baseline/syst_resim/npy_files_short/deep_syst_pure_noise.npy')

print('start')
while event_counter < args.number_of_events:
        
    #event, index1, index2 = noise_class.generate_noise()
    event = noise_class.generate_noise()

    filtered_event = np.fft.irfft(np.fft.rfft(event, axis=-1) * filter_total_output, axis=-1)
    full_data[event_counter] = np.concatenate((event, pure_noise_data[event_counter, -12:, :]))
    event_counter += 1
    print(event_counter)


np.save('final_data/deep_syst_thermal_30_68.npy', full_data)





labels_dict = {}
labels_dict['azimuths'] = [np.pi] * args.number_of_events
labels_dict['energies'] = [18] * args.number_of_events
labels_dict['event_group_ids'] = [0] * args.number_of_events
labels_dict['flavors'] = [0] * args.number_of_events
labels_dict['inelasticity'] = [1] * args.number_of_events
labels_dict['interaction_type'] = [0] * args.number_of_events
labels_dict['n_interaction'] = [0] * args.number_of_events
labels_dict['shower_energy'] = [18] * args.number_of_events
labels_dict['vertex_times'] = [0] * args.number_of_events
labels_dict['weights'] = [0] * args.number_of_events
labels_dict['xx'] = [100] * args.number_of_events
labels_dict['yy'] = [100] * args.number_of_events
labels_dict['zeniths'] = [np.pi] * args.number_of_events
labels_dict['zz'] = [100] * args.number_of_events
labels_dict['event_group_ids'] = [0] * args.number_of_events
labels_dict['event_ids'] = [0] * args.number_of_events
labels_dict['event_run_numbers'] = [0] * args.number_of_events
labels_dict['event_file_name'] = [0] * args.number_of_events

labels_dict['tr_times'] = [0] * args.number_of_events
labels_dict['sig_times'] = [0] * args.number_of_events

labels_dict['channel_0_max'] = [0] * args.number_of_events
labels_dict['channel_1_max'] = [0] * args.number_of_events
labels_dict['channel_2_max'] = [0] * args.number_of_events
labels_dict['channel_3_max'] = [0] * args.number_of_events
labels_dict['channel_4_max'] = [0] * args.number_of_events
labels_dict['channel_5_max'] = [0] * args.number_of_events
labels_dict['channel_6_max'] = [0] * args.number_of_events
labels_dict['channel_7_max'] = [0] * args.number_of_events
labels_dict['channel_8_max'] = [0] * args.number_of_events
labels_dict['channel_9_max'] = [0] * args.number_of_events
labels_dict['channel_10_max'] = [0] * args.number_of_events
labels_dict['channel_11_max'] = [0] * args.number_of_events
labels_dict['channel_12_max'] = [0] * args.number_of_events
labels_dict['channel_13_max'] = [0] * args.number_of_events
labels_dict['channel_14_max'] = [0] * args.number_of_events
labels_dict['channel_15_max'] = [0] * args.number_of_events


np.save('final_data/deep_syst_thermal_30_68_labels.npy', labels_dict)



fig, axs = plt.subplots(16, 1, figsize=(8, 15), sharex=True)

for i in range(16):
    axs[i].plot(full_data[0, i])
    
    

axs[-1].set_xlabel('Time samples')

plt.savefig('noise_example.png', dpi=300, bbox_inches='tight')


#print('SNR_event', np.max(abs(filtered_event))/vrms_full)