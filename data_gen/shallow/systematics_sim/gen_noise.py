from NuRadioReco.utilities import units, noise
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.detector import detector
from datetime import datetime
import numpy as np
from matplotlib import pylab as plt
from scipy import constants

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

kwargs = dict(json_filename="Gen2_shallow.json", antenna_by_depth=False)
det = detector.Detector(**kwargs)
det.update(datetime.now())

number_of_samples = 512
sampling_rate = 2.4 * units.GHz
     
time_coin = 5 * units.ns
n_majority = 2
time_coin_majority = 40 * units.ns
n_channels = 4
trigger_time =  75 * units.ns     


test_freq = np.fft.rfftfreq(number_of_samples, d=(1. / sampling_rate))

filter_trig_low = channelBandPassFilter.get_filter(test_freq, 0, 0, det, passband=[1 * units.MHz, 0.15],
                                filter_type='butter', order=10)

filter_high = channelBandPassFilter.get_filter(test_freq, 0, 0, det, passband=[0.08, 800 * units.GHz],
                                filter_type='butter', order=5)

filter_low = channelBandPassFilter.get_filter(test_freq, 0, 0, det, passband=[1 * units.MHz, 1000 * units.MHz],
                                filter_type='butter', order=10)

filter_total = filter_high * filter_low
filter_trig_total = filter_high * filter_low * filter_trig_low


# add thermal noise of fixed noise temperature
# add thermal noise of fixed noise temperature
Tnoise = 300  # in Kelvin

# calculate Vrms and normalize such that after filtering the correct Vrms is obtained
min_freq = 0
max_freq = 0.5 * sampling_rate

bandwidth = np.trapz(np.abs(filter_trig_total) ** 2, test_freq)
#bandwidth = np.trapz(np.abs(filter_trig_total) ** 2, test_freq)
vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5
#amplitude = Vrms / (bandwidth / max_freq) ** 0.5

bandwidth = np.trapz(np.abs(filter_total) ** 2, test_freq)
vrms_full = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5

print(vrms)
print(vrms_full)
#vrms = 0.00001382 * units.volt
threshold = 3.954 * vrms  
#threshold = 5 * vrms         


filt = filter_trig_total
noise_type="rayleigh"
keep_full_band=True



noise_class = noise.thermalNoiseGenerator(number_of_samples, 
                                            sampling_rate, 
                                            vrms, 
                                            threshold,
                                            time_coin, 
                                            n_majority,
                                            time_coin_majority, 
                                            n_channels, 
                                            trigger_time,
                                            filt,
                                            noise_type=noise_type,
                                            keep_full_band=keep_full_band
                                            )




full_data = np.zeros((1000, 5, 512))
event_counter = 0

pure_noise_data = np.load('~/shallow_station/syst_resim/npy_files_5k/shallow_syst_pure_noise.npy')

while event_counter < 1000:
        
    event = noise_class.generate_noise()

    print(event)
    print(event.shape)
    filtered_event = np.fft.irfft(np.fft.rfft(event, axis=-1) * filter_total, axis=-1)

    full_data[event_counter] = np.concatenate((filtered_event, pure_noise_data[event_counter, -1, :].reshape(1, 512)))
    event_counter += 1
    print(event_counter)
    #print(full_data)

np.save('final_data/shallow_syst_thermal_3_9.npy', full_data)





labels_dict = {}
labels_dict['azimuths'] = [np.pi] * 1000
labels_dict['energies'] = [18] * 1000
labels_dict['event_group_ids'] = [0] * 1000
labels_dict['flavors'] = [0] * 1000
labels_dict['inelasticity'] = [1] * 1000
labels_dict['interaction_type'] = [0] * 1000
labels_dict['n_interaction'] = [0] * 1000
labels_dict['shower_energy'] = [18] * 1000
labels_dict['vertex_times'] = [0] * 1000
labels_dict['weights'] = [0] * 1000
labels_dict['xx'] = [100] * 1000
labels_dict['yy'] = [100] * 1000
labels_dict['zeniths'] = [np.pi] * 1000
labels_dict['zz'] = [100] * 1000
labels_dict['event_group_ids'] = [0] * 1000
labels_dict['event_ids'] = [0] * 1000
labels_dict['event_run_numbers'] = [0] * 1000
labels_dict['event_file_name'] = [0] * 1000

labels_dict['tr_times'] = [0] * 1000
labels_dict['sig_times'] = [0] * 1000

labels_dict['channel_0_max'] = [0] * 1000
labels_dict['channel_1_max'] = [0] * 1000
labels_dict['channel_2_max'] = [0] * 1000
labels_dict['channel_3_max'] = [0] * 1000
labels_dict['channel_4_max'] = [0] * 1000


np.save('final_data/shallow_syst_thermal_3_9_labels.npy', labels_dict)



fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

for i in range(4):
    axs[i].plot(filtered_event[i])
    
    

axs[-1].set_xlabel('Time samples')

plt.savefig('noise_example.png', dpi=300, bbox_inches='tight')


print('SNR_event', np.max(abs(filtered_event))/vrms_full)