import os
import argparse
import numpy as np
import NuRadioReco.modules.io.NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.io.eventReader
import NuRadioReco.framework.parameters as parameters
import matplotlib.pyplot as plt
import collections


base_path = '~/shallow_station/npy_files/with_snr'
os.chdir(base_path)


nc_1600_data = np.load('shallow_baseline_nc_16.00eV.npy', mmap_mode='r')
nc_1620_data = np.load('shallow_baseline_nc_16.20eV.npy', mmap_mode='r')
nc_1640_data = np.load('shallow_baseline_nc_16.40eV.npy', mmap_mode='r')
nc_1660_data = np.load('shallow_baseline_nc_16.60eV.npy', mmap_mode='r')
nc_1680_data = np.load('shallow_baseline_nc_16.80eV.npy', mmap_mode='r')
nc_1700_data = np.load('shallow_baseline_nc_17.00eV.npy', mmap_mode='r')
nc_1720_data = np.load('shallow_baseline_nc_17.20eV.npy', mmap_mode='r')
nc_1740_data = np.load('shallow_baseline_nc_17.40eV.npy', mmap_mode='r')
nc_1760_data = np.load('shallow_baseline_nc_17.60eV.npy', mmap_mode='r')
nc_1780_data = np.load('shallow_baseline_nc_17.80eV.npy', mmap_mode='r')
nc_1800_data = np.load('shallow_baseline_nc_18.00eV.npy', mmap_mode='r')
nc_1820_data = np.load('shallow_baseline_nc_18.20eV.npy', mmap_mode='r')
nc_1840_data = np.load('shallow_baseline_nc_18.40eV.npy', mmap_mode='r')
nc_1860_data = np.load('shallow_baseline_nc_18.60eV.npy', mmap_mode='r')
nc_1880_data = np.load('shallow_baseline_nc_18.80eV.npy', mmap_mode='r')
nc_1900_data = np.load('shallow_baseline_nc_19.00eV.npy', mmap_mode='r')
nc_1920_data = np.load('shallow_baseline_nc_19.20eV.npy', mmap_mode='r')
nc_1940_data = np.load('shallow_baseline_nc_19.40eV.npy', mmap_mode='r')
nc_1960_data = np.load('shallow_baseline_nc_19.60eV.npy', mmap_mode='r')
nc_1980_data = np.load('shallow_baseline_nc_19.80eV.npy', mmap_mode='r')
nc_2000_data = np.load('shallow_baseline_nc_20.00eV.npy', mmap_mode='r')


nc_1600_label = np.load('shallow_baseline_nc_16.00eV_labels_array.npy', mmap_mode='r')
nc_1620_label = np.load('shallow_baseline_nc_16.20eV_labels_array.npy', mmap_mode='r')
nc_1640_label = np.load('shallow_baseline_nc_16.40eV_labels_array.npy', mmap_mode='r')
nc_1660_label = np.load('shallow_baseline_nc_16.60eV_labels_array.npy', mmap_mode='r')
nc_1680_label = np.load('shallow_baseline_nc_16.80eV_labels_array.npy', mmap_mode='r')
nc_1700_label = np.load('shallow_baseline_nc_17.00eV_labels_array.npy', mmap_mode='r')
nc_1720_label = np.load('shallow_baseline_nc_17.20eV_labels_array.npy', mmap_mode='r')
nc_1740_label = np.load('shallow_baseline_nc_17.40eV_labels_array.npy', mmap_mode='r')
nc_1760_label = np.load('shallow_baseline_nc_17.60eV_labels_array.npy', mmap_mode='r')
nc_1780_label = np.load('shallow_baseline_nc_17.80eV_labels_array.npy', mmap_mode='r')
nc_1800_label = np.load('shallow_baseline_nc_18.00eV_labels_array.npy', mmap_mode='r')
nc_1820_label = np.load('shallow_baseline_nc_18.20eV_labels_array.npy', mmap_mode='r')
nc_1840_label = np.load('shallow_baseline_nc_18.40eV_labels_array.npy', mmap_mode='r')
nc_1860_label = np.load('shallow_baseline_nc_18.60eV_labels_array.npy', mmap_mode='r')
nc_1880_label = np.load('shallow_baseline_nc_18.80eV_labels_array.npy', mmap_mode='r')
nc_1900_label = np.load('shallow_baseline_nc_19.00eV_labels_array.npy', mmap_mode='r')
nc_1920_label = np.load('shallow_baseline_nc_19.20eV_labels_array.npy', mmap_mode='r')
nc_1940_label = np.load('shallow_baseline_nc_19.40eV_labels_array.npy', mmap_mode='r')
nc_1960_label = np.load('shallow_baseline_nc_19.60eV_labels_array.npy', mmap_mode='r')
nc_1980_label = np.load('shallow_baseline_nc_19.80eV_labels_array.npy', mmap_mode='r')
nc_2000_label = np.load('shallow_baseline_nc_20.00eV_labels_array.npy', mmap_mode='r')


print('finished loading data')



label_list = [  nc_1600_label, 
                nc_1620_label, 
                nc_1640_label, 
                nc_1660_label, 
                nc_1680_label, 
                nc_1700_label, 
                nc_1720_label, 
                nc_1740_label, 
                nc_1760_label, 
                nc_1780_label, 
                nc_1800_label, 
                nc_1820_label, 
                nc_1840_label, 
                nc_1860_label,
                nc_1880_label, 
                nc_1900_label, 
                nc_1920_label, 
                nc_1940_label, 
                nc_1960_label, 
                nc_1980_label, 
                nc_2000_label
                                                
                ]

data_list = [  nc_1600_data,
                nc_1620_data, 
                nc_1640_data, 
                nc_1660_data, 
                nc_1680_data, 
                nc_1700_data, 
                nc_1720_data, 
                nc_1740_data, 
                nc_1760_data, 
                nc_1780_data,
                nc_1800_data, 
                nc_1820_data, 
                nc_1840_data, 
                nc_1860_data, 
                nc_1880_data, 
                nc_1900_data, 
                nc_1920_data, 
                nc_1940_data, 
                nc_1960_data, 
                nc_1980_data, 
                nc_2000_data
                                                
                ]

number_of_files_out = 105
number_of_events_per_file_in = 50000
number_of_events_per_file_out = 10000
number_of_files_in = len(data_list)

numbers = np.zeros(number_of_files_in)

print(numbers)

x = 0

for i in range(number_of_files_out):

    list_traces = []
    list_labels = []

    azimuths = []
    energies = []
    event_group_ids = []
    event_ids = []
    event_run_numbers = []
    event_file_name = []
    flavors = []
    inelasticity = []
    interaction_type = []
    n_interaction = []
    vertex_times = []
    weights = []
    xx = []
    yy = []
    zeniths = []
    zz = []
    shower_energy = []
    max_amp_0 = []
    max_amp_1 = []
    max_amp_2 = []
    max_amp_3 = []
    max_amp_4 = []
    

    for j in range(number_of_events_per_file_out):


        #x = np.random.randint(number_of_files_in)

        num = numbers[x]

        while num == number_of_events_per_file_in:
            print(numbers)
            #x = np.random.randint(number_of_files_in)
            
            num = numbers[x]

        #print(x)
        data_num = data_list[x]
        label_num = label_list[x]
        #print(file_num)
        list_traces.append(data_num[int(num)])
        list_labels.append(label_num[int(num)])

        azimuths.append(float(label_num[int(num)][0]))
        energies.append(float(label_num[int(num)][1]))
        event_group_ids.append(int(label_num[int(num)][2]))
        flavors.append(int(label_num[int(num)][3]))
        inelasticity.append(float(label_num[int(num)][4]))
        interaction_type.append(label_num[int(num)][5])
        n_interaction.append(int(label_num[int(num)][6]))
        shower_energy.append(float(label_num[int(num)][7]))
        vertex_times.append(float(label_num[int(num)][8]))
        weights.append(float(label_num[int(num)][9]))
        xx.append(float(label_num[int(num)][10]))
        yy.append(float(label_num[int(num)][11]))
        zeniths.append(float(label_num[int(num)][12]))
        zz.append(float(label_num[int(num)][13]))

        event_ids.append(float(label_num[int(num)][15]))
        event_run_numbers.append(float(label_num[int(num)][16]))
        event_file_name.append(label_num[int(num)][17])


        max_amp_0.append(float(label_num[int(num)][18]))
        max_amp_1.append(float(label_num[int(num)][19]))
        max_amp_2.append(float(label_num[int(num)][20]))
        max_amp_3.append(float(label_num[int(num)][21]))
        max_amp_4.append(float(label_num[int(num)][22]))

        numbers[x] += 1

        if x == number_of_files_in - 1:
            x = 0
        else:
            x += 1



    traces = np.dstack(list_traces)
    traces = np.transpose(traces, (2, 0, 1))

    idx = np.random.permutation(len(traces))

    labels = np.dstack(list_labels)
    labels = np.squeeze(labels)
    labels = np.transpose(labels, (1,0))


    traces_shuffled = traces[idx]
    labels_shuffled = labels[idx]

    azimuths_shuffled = np.array(azimuths)[idx]
    energies_shuffled = np.array(energies)[idx]
    event_group_ids_shuffled = np.array(event_group_ids)[idx]
    flavors_shuffled = np.array(flavors)[idx]
    inelasticity_shuffled = np.array(inelasticity)[idx]
    interaction_type_shuffled = np.array(interaction_type)[idx]
    n_interaction_shuffled = np.array(n_interaction)[idx]
    shower_energy_shuffled = np.array(shower_energy)[idx]
    vertex_times_shuffled = np.array(vertex_times)[idx]
    weights_shuffled = np.array(weights)[idx]
    xx_shuffled = np.array(xx)[idx]
    yy_shuffled = np.array(yy)[idx]
    zeniths_shuffled = np.array(zeniths)[idx]
    zz_shuffled = np.array(zz)[idx]
    
    event_ids_shuffled = np.array(event_ids)[idx]
    event_run_numbers_shuffled = np.array(event_run_numbers)[idx]
    event_file_name_shuffled = np.array(event_file_name)[idx]


    max_amp_0_shuffled = np.array(max_amp_0)[idx]
    max_amp_1_shuffled = np.array(max_amp_1)[idx]
    max_amp_2_shuffled = np.array(max_amp_2)[idx]
    max_amp_3_shuffled = np.array(max_amp_3)[idx]
    max_amp_4_shuffled = np.array(max_amp_4)[idx]

    print(traces_shuffled.shape)
    print(labels_shuffled.shape)
    print(azimuths_shuffled.shape)

    labels_dict_shuffled = {}
    labels_dict_shuffled['azimuths'] = azimuths_shuffled.tolist()
    labels_dict_shuffled['energies'] = energies_shuffled.tolist()
    labels_dict_shuffled['event_group_ids'] = event_group_ids_shuffled.tolist()
    labels_dict_shuffled['flavors'] = flavors_shuffled.tolist()
    labels_dict_shuffled['inelasticity'] = inelasticity_shuffled.tolist()
    labels_dict_shuffled['interaction_type'] = interaction_type_shuffled.tolist()
    labels_dict_shuffled['n_interaction'] = n_interaction_shuffled.tolist()
    labels_dict_shuffled['shower_energy'] = shower_energy_shuffled.tolist()
    labels_dict_shuffled['vertex_times'] = vertex_times_shuffled.tolist()
    labels_dict_shuffled['weights'] = weights_shuffled.tolist()
    labels_dict_shuffled['xx'] = xx_shuffled.tolist()
    labels_dict_shuffled['yy'] = yy_shuffled.tolist()
    labels_dict_shuffled['zeniths'] = zeniths_shuffled.tolist()
    labels_dict_shuffled['zz'] = zz_shuffled.tolist()

    labels_dict_shuffled['event_ids'] = event_ids_shuffled.tolist()
    labels_dict_shuffled['event_run_numbers'] = event_run_numbers_shuffled.tolist()
    labels_dict_shuffled['event_file_name'] = event_file_name_shuffled.tolist()

    labels_dict_shuffled['channel_0_max'] = max_amp_0_shuffled
    labels_dict_shuffled['channel_1_max'] = max_amp_1_shuffled
    labels_dict_shuffled['channel_2_max'] = max_amp_2_shuffled
    labels_dict_shuffled['channel_3_max'] = max_amp_3_shuffled
    labels_dict_shuffled['channel_4_max'] = max_amp_4_shuffled

    if np.isnan(traces_shuffled).any():
        print('nan in traces ', i)
        print(np.argwhere(np.isnan(traces_shuffled)).shape)
        print(np.unique(np.argwhere(np.isnan(traces_shuffled))[:,0]))
        print(len(np.unique(np.argwhere(np.isnan(traces_shuffled))[:,0])))

    if np.isnan(labels_shuffled[:, 7].astype(np.float)).any():
        print('nan in labels array  ', i)
        print(np.argwhere(np.isnan(labels_shuffled[:, 7].astype(np.float))))

    np.save('~/shallow_station/npy_files/with_snr/shuffled/shallow_baseline_nc_10k_' + str(i) + '_.npy', traces_shuffled)
    np.save('~/shallow_station/npy_files/with_snr/shuffled/shallow_baseline_nc_10k_' + str(i) + '_labels.npy', labels_dict_shuffled)
    np.save('~/shallow_station/npy_files/with_snr/shuffled/shallow_baseline_nc_10k_' + str(i) + '_labels_array.npy', labels_shuffled)

    print(i, numbers)

