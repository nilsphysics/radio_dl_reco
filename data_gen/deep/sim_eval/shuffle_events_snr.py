import os
import argparse
import numpy as np
import NuRadioReco.modules.io.NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.io.eventReader
import NuRadioReco.framework.parameters as parameters
import matplotlib.pyplot as plt
import collections


base_path = '~/single_station_baseline/npy_files/with_snr'
os.chdir(base_path)


nc_1600_data = np.load('deep_baseline_nc_16.00eV.npy', mmap_mode='r')
nc_1620_data = np.load('deep_baseline_nc_16.20eV.npy', mmap_mode='r')
nc_1640_data = np.load('deep_baseline_nc_16.40eV.npy', mmap_mode='r')
nc_1660_data = np.load('deep_baseline_nc_16.60eV.npy', mmap_mode='r')
nc_1680_data = np.load('deep_baseline_nc_16.80eV.npy', mmap_mode='r')
nc_1700_data = np.load('deep_baseline_nc_17.00eV.npy', mmap_mode='r')
nc_1720_data = np.load('deep_baseline_nc_17.20eV.npy', mmap_mode='r')
nc_1740_data = np.load('deep_baseline_nc_17.40eV.npy', mmap_mode='r')
nc_1760_data = np.load('deep_baseline_nc_17.60eV.npy', mmap_mode='r')
nc_1780_data = np.load('deep_baseline_nc_17.80eV.npy', mmap_mode='r')
nc_1800_data = np.load('deep_baseline_nc_18.00eV.npy', mmap_mode='r')
nc_1820_data = np.load('deep_baseline_nc_18.20eV.npy', mmap_mode='r')
nc_1840_data = np.load('deep_baseline_nc_18.40eV.npy', mmap_mode='r')
nc_1860_data = np.load('deep_baseline_nc_18.60eV.npy', mmap_mode='r')
nc_1880_data = np.load('deep_baseline_nc_18.80eV.npy', mmap_mode='r')
nc_1900_data = np.load('deep_baseline_nc_19.00eV.npy', mmap_mode='r')
nc_1920_data = np.load('deep_baseline_nc_19.20eV.npy', mmap_mode='r')
nc_1940_data = np.load('deep_baseline_nc_19.40eV.npy', mmap_mode='r')
nc_1960_data = np.load('deep_baseline_nc_19.60eV.npy', mmap_mode='r')
nc_1980_data = np.load('deep_baseline_nc_19.80eV.npy', mmap_mode='r')
nc_2000_data = np.load('deep_baseline_nc_20.00eV.npy', mmap_mode='r')

print('nc data loaded')

cc_1600_data = np.load('deep_baseline_cc_16.00eV.npy', mmap_mode='r')
cc_1620_data = np.load('deep_baseline_cc_16.20eV.npy', mmap_mode='r')
cc_1640_data = np.load('deep_baseline_cc_16.40eV.npy', mmap_mode='r')
cc_1660_data = np.load('deep_baseline_cc_16.60eV.npy', mmap_mode='r')
cc_1680_data = np.load('deep_baseline_cc_16.80eV.npy', mmap_mode='r')
cc_1700_data = np.load('deep_baseline_cc_17.00eV.npy', mmap_mode='r')
cc_1720_data = np.load('deep_baseline_cc_17.20eV.npy', mmap_mode='r')
cc_1740_data = np.load('deep_baseline_cc_17.40eV.npy', mmap_mode='r')
cc_1760_data = np.load('deep_baseline_cc_17.60eV.npy', mmap_mode='r')
cc_1780_data = np.load('deep_baseline_cc_17.80eV.npy', mmap_mode='r')
cc_1800_data = np.load('deep_baseline_cc_18.00eV.npy', mmap_mode='r')
cc_1820_data = np.load('deep_baseline_cc_18.20eV.npy', mmap_mode='r')
cc_1840_data = np.load('deep_baseline_cc_18.40eV.npy', mmap_mode='r')
cc_1860_data = np.load('deep_baseline_cc_18.60eV.npy', mmap_mode='r')
cc_1880_data = np.load('deep_baseline_cc_18.80eV.npy', mmap_mode='r')
cc_1900_data = np.load('deep_baseline_cc_19.00eV.npy', mmap_mode='r')
cc_1920_data = np.load('deep_baseline_cc_19.20eV.npy', mmap_mode='r')
cc_1940_data = np.load('deep_baseline_cc_19.40eV.npy', mmap_mode='r')
cc_1960_data = np.load('deep_baseline_cc_19.60eV.npy', mmap_mode='r')
cc_1980_data = np.load('deep_baseline_cc_19.80eV.npy', mmap_mode='r')
cc_2000_data = np.load('deep_baseline_cc_20.00eV.npy', mmap_mode='r')

print('cc data loaded')

nc_1600_label = np.load('deep_baseline_nc_16.00eV_labels_array.npy', mmap_mode='r')
nc_1620_label = np.load('deep_baseline_nc_16.20eV_labels_array.npy', mmap_mode='r')
nc_1640_label = np.load('deep_baseline_nc_16.40eV_labels_array.npy', mmap_mode='r')
nc_1660_label = np.load('deep_baseline_nc_16.60eV_labels_array.npy', mmap_mode='r')
nc_1680_label = np.load('deep_baseline_nc_16.80eV_labels_array.npy', mmap_mode='r')
nc_1700_label = np.load('deep_baseline_nc_17.00eV_labels_array.npy', mmap_mode='r')
nc_1720_label = np.load('deep_baseline_nc_17.20eV_labels_array.npy', mmap_mode='r')
nc_1740_label = np.load('deep_baseline_nc_17.40eV_labels_array.npy', mmap_mode='r')
nc_1760_label = np.load('deep_baseline_nc_17.60eV_labels_array.npy', mmap_mode='r')
nc_1780_label = np.load('deep_baseline_nc_17.80eV_labels_array.npy', mmap_mode='r')
nc_1800_label = np.load('deep_baseline_nc_18.00eV_labels_array.npy', mmap_mode='r')
nc_1820_label = np.load('deep_baseline_nc_18.20eV_labels_array.npy', mmap_mode='r')
nc_1840_label = np.load('deep_baseline_nc_18.40eV_labels_array.npy', mmap_mode='r')
nc_1860_label = np.load('deep_baseline_nc_18.60eV_labels_array.npy', mmap_mode='r')
nc_1880_label = np.load('deep_baseline_nc_18.80eV_labels_array.npy', mmap_mode='r')
nc_1900_label = np.load('deep_baseline_nc_19.00eV_labels_array.npy', mmap_mode='r')
nc_1920_label = np.load('deep_baseline_nc_19.20eV_labels_array.npy', mmap_mode='r')
nc_1940_label = np.load('deep_baseline_nc_19.40eV_labels_array.npy', mmap_mode='r')
nc_1960_label = np.load('deep_baseline_nc_19.60eV_labels_array.npy', mmap_mode='r')
nc_1980_label = np.load('deep_baseline_nc_19.80eV_labels_array.npy', mmap_mode='r')
nc_2000_label = np.load('deep_baseline_nc_20.00eV_labels_array.npy', mmap_mode='r')

cc_1600_label = np.load('deep_baseline_cc_16.00eV_labels_array.npy', mmap_mode='r')
cc_1620_label = np.load('deep_baseline_cc_16.20eV_labels_array.npy', mmap_mode='r')
cc_1640_label = np.load('deep_baseline_cc_16.40eV_labels_array.npy', mmap_mode='r')
cc_1660_label = np.load('deep_baseline_cc_16.60eV_labels_array.npy', mmap_mode='r')
cc_1680_label = np.load('deep_baseline_cc_16.80eV_labels_array.npy', mmap_mode='r')
cc_1700_label = np.load('deep_baseline_cc_17.00eV_labels_array.npy', mmap_mode='r')
cc_1720_label = np.load('deep_baseline_cc_17.20eV_labels_array.npy', mmap_mode='r')
cc_1740_label = np.load('deep_baseline_cc_17.40eV_labels_array.npy', mmap_mode='r')
cc_1760_label = np.load('deep_baseline_cc_17.60eV_labels_array.npy', mmap_mode='r')
cc_1780_label = np.load('deep_baseline_cc_17.80eV_labels_array.npy', mmap_mode='r')
cc_1800_label = np.load('deep_baseline_cc_18.00eV_labels_array.npy', mmap_mode='r')
cc_1820_label = np.load('deep_baseline_cc_18.20eV_labels_array.npy', mmap_mode='r')
cc_1840_label = np.load('deep_baseline_cc_18.40eV_labels_array.npy', mmap_mode='r')
cc_1860_label = np.load('deep_baseline_cc_18.60eV_labels_array.npy', mmap_mode='r')
cc_1880_label = np.load('deep_baseline_cc_18.80eV_labels_array.npy', mmap_mode='r')
cc_1900_label = np.load('deep_baseline_cc_19.00eV_labels_array.npy', mmap_mode='r')
cc_1920_label = np.load('deep_baseline_cc_19.20eV_labels_array.npy', mmap_mode='r')
cc_1940_label = np.load('deep_baseline_cc_19.40eV_labels_array.npy', mmap_mode='r')
cc_1960_label = np.load('deep_baseline_cc_19.60eV_labels_array.npy', mmap_mode='r')
cc_1980_label = np.load('deep_baseline_cc_19.80eV_labels_array.npy', mmap_mode='r')
cc_2000_label = np.load('deep_baseline_cc_20.00eV_labels_array.npy', mmap_mode='r')

print('finished loading data')




label_list = [  nc_1600_label, cc_1600_label,
                nc_1620_label, cc_1620_label,
                nc_1640_label, cc_1640_label,
                nc_1660_label, cc_1660_label,
                nc_1680_label, cc_1680_label,
                nc_1700_label, cc_1700_label,
                nc_1720_label, cc_1720_label,
                nc_1740_label, cc_1740_label,
                nc_1760_label, cc_1760_label,
                nc_1780_label, cc_1780_label,
                nc_1800_label, cc_1800_label,
                nc_1820_label, cc_1820_label,
                nc_1840_label, cc_1840_label,
                nc_1860_label, cc_1860_label,
                nc_1880_label, cc_1880_label,
                nc_1900_label, cc_1900_label,
                nc_1920_label, cc_1920_label,
                nc_1940_label, cc_1940_label,
                nc_1960_label, cc_1960_label,
                nc_1980_label, cc_1980_label,
                nc_2000_label, cc_2000_label
                                                
                ]

data_list = [  nc_1600_data, cc_1600_data,
                nc_1620_data, cc_1620_data,
                nc_1640_data, cc_1640_data,
                nc_1660_data, cc_1660_data,
                nc_1680_data, cc_1680_data,
                nc_1700_data, cc_1700_data,
                nc_1720_data, cc_1720_data,
                nc_1740_data, cc_1740_data,
                nc_1760_data, cc_1760_data,
                nc_1780_data, cc_1780_data,
                nc_1800_data, cc_1800_data,
                nc_1820_data, cc_1820_data,
                nc_1840_data, cc_1840_data,
                nc_1860_data, cc_1860_data,
                nc_1880_data, cc_1880_data,
                nc_1900_data, cc_1900_data,
                nc_1920_data, cc_1920_data,
                nc_1940_data, cc_1940_data,
                nc_1960_data, cc_1960_data,
                nc_1980_data, cc_1980_data,
                nc_2000_data, cc_2000_data
                                                
                ]

#for i in range(len(label_list)):
#    label_list[i] = np.transpose(label_list[i], (1,0))


number_of_files_out = 210
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
    max_amp_5 = []
    max_amp_6 = []
    max_amp_7 = []
    max_amp_8 = []
    max_amp_9 = []
    max_amp_10 = []
    max_amp_11 = []
    max_amp_12 = []
    max_amp_13 = []
    max_amp_14 = []
    max_amp_15 = []


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
        max_amp_5.append(float(label_num[int(num)][23]))
        max_amp_6.append(float(label_num[int(num)][24]))
        max_amp_7.append(float(label_num[int(num)][25]))
        max_amp_8.append(float(label_num[int(num)][26]))
        max_amp_9.append(float(label_num[int(num)][27]))
        max_amp_10.append(float(label_num[int(num)][28]))
        max_amp_11.append(float(label_num[int(num)][29]))
        max_amp_12.append(float(label_num[int(num)][30]))
        max_amp_13.append(float(label_num[int(num)][31]))
        max_amp_14.append(float(label_num[int(num)][32]))
        max_amp_15.append(float(label_num[int(num)][33]))


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
    max_amp_5_shuffled = np.array(max_amp_5)[idx]
    max_amp_6_shuffled = np.array(max_amp_6)[idx]
    max_amp_7_shuffled = np.array(max_amp_7)[idx]
    max_amp_8_shuffled = np.array(max_amp_8)[idx]
    max_amp_9_shuffled = np.array(max_amp_9)[idx]
    max_amp_10_shuffled = np.array(max_amp_10)[idx]
    max_amp_11_shuffled = np.array(max_amp_11)[idx]
    max_amp_12_shuffled = np.array(max_amp_12)[idx]
    max_amp_13_shuffled = np.array(max_amp_13)[idx]
    max_amp_14_shuffled = np.array(max_amp_14)[idx]
    max_amp_15_shuffled = np.array(max_amp_15)[idx]
    

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
    labels_dict_shuffled['channel_5_max'] = max_amp_5_shuffled
    labels_dict_shuffled['channel_6_max'] = max_amp_6_shuffled
    labels_dict_shuffled['channel_7_max'] = max_amp_7_shuffled
    labels_dict_shuffled['channel_8_max'] = max_amp_8_shuffled
    labels_dict_shuffled['channel_9_max'] = max_amp_9_shuffled
    labels_dict_shuffled['channel_10_max'] = max_amp_10_shuffled
    labels_dict_shuffled['channel_11_max'] = max_amp_11_shuffled
    labels_dict_shuffled['channel_12_max'] = max_amp_12_shuffled
    labels_dict_shuffled['channel_13_max'] = max_amp_13_shuffled
    labels_dict_shuffled['channel_14_max'] = max_amp_14_shuffled
    labels_dict_shuffled['channel_15_max'] = max_amp_15_shuffled


    if np.isnan(traces_shuffled).any():
        print('nan in traces ', i)
        print(np.argwhere(np.isnan(traces_shuffled)).shape)
        print(np.unique(np.argwhere(np.isnan(traces_shuffled))[:,0]))
        print(len(np.unique(np.argwhere(np.isnan(traces_shuffled))[:,0])))

    if np.isnan(labels_shuffled[:, 7].astype(np.float)).any():
        print('nan in labels array  ', i)
        print(np.argwhere(np.isnan(labels_shuffled[:, 7].astype(np.float))))

    np.save('~/single_station_baseline/npy_files/with_snr/shuffled/deep_baseline_10k_' + str(i) + '_.npy', traces_shuffled)
    np.save('~/single_station_baseline/npy_files/with_snr/shuffled/deep_baseline_10k_' + str(i) + '_labels.npy', labels_dict_shuffled)
    np.save('~/single_station_baseline/npy_files/with_snr/shuffled/deep_baseline_10k_' + str(i) + '_labels_array.npy', labels_shuffled)

    print(i, numbers)


