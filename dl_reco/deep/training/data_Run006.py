import os
import numpy as np
import hyperparameters
import torch
from queue import Queue
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from typing import Any, Iterable, List, Optional, Tuple, Union

import threading

"""
Prepare_Dataset makes it possible to read large amounts of data without losing too much time during training. 
"""

class Prepare_Dataset_direction(Dataset):
    def __init__(self, file_ids, points=10000, data_type = '_training', worker_num=0, transform=None, target_transform=None, shuffle=True):

        self.file_ids = file_ids
        self.data = None
        self.direction = None
        self.snr = None
        self.times = None
        self.energy_and_vertex = None
        self.points = points
        self.transform = transform
        self.target_transform = target_transform
        self.data_type = data_type
        self.worker_num = worker_num
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.file_ids)*self.points

    def __getitem__(self, idx):


        worker_info = torch.utils.data.get_worker_info()

        file_idx = idx // self.points

        if np.mod((idx+self.points),self.points) == 0:
            self.data, self.direction, self.flavor, self.energy_and_vertex, self.snr, self.times  = self.get_data(file_idx)
            
        #print('data', self.data[idx-self.points*file_idx].to(torch.float32).shape)
        #return self.data[idx-self.points*file_idx].to(torch.float32), self.direction[:, idx-self.points*file_idx].to(torch.float32), self.flavor[idx-self.points*file_idx].to(torch.float32), self.energy_and_vertex[:, idx-self.points*file_idx].to(torch.float32), self.snr[:, idx-self.points*file_idx].to(torch.float32)
        #return self.data[idx-self.points*file_idx].to(torch.bfloat16), self.direction[:, idx-self.points*file_idx].to(torch.bfloat16), self.energy[idx-self.points*file_idx].to(torch.bfloat16), self.flavor[idx-self.points*file_idx].to(torch.bfloat16)
        #return self.data[idx-self.points*file_idx].to(torch.double), self.direction[:, idx-self.points*file_idx].to(torch.double), self.energy[idx-self.points*file_idx].to(torch.double), self.flavor[idx-self.points*file_idx].to(torch.double)
        return self.data[idx-self.points*file_idx].to(torch.float32), self.direction[:, idx-self.points*file_idx].to(torch.float32), self.flavor[idx-self.points*file_idx].to(torch.float32), self.energy_and_vertex[:, idx-self.points*file_idx].to(torch.float32), self.snr[:, idx-self.points*file_idx].to(torch.float32), self.times[:, idx-self.points*file_idx].to(torch.float32)
        


    def get_data(self,file_idx):
        # Load data from file

        data, direction, flavor, en_vertex, snr, times = self.load_file(self.file_ids[file_idx],hyperparameters.norm)


        # randomly choose the points in a file 
        #np.random.seed(hyperparameters.seed)

        if self.shuffle:
            idx = np.random.choice(direction.shape[1], size=self.points, replace=False)

            data = np.swapaxes(data,1,3)
            data = np.swapaxes(data,2,3)

            
            direction = direction[:,idx]
            data = data[idx,:]
            snr = snr[:,idx]
            times = times[:,idx]
            
            data = torch.from_numpy(data)
            direction = torch.from_numpy(direction)
            snr = torch.from_numpy(snr)
            times = torch.from_numpy(times)

            flavor = np.expand_dims(flavor,1)
            flavor = flavor[idx, :]
            flavor = torch.from_numpy(flavor)

            en_vertex = en_vertex[:,idx]
            en_vertex = torch.from_numpy(en_vertex)

        else:
            data = np.swapaxes(data,1,3)
            data = np.swapaxes(data,2,3)

            data = torch.from_numpy(data)
            direction = torch.from_numpy(direction)
            snr = torch.from_numpy(snr)
            times = torch.from_numpy(times)

            flavor = np.expand_dims(flavor,1)
            flavor = torch.from_numpy(flavor)

            en_vertex = torch.from_numpy(en_vertex)

        return data, direction, flavor, en_vertex, snr, times


    """
    def dev_workers(self):

        workers = np.arange(self.worker_num)
        
        for i in workers:
            worker_files = 
    """



    # Loading data and label files
    def load_file(self, i_file, norm=1e-6):

        # Load 500 MHz filter (not used here but might be used in the future)
        """
        filt = np.load(f"{common_dir()}/bandpass_filters/500MHz_filter.npy")

        #     t0 = time.time()
        #     print(f"loading file {i_file}", flush=True)
        data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True)
        data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
        """
        if self.data_type == '_training':
            data_location = hyperparameters.data_location_training

        elif self.data_type == '_validating':
            data_location = hyperparameters.data_location_validating

        elif self.data_type == '_testing':
            data_location = hyperparameters.data_location_testing


        #print('this is the data file')
        #print(f"{hyperparameters.data_name}{hyperparameters.interaction}{self.data_type}{'_10k_'}{i_file}_.npy")

        data = np.load(os.path.join(data_location, f"{hyperparameters.data_name}{hyperparameters.interaction}{self.data_type}{'_10k_'}{i_file}_.npy"), allow_pickle=True)
        data = data[:, :, :, np.newaxis]
        labels_tmp = np.load(os.path.join(data_location, f"{hyperparameters.label_name}{hyperparameters.interaction}{self.data_type}{'_10k_'}{i_file}_labels.npy"), allow_pickle=True)

        azimuths = np.array(labels_tmp.item().get('azimuths'))
        zeniths = np.array(labels_tmp.item().get('zeniths'))

        vert_x = np.array(labels_tmp.item().get('xx'))
        vert_y = np.array(labels_tmp.item().get('yy'))
        vert_z = np.array(labels_tmp.item().get('zz'))

        #print(labels_tmp.item().keys())
        #quit()

        # check for nans and remove thempython 
        idx = ~(np.isnan(data))
        idx = np.all(idx, axis=1)
        idx = np.all(idx, axis=1)
        idx = np.all(idx, axis=1)
        data = data[idx, :, :, :]

        #print(norm)
        
        data /= norm

        if hyperparameters.coordinates == 'cartesian':
            direction = np.array([np.sin(zeniths)*np.cos(azimuths), np.sin(zeniths)*np.sin(azimuths), np.cos(zeniths)])
            direction = direction[:, idx]
            return data, direction

        elif hyperparameters.coordinates == 'spherical':
            #angle = np.array([azimuths, zeniths])
            angle = np.array([zeniths, azimuths])
            angle = angle[:, idx]

            neutrino_energy_data = np.array(labels_tmp.item().get("energies"))
            inelasticity_data = np.array(labels_tmp.item().get("inelasticity"))
            interaction_type_data = np.array(labels_tmp.item().get("interaction_type"))

            channel_0_max = np.array(labels_tmp.item().get("channel_0_max"))
            channel_1_max = np.array(labels_tmp.item().get("channel_1_max"))
            channel_2_max = np.array(labels_tmp.item().get("channel_2_max"))
            channel_3_max = np.array(labels_tmp.item().get("channel_3_max"))
            channel_4_max = np.array(labels_tmp.item().get("channel_4_max"))
            channel_5_max = np.array(labels_tmp.item().get("channel_5_max"))
            channel_6_max = np.array(labels_tmp.item().get("channel_6_max"))
            channel_7_max = np.array(labels_tmp.item().get("channel_7_max"))
            channel_8_max = np.array(labels_tmp.item().get("channel_8_max"))
            channel_9_max = np.array(labels_tmp.item().get("channel_9_max"))
            channel_10_max = np.array(labels_tmp.item().get("channel_10_max"))
            channel_11_max = np.array(labels_tmp.item().get("channel_11_max"))
            channel_12_max = np.array(labels_tmp.item().get("channel_12_max"))
            channel_13_max = np.array(labels_tmp.item().get("channel_13_max"))
            channel_14_max = np.array(labels_tmp.item().get("channel_14_max"))
            channel_15_max = np.array(labels_tmp.item().get("channel_15_max"))

            snr = np.array([channel_0_max, channel_1_max, 
            channel_2_max, channel_3_max, 
            channel_4_max, channel_5_max, 
            channel_6_max, channel_7_max, 
            channel_8_max, channel_9_max, 
            channel_10_max, channel_11_max, 
            channel_12_max, channel_13_max, 
            channel_14_max, channel_15_max])
            
            snr = snr[:, idx]

            trigger_time = np.array(labels_tmp.item().get("trigger_time"))
            signal_time = np.array(labels_tmp.item().get("signal_time"))
            times = np.array([trigger_time, signal_time])
            times = times[:, idx]

            inelastic_energy = neutrino_energy_data * inelasticity_data
            mask_of_types = interaction_type_data == 'cc'
            shower_energy_data = np.where(mask_of_types, neutrino_energy_data, inelastic_energy)
            
            shower_energy_log10 = np.log10(shower_energy_data)
            labels_flavor = np.where(mask_of_types, 1, 0)
            #shower_energy_data = shower_energy_data[idx]
            labels_flavor = labels_flavor[idx]



            energy_vertex = np.array([shower_energy_log10, vert_x, vert_y, vert_z])
            energy_vertex = energy_vertex[:, idx]



            return data, angle, labels_flavor, energy_vertex, snr, times






class Prepare_Dataset_systematics(IterableDataset):
    def __init__(self, syst_name = 'no_change'):

        self.syst_name = syst_name

    def filter_fn(self, data, times, energies):

        all_masks = {}

        if isinstance(data, np.ndarray):
            data = torch.tensor(data)

        nan_mask = torch.isnan(data)
        all_masks['nan_mask'] = ~nan_mask.any(dim=1).any(dim=1).any(dim=1)#.squeeze()

        if hyperparameters.filter_times:
            times_tensor = torch.tensor(times, dtype=torch.float32)
            diff_time = times_tensor[1] - times_tensor[0]
            all_masks['time_mask'] = (diff_time > -120) & (diff_time < 20)

        final_mask = torch.ones(data.shape[0], dtype=torch.bool)



        for key in all_masks.keys():

            final_mask &= all_masks[key]

        return final_mask

    def load_file(self, norm=1e-6):

        data_location = '~/data/gen2_deep_snr/syst_files'

        data = np.load(os.path.join(data_location, 'deep_syst_' + self.syst_name + '.npy' ), allow_pickle=True)
        labels_tmp = np.load(os.path.join(data_location, 'deep_syst_' + self.syst_name + '_labels.npy' ), allow_pickle=True)

        if hyperparameters.filter_500:
            filt = np.load("~/data/filters/500MHz_filter.npy")
            data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)

        
        data = data[:, :, :, np.newaxis]
        data /= norm
        data = np.swapaxes(data,1,3)
        data = np.swapaxes(data,2,3)
        

        trigger_time = np.array(labels_tmp.item().get('tr_times'))
        signal_time = np.array(labels_tmp.item().get('sig_times'))
        
        azimuths = np.array(labels_tmp.item().get('azimuths'))
        zeniths = np.array(labels_tmp.item().get('zeniths'))

        vert_x = np.array(labels_tmp.item().get('xx'))/1000
        vert_y = np.array(labels_tmp.item().get('yy'))/1000
        vert_z = np.array(labels_tmp.item().get('zz'))/1000

        neutrino_energy_data = np.array(labels_tmp.item().get("energies"))
        inelasticity_data = np.array(labels_tmp.item().get("inelasticity"))
        interaction_type_data = np.array(labels_tmp.item().get("interaction_type"))

        channel_0_max = np.array(labels_tmp.item().get("channel_0_max"))
        channel_1_max = np.array(labels_tmp.item().get("channel_1_max"))
        channel_2_max = np.array(labels_tmp.item().get("channel_2_max"))
        channel_3_max = np.array(labels_tmp.item().get("channel_3_max"))
        channel_4_max = np.array(labels_tmp.item().get("channel_4_max"))
        channel_5_max = np.array(labels_tmp.item().get("channel_5_max"))
        channel_6_max = np.array(labels_tmp.item().get("channel_6_max"))
        channel_7_max = np.array(labels_tmp.item().get("channel_7_max"))
        channel_8_max = np.array(labels_tmp.item().get("channel_8_max"))
        channel_9_max = np.array(labels_tmp.item().get("channel_9_max"))
        channel_10_max = np.array(labels_tmp.item().get("channel_10_max"))
        channel_11_max = np.array(labels_tmp.item().get("channel_11_max"))
        channel_12_max = np.array(labels_tmp.item().get("channel_12_max"))
        channel_13_max = np.array(labels_tmp.item().get("channel_13_max"))
        channel_14_max = np.array(labels_tmp.item().get("channel_14_max"))
        channel_15_max = np.array(labels_tmp.item().get("channel_15_max"))

        angle = np.array([zeniths, azimuths])

        inelastic_energy = neutrino_energy_data * inelasticity_data
        mask_of_types = interaction_type_data == 'cc'
        shower_energy_data = np.where(mask_of_types, neutrino_energy_data, inelastic_energy)
        shower_energy_log10 = np.log10(shower_energy_data)
        labels_flavor = np.where(mask_of_types, 1, 0)
        energy_vertex = np.array([shower_energy_log10, vert_x, vert_y, vert_z])

        snr = np.array([channel_0_max, channel_1_max, channel_2_max, channel_3_max, channel_4_max, channel_5_max, channel_6_max, channel_7_max, channel_8_max, channel_9_max, channel_10_max, channel_11_max, channel_12_max, channel_13_max, channel_14_max, channel_15_max])
        times = np.array([trigger_time, signal_time])

        return data, angle, labels_flavor, energy_vertex, snr, times

    def __iter__(self):


        data, direction, flavor, energy_and_vertex, snr, times = self.load_file(hyperparameters.norm)


        # Apply filtering
        filter_mask = self.filter_fn(data, times, energy_and_vertex[0, :])
        filtered_data = data[filter_mask]
        filtered_direction = direction[:, filter_mask]
        filtered_flavor = flavor[filter_mask]
        filtered_energy_and_vertex = energy_and_vertex[:, filter_mask]
        filtered_snr = snr[:, filter_mask]
        filtered_times = times[:, filter_mask]

        
        for i in range(filtered_data.shape[0]):  # Iterate sequentially
            yield (
                torch.from_numpy(filtered_data[i]).to(torch.float32),
                torch.from_numpy(filtered_direction[:, i]).to(torch.float32),
                torch.tensor(filtered_flavor[i], dtype=torch.float32),  # Handle the scalar as a tensor
                torch.from_numpy(filtered_energy_and_vertex[:, i]).to(torch.float32),
                torch.from_numpy(filtered_snr[:, i]).to(torch.float32),
                torch.from_numpy(filtered_times[:, i]).to(torch.float32),
            )







