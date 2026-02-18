import os
import argparse
import numpy as np
import NuRadioReco.modules.io.NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.io.eventReader
import NuRadioReco.framework.parameters as parameters
import matplotlib.pyplot as plt
import collections


parser = argparse.ArgumentParser(description='read files and save them to a numpy file')

parser.add_argument('interaction', type=str,
                    help='interaction')
parser.add_argument('energy', type=str,
                    help='run')

events_per_file = 50000

args = parser.parse_args()
interaction_arg = args.interaction
energy_arg = args.energy

base_path = '~/shallow_station/nu_e_'+ interaction_arg +'/' + energy_arg + 'eV'
os.chdir(base_path)

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

max_amp_dict = {}
for ch_idx in range(5):
    max_amp_dict[ch_idx] = []
    print(ch_idx)

event_reader = NuRadioReco.modules.io.eventReader.eventReader()

list_traces = []


if 1:
    for run in sorted(os.listdir()):
        print(run)
        os.chdir(run)

        #print(sorted(os.listdir()))

        for nur in sorted(os.listdir()):

            if nur[-1] != '5':
                print(nur)
               
                event_reader.begin([nur])
                # iterate over events

                try:
                    for event in event_reader.run():
                        station = event.get_station(1001)
                        break
                except:
                    continue

                for event in event_reader.run():
                    station = event.get_station(1001)

                    sim_station = station.get_sim_station()

                    list_trace = []
                    for channel in station.iter_channels():
                        trace = channel.get_trace()
                        list_trace.append(trace)
                    event_traces = np.vstack(list_trace)

                    if np.isnan(event_traces).any():
                        continue



                    for ch_idx in range(5):

                        channel_obj = sim_station.get_channels_by_channel_id(ch_idx)
                        max_amps_list = []

                        for obj in channel_obj:
                            
                            max_amp = obj.get_parameter(parameters.channelParameters.maximum_amplitude_envelope)
                            max_amps_list.append(max_amp)

                            #print(obj.get_id(), max_amp)

                        max_per_channel = max(max_amps_list, default=0)
                        max_amp_dict[ch_idx].append(max_per_channel)





                    list_traces.append(event_traces)

                    azimuths.append(event.get_primary().get_parameter(parameters.particleParameters.azimuth))
                    energies.append(event.get_primary().get_parameter(parameters.particleParameters.energy))
                    event_group_ids.append(event.get_primary().get_id())
                    event_ids.append(event.get_id())
                    event_run_numbers.append(event.get_run_number())
                    event_file_name.append(nur)
                    flavors.append(event.get_primary().get_parameter(parameters.particleParameters.flavor))
                    inelasticity.append(event.get_primary().get_parameter(parameters.particleParameters.inelasticity))
                    interaction_type.append(event.get_primary().get_parameter(parameters.particleParameters.interaction_type))
                    n_interaction.append(event.get_primary().get_parameter(parameters.particleParameters.n_interaction))
                    vertex_times.append(event.get_primary().get_parameter(parameters.particleParameters.vertex_time))
                    weights.append(event.get_primary().get_parameter(parameters.particleParameters.weight))
                    xx.append(event.get_primary().get_parameter(parameters.particleParameters.vertex)[0])
                    yy.append(event.get_primary().get_parameter(parameters.particleParameters.vertex)[1])
                    zeniths.append(event.get_primary().get_parameter(parameters.particleParameters.zenith))
                    zz.append(event.get_primary().get_parameter(parameters.particleParameters.vertex)[2])

                    
      

                    if event.get_primary().get_parameter(parameters.particleParameters.interaction_type) == 'nc':
                        shower_energy.append(np.log10(event.get_primary().get_parameter(parameters.particleParameters.energy) * event.get_primary().get_parameter(parameters.particleParameters.inelasticity)))

                    elif event.get_primary().get_parameter(parameters.particleParameters.interaction_type) == 'cc':
                        shower_energy.append(np.log10(event.get_primary().get_parameter(parameters.particleParameters.energy)))

                    if len(shower_energy) == events_per_file:
                        break

            

            if len(shower_energy) == events_per_file:
                break

        os.chdir(base_path)

        if len(shower_energy) == events_per_file:
            break


print('reached ' + str(len(shower_energy)) + ' events')

labels_dict = {}
labels_dict['azimuths'] = azimuths
labels_dict['energies'] = energies
labels_dict['event_group_ids'] = event_group_ids
labels_dict['flavors'] = flavors
labels_dict['inelasticity'] = inelasticity
labels_dict['interaction_type'] = interaction_type
labels_dict['n_interaction'] = n_interaction
labels_dict['shower_energy'] = shower_energy
labels_dict['vertex_times'] = vertex_times
labels_dict['weights'] = weights
labels_dict['xx'] = xx
labels_dict['yy'] = yy
labels_dict['zeniths'] = zeniths
labels_dict['zz'] = zz

labels_dict['event_group_ids'] = event_group_ids
labels_dict['event_ids'] = event_ids
labels_dict['event_run_numbers'] = event_run_numbers
labels_dict['event_file_name'] = event_file_name


labels_dict['channel_0_max'] = max_amp_dict[0]
labels_dict['channel_1_max'] = max_amp_dict[1]
labels_dict['channel_2_max'] = max_amp_dict[2]
labels_dict['channel_3_max'] = max_amp_dict[3]
labels_dict['channel_4_max'] = max_amp_dict[4]


labels_array = np.array([azimuths, energies, event_group_ids, flavors, inelasticity, interaction_type, n_interaction, shower_energy, vertex_times, weights, xx, yy, zeniths, zz, event_group_ids, event_ids, event_run_numbers, event_file_name,
        max_amp_dict[0],
        max_amp_dict[1],
        max_amp_dict[2],
        max_amp_dict[3],
        max_amp_dict[4]])



traces = np.dstack(list_traces)
traces = np.transpose(traces, (2, 0, 1))

labels_array = np.transpose(labels_array, (1, 0))

base_path = '~shallow_station/npy_files/with_snr'
os.chdir(base_path)

if np.isnan(traces).any():
    print('nan detected in final file')

print(traces.shape)
print(labels_array.shape)

np.save('shallow_baseline_' + interaction_arg + '_' + energy_arg + 'eV.npy', traces)
np.save('shallow_baseline_' + interaction_arg + '_' + energy_arg + 'eV_labels.npy', labels_dict)
np.save('shallow_baseline_' + interaction_arg + '_' + energy_arg + 'eV_labels_array.npy', labels_array)


print('done saving')