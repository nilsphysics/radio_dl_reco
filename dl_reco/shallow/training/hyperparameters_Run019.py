import numpy as np

"""
The hyperparameters make it possible to alter and change the tunable parameters in a specific model class.
After training the hyperparameters are saved in results/RunXXX.
"""
"""
#file numbers
n_files_train = 90
n_files_val = 10
n_files_test = 5
"""

nc = False
cc = False
nccc = True


if nccc:

    n_files_train = 180
    n_files_val = 20
    #n_files_train = 1
    #n_files_val = 1
    n_files_test = 10

    #datafile and path name NCCC
    data_location_training = '~/data/gen2_shallow_snr/nc_cc/training'
    data_location_validating = '~/data/gen2_shallow_snr/nc_cc/validating'
    data_location_testing = '~/data/gen2_shallow_snr/nc_cc/testing'

    interaction = 'nccc'


n_events_per_file = 10000
data_name = 'shallow_snr_'     #deep_baseline_10k_00000X.npy
label_name = 'shallow_snr_times_'

filter_path = '~/data/filters/500MHz_filter.npy'

loss_function = 'bce_ggt_15f_split'

coordinates = 'spherical'
auto_opt = False


#events per data set
train_data_points = n_events_per_file * n_files_train
val_data_points = n_events_per_file * n_files_val
test_data_points = n_events_per_file * n_files_test 

#training parameters
epochs = 250
model_name = "Split"
learning_rate = 0.0005
es_patience = 10
rd_patience = 2
es_min_delta = 0.0001 # Old value: es_min_delta = 0.0001
norm = 1e-6
batchSize = 128
min_lr = 0.000001

worker_num = 0

#model parameters
conv2D_filter_size = 16
pooling_size = 4
amount_Conv2D_layers_per_block = 3 
amount_Conv2D_blocks = 4
conv2D_filter_amount = 32
batch_eps = 0.001
opt_eps = 1e-7
sch_factor = 0.2
momentum = 0.99
padding = 'same'
filter_500 = 0
filter_times = 1

conv1_layers = 32
conv2_layers = 64
conv3_layers = 128
conv4_layers = 256

class_weight = 2
energy_weight = 1
direction_weight = 1.5

#dense0 = 2560
#dense0 = 1000
#dense1 = 1024
#dense2 = 1024
#dense3 = 512
#dense4 = 256
#dense5 = 128

output_layer = 1

#cond_input = 2560
#cond_input = 1000
cond_input = 1024
#cond_input = 1024
#mlp_layers = '1000'

numb_of_classes = 2
skip_connections = False