import hyperparameters
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import jammy_flows
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnext101_32x8d
from torchsummary import summary
import numpy as np


"""
Every class is a different 'class' of model that can be tuned and optimized by changing the hyperparameters in hyperparameters.py.
After training the model is saved in results/RunXXX/model.
"""


def overwrite(cur_flow_def, flow_options_overwrite_extraf, order=1):
    flow_options_overwrite_extraf[(order,len(cur_flow_def))]=dict()
    flow_options_overwrite_extraf[(order,len(cur_flow_def))]["f"]=dict()
    flow_options_overwrite_extraf[(order,len(cur_flow_def))]["f"]["add_vertical_rq_spline_flow"]=0
    flow_options_overwrite_extraf[(order,len(cur_flow_def))]["f"]["add_circular_rq_spline_flow"]=0
    flow_options_overwrite_extraf[(order,len(cur_flow_def))]["f"]["add_rotation"]=1
    ## copy kappa modelling from options overwrite, otherwise take default
    if("f" in flow_options_overwrite_extraf):
        if("kappa_prediction" in flow_options_overwrite_extraf["f"]):
            flow_options_overwrite_extraf[(order,len(cur_flow_def))]["f"]["kappa_prediction"]=flow_options_overwrite_extraf["f"]["kappa_prediction"]
            
        if("min_kappa" in flow_options_overwrite_extraf["f"]):
            flow_options_overwrite_extraf[(order,len(cur_flow_def))]["f"]["min_kappa"]=flow_options_overwrite_extraf["f"]["min_kappa"]
        
        if("kappa_clamping" in flow_options_overwrite_extraf["f"]):
            flow_options_overwrite_extraf[(order,len(cur_flow_def))]["f"]["kappa_clamping"]=flow_options_overwrite_extraf["f"]["kappa_clamping"]
        
        if("num_householder_iter" in flow_options_overwrite_extraf["f"]):
            flow_options_overwrite_extraf[(order,len(cur_flow_def))]["f"]["num_householder_iter"]=flow_options_overwrite_extraf["f"]["num_householder_iter"]
                

    return cur_flow_def+"f", flow_options_overwrite_extraf


class Split(pl.LightningModule):

    def __init__(self, loss_function = 'bce_ggt_15f_split'):
        """
        Defines the layes used in the model.
        """

        super().__init__()

        self.loss_function = loss_function
        self.class_criterion = nn.BCELoss()

        self.num_classes = 1000
        #self.resnet34_5 = ResNet(ResidualBlock, [3, 4, 6, 3, 1, 1], self.num_classes)
        self.resnet34_5 = ResNet(ResidualBlock, [3, 4, 6, 3, 1], self.num_classes)

        self.automatic_optimization = hyperparameters.auto_opt

        if self.loss_function == 'bce_ggt_15f_split':
            device = torch.device(f"cuda:0")

            """old f-flow settings changed after Run004
            flow_options_overwrite = {}
            flow_options_overwrite['f'] = dict()
            flow_options_overwrite['f']['add_vertical_rq_spline_flow'] = 1
            flow_options_overwrite['f']["vertical_smooth"] = 1
            flow_options_overwrite['f']["vertical_fix_boundary_derivative"] = 1
            flow_options_overwrite['f']["spline_num_basis_functions"] = -1
            flow_options_overwrite["f"]["boundary_cos_theta_identity_region"] = 0
            flow_options_overwrite["f"]["circular_add_rotation"]=0
            #flow_options_overwrite["f"]["vertical_fix_first_width_n_height_to_zero"] = 1
            #flow_options_overwrite["f"]["vertical_independent_width_height_parametrization"] = 1
            #flow_options_overwrite["f"]["add_circular_rq_spline_flow"] = 1
            #flow_options_overwrite["f"]["vertical_also_fix_second_width_to_zero"] = 1
            """
            flow_options_overwrite = {}
            flow_options_overwrite["f"]=dict()
            flow_options_overwrite["g"]=dict()
            flow_options_overwrite["t"]=dict()
            
            flow_options_overwrite["f"]["add_vertical_rq_spline_flow"]=1
            flow_options_overwrite["f"]["spline_num_basis_functions"]=-1
            flow_options_overwrite["f"]["vertical_smooth"]=1
            flow_options_overwrite["f"]["vertical_flow_defs"]="rr"
            flow_options_overwrite["f"]["vertical_fix_boundary_derivative"]=1
            flow_options_overwrite["f"]["min_kappa"]=1e-10
            flow_options_overwrite["f"]["kappa_prediction"]="direct_log_real_bounded"
            flow_options_overwrite["f"]["kappa_clamping"]=0
            flow_options_overwrite["f"]["vertical_restrict_max_min_width_height_ratio"]=-1.0
            flow_options_overwrite["f"]["vertical_fix_first_width_n_height_to_zero"]=1 # fix the first width/height to 0
            flow_options_overwrite["f"]["vertical_independent_width_height_parametrization"]=1
            flow_options_overwrite["f"]["add_circular_rq_spline_flow"]=1
            flow_options_overwrite["f"]["circular_add_rotation"]=0
            flow_options_overwrite["f"]["vertical_also_fix_second_width_to_zero"]=1

            flow_options_overwrite["g"]["upper_bound_for_widths"]=1
            flow_options_overwrite["g"]["lower_bound_for_widths"]=0.01
            flow_options_overwrite["g"]["fit_normalization"]=0
            flow_options_overwrite['t']['cov_type'] = 'full'


            #self.pdf_energy = jammy_flows.pdf("e1", "ggt", conditional_input_dim=hyperparameters.cond_input, options_overwrite=flow_options_overwrite).to(device)
            #self.pdf_energy = jammy_flows.pdf("e1", "ggt", conditional_input_dim=hyperparameters.cond_input, options_overwrite=flow_options_overwrite).to(device)
            #self.pdf_energy_vertex = jammy_flows.pdf("e1+e1+e1+e1", "ggggt+ggggt+ggggt+ggggt", conditional_input_dim=hyperparameters.cond_input, options_overwrite=flow_options_overwrite).to(device)
            self.pdf_energy_vertex = jammy_flows.pdf("e4", "ggggt", conditional_input_dim=hyperparameters.cond_input, options_overwrite=flow_options_overwrite).to(device)
            print('################hello#################')
            self.pdf_direction = jammy_flows.pdf("s2", "fffffffffffffff", conditional_input_dim=hyperparameters.cond_input, options_overwrite=flow_options_overwrite).to(device)



        self.classifier = nn.Sequential(
                            #nn.Linear(1000, 512),
                            #nn.ReLU(), 
                            #nn.Linear(512, 512),
                            #nn.ReLU(), 
                            nn.Linear(hyperparameters.cond_input, 1),
                            nn.Sigmoid() 
                            )
        
        
        self.model1D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2)),


            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            
            )
        


    def forward(self, x):
        x = self.model1D(x)

        #print('summary 1d', summary(self.model1D, (1, 5, 512)))

        x = x.permute((0, 2, 1, 3))
        
        #print(np.shape(x))
        #quit()
        x = self.resnet34_5(x)

        #print(np.shape(x))
        #quit()

        #print('summary resnet', summary(self.resnet34_5, (5, 256, 256)))

        return x

  
    def second_init(self, data):
        self.pdf.init_params(data)

    """
    def configure_optimizers(self):
        
        
        #Defines the optimizer used in the model.
        
        print('this is the learning rate:', hyperparameters.learning_rate)
    
        optimizer = torch.optim.Adam(self.parameters(), lr=hyperparameters.learning_rate, eps=hyperparameters.opt_eps) 
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=hyperparameters.sch_factor, patience=hyperparameters.rd_patience, min_lr=hyperparameters.min_lr, verbose=1)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_total_loss'}
    """


    def configure_optimizers(self):
        
        """
        Defines the optimizer used in the model.
        """
        print('this is the learning rate:', hyperparameters.learning_rate)
    
        optimizer = torch.optim.Adam(self.parameters(), lr=hyperparameters.learning_rate, eps=hyperparameters.opt_eps) 
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=hyperparameters.sch_factor, patience=hyperparameters.rd_patience, min_lr=hyperparameters.min_lr, verbose=1)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_weighted_loss'}

    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_weighted_loss"])
    

        
    def training_step(self, train_batch, batch_idx):


        try:

            x, direction, flavor, energy_vertex, _, _ = train_batch


            conv_out = self.forward(x)

            flavor_pred = self.classifier(conv_out) 

            log_pdf_energy_vertex, _,_= self.pdf_energy_vertex(energy_vertex.to(torch.double), conditional_input=conv_out.to(torch.double))

            log_pdf_direction, _,_= self.pdf_direction(direction.to(torch.double), conditional_input=conv_out.to(torch.double))
            loss_classifier = self.class_criterion(flavor_pred.squeeze(), flavor.squeeze())

            final_loss = loss_classifier - log_pdf_energy_vertex.mean() - log_pdf_direction.mean()

            weighted_loss =  hyperparameters.class_weight * loss_classifier - hyperparameters.energy_weight * log_pdf_energy_vertex.mean() - hyperparameters.direction_weight * log_pdf_direction.mean()

            self.log('train_class_loss', loss_classifier)
            self.log('train_pdf_energy_vertex', -log_pdf_energy_vertex.mean())
            self.log('train_pdf_direction', -log_pdf_direction.mean())

            self.log('train_total_loss', final_loss)
            self.log('train_weighted_loss', weighted_loss)

            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(weighted_loss)

            grads = []

            for param in Split.parameters(self):

                if param.grad == None:
                    continue

                grads.append(param.grad.view(-1))

            grads = torch.cat(grads)

            if torch.isnan(grads).any():

                torch.set_printoptions(threshold=10_000)
                f = open("output.txt", "a")

                print('########################')
                print('nan occured here')
                print('########################')

                
                for name, param in Split.named_parameters(self):

                    if torch.isnan(param).any():
                        print(name, file=f)
                        print(param, file=f)

                f.close()

            assert(not torch.isnan(grads).any())
            

            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()

        
        except:
            print('error in training step')
        
            
    def validation_step(self, val_batch, batch_idx):


        x, direction, flavor, energy_vertex, _, _ = val_batch

        conv_out = self.forward(x)

        flavor_pred = self.classifier(conv_out) 

        log_pdf_energy_vertex, _,_= self.pdf_energy_vertex(energy_vertex.to(torch.double), conditional_input=conv_out.to(torch.double))

        log_pdf_direction, _,_= self.pdf_direction(direction.to(torch.double), conditional_input=conv_out.to(torch.double))
        loss_classifier = self.class_criterion(flavor_pred.squeeze(), flavor.squeeze())

        final_loss = loss_classifier - log_pdf_energy_vertex.mean() - log_pdf_direction.mean() 
        weighted_loss =  hyperparameters.class_weight * loss_classifier - hyperparameters.energy_weight * log_pdf_energy_vertex.mean() - hyperparameters.direction_weight * log_pdf_direction.mean()

        self.log('val_class_loss', loss_classifier)
        self.log('val_pdf_energy_vertex', -log_pdf_energy_vertex.mean())
        self.log('val_pdf_direction', -log_pdf_direction.mean())
        self.log('val_total_loss', final_loss)
        self.log('val_weighted_loss', weighted_loss)
        



class AdaptiveConcatPool2d(nn.Module):
    "Concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self):
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)




class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(5, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.layer4 = self._make_layer(block, 1024, layers[4], stride = 2)
        #self.layer5 = self._make_layer(block, 2048, layers[5], stride = 2)

        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        #print('start', x.shape)
        x = self.conv1(x)
        #print('conv1', x.shape)
        x = self.maxpool(x)
        #print('max1', x.shape)
        x = self.layer0(x)
        #print('layer0', x.shape)
        x = self.layer1(x)
        #print('layer1', x.shape)
        x = self.layer2(x)
        #print('layer2', x.shape)
        x = self.layer3(x)
        #print('layer3', x.shape)
        x = self.layer4(x)

        #print('layer4', x.shape)

        #x = self.layer5(x)

        #print('layer5', x.shape)
        #quit()
        #print('layer4', x.shape)

        x = self.avgpool(x)
        #print('avpool', x.shape)
        x = x.view(x.size(0), -1)
        #print('view', x.shape)
        #x = self.fc(x)
        #print('fc', x.shape)

        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
            #print('downsampling', self.downsample)
            #print('conv1', self.conv1)
            #print('conv2', self.conv2)
        out += residual
        out = self.relu(out)
        return out


def _make_mlp(input_dim, hidden_dims, output_dim, dtype=torch.float64, layer_norm=0):

    device = torch.device(f"cuda:0")

    mlp_in_dims = [input_dim]
    if(hidden_dims!=""):
        mlp_in_dims = mlp_in_dims + [int(i) for i in hidden_dims.split("-")]

    mlp_out_dims = [output_dim]
    if(hidden_dims != ""):
        mlp_out_dims =  [int(i) for i in hidden_dims.split("-")] + mlp_out_dims
   
    nn_list = []

    for i in range(len(mlp_in_dims)):
       
        l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i], dtype=dtype)
        #print("L ", l, l.weight.shape)
        nn_list.append(l)
        
        if i < (len(mlp_in_dims) - 1):
            if(layer_norm):
                nn_list.append(torch.nn.LayerNorm(mlp_out_dims[i]))
            nn_list.append(torch.nn.Tanh())
    
    return torch.nn.Sequential(*nn_list).to(device)

class SkipMLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dtype=torch.float64, add_skip_connection=False, layer_norm=0):
        super(SkipMLP, self).__init__()
        device = torch.device(f"cuda:0")
        self.mlp=_make_mlp(input_dim,hidden_dims,output_dim, dtype=dtype, layer_norm=layer_norm)

        self.skip_connection=None
        if(add_skip_connection):
            
            self.skip_connection=torch.nn.Linear(input_dim, output_dim, dtype=dtype, bias=False).to(device)

    def skip_forward(self, x):
        
        if(self.skip_connection is not None):

            return self.skip_connection(x)+self.mlp(x)

        else:
        
            return self.mlp(x)
