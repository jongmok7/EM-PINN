import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

from torch import Tensor
from torch.nn.parameter import Parameter
from pyDOE import lhs

class GauossianFourierFeatureTransform_1D(torch.nn.Module):
    '''
    An inplementation of Gaussian Fourier feature mapping.
    
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
    
    Given an input of size [batches, num_input_channels, width, height],
        returns a tensor of size [batches, mapping_size*2, width, height].
    '''
    
    def __init__(self, num_input_channels, mapping_size, scale):
        super().__init__()
        
        self._num_input_channels = num_input_channels
        self.mapping_size = mapping_size
 
        _B = []
        for i in range(len(scale)):

            _B.append(torch.randn((num_input_channels, mapping_size)) * scale[i])    
            
        self._B = _B   
        
    def forward(self, x):
        
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        batches, channels = x.shape
        
        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)
        
        # Make shape compatible for matmul with_B.
        # From [B, C, W, H] to [(B*W*H), C].
        
        X = x.float() @ self._B[0].to(x.device)
        
        X0 = x.float()
        for i in range(len(self._B)-1):
            X = torch.concat([X, X0 @ self._B[i+1].to(x.device)], dim = 1)
            X.shape     
        
        X = 2 * np.pi * X
        
        return torch.cat([torch.sin(X), torch.cos(X)], dim = 1)
    

class MODEL(nn.Module):
    def __init__(self, layers, mapping_size, scale, FF = False):
        super().__init__()
        
        self.FF = FF
        self.FourierFeature = GauossianFourierFeatureTransform_1D(2, mapping_size, scale)
        
        'activation function'
        self.activation = nn.Tanh()
        
        self.layers = layers
        'initialize neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        
        self.iter = 0
        
        'Xavier Normal initialization'
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain = 1.0) # weight from a normal distribution with recommanded gain value
            nn.init.zeros_(self.linears[i].bias.data) # set biases to zero
            
    def forward(self, x, Re):
        if torch.is_tensor(x) != True:
            print('Input is not tensor')
        
        a = x.float()
        
        if self.FF == True:
            
            a = self.FourierFeature(a)
        
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        
        a = self.linears[-1](a)
        
        return a
    
class MODEL2(nn.Module):
    def __init__(self, layers, mapping_size, scale, FF = False):
        super().__init__()
        
        self.FF = FF
        self.FourierFeature = GauossianFourierFeatureTransform_1D(2, mapping_size, scale)
        
        'activation function'
        self.activation = nn.Tanh()
        
        self.layers = layers
        'initialize neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        
        self.iter = 0
        
        'Xavier Normal initialization'
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain = 1.0) # weight from a normal distribution with recommanded gain value
            nn.init.zeros_(self.linears[i].bias.data) # set biases to zero
            
    def forward(self, x):
        if torch.is_tensor(x) != True:
            print('Input is not tensor')
        
        a = x.float()
        
        if self.FF == True:
            
            a = self.FourierFeature(a)
        
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        
        a = self.linears[-1](a)
        
        return a
    
def COLLOCATION_POINTS(density_surface, density_line, Transient = False, Plot=True):
    '''
    m 단위로 기입
    단위 길이, 단위 면적당 collocation points density
    Return : (label, x, y, t)
    '''
    collocation_indexing = {'domain_fluid': 0, 'bc_fluid_top': 1, 'bc_fluid_bottom': 2, 'bc_fluid_left': 3, 
                            'bc_fluid_right': 4, 'bc_cylinder': 5, 'domain_saga': 6, 'bc_saga': 7, 'sb': 8}
    
    'Domain Fluid'
    bound_domain_fluid_lower = np.array([-0.375, -0.5])
    bound_domain_fluid_upper = np.array([1.125, 0.5])
    area_fluid = (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0]) * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1])
    domain_fluid = bound_domain_fluid_lower + (bound_domain_fluid_upper - bound_domain_fluid_lower)*lhs(2, int(area_fluid*density_surface))
    
    'Heatsource Region Remove'
    radius = 0.05
    cylinder_center = [0, 0]

    cylinder_idx = np.where((domain_fluid[:, 0] - cylinder_center[0])**2 + (domain_fluid[:, 1] - cylinder_center[1])**2 <= radius**2)[0]
    domain_fluid = np.delete(domain_fluid, cylinder_idx, axis=0)

    'Saga Region Remove'
    bound_domain_saga_lower = np.array([0.4, -0.1])
    bound_domain_saga_upper = np.array([0.5, 0.1])   
    index_saga = np.logical_and((np.logical_and((domain_fluid[:, 0] > bound_domain_saga_lower[0]), 
                                                (domain_fluid[:, 0] < bound_domain_saga_upper[0]))), 
                                (np.logical_and((domain_fluid[:, 1] > bound_domain_saga_lower[1]), 
                                                (domain_fluid[:, 1] < bound_domain_saga_upper[1]))))
    domain_fluid = np.delete(domain_fluid, index_saga, axis=0)

    'Domain Saga'
    area_saga = (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]) * (bound_domain_saga_upper[1]- bound_domain_saga_lower[1])
    domain_saga = bound_domain_saga_lower + (bound_domain_saga_upper - bound_domain_saga_lower)*lhs(2, int(area_saga * density_surface))

    'BC of Domain Fluid'
    bc_fluid_top_x = (np.linspace(bound_domain_fluid_lower[0], bound_domain_fluid_upper[0], int(density_line * (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0])))).reshape(-1, 1)
    bc_fluid_top_y = (np.ones(int(density_line * (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0]))) * bound_domain_fluid_upper[1]).reshape(-1, 1)

    bc_fluid_bottom_x = (np.linspace(bound_domain_fluid_lower[0], bound_domain_fluid_upper[0], int(density_line * (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0])))).reshape(-1, 1)
    bc_fluid_bottom_y = (np.ones(int(density_line * (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0]))) * bound_domain_fluid_lower[1]).reshape(-1, 1)

    bc_fluid_left_x = (np.ones(int(density_line * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1]))) * bound_domain_fluid_lower[0]).reshape(-1, 1)
    bc_fluid_left_y = np.linspace(bound_domain_fluid_lower[1], bound_domain_fluid_upper[1], int(density_line * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1]))).reshape(-1, 1)

    bc_fluid_right_x = (np.ones(int(density_line * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1])))*bound_domain_fluid_upper[0]).reshape(-1, 1)
    bc_fluid_right_y = np.linspace(bound_domain_fluid_lower[1], bound_domain_fluid_upper[1], int(density_line * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1]))).reshape(-1, 1)

    bc_fluid_left = np.hstack([bc_fluid_left_x, bc_fluid_left_y])
    bc_fluid_right = np.hstack([bc_fluid_right_x, bc_fluid_right_y])
    bc_fluid_top = np.hstack([bc_fluid_top_x, bc_fluid_top_y])
    bc_fluid_bottom = np.hstack([bc_fluid_bottom_x, bc_fluid_bottom_y])

    'BC of Cylinder'
    num_bc_cylinder = 2 * np.pi * radius * density_line
    theta = np.linspace(0, 2 * np.pi, int(num_bc_cylinder))
    bc_cylinder_x = (cylinder_center[0] + radius * np.cos(theta)).reshape(-1, 1)
    bc_cylinder_y = (cylinder_center[1] + radius * np.sin(theta)).reshape(-1, 1)

    bc_cylinder = np.hstack([bc_cylinder_x, bc_cylinder_y])

    'BC of Saga'
    bc_saga_top_x = (np.linspace(bound_domain_saga_lower[0], bound_domain_saga_upper[0], int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0])))).reshape(-1, 1)
    bc_saga_top_y = (np.ones(int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]))) * bound_domain_saga_upper[1]).reshape(-1, 1)
    
    bc_saga_bottom_x = (np.linspace(bound_domain_saga_lower[0], bound_domain_saga_upper[0], int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0])))).reshape(-1, 1)
    bc_saga_bottom_y = (np.ones(int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]))) * bound_domain_saga_lower[1]).reshape(-1, 1)

    bc_saga_left_x = (np.ones(int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))) * bound_domain_saga_lower[0]).reshape(-1, 1)
    bc_saga_left_y = np.linspace(bound_domain_saga_lower[1], bound_domain_saga_upper[1], int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))).reshape(-1, 1)

    bc_saga_right_x = (np.ones(int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1])))*bound_domain_saga_upper[0]).reshape(-1, 1)
    bc_saga_right_y = np.linspace(bound_domain_saga_lower[1], bound_domain_saga_upper[1], int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))).reshape(-1, 1)

    bc_saga_left = np.hstack([bc_saga_left_x, bc_saga_left_y])
    bc_saga_right = np.hstack([bc_saga_right_x, bc_saga_right_y])
    bc_saga_top = np.hstack([bc_saga_top_x, bc_saga_top_y])
    bc_saga_bottom = np.hstack([bc_saga_bottom_x, bc_saga_bottom_y])
    bc_saga = np.vstack([bc_saga_left, bc_saga_right, bc_saga_top, bc_saga_bottom])

    'SB'
    sb_saga_top_x = (np.linspace(bound_domain_saga_lower[0], bound_domain_saga_upper[0], int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0])))).reshape(-1, 1)
    sb_saga_top_y = (np.ones(int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]))) * bound_domain_saga_upper[1]).reshape(-1, 1)
    
    sb_saga_bottom_x = (np.linspace(bound_domain_saga_lower[0], bound_domain_saga_upper[0], int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0])))).reshape(-1, 1)
    sb_saga_bottom_y = (np.ones(int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]))) * bound_domain_saga_lower[1]).reshape(-1, 1)

    sb_saga_left_x = (np.ones(int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))) * bound_domain_saga_lower[0]).reshape(-1, 1)
    sb_saga_left_y = np.linspace(bound_domain_saga_lower[1], bound_domain_saga_upper[1], int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))).reshape(-1, 1)

    sb_saga_right_x = (np.ones(int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1])))*bound_domain_saga_upper[0]).reshape(-1, 1)
    sb_saga_right_y = np.linspace(bound_domain_saga_lower[1], bound_domain_saga_upper[1], int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))).reshape(-1, 1)

    sb_saga_left = np.hstack([sb_saga_left_x, sb_saga_left_y])
    sb_saga_right = np.hstack([sb_saga_right_x, sb_saga_right_y])
    sb_saga_top = np.hstack([sb_saga_top_x, sb_saga_top_y])
    sb_saga_bottom = np.hstack([sb_saga_bottom_x, sb_saga_bottom_y])
    sb_saga = np.vstack([sb_saga_left, sb_saga_right, sb_saga_top, sb_saga_bottom])

    'Indexing'
    index_domain_fluid = (np.ones(len(domain_fluid))*collocation_indexing['domain_fluid']).reshape(-1, 1)
    index_bc_fluid_top = (np.ones(len(bc_fluid_top))*collocation_indexing['bc_fluid_top']).reshape(-1, 1)
    index_bc_fluid_bottom = (np.ones(len(bc_fluid_bottom))*collocation_indexing['bc_fluid_bottom']).reshape(-1, 1)
    index_bc_fluid_right = (np.ones(len(bc_fluid_right))*collocation_indexing['bc_fluid_right']).reshape(-1, 1)
    index_bc_fluid_left = (np.ones(len(bc_fluid_left))*collocation_indexing['bc_fluid_left']).reshape(-1, 1)
    index_bc_cylinder = (np.ones(len(bc_cylinder))*collocation_indexing['bc_cylinder']).reshape(-1, 1)
    index_domain_saga = (np.ones(len(domain_saga))*collocation_indexing['domain_saga']).reshape(-1, 1)
    index_bc_saga = (np.ones(len(bc_saga))*collocation_indexing['bc_saga']).reshape(-1, 1)
    index_sb_saga = (np.ones(len(sb_saga))*collocation_indexing['sb']).reshape(-1, 1)

    domain_fluid = np.hstack([index_domain_fluid, domain_fluid])
    bc_fluid_top = np.hstack([index_bc_fluid_top, bc_fluid_top])
    bc_fluid_bottom = np.hstack([index_bc_fluid_bottom, bc_fluid_bottom])
    bc_fluid_right = np.hstack([index_bc_fluid_right, bc_fluid_right])
    bc_fluid_left = np.hstack([index_bc_fluid_left, bc_fluid_left ])
    bc_cylinder = np.hstack([index_bc_cylinder, bc_cylinder])
    domain_saga = np.hstack([index_domain_saga, domain_saga])
    bc_saga = np.hstack([index_bc_saga, bc_saga])
    sb_saga = np.hstack([index_sb_saga, sb_saga])

    collocation_points = np.vstack([domain_fluid, bc_fluid_top, bc_fluid_bottom, bc_fluid_right, 
                                    bc_fluid_left, bc_cylinder, domain_saga, bc_saga, sb_saga])
    # collocation_points[:, 1] = collocation_points[:, 1] - 0.55
#     collocation_points[:, 2] = collocation_points[:, 2] - (bound_domain_fluid_upper[1]/2)
    
    'Time Addition'
    if Transient == True:
        time_end = 1
        t = np.random.uniform(0, time_end, len(collocation_points)).reshape(-1, 1)
        collocation_points = np.hstack([collocation_points, t])
    
        'Plot'
    if Plot == True:
        data = collocation_points[:, 1:]
        label = collocation_points[:, 0]
        for i in range(len(collocation_indexing)):
            a = data[np.where(label == i)]
            if i == 0 or i == 6:
                plt.scatter(a[:, 0], a[:, 1], marker='*', s=2)
            else:
                plt.scatter(a[:, 0], a[:, 1], s=2)
        plt.axis('equal') 
        plt.show()
    
    return collocation_points

def COLLOCATION_POINTS2(density_surface, density_line, Transient = False, Plot=True):
    '''
    m 단위로 기입
    단위 길이, 단위 면적당 collocation points density
    Return : (label, x, y, t)
    '''
    collocation_indexing = {'domain_fluid': 0, 'bc_fluid_top': 1, 'bc_fluid_bottom': 2, 'bc_fluid_left': 3, 
                            'bc_fluid_right': 4, 'bc_cylinder': 5, 'domain_saga': 6, 'bc_saga': 7, 'sb1': 8, 'sb2': 9}
    
    'Domain Fluid'
    bound_domain_fluid_lower = np.array([-0.375, -0.5])
    bound_domain_fluid_upper = np.array([1.125, 0.5])
    area_fluid = (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0]) * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1])
    domain_fluid = bound_domain_fluid_lower + (bound_domain_fluid_upper - bound_domain_fluid_lower)*lhs(2, int(area_fluid*density_surface))
    
    'Heatsource Region Remove'
    radius = 0.05
    cylinder_center = [0, 0]

    cylinder_idx = np.where((domain_fluid[:, 0] - cylinder_center[0])**2 + (domain_fluid[:, 1] - cylinder_center[1])**2 <= radius**2)[0]
    domain_fluid = np.delete(domain_fluid, cylinder_idx, axis=0)

    'Saga Region Remove'
    bound_domain_saga_lower = np.array([0.4, -0.1])
    bound_domain_saga_upper = np.array([0.5, 0.1])   
    index_saga = np.logical_and((np.logical_and((domain_fluid[:, 0] > bound_domain_saga_lower[0]), 
                                                (domain_fluid[:, 0] < bound_domain_saga_upper[0]))), 
                                (np.logical_and((domain_fluid[:, 1] > bound_domain_saga_lower[1]), 
                                                (domain_fluid[:, 1] < bound_domain_saga_upper[1]))))
    domain_fluid = np.delete(domain_fluid, index_saga, axis=0)

    'Domain Saga'
    area_saga = (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]) * (bound_domain_saga_upper[1]- bound_domain_saga_lower[1])
    domain_saga = bound_domain_saga_lower + (bound_domain_saga_upper - bound_domain_saga_lower)*lhs(2, int(area_saga * density_surface))

    'BC of Domain Fluid'
    bc_fluid_top_x = (np.linspace(bound_domain_fluid_lower[0], bound_domain_fluid_upper[0], int(density_line * (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0])))).reshape(-1, 1)
    bc_fluid_top_y = (np.ones(int(density_line * (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0]))) * bound_domain_fluid_upper[1]).reshape(-1, 1)

    bc_fluid_bottom_x = (np.linspace(bound_domain_fluid_lower[0], bound_domain_fluid_upper[0], int(density_line * (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0])))).reshape(-1, 1)
    bc_fluid_bottom_y = (np.ones(int(density_line * (bound_domain_fluid_upper[0]-bound_domain_fluid_lower[0]))) * bound_domain_fluid_lower[1]).reshape(-1, 1)

    bc_fluid_left_x = (np.ones(int(density_line * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1]))) * bound_domain_fluid_lower[0]).reshape(-1, 1)
    bc_fluid_left_y = np.linspace(bound_domain_fluid_lower[1], bound_domain_fluid_upper[1], int(density_line * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1]))).reshape(-1, 1)

    bc_fluid_right_x = (np.ones(int(density_line * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1])))*bound_domain_fluid_upper[0]).reshape(-1, 1)
    bc_fluid_right_y = np.linspace(bound_domain_fluid_lower[1], bound_domain_fluid_upper[1], int(density_line * (bound_domain_fluid_upper[1]-bound_domain_fluid_lower[1]))).reshape(-1, 1)

    bc_fluid_left = np.hstack([bc_fluid_left_x, bc_fluid_left_y])
    bc_fluid_right = np.hstack([bc_fluid_right_x, bc_fluid_right_y])
    bc_fluid_top = np.hstack([bc_fluid_top_x, bc_fluid_top_y])
    bc_fluid_bottom = np.hstack([bc_fluid_bottom_x, bc_fluid_bottom_y])

    'BC of Cylinder'
    num_bc_cylinder = 2 * np.pi * radius * density_line
    theta = np.linspace(0, 2 * np.pi, int(num_bc_cylinder))
    bc_cylinder_x = (cylinder_center[0] + radius * np.cos(theta)).reshape(-1, 1)
    bc_cylinder_y = (cylinder_center[1] + radius * np.sin(theta)).reshape(-1, 1)

    bc_cylinder = np.hstack([bc_cylinder_x, bc_cylinder_y])

    'BC of Saga'
    bc_saga_top_x = (np.linspace(bound_domain_saga_lower[0], bound_domain_saga_upper[0], int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0])))).reshape(-1, 1)
    bc_saga_top_y = (np.ones(int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]))) * bound_domain_saga_upper[1]).reshape(-1, 1)
    
    bc_saga_bottom_x = (np.linspace(bound_domain_saga_lower[0], bound_domain_saga_upper[0], int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0])))).reshape(-1, 1)
    bc_saga_bottom_y = (np.ones(int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]))) * bound_domain_saga_lower[1]).reshape(-1, 1)

    bc_saga_left_x = (np.ones(int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))) * bound_domain_saga_lower[0]).reshape(-1, 1)
    bc_saga_left_y = np.linspace(bound_domain_saga_lower[1], bound_domain_saga_upper[1], int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))).reshape(-1, 1)

    bc_saga_right_x = (np.ones(int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1])))*bound_domain_saga_upper[0]).reshape(-1, 1)
    bc_saga_right_y = np.linspace(bound_domain_saga_lower[1], bound_domain_saga_upper[1], int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))).reshape(-1, 1)

    bc_saga_left = np.hstack([bc_saga_left_x, bc_saga_left_y])
    bc_saga_right = np.hstack([bc_saga_right_x, bc_saga_right_y])
    bc_saga_top = np.hstack([bc_saga_top_x, bc_saga_top_y])
    bc_saga_bottom = np.hstack([bc_saga_bottom_x, bc_saga_bottom_y])
    bc_saga = np.vstack([bc_saga_left, bc_saga_right, bc_saga_top, bc_saga_bottom])

    'SB'
    sb_saga_top_x = (np.linspace(bound_domain_saga_lower[0], bound_domain_saga_upper[0], int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0])))).reshape(-1, 1)
    sb_saga_top_y = (np.ones(int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]))) * bound_domain_saga_upper[1]).reshape(-1, 1)
    
    sb_saga_bottom_x = (np.linspace(bound_domain_saga_lower[0], bound_domain_saga_upper[0], int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0])))).reshape(-1, 1)
    sb_saga_bottom_y = (np.ones(int(density_line * (bound_domain_saga_upper[0]-bound_domain_saga_lower[0]))) * bound_domain_saga_lower[1]).reshape(-1, 1)

    sb_saga_left_x = (np.ones(int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))) * bound_domain_saga_lower[0]).reshape(-1, 1)
    sb_saga_left_y = np.linspace(bound_domain_saga_lower[1], bound_domain_saga_upper[1], int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))).reshape(-1, 1)

    sb_saga_right_x = (np.ones(int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1])))*bound_domain_saga_upper[0]).reshape(-1, 1)
    sb_saga_right_y = np.linspace(bound_domain_saga_lower[1], bound_domain_saga_upper[1], int(density_line * (bound_domain_saga_upper[1]-bound_domain_saga_lower[1]))).reshape(-1, 1)

    sb_saga_left = np.hstack([sb_saga_left_x, sb_saga_left_y])
    sb_saga_right = np.hstack([sb_saga_right_x, sb_saga_right_y])
    sb_saga_top = np.hstack([sb_saga_top_x, sb_saga_top_y])
    sb_saga_bottom = np.hstack([sb_saga_bottom_x, sb_saga_bottom_y])
    sb_saga1 = np.vstack([sb_saga_left, sb_saga_right])
    sb_saga2 = np.vstack([sb_saga_top, sb_saga_bottom])

    'Indexing'
    index_domain_fluid = (np.ones(len(domain_fluid))*collocation_indexing['domain_fluid']).reshape(-1, 1)
    index_bc_fluid_top = (np.ones(len(bc_fluid_top))*collocation_indexing['bc_fluid_top']).reshape(-1, 1)
    index_bc_fluid_bottom = (np.ones(len(bc_fluid_bottom))*collocation_indexing['bc_fluid_bottom']).reshape(-1, 1)
    index_bc_fluid_right = (np.ones(len(bc_fluid_right))*collocation_indexing['bc_fluid_right']).reshape(-1, 1)
    index_bc_fluid_left = (np.ones(len(bc_fluid_left))*collocation_indexing['bc_fluid_left']).reshape(-1, 1)
    index_bc_cylinder = (np.ones(len(bc_cylinder))*collocation_indexing['bc_cylinder']).reshape(-1, 1)
    index_domain_saga = (np.ones(len(domain_saga))*collocation_indexing['domain_saga']).reshape(-1, 1)
    index_bc_saga = (np.ones(len(bc_saga))*collocation_indexing['bc_saga']).reshape(-1, 1)
    index_sb_saga1 = (np.ones(len(sb_saga1))*collocation_indexing['sb1']).reshape(-1, 1)
    index_sb_saga2 = (np.ones(len(sb_saga2))*collocation_indexing['sb2']).reshape(-1, 1)

    domain_fluid = np.hstack([index_domain_fluid, domain_fluid])
    bc_fluid_top = np.hstack([index_bc_fluid_top, bc_fluid_top])
    bc_fluid_bottom = np.hstack([index_bc_fluid_bottom, bc_fluid_bottom])
    bc_fluid_right = np.hstack([index_bc_fluid_right, bc_fluid_right])
    bc_fluid_left = np.hstack([index_bc_fluid_left, bc_fluid_left ])
    bc_cylinder = np.hstack([index_bc_cylinder, bc_cylinder])
    domain_saga = np.hstack([index_domain_saga, domain_saga])
    bc_saga = np.hstack([index_bc_saga, bc_saga])
    sb_saga1 = np.hstack([index_sb_saga1, sb_saga1])
    sb_saga2 = np.hstack([index_sb_saga2, sb_saga2])

    collocation_points = np.vstack([domain_fluid, bc_fluid_top, bc_fluid_bottom, bc_fluid_right, 
                                    bc_fluid_left, bc_cylinder, domain_saga, bc_saga, sb_saga1, sb_saga2])
    # collocation_points[:, 1] = collocation_points[:, 1] - 0.55
#     collocation_points[:, 2] = collocation_points[:, 2] - (bound_domain_fluid_upper[1]/2)
    
    'Time Addition'
    if Transient == True:
        time_end = 1
        t = np.random.uniform(0, time_end, len(collocation_points)).reshape(-1, 1)
        collocation_points = np.hstack([collocation_points, t])
    
        'Plot'
    if Plot == True:
        data = collocation_points[:, 1:]
        label = collocation_points[:, 0]
        for i in range(len(collocation_indexing)):
            a = data[np.where(label == i)]
            if i == 0 or i == 6:
                plt.scatter(a[:, 0], a[:, 1], marker='*', s=2)
            else:
                plt.scatter(a[:, 0], a[:, 1], s=2)
        plt.axis('equal') 
        plt.show()
    
    return collocation_points


class Linear(nn.Module):
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()

#         self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
#         self.bias = Parameter(torch.empty(out_features, **factory_kwargs))

    def forward(self, input: Tensor, weight, bias) -> Tensor:
        return F.linear(input, weight, bias)

class HyperNetwork(nn.Module):
    def __init__(self, z_dim, input_dim, output_dim, device):
        super(HyperNetwork, self).__init__()

        self.z_dim = z_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w1 = Parameter(torch.empty((self.z_dim, self.input_dim)).to(device))
        self.b1 = Parameter(torch.empty((self.input_dim)).to(device))

        self.w2 = Parameter(torch.empty((self.input_dim, self.input_dim + self.output_dim)).to(device))
        self.b2 = Parameter(torch.empty((self.input_dim + self.output_dim)).to(device))

        self.w3 = Parameter(torch.empty((self.input_dim + self.output_dim, self.input_dim * self.output_dim)).to(device))
        self.b3 = Parameter(torch.empty((self.input_dim * self.output_dim)).to(device))

        self.b = Parameter(torch.empty((self.z_dim, self.output_dim)).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.b, a=math.sqrt(5))

        if self.b1 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b1, -bound, bound)
        
        if self.b2 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w2)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b2, -bound, bound)
        
        if self.b3 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w3)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b3, -bound, bound)

    def forward(self, x):

        w1 = torch.matmul(x, self.w1) + self.b1
        w2 = torch.matmul(w1, self.w2) + self.b2
        w3 = torch.matmul(w2, self.w3) + self.b3
        w = w3.view(self.output_dim, self.input_dim)

        b = torch.matmul(x, self.b)

        return w, b

class MainNetwork(nn.Module):
    def __init__(self, z_dim, m_layers):
        super(MainNetwork, self).__init__()

        self.activation = nn.Tanh()
        self.z_dim = z_dim
        self.m_layers = m_layers

        self.hyper = nn.ModuleList([HyperNetwork(self.z_dim, self.m_layers[i], self.m_layers[i + 1]) for i in range(len(self.m_layers))])
        self.linears = nn.ModuleList([Linear() for _ in range(len(self.m_layers) - 1)])

    def forward(self, x, RE):
        x = x.float()

        for i in range(len(self.linears) - 1):
            w, b = self.hyper[i](RE)
            z = self.linears[i](x, w, b)
            x = self.activation(z)

        w, b = self.hyper[-1](RE)
        z = self.linears[-1](x, w, b)

        return z