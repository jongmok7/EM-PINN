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
    collocation_indexing = {'domain_fluid1': 0, 'domain_fluid2': 1, 'bc_fluid1_top': 2, 'bc_fluid1_bottom': 3, 'bc_fluid2_top': 4, 'bc_fluid2_bottom': 5, 'inlet_fluid1': 6, 
                            'outlet_fluid1': 7, 'inlet_fluid2': 8, 'outlet_fluid2': 9, 'domain_solid': 10, 'bc_solid_top': 11, 'bc_solid_bottom': 12, 'bc_solid_left': 13, 'bc_solid_right': 14}
    
    'Domain Fluid1'
    bound_domain_fluid1_lower = np.array([0, 0.6])
    bound_domain_fluid1_upper = np.array([1.4, 1.05])
    area_fluid1 = (bound_domain_fluid1_upper[0]-bound_domain_fluid1_lower[0]) * (bound_domain_fluid1_upper[1]-bound_domain_fluid1_lower[1])
    domain_fluid1 = bound_domain_fluid1_lower + (bound_domain_fluid1_upper - bound_domain_fluid1_lower)*lhs(2, int(area_fluid1*density_surface))
    index_fluid1 = np.logical_or((np.logical_and((domain_fluid1[:, 0] < 0.5), 
                                                  (domain_fluid1[:, 1] > 1))), 
                                  (np.logical_and((domain_fluid1[:, 0] > 0.9), 
                                                  (domain_fluid1[:, 1] > 1))))
    domain_fluid1 = np.delete(domain_fluid1, index_fluid1, axis=0)

    'Domain Fluid1'
    bound_domain_fluid2_lower = np.array([0, -0.05])
    bound_domain_fluid2_upper = np.array([1.4, 0.4])
    area_fluid2 = (bound_domain_fluid2_upper[0]-bound_domain_fluid2_lower[0]) * (bound_domain_fluid2_upper[1]-bound_domain_fluid2_lower[1])
    domain_fluid2 = bound_domain_fluid2_lower + (bound_domain_fluid2_upper - bound_domain_fluid2_lower)*lhs(2, int(area_fluid2*density_surface))
    index_fluid2 = np.logical_or((np.logical_and((domain_fluid2[:, 0] < 0.5), 
                                                  (domain_fluid2[:, 1] < 0))), 
                                  (np.logical_and((domain_fluid2[:, 0] > 0.9), 
                                                  (domain_fluid2[:, 1] < 0))))
    domain_fluid2 = np.delete(domain_fluid2, index_fluid2, axis=0)
    bound_domain_fluid2_lower = np.array([0, 0])
    bound_domain_fluid2_upper = np.array([1.4, 0.4])

    'Domain Solid'
    bound_domain_solid_lower = np.array([0, 0.4])
    bound_domain_solid_upper = np.array([1.4, 0.6])   
    area_solid = (bound_domain_solid_upper[0]-bound_domain_solid_lower[0]) * (bound_domain_solid_upper[1]- bound_domain_solid_lower[1])
    domain_solid = bound_domain_solid_lower + (bound_domain_solid_upper - bound_domain_solid_lower)*lhs(2, int(area_solid * density_surface))

    'BC of Domain Fluid1'
    bc_fluid1_top1_x = np.linspace(bound_domain_fluid1_lower[0], 0.5, int(density_line * (0.5-bound_domain_fluid1_lower[0]))).reshape(-1, 1)
    bc_fluid1_top1_y = (np.ones(int(density_line * (0.5-bound_domain_fluid1_lower[0]))) * 1).reshape(-1, 1)
    bc_fluid1_top1 = np.hstack([bc_fluid1_top1_x, bc_fluid1_top1_y])

    bc_fluid1_top2_x = np.linspace(0.9, bound_domain_fluid1_upper[0], int(density_line * (bound_domain_fluid1_upper[0] - 0.9))).reshape(-1, 1)
    bc_fluid1_top2_y = (np.ones(int(density_line * (bound_domain_fluid1_upper[0] - 0.9))) * 1).reshape(-1, 1)
    bc_fluid1_top2 = np.hstack([bc_fluid1_top2_x, bc_fluid1_top2_y])

    bc_fluid1_top3_x = (np.ones(int(density_line * 0.05)) * 0.5).reshape(-1, 1)
    bc_fluid1_top3_y = np.linspace(1, bound_domain_fluid1_upper[1], int(density_line * (bound_domain_fluid1_upper[1] - 1))).reshape(-1, 1)
    bc_fluid1_top3 = np.hstack([bc_fluid1_top3_x, bc_fluid1_top3_y])

    bc_fluid1_top4_x = (np.ones(int(density_line * 0.05)) * 0.9).reshape(-1, 1)
    bc_fluid1_top4_y = np.linspace(1, bound_domain_fluid1_upper[1], int(density_line * (bound_domain_fluid1_upper[1] - 1))).reshape(-1, 1)
    bc_fluid1_top4 = np.hstack([bc_fluid1_top4_x, bc_fluid1_top4_y])

    bc_fluid1_bottom_x = np.linspace(bound_domain_fluid1_lower[0], bound_domain_fluid1_upper[0], int(density_line * (bound_domain_fluid1_upper[0]-bound_domain_fluid1_lower[0]))).reshape(-1, 1)
    bc_fluid1_bottom_y = (np.ones(int(density_line * (bound_domain_fluid1_upper[0]-bound_domain_fluid1_lower[0])))*bound_domain_fluid1_lower[1]).reshape(-1, 1)

    bc_fluid1_top = np.vstack([bc_fluid1_top1, bc_fluid1_top2, bc_fluid1_top3, bc_fluid1_top4])
    bc_fluid1_bottom = np.hstack([bc_fluid1_bottom_x, bc_fluid1_bottom_y])

    'Inlet Fluid1'
    inlet_fluid1_x = (np.linspace(0.5, 0.9, int(density_line * 0.4))).reshape(-1, 1)
    inlet_fluid1_y = (np.ones(int(density_line * 0.4)) * bound_domain_fluid1_upper[1]).reshape(-1, 1)
    inlet_fluid1 = np.hstack([inlet_fluid1_x, inlet_fluid1_y])

    'Outlet Fluid1'
    outlet_fluid1_left_x = (np.ones(int(density_line * (1 - bound_domain_fluid1_lower[1]))) * bound_domain_fluid1_lower[0]).reshape(-1, 1)
    outlet_fluid1_left_y = (np.linspace(bound_domain_fluid1_lower[1], 1, int(density_line * (1 - bound_domain_fluid1_lower[1])))).reshape(-1, 1)
    outlet_fluid1_left = np.hstack([outlet_fluid1_left_x, outlet_fluid1_left_y])
    outlet_fluid1_right_x = (np.ones(int(density_line * (1 - bound_domain_fluid1_lower[1]))) * bound_domain_fluid1_upper[0]).reshape(-1, 1)
    outlet_fluid1_right_y = (np.linspace(bound_domain_fluid1_lower[1], 1, int(density_line * (1 - bound_domain_fluid1_lower[1])))).reshape(-1, 1)
    outlet_fluid1_right = np.hstack([outlet_fluid1_right_x, outlet_fluid1_right_y])
    outlet_fluid1 = np.vstack([outlet_fluid1_left, outlet_fluid1_right])

    'BC of Domain Fluid2'
    bc_fluid2_top_x = np.linspace(bound_domain_fluid2_lower[0], bound_domain_fluid2_upper[0], int(density_line * (bound_domain_fluid2_upper[0]-bound_domain_fluid2_lower[0]))).reshape(-1, 1)
    bc_fluid2_top_y = (np.ones(int(density_line * (bound_domain_fluid2_upper[0]-bound_domain_fluid2_lower[0]))) * bound_domain_fluid2_upper[1]).reshape(-1, 1)

    bc_fluid2_bottom1_x = np.linspace(bound_domain_fluid2_lower[0], 0.5, int(density_line * (0.5-bound_domain_fluid2_lower[0]))).reshape(-1, 1)
    bc_fluid2_bottom1_y = (np.ones(int(density_line * (0.5-bound_domain_fluid2_lower[0]))) * bound_domain_fluid2_lower[1]).reshape(-1, 1)
    bc_fluid2_bottom1 = np.hstack([bc_fluid2_bottom1_x, bc_fluid2_bottom1_y])

    bc_fluid2_bottom2_x = np.linspace(0.9, bound_domain_fluid2_upper[0], int(density_line * (bound_domain_fluid2_upper[0] - 0.9))).reshape(-1, 1)
    bc_fluid2_bottom2_y = (np.ones(int(density_line * (bound_domain_fluid2_upper[0] - 0.9))) * bound_domain_fluid2_lower[1]).reshape(-1, 1)
    bc_fluid2_bottom2 = np.hstack([bc_fluid2_bottom2_x, bc_fluid2_bottom2_y])

    bc_fluid2_bottom3_x = (np.ones(int(density_line * 0.05)) * 0.5).reshape(-1, 1)
    bc_fluid2_bottom3_y = np.linspace(-0.05, bound_domain_fluid2_lower[1], int(density_line * (bound_domain_fluid2_lower[1] + 0.05))).reshape(-1, 1)
    bc_fluid2_bottom3 = np.hstack([bc_fluid2_bottom3_x, bc_fluid2_bottom3_y])

    bc_fluid2_bottom4_x = (np.ones(int(density_line * 0.05)) * 0.9).reshape(-1, 1)
    bc_fluid2_bottom4_y = np.linspace(-0.05, bound_domain_fluid2_lower[1], int(density_line * (bound_domain_fluid2_lower[1] + 0.05))).reshape(-1, 1)
    bc_fluid2_bottom4 = np.hstack([bc_fluid2_bottom4_x, bc_fluid2_bottom4_y])

    bc_fluid2_top = np.hstack([bc_fluid2_top_x, bc_fluid2_top_y])
    bc_fluid2_bottom = np.vstack([bc_fluid2_bottom1, bc_fluid2_bottom2, bc_fluid2_bottom3, bc_fluid2_bottom4])

    'Inlet Fluid2'
    inlet_fluid2_x = (np.linspace(0.5, 0.9, int(density_line * (0.4)))).reshape(-1, 1)
    inlet_fluid2_y = (np.ones(int(density_line * (0.4))) * (-0.05)).reshape(-1, 1)
    inlet_fluid2 = np.hstack([inlet_fluid2_x, inlet_fluid2_y])

    'Outlet Fluid2'
    outlet_fluid2_left_x = (np.ones(int(density_line * (bound_domain_fluid2_upper[1] - bound_domain_fluid2_lower[1]))) * bound_domain_fluid2_lower[0]).reshape(-1, 1)
    outlet_fluid2_left_y = (np.linspace(bound_domain_fluid2_lower[1], bound_domain_fluid2_upper[1], int(density_line * (bound_domain_fluid2_upper[1] - bound_domain_fluid2_lower[1])))).reshape(-1, 1)
    outlet_fluid2_left = np.hstack([outlet_fluid2_left_x, outlet_fluid2_left_y])
    outlet_fluid2_right_x = (np.ones(int(density_line * (bound_domain_fluid2_upper[1] - bound_domain_fluid2_lower[1]))) * bound_domain_fluid2_upper[0]).reshape(-1, 1)
    outlet_fluid2_right_y = (np.linspace(bound_domain_fluid2_lower[1], bound_domain_fluid2_upper[1], int(density_line * (bound_domain_fluid2_upper[1] - bound_domain_fluid2_lower[1])))).reshape(-1, 1)
    outlet_fluid2_right = np.hstack([outlet_fluid2_right_x, outlet_fluid2_right_y])
    outlet_fluid2 = np.vstack([outlet_fluid2_left, outlet_fluid2_right])

    'BC of Solid'
    bc_solid_top_x = (np.linspace(bound_domain_solid_lower[0], bound_domain_solid_upper[0], int(density_line * (bound_domain_solid_upper[0]-bound_domain_solid_lower[0])))).reshape(-1, 1)
    bc_solid_top_y = (np.ones(int(density_line * (bound_domain_solid_upper[0]-bound_domain_solid_lower[0]))) * bound_domain_solid_upper[1]).reshape(-1, 1)
    
    bc_solid_bottom_x = (np.linspace(bound_domain_solid_lower[0], bound_domain_solid_upper[0], int(density_line * (bound_domain_solid_upper[0]-bound_domain_solid_lower[0])))).reshape(-1, 1)
    bc_solid_bottom_y = (np.ones(int(density_line * (bound_domain_solid_upper[0]-bound_domain_solid_lower[0]))) * bound_domain_solid_lower[1]).reshape(-1, 1)

    bc_solid_left_x = (np.ones(int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))) * bound_domain_solid_lower[0]).reshape(-1, 1)
    bc_solid_left_y = np.linspace(bound_domain_solid_lower[1], bound_domain_solid_upper[1], int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))).reshape(-1, 1)

    bc_solid_right_x = (np.ones(int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1])))*bound_domain_solid_upper[0]).reshape(-1, 1)
    bc_solid_right_y = np.linspace(bound_domain_solid_lower[1], bound_domain_solid_upper[1], int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))).reshape(-1, 1)
    
    bc_solid_top = np.hstack([bc_solid_top_x, bc_solid_top_y])
    bc_solid_bottom = np.hstack([bc_solid_bottom_x, bc_solid_bottom_y])
    bc_solid_left = np.hstack([bc_solid_left_x, bc_solid_left_y])
    bc_solid_right = np.hstack([bc_solid_right_x, bc_solid_right_y])
    sb_solid = np.vstack([bc_solid_top, bc_solid_bottom, bc_solid_left, bc_solid_right])

    # 'SB1_Solid'
    # sb1_solid_left_x = (np.ones(int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))) * bound_domain_solid_lower[0]).reshape(-1, 1)
    # sb1_solid_left_y = np.linspace(bound_domain_solid_lower[1], bound_domain_solid_upper[1], int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))).reshape(-1, 1)
    # sb1_solid = np.hstack([sb1_solid_left_x, sb1_solid_left_y])

    # 'SB2_Solid'
    # sb2_solid_right_x = (np.ones(int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1])))*bound_domain_solid_upper[0]).reshape(-1, 1)
    # sb2_solid_right_y = np.linspace(bound_domain_solid_lower[1], bound_domain_solid_upper[1], int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))).reshape(-1, 1)
    # sb2_solid = np.hstack([sb2_solid_right_x, sb2_solid_right_y])

    'Indexing'
    index_domain_fluid1 = (np.ones(len(domain_fluid1))*collocation_indexing['domain_fluid1']).reshape(-1, 1)
    index_domain_fluid2 = (np.ones(len(domain_fluid2))*collocation_indexing['domain_fluid2']).reshape(-1, 1)
    index_bc_fluid1_top = (np.ones(len(bc_fluid1_top))*collocation_indexing['bc_fluid1_top']).reshape(-1, 1)
    index_bc_fluid1_bottom = (np.ones(len(bc_fluid1_bottom))*collocation_indexing['bc_fluid1_bottom']).reshape(-1, 1)
    index_bc_fluid2_top = (np.ones(len(bc_fluid2_top))*collocation_indexing['bc_fluid2_top']).reshape(-1, 1)
    index_bc_fluid2_bottom = (np.ones(len(bc_fluid2_bottom))*collocation_indexing['bc_fluid2_bottom']).reshape(-1, 1)
    index_inlet_fluid1 = (np.ones(len(inlet_fluid1))*collocation_indexing['inlet_fluid1']).reshape(-1, 1)
    index_outlet_fluid1 = (np.ones(len(outlet_fluid1))*collocation_indexing['outlet_fluid1']).reshape(-1, 1)
    index_inlet_fluid2 = (np.ones(len(inlet_fluid2))*collocation_indexing['inlet_fluid2']).reshape(-1, 1)
    index_outlet_fluid2 = (np.ones(len(outlet_fluid2))*collocation_indexing['outlet_fluid2']).reshape(-1, 1)
    index_domain_solid = (np.ones(len(domain_solid))*collocation_indexing['domain_solid']).reshape(-1, 1)
    index_bc_solid_top = (np.ones(len(bc_solid_top))*collocation_indexing['bc_solid_top']).reshape(-1, 1)
    index_bc_solid_bottom = (np.ones(len(bc_solid_bottom))*collocation_indexing['bc_solid_bottom']).reshape(-1, 1)
    index_bc_solid_left = (np.ones(len(bc_solid_left))*collocation_indexing['bc_solid_left']).reshape(-1, 1)
    index_bc_solid_right = (np.ones(len(bc_solid_right))*collocation_indexing['bc_solid_right']).reshape(-1, 1)

    domain_fluid1 = np.hstack([index_domain_fluid1, domain_fluid1])
    domain_fluid2 = np.hstack([index_domain_fluid2, domain_fluid2])
    bc_fluid1_top = np.hstack([index_bc_fluid1_top, bc_fluid1_top])
    bc_fluid1_bottom = np.hstack([index_bc_fluid1_bottom, bc_fluid1_bottom])
    bc_fluid2_top = np.hstack([index_bc_fluid2_top, bc_fluid2_top])
    bc_fluid2_bottom = np.hstack([index_bc_fluid2_bottom, bc_fluid2_bottom])
    inlet_fluid1 = np.hstack([index_inlet_fluid1, inlet_fluid1])
    outlet_fluid1 = np.hstack([index_outlet_fluid1, outlet_fluid1])
    inlet_fluid2 = np.hstack([index_inlet_fluid2, inlet_fluid2])
    outlet_fluid2 = np.hstack([index_outlet_fluid2, outlet_fluid2])
    domain_solid = np.hstack([index_domain_solid, domain_solid])
    bc_solid_top = np.hstack([index_bc_solid_top, bc_solid_top])
    bc_solid_bottom = np.hstack([index_bc_solid_bottom, bc_solid_bottom])
    bc_solid_left = np.hstack([index_bc_solid_left, bc_solid_left])
    bc_solid_right = np.hstack([index_bc_solid_right, bc_solid_right])

    collocation_points = np.vstack([domain_fluid1, domain_fluid2, bc_fluid1_top, bc_fluid1_bottom, bc_fluid2_top, bc_fluid2_bottom, inlet_fluid1, outlet_fluid1, 
                                    inlet_fluid2, outlet_fluid2, domain_solid, bc_solid_top, bc_solid_bottom, bc_solid_left, bc_solid_right])
    collocation_points[:, 1] -= 0.7
    collocation_points[:, 2] -= 0.5
    
    'Time Addition'
    if Transient == True:
        time_end = 1
        t = np.random.uniform(0, time_end, len(collocation_points)).reshape(-1, 1)
        collocation_points = np.hstack([collocation_points, t])
    
    'Plot'
    if Plot == True:
        data = collocation_points[:, 1:]
        label = collocation_points[:,0]
        for i in range(len(collocation_indexing)):
            a = data[np.where(label == i)]
            plt.scatter(a[:, 0], a[:, 1], s=0.1)
        plt.axis('equal') 
        plt.show()
    
    return collocation_points

def COLLOCATION_POINTS2(density_surface, density_line, Transient = False, Plot=True):
    '''
    m 단위로 기입
    단위 길이, 단위 면적당 collocation points density
    Return : (label, x, y, t)
    '''
    collocation_indexing = {'domain_fluid1': 0, 'domain_fluid2': 1, 'bc_fluid1_top': 2, 'bc_fluid1_bottom': 3, 'bc_fluid2_top': 4, 'bc_fluid2_bottom': 5, 'inlet_fluid1': 6, 
                            'outlet_fluid1': 7, 'inlet_fluid2': 8, 'outlet_fluid2': 9, 'domain_solid': 10, 'bc_solid_top': 11, 'bc_solid_bottom': 12, 'sb_solid1': 13, 'sb_solid2': 14}
    
    'Domain Fluid1'
    bound_domain_fluid1_lower = np.array([0, 0.6])
    bound_domain_fluid1_upper = np.array([1.4, 1.05])
    area_fluid1 = (bound_domain_fluid1_upper[0]-bound_domain_fluid1_lower[0]) * (bound_domain_fluid1_upper[1]-bound_domain_fluid1_lower[1])
    domain_fluid1 = bound_domain_fluid1_lower + (bound_domain_fluid1_upper - bound_domain_fluid1_lower)*lhs(2, int(area_fluid1*density_surface))
    index_fluid1 = np.logical_or((np.logical_and((domain_fluid1[:, 0] < 0.5), 
                                                  (domain_fluid1[:, 1] > 1))), 
                                  (np.logical_and((domain_fluid1[:, 0] > 0.9), 
                                                  (domain_fluid1[:, 1] > 1))))
    domain_fluid1 = np.delete(domain_fluid1, index_fluid1, axis=0)

    'Domain Fluid1'
    bound_domain_fluid2_lower = np.array([0, -0.05])
    bound_domain_fluid2_upper = np.array([1.4, 0.4])
    area_fluid2 = (bound_domain_fluid2_upper[0]-bound_domain_fluid2_lower[0]) * (bound_domain_fluid2_upper[1]-bound_domain_fluid2_lower[1])
    domain_fluid2 = bound_domain_fluid2_lower + (bound_domain_fluid2_upper - bound_domain_fluid2_lower)*lhs(2, int(area_fluid2*density_surface))
    index_fluid2 = np.logical_or((np.logical_and((domain_fluid2[:, 0] < 0.5), 
                                                  (domain_fluid2[:, 1] < 0))), 
                                  (np.logical_and((domain_fluid2[:, 0] > 0.9), 
                                                  (domain_fluid2[:, 1] < 0))))
    domain_fluid2 = np.delete(domain_fluid2, index_fluid2, axis=0)
    bound_domain_fluid2_lower = np.array([0, 0])
    bound_domain_fluid2_upper = np.array([1.4, 0.4])

    'Domain Solid'
    bound_domain_solid_lower = np.array([0, 0.4])
    bound_domain_solid_upper = np.array([1.4, 0.6])   
    area_solid = (bound_domain_solid_upper[0]-bound_domain_solid_lower[0]) * (bound_domain_solid_upper[1]- bound_domain_solid_lower[1])
    domain_solid = bound_domain_solid_lower + (bound_domain_solid_upper - bound_domain_solid_lower)*lhs(2, int(area_solid * density_surface))

    'BC of Domain Fluid1'
    bc_fluid1_top1_x = np.linspace(bound_domain_fluid1_lower[0], 0.5, int(density_line * (0.5-bound_domain_fluid1_lower[0]))).reshape(-1, 1)
    bc_fluid1_top1_y = (np.ones(int(density_line * (0.5-bound_domain_fluid1_lower[0]))) * 1).reshape(-1, 1)
    bc_fluid1_top1 = np.hstack([bc_fluid1_top1_x, bc_fluid1_top1_y])

    bc_fluid1_top2_x = np.linspace(0.9, bound_domain_fluid1_upper[0], int(density_line * (bound_domain_fluid1_upper[0] - 0.9))).reshape(-1, 1)
    bc_fluid1_top2_y = (np.ones(int(density_line * (bound_domain_fluid1_upper[0] - 0.9))) * 1).reshape(-1, 1)
    bc_fluid1_top2 = np.hstack([bc_fluid1_top2_x, bc_fluid1_top2_y])

    bc_fluid1_top3_x = (np.ones(int(density_line * 0.05)) * 0.5).reshape(-1, 1)
    bc_fluid1_top3_y = np.linspace(1, bound_domain_fluid1_upper[1], int(density_line * (bound_domain_fluid1_upper[1] - 1))).reshape(-1, 1)
    bc_fluid1_top3 = np.hstack([bc_fluid1_top3_x, bc_fluid1_top3_y])

    bc_fluid1_top4_x = (np.ones(int(density_line * 0.05)) * 0.9).reshape(-1, 1)
    bc_fluid1_top4_y = np.linspace(1, bound_domain_fluid1_upper[1], int(density_line * (bound_domain_fluid1_upper[1] - 1))).reshape(-1, 1)
    bc_fluid1_top4 = np.hstack([bc_fluid1_top4_x, bc_fluid1_top4_y])

    bc_fluid1_bottom_x = np.linspace(bound_domain_fluid1_lower[0], bound_domain_fluid1_upper[0], int(density_line * (bound_domain_fluid1_upper[0]-bound_domain_fluid1_lower[0]))).reshape(-1, 1)
    bc_fluid1_bottom_y = (np.ones(int(density_line * (bound_domain_fluid1_upper[0]-bound_domain_fluid1_lower[0])))*bound_domain_fluid1_lower[1]).reshape(-1, 1)

    bc_fluid1_top = np.vstack([bc_fluid1_top1, bc_fluid1_top2, bc_fluid1_top3, bc_fluid1_top4])
    bc_fluid1_bottom = np.hstack([bc_fluid1_bottom_x, bc_fluid1_bottom_y])

    'Inlet Fluid1'
    inlet_fluid1_x = (np.linspace(0.5, 0.9, int(density_line * 0.4))).reshape(-1, 1)
    inlet_fluid1_y = (np.ones(int(density_line * 0.4)) * bound_domain_fluid1_upper[1]).reshape(-1, 1)
    inlet_fluid1 = np.hstack([inlet_fluid1_x, inlet_fluid1_y])

    'Outlet Fluid1'
    outlet_fluid1_left_x = (np.ones(int(density_line * (1 - bound_domain_fluid1_lower[1]))) * bound_domain_fluid1_lower[0]).reshape(-1, 1)
    outlet_fluid1_left_y = (np.linspace(bound_domain_fluid1_lower[1], 1, int(density_line * (1 - bound_domain_fluid1_lower[1])))).reshape(-1, 1)
    outlet_fluid1_left = np.hstack([outlet_fluid1_left_x, outlet_fluid1_left_y])
    outlet_fluid1_right_x = (np.ones(int(density_line * (1 - bound_domain_fluid1_lower[1]))) * bound_domain_fluid1_upper[0]).reshape(-1, 1)
    outlet_fluid1_right_y = (np.linspace(bound_domain_fluid1_lower[1], 1, int(density_line * (1 - bound_domain_fluid1_lower[1])))).reshape(-1, 1)
    outlet_fluid1_right = np.hstack([outlet_fluid1_right_x, outlet_fluid1_right_y])
    outlet_fluid1 = np.vstack([outlet_fluid1_left, outlet_fluid1_right])

    'BC of Domain Fluid2'
    bc_fluid2_top_x = np.linspace(bound_domain_fluid2_lower[0], bound_domain_fluid2_upper[0], int(density_line * (bound_domain_fluid2_upper[0]-bound_domain_fluid2_lower[0]))).reshape(-1, 1)
    bc_fluid2_top_y = (np.ones(int(density_line * (bound_domain_fluid2_upper[0]-bound_domain_fluid2_lower[0]))) * bound_domain_fluid2_upper[1]).reshape(-1, 1)

    bc_fluid2_bottom1_x = np.linspace(bound_domain_fluid2_lower[0], 0.5, int(density_line * (0.5-bound_domain_fluid2_lower[0]))).reshape(-1, 1)
    bc_fluid2_bottom1_y = (np.ones(int(density_line * (0.5-bound_domain_fluid2_lower[0]))) * bound_domain_fluid2_lower[1]).reshape(-1, 1)
    bc_fluid2_bottom1 = np.hstack([bc_fluid2_bottom1_x, bc_fluid2_bottom1_y])

    bc_fluid2_bottom2_x = np.linspace(0.9, bound_domain_fluid2_upper[0], int(density_line * (bound_domain_fluid2_upper[0] - 0.9))).reshape(-1, 1)
    bc_fluid2_bottom2_y = (np.ones(int(density_line * (bound_domain_fluid2_upper[0] - 0.9))) * bound_domain_fluid2_lower[1]).reshape(-1, 1)
    bc_fluid2_bottom2 = np.hstack([bc_fluid2_bottom2_x, bc_fluid2_bottom2_y])

    bc_fluid2_bottom3_x = (np.ones(int(density_line * 0.05)) * 0.5).reshape(-1, 1)
    bc_fluid2_bottom3_y = np.linspace(-0.05, bound_domain_fluid2_lower[1], int(density_line * (bound_domain_fluid2_lower[1] + 0.05))).reshape(-1, 1)
    bc_fluid2_bottom3 = np.hstack([bc_fluid2_bottom3_x, bc_fluid2_bottom3_y])

    bc_fluid2_bottom4_x = (np.ones(int(density_line * 0.05)) * 0.9).reshape(-1, 1)
    bc_fluid2_bottom4_y = np.linspace(-0.05, bound_domain_fluid2_lower[1], int(density_line * (bound_domain_fluid2_lower[1] + 0.05))).reshape(-1, 1)
    bc_fluid2_bottom4 = np.hstack([bc_fluid2_bottom4_x, bc_fluid2_bottom4_y])

    bc_fluid2_top = np.hstack([bc_fluid2_top_x, bc_fluid2_top_y])
    bc_fluid2_bottom = np.vstack([bc_fluid2_bottom1, bc_fluid2_bottom2, bc_fluid2_bottom3, bc_fluid2_bottom4])

    'Inlet Fluid2'
    inlet_fluid2_x = (np.linspace(0.5, 0.9, int(density_line * (0.4)))).reshape(-1, 1)
    inlet_fluid2_y = (np.ones(int(density_line * (0.4))) * (-0.05)).reshape(-1, 1)
    inlet_fluid2 = np.hstack([inlet_fluid2_x, inlet_fluid2_y])

    'Outlet Fluid2'
    outlet_fluid2_left_x = (np.ones(int(density_line * (bound_domain_fluid2_upper[1] - bound_domain_fluid2_lower[1]))) * bound_domain_fluid2_lower[0]).reshape(-1, 1)
    outlet_fluid2_left_y = (np.linspace(bound_domain_fluid2_lower[1], bound_domain_fluid2_upper[1], int(density_line * (bound_domain_fluid2_upper[1] - bound_domain_fluid2_lower[1])))).reshape(-1, 1)
    outlet_fluid2_left = np.hstack([outlet_fluid2_left_x, outlet_fluid2_left_y])
    outlet_fluid2_right_x = (np.ones(int(density_line * (bound_domain_fluid2_upper[1] - bound_domain_fluid2_lower[1]))) * bound_domain_fluid2_upper[0]).reshape(-1, 1)
    outlet_fluid2_right_y = (np.linspace(bound_domain_fluid2_lower[1], bound_domain_fluid2_upper[1], int(density_line * (bound_domain_fluid2_upper[1] - bound_domain_fluid2_lower[1])))).reshape(-1, 1)
    outlet_fluid2_right = np.hstack([outlet_fluid2_right_x, outlet_fluid2_right_y])
    outlet_fluid2 = np.vstack([outlet_fluid2_left, outlet_fluid2_right])

    'BC of Solid'
    bc_solid_top_x = (np.linspace(bound_domain_solid_lower[0], bound_domain_solid_upper[0], int(density_line * (bound_domain_solid_upper[0]-bound_domain_solid_lower[0])))).reshape(-1, 1)
    bc_solid_top_y = (np.ones(int(density_line * (bound_domain_solid_upper[0]-bound_domain_solid_lower[0]))) * bound_domain_solid_upper[1]).reshape(-1, 1)
    
    bc_solid_bottom_x = (np.linspace(bound_domain_solid_lower[0], bound_domain_solid_upper[0], int(density_line * (bound_domain_solid_upper[0]-bound_domain_solid_lower[0])))).reshape(-1, 1)
    bc_solid_bottom_y = (np.ones(int(density_line * (bound_domain_solid_upper[0]-bound_domain_solid_lower[0]))) * bound_domain_solid_lower[1]).reshape(-1, 1)

    bc_solid_left_x = (np.ones(int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))) * bound_domain_solid_lower[0]).reshape(-1, 1)
    bc_solid_left_y = np.linspace(bound_domain_solid_lower[1], bound_domain_solid_upper[1], int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))).reshape(-1, 1)

    bc_solid_right_x = (np.ones(int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1])))*bound_domain_solid_upper[0]).reshape(-1, 1)
    bc_solid_right_y = np.linspace(bound_domain_solid_lower[1], bound_domain_solid_upper[1], int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))).reshape(-1, 1)
    
    bc_solid_top = np.hstack([bc_solid_top_x, bc_solid_top_y])
    bc_solid_bottom = np.hstack([bc_solid_bottom_x, bc_solid_bottom_y])
    bc_solid_left = np.hstack([bc_solid_left_x, bc_solid_left_y])
    bc_solid_right = np.hstack([bc_solid_right_x, bc_solid_right_y])
    sb_solid = np.vstack([bc_solid_top, bc_solid_bottom, bc_solid_left, bc_solid_right])
    sb_solid1 = bc_solid_top
    sb_solid2 = bc_solid_bottom
    # 'SB1_Solid'
    # sb1_solid_left_x = (np.ones(int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))) * bound_domain_solid_lower[0]).reshape(-1, 1)
    # sb1_solid_left_y = np.linspace(bound_domain_solid_lower[1], bound_domain_solid_upper[1], int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))).reshape(-1, 1)
    # sb1_solid = np.hstack([sb1_solid_left_x, sb1_solid_left_y])

    # 'SB2_Solid'
    # sb2_solid_right_x = (np.ones(int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1])))*bound_domain_solid_upper[0]).reshape(-1, 1)
    # sb2_solid_right_y = np.linspace(bound_domain_solid_lower[1], bound_domain_solid_upper[1], int(density_line * (bound_domain_solid_upper[1]-bound_domain_solid_lower[1]))).reshape(-1, 1)
    # sb2_solid = np.hstack([sb2_solid_right_x, sb2_solid_right_y])

    'Indexing'
    index_domain_fluid1 = (np.ones(len(domain_fluid1))*collocation_indexing['domain_fluid1']).reshape(-1, 1)
    index_domain_fluid2 = (np.ones(len(domain_fluid2))*collocation_indexing['domain_fluid2']).reshape(-1, 1)
    index_bc_fluid1_top = (np.ones(len(bc_fluid1_top))*collocation_indexing['bc_fluid1_top']).reshape(-1, 1)
    index_bc_fluid1_bottom = (np.ones(len(bc_fluid1_bottom))*collocation_indexing['bc_fluid1_bottom']).reshape(-1, 1)
    index_bc_fluid2_top = (np.ones(len(bc_fluid2_top))*collocation_indexing['bc_fluid2_top']).reshape(-1, 1)
    index_bc_fluid2_bottom = (np.ones(len(bc_fluid2_bottom))*collocation_indexing['bc_fluid2_bottom']).reshape(-1, 1)
    index_inlet_fluid1 = (np.ones(len(inlet_fluid1))*collocation_indexing['inlet_fluid1']).reshape(-1, 1)
    index_outlet_fluid1 = (np.ones(len(outlet_fluid1))*collocation_indexing['outlet_fluid1']).reshape(-1, 1)
    index_inlet_fluid2 = (np.ones(len(inlet_fluid2))*collocation_indexing['inlet_fluid2']).reshape(-1, 1)
    index_outlet_fluid2 = (np.ones(len(outlet_fluid2))*collocation_indexing['outlet_fluid2']).reshape(-1, 1)
    index_domain_solid = (np.ones(len(domain_solid))*collocation_indexing['domain_solid']).reshape(-1, 1)
    index_bc_solid_top = (np.ones(len(bc_solid_top))*collocation_indexing['bc_solid_top']).reshape(-1, 1)
    index_bc_solid_bottom = (np.ones(len(bc_solid_bottom))*collocation_indexing['bc_solid_bottom']).reshape(-1, 1)

    domain_fluid1 = np.hstack([index_domain_fluid1, domain_fluid1])
    domain_fluid2 = np.hstack([index_domain_fluid2, domain_fluid2])
    bc_fluid1_top = np.hstack([index_bc_fluid1_top, bc_fluid1_top])
    bc_fluid1_bottom = np.hstack([index_bc_fluid1_bottom, bc_fluid1_bottom])
    bc_fluid2_top = np.hstack([index_bc_fluid2_top, bc_fluid2_top])
    bc_fluid2_bottom = np.hstack([index_bc_fluid2_bottom, bc_fluid2_bottom])
    inlet_fluid1 = np.hstack([index_inlet_fluid1, inlet_fluid1])
    outlet_fluid1 = np.hstack([index_outlet_fluid1, outlet_fluid1])
    inlet_fluid2 = np.hstack([index_inlet_fluid2, inlet_fluid2])
    outlet_fluid2 = np.hstack([index_outlet_fluid2, outlet_fluid2])
    domain_solid = np.hstack([index_domain_solid, domain_solid])
    bc_solid_top = np.hstack([index_bc_solid_top, bc_solid_top])
    bc_solid_bottom = np.hstack([index_bc_solid_bottom, bc_solid_bottom])

    collocation_points = np.vstack([domain_fluid1, domain_fluid2, bc_fluid1_top, bc_fluid1_bottom, bc_fluid2_top, bc_fluid2_bottom, inlet_fluid1, outlet_fluid1, 
                                    inlet_fluid2, outlet_fluid2, domain_solid, bc_solid_top, bc_solid_bottom])
    collocation_points[:, 1] -= 0.7
    collocation_points[:, 2] -= 0.5
    
    'Time Addition'
    if Transient == True:
        time_end = 1
        t = np.random.uniform(0, time_end, len(collocation_points)).reshape(-1, 1)
        collocation_points = np.hstack([collocation_points, t])
    
    'Plot'
    if Plot == True:
        data = collocation_points[:, 1:]
        label = collocation_points[:,0]
        for i in range(len(collocation_indexing)):
            a = data[np.where(label == i)]
            plt.scatter(a[:, 0], a[:, 1], s=0.1)
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