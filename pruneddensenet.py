# Jack DeLano
# Heavily modified from https://github.com/andreasveit/densenet-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MaskedBottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, mask_holes=24):
        super(MaskedBottleneckBlock, self).__init__()
        self.mask = nn.Parameter(torch.BoolTensor(in_planes, 1, 1).fill_(True), requires_grad=False)
        self.maskHoles = mask_holes
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*out_planes, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(4*out_planes)
        self.conv2 = nn.Conv2d(4*out_planes, out_planes, kernel_size=3, stride=1, padding=1)
    
    def createMask(self):
        with torch.no_grad():
            # get abs of kernel weights (out_chans x in_chans)
            w = self.conv1.weight.data.squeeze()
            
            # scale each row (i.e. out channel "sub kernel") to be a unit vector
            w = (w*(1./(w.norm(dim=1)).unsqueeze(1)))
            
            # "importance" of each input channel = norm of each column vector (of row-wise unit-scaled weight matrix)
            importance = w.norm(dim=0)
            
            bestPlaneIdxs = importance.topk(self.maskHoles, sorted=False)[1]
            
            in_planes = self.mask.size(0)
            
            self.mask.data.fill_(False)
            self.mask.data.index_fill_(0, bestPlaneIdxs, True)
    
    def forward(self, x):
        out = x*self.mask
        out = self.conv1(self.relu(self.bn1(out)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)

class PrunedBottleneckLayer(nn.Module):
    def __init__(self, in_planes, out_planes, prune_factor):
        super(PrunedBottleneckLayer, self).__init__()
        self.inPlanes = in_planes
        self.outPlanes = out_planes
        self.prunedInPlanes = int(prune_factor*in_planes)
        
        self.bestInPlanes = nn.Parameter(torch.tensor(list(range(in_planes))), requires_grad=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        #initialized here to ensure it's on the correct device (i.e. gpu, if desired)
        self.bn1Pruned = nn.BatchNorm2d(self.prunedInPlanes)
        
        stdv = 1./math.sqrt(in_planes)
        self.conv1Weight = nn.Parameter(torch.Tensor(4*out_planes, in_planes, 1, 1).uniform_(-stdv, stdv))
        self.conv1Bias = nn.Parameter(torch.Tensor(4*out_planes).uniform_(-stdv, stdv))
        
        self.bn2 = nn.BatchNorm2d(4*out_planes)
        self.conv2 = nn.Conv2d(4*out_planes, out_planes, kernel_size=3, stride=1, padding=1)
    
    def prune(self):
        with torch.no_grad():
            # get abs of kernel weights (out_chans x in_chans)
            w = self.conv1Weight.squeeze()
            
            # scale each row (i.e. out channel "sub kernel") to be a unit vector
            w = (w*(1./(w.norm(dim=1)).unsqueeze(1)))
            
            # "importance" of each input channel = norm of each column vector (of row-wise unit-scaled weight matrix)
            importance = w.norm(dim=0)
            
            bestInPlanesPruned = nn.Parameter(self.bestInPlanes.new_empty(self.prunedInPlanes), requires_grad=False)
            bestInPlanesPruned.copy_(importance.topk(self.prunedInPlanes, sorted=True)[1])
            self.bestInPlanes = bestInPlanesPruned
            
            conv1WeightPruned = nn.Parameter(self.conv1Weight.new_empty(4*self.outPlanes, self.prunedInPlanes, 1, 1))
            conv1WeightPruned.copy_(self.conv1Weight.index_select(1, self.bestInPlanes))
            self.conv1Weight = conv1WeightPruned
            
            bn1p_sd = self.bn1Pruned.state_dict()
            bn1p_sd['weight'] = self.bn1.weight.index_select(0, self.bestInPlanes)
            bn1p_sd['bias'] = self.bn1.bias.index_select(0, self.bestInPlanes)
            bn1p_sd['running_mean'] = self.bn1.running_mean.index_select(0, self.bestInPlanes)
            bn1p_sd['running_var'] = self.bn1.running_var.index_select(0, self.bestInPlanes)
            bn1p_sd['num_batches_tracked'] = self.bn1.num_batches_tracked
            self.bn1Pruned.load_state_dict(bn1p_sd)
            self.bn1 = self.bn1Pruned
    
    def forward(self, x):
        out = x.index_select(1, self.bestInPlanes)
        out = self.relu(self.bn1(out))
        out = F.conv2d(out, self.conv1Weight, self.conv1Bias, stride=1, padding=0)
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)

class PrunedDenseBlock(nn.Module):
    def __init__(self, num_layers, in_planes, growth_rate, prune_factor):
        super(PrunedDenseBlock, self).__init__()
        self.pbls = nn.ModuleList()
        for i in range(num_layers):
            self.pbls.append(PrunedBottleneckLayer(in_planes+i*growth_rate, growth_rate, prune_factor))
    
    def prune(self):
        for pbl in self.pbls:
            pbl.prune()
    
    def forward(self, x):
        for pbl in self.pbls:
            x = pbl(x)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
            
        return out

class PrunedDenseNet(nn.Module):
    def __init__(self, prune_factor):
        super(PrunedDenseNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
        
        self.pdb1 = PrunedDenseBlock(6, 24, 12, prune_factor) # 24 + 6*12 = 96
        self.pdb2 = PrunedDenseBlock(6, 96, 12, prune_factor) # 96 + 6*12 = 168
        self.pdb3 = PrunedDenseBlock(6, 168, 12, prune_factor) # 168 + 6*12 = 240
        
        self.bn = nn.BatchNorm2d(240)
        self.fc = nn.Linear(240, 100)
        self.relu = nn.ReLU(inplace=True)
        
    def prune(self):
        self.pdb1.prune()
        self.pdb2.prune()
        self.pdb3.prune()
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.pdb1(x)
        x = F.avg_pool2d(x, 2)
        x = self.pdb2(x)
        x = F.avg_pool2d(x, 2)
        x = self.pdb3(x)
        
        x = self.relu(self.bn(x))
        x = F.avg_pool2d(x, 8)
        
        x = x.view(-1, 240)
        x = self.fc(x)
        
        return x

class PrunedDenseNetWithCompression(nn.Module):
    def __init__(self, prune_factor):
        super(PrunedDenseNetWithCompression, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
        
        self.pdb1 = PrunedDenseBlock(6, 24, 12, prune_factor) # 24 + 6*12 = 96
        self.tb1 = TransitionBlock(96, 48)
        self.pdb2 = PrunedDenseBlock(6, 48, 12, prune_factor) # 48 + 6*12 = 120
        self.tb2 = TransitionBlock(120, 60)
        self.pdb3 = PrunedDenseBlock(6, 60, 12, prune_factor) # 60 + 6*12 = 132
        
        self.bn = nn.BatchNorm2d(132)
        self.fc = nn.Linear(132, 100)
        self.relu = nn.ReLU(inplace=True)
        
    def prune(self):
        self.pdb1.prune()
        self.pdb2.prune()
        self.pdb3.prune()
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.pdb1(x)
        x = self.tb1(x)
        x = self.pdb2(x)
        x = self.tb2(x)
        x = self.pdb3(x)
        
        x = self.relu(self.bn(x))
        x = F.avg_pool2d(x, 8)
        
        x = x.view(-1, 132)
        x = self.fc(x)
        
        return x