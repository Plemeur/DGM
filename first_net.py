import time
import torch
import torch.nn as nn
import numpy as np

#from derivatives import gradient, hessian
import matplotlib.pyplot as plt
import math

np.random.seed(42)
torch.manual_seed(42)


class Linear(nn.Module):
    """ Copy of linear module from Pytorch, modified to have a Xavier init,
        TODO : figure out what to do with the bias"
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -1, 1) #boundary matter?

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class DGM_layer(nn.Module):
    """ See readme for paper source"""
    def __init__(self, in_features, out_feature, residual=False):
        super(DGM_layer, self).__init__()
        self.residual = residual

        self.Z = Linear(out_feature, out_feature)
        self.UZ = Linear(in_features, out_feature, bias=False)
        self.G = Linear(out_feature, out_feature)
        self.UG = Linear(in_features, out_feature, bias=False)
        self.R = Linear(out_feature, out_feature)
        self.UR = Linear(in_features, out_feature, bias=False)
        self.H = Linear(out_feature, out_feature)
        self.UH = Linear(in_features, out_feature, bias=False)

    def forward(self, x, s):
        z = torch.tanh(self.UZ(x) + self.Z(s))
        g = torch.tanh(self.UG(x) + self.G(s))
        r = torch.tanh(self.UR(x) + self.R(s))
        h = torch.tanh(self.UH(x) + self.H(s * r))
        return (1 - g) * h + z * s


class Net(nn.Module):

    def __init__(self, in_size, out_size, neurons, depth):
        super(Net, self).__init__()
        self.dim = in_size
        self.input_layer = Linear(in_size, neurons)
        self.middle_layer = nn.ModuleList([DGM_layer(in_size, neurons) for i in range(depth)])
        self.final_layer = Linear(neurons, out_size)

    def forward(self, X):
        s = torch.tanh(self.input_layer(X))
        for i, layer in enumerate(self.middle_layer):
            s = torch.tanh(layer(X, s))

        return self.final_layer(s)

