#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:21:05 2020

@author: jethro
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf

# geometry parameters
xmin = 0.0
ymin = 0.0
xmax = 1.0
ymax = 1.0

# input parameters
t0 = 0.0
te = 0.3
x_start = 0.0 # laser start position

# dnn parameters
num_hidden_layer = 3 # number of hidden layers for DNN
hidden_layer_size = 60 # size of each hidden layers
num_domain=1000 # number of training points within domain Tf: random points (spatio-temporal domain)
num_boundary=1000 # number of training boundary condition points on the geometry boundary: Tb
num_initial= 1000 # number of training initial condition points: Tb
num_test=None # number of testing points within domain: uniform generated
epochs=20000 # number of epochs for training
lr=0.001 # learning rate