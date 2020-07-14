#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:19:39 2020

@author: jethro
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d

# geometry parameters
xdim = 1
ydim = 1
xmin = 0.0
ymin = 0.0
xmax = 1.0
ymax = 1.0
# input parameters

rho =1
cp = 1
k = 1
T0 = 0

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

def gen_testdata():
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    xx, yy = np.meshgrid(x, y) 
    X = np.vstack((np.ravel(xx), np.ravel(yy))).T 
    X = np.hstack((X,0.2*np.ones((10000,1))))
    return X


def main():
    def pde(x, T):
        dT_x = tf.gradients(T, x)[0]
        dT_x, dT_y, dT_t = dT_x[:,0:1], dT_x[:,1:2], dT_x[:,2:]
        dT_xx = tf.gradients(dT_x, x)[0][:, 0:1]
        dT_yy = tf.gradients(dT_y, x)[0][:, 1:2]
        return rho*cp*dT_t -  k*dT_xx - k*dT_yy

    def boundary_x_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], xmin)

    def boundary_x_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], xmax)

    def boundary_y_b(x, on_boundary):
        return on_boundary and np.isclose(x[1], ymin)

    def boundary_y_u(x, on_boundary):
        return on_boundary and np.isclose(x[1], ymax)

    def func(x):
        return np.zeros((len(x),1), dtype=np.float32)*T0

    def func_n(x):
        return np.sin(2 * np.pi * x[:, 0:1])**2#np.zeros((len(x),1), dtype=np.float32)


    geom = dde.geometry.Rectangle([0, 0], [xmax, ymax])
    timedomain = dde.geometry.TimeDomain(t0, te)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_x_l = dde.DirichletBC(geomtime, func, boundary_x_l)
    bc_x_r = dde.DirichletBC(geomtime, func, boundary_x_r)
    bc_y_b = dde.DirichletBC(geomtime, func_n, boundary_y_b)
    bc_y_u = dde.DirichletBC(geomtime, func, boundary_y_u)
    ic = dde.IC(geomtime, func, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_x_l, bc_x_r, bc_y_b, bc_y_u, ic],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        # train_distribution="uniform",
        num_test=num_test
    )
    net = dde.maps.FNN([3] + [hidden_layer_size] * num_hidden_layer + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=lr)
    losshistory, train_state = model.train(epochs=epochs)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=False, isplot=False)
    
    X = gen_testdata()
    y_pred = model.predict(X)
    np.savetxt("test.dat", np.hstack((X, y_pred)))
    
if __name__ == "__main__":
    main()