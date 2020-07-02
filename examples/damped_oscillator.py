#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:38:34 2020

@author: jethro
"""
#Damped oscillator

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    def ode(t, y):
        """ODE.
        dy_tt + 0.5*dy_t + y = 0
        """
        dy_t = tf.gradients(y, t)[0]
        dy_tt = tf.gradients(dy_t, t)[0]
        
        return dy_tt + 0.5*dy_t + y
    
    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

    def func(t):

        x = np.sqrt(1-(0.5**2)/4)*t 
        d = np.sqrt(4/(0.5**2)-1)
        return np.exp(-0.25*t)*(np.cos(x) + (np.sin(x)/d))

    geom = dde.geometry.Interval(0, 32)
    bc_l = dde.DirichletBC(geom, lambda x: np.ones((len(x), 1)), boundary)
    bc_r = dde.NeumannBC(geom, lambda x: np.zeros((len(x), 1)), boundary)
    data = dde.data.PDE(geom, ode, [bc_l, bc_r], 200, 1, solution=func, num_test=100)

    layer_size = [1] + [100] * 5 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=20000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()