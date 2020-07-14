#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:08:37 2020

@author: jethro
"""

from deepxde.boundary_conditions import DirichletBC
from deepxde.geometry.geometry_2d import Polygon
from deepxde.callbacks import EarlyStopping
from importlib import import_module
from deepxde.maps.fnn import FNN
from deepxde.data.pde import PDE
from deepxde.model import Model
import matplotlib.pyplot as plt
from deepxde.backend import tf
import numpy as np


def plot_points(points, color="k", marker="."):
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.scatter(points[:, 0], points[:, 1], color=color, marker=marker)
    plt.show()


def wall_top_boundary(x, on_boundary):
    """Checks for points on top wall boundary"""
    return on_boundary and np.isclose(x[1], 2.0)  #x[1] refers to y 


def wall_bottom_boundary(x, on_boundary):
    """Checks for points on bottom wall boundary"""
    return on_boundary and np.isclose(x[1], 0.0)


def wall_mid_horizontal_boundary(x, on_boundary):
    """Check for points on step horizontal boundary"""
    return on_boundary and (np.isclose(x[1], 1.0) and x[0] < 2.0)


def wall_mid_vertical_boundary(x, on_boundary):
    """Check for points on step horizontal boundary"""
    return on_boundary and (x[1] < 1.0 and np.isclose(x[0], 2.0))


def outlet_boundary(x, on_boundary):
    """Implements the outlet boundary with zero y-velocity component"""
    return on_boundary and np.isclose(x[0], 12.0)  #x[0] refers to x


def inlet_boundary(x, on_boundary):
    """Implements the inlet boundary with parabolic x-velocity component"""
    return on_boundary and np.isclose(x[0], 0.0)


def parabolic_velocity(x):
    """Parabolic velocity"""
    return (6 * (x[:, 1] - 1) * (2 - x[:, 1])).reshape(-1, 1)


def zero_velocity(x):
    """Zero velocity"""
    return np.zeros((x.shape[0], 1))


def navier_stokes(x, y): #x is input to NN and y is output
    """Navier-Stokes equation"""
    rho = 1.0
    nu = 0.01
    eps = 1e-8

    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    du = tf.gradients(u, x)[0]
    dv = tf.gradients(v, x)[0]
    dp = tf.gradients(p, x)[0]

    p_x, p_y = dp[:, 0:1], dp[:, 1:2]
    u_x, u_y = du[:, 0:1], du[:, 1:2]
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]

    u_xx = tf.gradients(u_x, x)[0][:, 0:1]
    u_yy = tf.gradients(u_y, x)[0][:, 1:2]

    v_xx = tf.gradients(v_x, x)[0][:, 0:1]
    v_yy = tf.gradients(v_y, x)[0][:, 1:2]

    continuity = u_x + v_y + eps * p
    x_momentum = u * u_x + v * u_y + 1 / rho * p_x - nu * (u_xx + u_yy)
    y_momentum = u * v_x + v * v_y + 1 / rho * p_y - nu * (v_xx + v_yy)

    return [continuity, x_momentum, y_momentum]


if __name__ == '__main__':
    geom = Polygon([
        [0.0, 2.0], [12.0, 2.0], [12.0, 0.0], [2.0, 0.0], [2.0, 1.0],
        [0.0, 1.0]
    ])

    inlet_x = DirichletBC(geom, parabolic_velocity, inlet_boundary,
                          component=0)
    inlet_y = DirichletBC(geom, zero_velocity, inlet_boundary, component=1)
    outlet = DirichletBC(geom, zero_velocity, outlet_boundary, component=1)
    wallt_x = DirichletBC(geom, zero_velocity, wall_top_boundary, component=0)
    wallt_y = DirichletBC(geom, zero_velocity, wall_top_boundary, component=1)
    wallb_x = DirichletBC(geom, zero_velocity, wall_bottom_boundary,
                          component=0)
    wallb_y = DirichletBC(geom, zero_velocity, wall_bottom_boundary,
                          component=1)
    wallsh_x = DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary,
                           component=0)
    wallsh_y = DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary,
                           component=1)
    wallsv_x = DirichletBC(geom, zero_velocity, wall_mid_vertical_boundary,
                           component=0)
    wallsv_y = DirichletBC(geom, zero_velocity, wall_mid_vertical_boundary,
                           component=1)

    data = PDE(
        geom, navier_stokes,
        [inlet_x, inlet_y, outlet, wallb_x, wallb_y, wallsh_x, wallsh_y,
         wallsv_x, wallsv_y, wallt_x, wallt_y],
        num_domain=1000, num_boundary=3000,
        num_test=5000
    )

    layer_size = [2] + [50] * 3 + [3]
    net = FNN(layer_size, "tanh", "Glorot uniform")

    model = Model(data, net)
    model.compile("adam", lr=0.001)

    early_stopping = EarlyStopping(min_delta=1e-8, patience=10000)
    model.train(epochs=25000, display_every=100,
                disregard_previous_best=True, callbacks=[early_stopping])