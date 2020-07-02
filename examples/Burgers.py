from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def gen_testdata():
    data = np.load("dataset/Burgers.npz") # load the data from simulations
    t, x, exact = data["t"], data["x"], data["usol"].T # data: time, x-location, u velocity ##.T:transpose
    xx, tt = np.meshgrid(x, t) 
    ##ravel: converts matrix to array, vstack: stacks the array one on top of the other
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T  
    y = exact.flatten()[:, None] ##flatten does the same thing as ravel
    return X, y


def main():
    #the pde
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_x, dy_t = dy_x[:, 0:1], dy_x[:, 1:2]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    geom = dde.geometry.Interval(-1, 1) #physical domain
    timedomain = dde.geometry.TimeDomain(0, 0.99) #time domain
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    #boundary condition : 0 for all boundaries
    bc = dde.DirichletBC(
        geomtime, lambda x: np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary
    )
    #initial condition: -sin(pi*x)
    ic = dde.IC(
        geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
    )

    #specify the number of training points and their location
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
    )
    
    #NN structure
    net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    #train the model
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    model.train(epochs=15000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    #Test the model
    X, y_true = gen_testdata()
    #Predict values
    y_pred = model.predict(X)
    f = model.predict(X, operator=pde)
    print("Mean residual:", np.mean(np.absolute(f)))
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))


if __name__ == "__main__":
    main()
