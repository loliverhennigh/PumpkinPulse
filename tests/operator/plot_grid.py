

import numpy as np
import matplotlib.pyplot as plt

f_00 = -0.0
f_01 = -0.0
f_02 =  1.0
f_10 = -0.0
f_11 =  1.0
f_12 =  1.0
f_20 = -0.0
f_21 = -0.0
f_22 =  1.0


#def shape_func(x, x0):
#    w = 1.0 - np.abs(x - x0)
#    w = np.maximum(w, np.zeros_like(w))
#    return w
#
#def l1_sdf(x, y):
#    dist_00 = shape_func(x, 0.0) * shape_func(y, 0.0)
#    dist_01 = shape_func(x, 0.0) * shape_func(y, 1.0)
#    dist_02 = shape_func(x, 0.0) * shape_func(y, 2.0)
#    dist_10 = shape_func(x, 1.0) * shape_func(y, 0.0)
#    dist_11 = shape_func(x, 1.0) * shape_func(y, 1.0)
#    dist_12 = shape_func(x, 1.0) * shape_func(y, 2.0)
#    dist_20 = shape_func(x, 2.0) * shape_func(y, 0.0)
#    dist_21 = shape_func(x, 2.0) * shape_func(y, 1.0)
#    dist_22 = shape_func(x, 2.0) * shape_func(y, 2.0)
#    return dist_00 * f_00 + dist_01 * f_01 + dist_02 * f_02 + dist_10 * f_10 + dist_11 * f_11 + dist_12 * f_12 + dist_20 * f_20 + dist_21 * f_21 + dist_22 * f_22

def shape_func(x, x0, y, y0):
    w = 1.0 - np.minimum(np.abs(x - x0), np.ones_like(x)) - np.minimum(np.abs(y - y0), np.ones_like(x)) + np.minimum(np.abs(x - x0), np.ones_like(x)) * np.minimum(np.abs(y - y0), np.ones_like(x))
    #w = 1.0 - np.minimum(np.abs(x - x0), np.ones_like(x)) - np.minimum(np.abs(y - y0), np.ones_like(x)) + np.maximum(np.minimum(np.abs(x - x0), np.ones_like(x)), np.minimum(np.abs(y - y0), np.ones_like(x)))
    w = np.maximum(w, np.zeros_like(w))
    return w

def l1_sdf(x, y):
    dist_00 = shape_func(x, 0.0, y, 0.0)
    dist_01 = shape_func(x, 0.0, y, 1.0)
    dist_02 = shape_func(x, 0.0, y, 2.0)
    dist_10 = shape_func(x, 1.0, y, 0.0)
    dist_11 = shape_func(x, 1.0, y, 1.0)
    dist_12 = shape_func(x, 1.0, y, 2.0)
    dist_20 = shape_func(x, 2.0, y, 0.0)
    dist_21 = shape_func(x, 2.0, y, 1.0)
    dist_22 = shape_func(x, 2.0, y, 2.0)
    return dist_00 * f_00 + dist_01 * f_01 + dist_02 * f_02 + dist_10 * f_10 + dist_11 * f_11 + dist_12 * f_12 + dist_20 * f_20 + dist_21 * f_21 + dist_22 * f_22






x_lin = np.linspace(0.0, 2.0, 1000)
y_lin = np.linspace(0.0, 2.0, 1000)
X, Y = np.meshgrid(x_lin, y_lin)
sdf = l1_sdf(X, Y)
sdf[375, 375] = 0.0
plt.imshow(sdf, extent=(0, 2, 0, 2), origin='lower')
plt.colorbar()
#plt.clim(-0.01, 0.01)
plt.clim(0.50, 0.501)
plt.show()
