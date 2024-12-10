import numpy as np
import matplotlib.pyplot as plt

colors = [
    '#e66101',
    '#fdb863',
    '#b2abd2',
    '#5e3c99'
    ]

def eval_3d(func,x_range,y_range,step):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(x, y)
    flat = np.vstack([xx.ravel(), yy.ravel()])
    z = np.array([func(point) for point in flat.T]).reshape(xx.shape)
    return xx,yy,z

def contour_plot(x_range, y_range, step, function, color='black',levels=None, points=False):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(x, y)
    flat = np.vstack([xx.ravel(), yy.ravel()])
    z = np.array([function(point) for point in flat.T]).reshape(xx.shape)
    points = plt.contour(xx, yy, z, levels=levels,colors=color)
    if points:
        return points