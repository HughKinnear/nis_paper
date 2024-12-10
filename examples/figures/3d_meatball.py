import matplotlib.pyplot as plt
from examples.numerical_examples.performance_functions import meatball
import numpy as np
from examples.figures.utils import contour_plot, eval_3d, colors


plt.style.use('default')

points = contour_plot((-8,8), (-8,8), 0.1, meatball, color='black',levels=[0])
lss_xy = points.get_paths()[0].vertices.T
lss_x = lss_xy[0]
lss_y = lss_xy[1]
lss_z = np.array([meatball(pt) for pt in lss_xy.T])
plt.close()


path_y_1 = np.arange(-5.5,0,0.1)
path_x_1 = np.zeros(len(path_y_1)) + 0.3
path_z_1 = np.array([meatball(pt) for pt in np.vstack((path_x_1,path_y_1)).T])


path_y_2 = np.arange(2,6,0.1)
path_x_2 = np.zeros(len(path_y_2)) + 0.3
path_z_2 = np.array([meatball(pt) for pt in np.vstack((path_x_2,path_y_2)).T])


performance_function = meatball

stride = 4
lw = 0.05
alpha = 1
alias = True

xx,yy,z = eval_3d(performance_function,(-6,6),(-8,8),0.1)


plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d',computed_zorder=False)
ax.view_init(40,250)
ax.grid(False)
plt.xlabel(u'x\u2081', fontsize=20,labelpad=10)
plt.ylabel(u'x\u2082', fontsize=20,labelpad=10)
ax.text2D(0.08, 0.83, 'Performance', rotation=13, fontsize=14,  transform=ax.transAxes) 

ax.yaxis.set_ticks([-8,-4,0,4,8])

ax.zaxis.set_ticks([-20,-10,0])



ax.plot_surface(xx,
                yy,
                z,
                rstride=stride,
                cstride=stride,
                antialiased=alias,
                color=colors[2],
                alpha=alpha,
                edgecolors='white',
                lw=lw)


ax.plot3D(lss_x,
          lss_y,
          lss_z,
          alpha=1,
          color='black',
          lw=0.8)

ax.scatter(-4.265,
           0,
           0,
           s=50,
           marker = 's',
           color='black',
           zorder=1)



ax.plot3D(path_x_1,
          path_y_1,
          path_z_1,
          alpha=1,
          color=colors[0],
          linestyle='dotted',
          lw=3)

ax.plot3D(path_x_2,
          path_y_2,
          path_z_2,
          alpha=1,
          color=colors[0],
          linestyle='dotted',
          lw=3)


ax.scatter(0.3,
              5.8,
              performance_function((0.3,5.8)),
              s=50,
              color=colors[0],
              marker='s',
              zorder=3,
              edgecolor='white'
              )

ax.scatter(0.3,
              0.3,
              performance_function((0.3,0.3)),
              s=50,
              color=colors[0],
              marker='o',
              zorder=3,
              edgecolors='white'
              )


ax.scatter(0.3,
              2,
              performance_function((0.3,2)),
              s=50,
              color=colors[0],
              marker='o',
              zorder=3,
              edgecolors='white'
              )

ax.scatter(0.3,
              -5.2,
              performance_function((0.3,-5.2)),
              s=50,
              color=colors[0],
              marker='s',
              zorder=3,
              edgecolors='white'
              )


plt.savefig('examples/figures/images/3d_meatball.pdf',bbox_inches='tight')

plt.show()
