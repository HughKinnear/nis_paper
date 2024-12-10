import matplotlib.pyplot as plt
from examples.numerical_examples.performance_functions import pwl
import numpy as np
from examples.figures.utils import contour_plot, eval_3d, colors



path_x_1 = np.arange(0,4,0.1)
path_y_1 = np.zeros(len(path_x_1)) - 1.5
path_z_1 = np.array([pwl(pt) for pt in np.vstack((path_x_1,path_y_1)).T])


path_y_2 = np.arange(1.5,5,0.1)
path_x_2 = np.zeros(len(path_y_2))
path_z_2 = np.array([pwl(pt) for pt in np.vstack((path_x_2,path_y_2)).T])

points = contour_plot((-4,6), (-4,6), 0.1, pwl, color='black',levels=[0])
lss_xy = points.get_paths()[0].vertices.T
lss_x = lss_xy[0]
lss_y = lss_xy[1]
lss_z = np.array([pwl(pt) for pt in lss_xy.T])
plt.close()


performance_function = pwl

stride = 4
lw = 0.05
alpha = 1
alias = True

xx,yy,z = eval_3d(performance_function,(-4,6),(-4,6),0.1)


plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d',computed_zorder=False)
ax.view_init(30,-120)
ax.grid(False)

plt.xlabel(u'x\u2081', fontsize=20,labelpad=10)
plt.ylabel(u'x\u2082', fontsize=20,labelpad=10)
ax.text2D(0.07, 0.8, 'Performance', rotation=13, fontsize=14,  transform=ax.transAxes) 


ax.zaxis.set_ticks([-1,0,1,2])


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
          label='Limit state surface',
          lw=0.8)


ax.scatter(4,
           0,
           0,
           s=50,
           marker = 's',
           color='black',
           label = 'Design point',
           zorder=1)


ax.plot3D(path_x_1,
          path_y_1,
          path_z_1,
          alpha=1,
          color=colors[0],
          linestyle='dotted',
          lw=3)

ax.scatter(0,
              -1.5,
              performance_function((0,-1.5)),
              s=50,
              color=colors[0],
              marker='o',
              zorder=3,
              edgecolors='white'
              )

ax.scatter(4,
              -1.5,
              performance_function((4,-1.5)),
              s=50,
              color=colors[0],
              marker='s',
              zorder=3,
              edgecolors='white'
              )


ax.plot3D(path_x_2,
          path_y_2,
          path_z_2,
          alpha=1,
          color=colors[0],
          linestyle='dotted',
          label='Solution trajectory',
          lw=3)



ax.scatter(0,
              1.5,
              performance_function((0,1.5)),
              s=50,
              color=colors[0],
              marker='o',
              zorder=3,
              edgecolors='white',
              label='Initial sample'
              )

ax.scatter(0,
              5,
              performance_function((0,5)),
              s=50,
              color=colors[0],
              marker='s',
              zorder=3,
              edgecolors='white',
              label='Terminal point'
              )

plt.savefig('examples/figures/images/3d_pwl.pdf',bbox_inches='tight')

plt.show()

plt.style.use('default')
plt.figure(figsize=(1,1))
plt.axis('off')
plt.legend(*ax.get_legend_handles_labels(),ncols=2)
plt.savefig('examples/figures/images/3d_legend.pdf',bbox_inches='tight')
plt.show()
