from examples.sis.implementation import sis
from examples.numerical_examples.performance_functions import c_pwl, c_meatball
import matplotlib.pyplot as plt
from examples.figures.utils import contour_plot, colors


sis_results = sis(d=2,
                  g=c_pwl,
                  p=0.1,
                  N=1000,
                  burn=0,
                  tarCOV=1.5,
                  seed=0)

plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
plt.grid(False)
plt.xlim(-4,6)
plt.ylim(-4,6)

contour_plot((-4,6), (-4,6), 0.1, c_pwl, color='black',levels=[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.scatter(4,0,s=200,c='black',marker='s',zorder=3)

for samples in sis_results.samples:
    samples = samples[::5]
    plt.scatter(samples[:,0],samples[:,1],color=colors[0],s=10)

plt.savefig('examples/figures/images/sis_pwl.pdf',bbox_inches='tight')

plt.show()


sis_results = sis(d=2,
                  g=c_meatball,
                  p=0.1,
                  N=1000,
                  burn=0,
                  tarCOV=1.5,
                  seed=0)

plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
ax = plt.gca()
plt.grid(False)
plt.xlim(-6,6.5)
plt.ylim(-8,8)

contour_plot((-6,6), (-8,8), 0.1, c_meatball, color='black',levels=[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.scatter(-4.265,0,s=200,c='black',marker='s',zorder=3,label='Design point')

first = True
for samples in sis_results.samples:
    samples = samples[::5]
    if first:
        first = False
        plt.scatter(samples[:,0],samples[:,1],color=colors[0],s=10,label='Samples')
    else:
        plt.scatter(samples[:,0],samples[:,1],color=colors[0],s=10)

plt.plot([], [], color='black', linewidth=2, label='Limit state surface')


plt.savefig('examples/figures/images/sis_meatball.pdf',bbox_inches='tight')

plt.show()


plt.style.use('default')
plt.figure(figsize=(1,1))
plt.axis('off')
plt.legend(*ax.get_legend_handles_labels(),ncols=2)
plt.savefig('examples/figures/images/sis_fail_legend.pdf',bbox_inches='tight')
plt.show()
