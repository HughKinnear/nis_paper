from nis.nis import NichingImportanceSampling
from rough.benchmarks import c_meatball
import matplotlib.pyplot as plt
from examples.figures.utils import contour_plot, colors
import numpy as np


nis = NichingImportanceSampling(performance_function=c_meatball,
                                dimension=2,
                                seed=17,
                                fitting_multiplier=30)


nis.run()



plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
plt.grid(False)
plt.xlim(-6,6.5)
plt.ylim(-8,8)

contour_plot((-6,6), (-8,8), 0.1, c_meatball, color='black',levels=[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.scatter(-4.265,0,s=200,c='black',marker='s',zorder=3,label='Design point')

is_first = True
for chain_run in nis.initial_sampler.chain_runs:
    label = 'Markov chain' if is_first else None
    is_first = False
    chain_array = np.array([samp.array for samp in chain_run.all_samples])
    plt.plot(chain_array[:,0],chain_array[:,1],c=colors[3],zorder=1,marker='o',markersize=3,label=label)

is_first = True
for initial in nis.initial_sampler.initial_samples:
    label = 'Initial sample' if is_first else None
    is_first = False
    plt.scatter(initial.array[0],initial.array[1],c=colors[0],marker='o',
                s=100,zorder=2,label=label)
    
plt.plot([], [], color='black', linewidth=2, label='Limit state surface')

ax = plt.gca()

    
plt.savefig('examples/figures/images/nis_meatball_initial.pdf',bbox_inches='tight')

plt.show()

plt.style.use('default')
plt.figure(figsize=(1,1))
plt.axis('off')
plt.legend(*ax.get_legend_handles_labels(),ncols=2)
plt.savefig('examples/figures/images/nis_initial_legend.pdf',bbox_inches='tight')
plt.show()




plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
plt.grid(False)
plt.xlim(-6,6.5)
plt.ylim(-8,8)

plt.plot([], [], color='black', linewidth=2, label='Limit state surface')

contour_plot((-6,6), (-8,8), 0.1, c_meatball, color='black',levels=[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.scatter(-4.265,0,s=200,c='black',marker='s',zorder=3,label='Design point')

samples= np.vstack([chain_data.all_accept_samples for chain_data in nis.mixture_chain_data])

plt.scatter(samples[:,0],samples[:,1],c=colors[0],s=20,label='Markov chain samples')


ax = plt.gca()


plt.savefig('examples/figures/images/nis_meatball_chain.pdf',bbox_inches='tight')

plt.show()

plt.style.use('default')
plt.figure(figsize=(1,1))
plt.axis('off')
plt.legend(*ax.get_legend_handles_labels(),ncols=2)
plt.savefig('examples/figures/images/nis_chain_legend.pdf',bbox_inches='tight')
plt.show()





plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
plt.grid(False)
plt.xlim(-6,6.5)
plt.ylim(-8,8)

plt.plot([], [], color='black', linewidth=2, label='Limit state surface')

contour_plot((-6,6), (-8,8), 0.1, c_meatball, color='black',levels=[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.scatter(-4.265,0,s=200,c='black',marker='s',zorder=3,label='Design point')

samples = nis.importance_sampler.importance_samples

plt.scatter(samples[:,0],samples[:,1],c=colors[0],s=20,label='Importance samples')

ax = plt.gca()





plt.savefig('examples/figures/images/nis_meatball_import.pdf',bbox_inches='tight')

plt.show()


plt.style.use('default')
plt.figure(figsize=(1,1))
plt.axis('off')
plt.legend(*ax.get_legend_handles_labels(),ncols=2)
plt.savefig('examples/figures/images/nis_import_legend.pdf',bbox_inches='tight')
plt.show()