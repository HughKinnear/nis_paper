from nis.nis import NichingImportanceSampling
from rough.benchmarks import c_pwl
import matplotlib.pyplot as plt
from examples.figures.utils import contour_plot, colors
import numpy as np

nis = NichingImportanceSampling(performance_function=c_pwl,
                                dimension=2,
                                seed=0,
                                fitting_multiplier=30)

nis.run()

plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
plt.grid(False)
plt.xlim(-4,6)
plt.ylim(-4,6)

contour_plot((-4,6), (-4,6), 0.1, c_pwl, color='black',levels=[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.scatter(4,0,s=200,c='black',marker='s',zorder=3)


for chain_run in nis.initial_sampler.chain_runs:
    chain_array = np.array([samp.array for samp in chain_run.all_samples])
    plt.plot(chain_array[:,0],chain_array[:,1],c=colors[3],zorder=1,marker='o',markersize=3)

for initial in nis.initial_sampler.initial_samples:
    plt.scatter(initial.array[0],initial.array[1],c=colors[0],marker='o',
                s=100,zorder=2)

    
plt.savefig('examples/figures/images/nis_pwl_initial.pdf',bbox_inches='tight')

plt.show()

plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
plt.grid(False)
plt.xlim(-4,6)
plt.ylim(-4,6)

contour_plot((-4,6), (-4,6), 0.1, c_pwl, color='black',levels=[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.scatter(4,0,s=200,c='black',marker='s',zorder=3)

samples= np.vstack([chain_data.all_accept_samples for chain_data in nis.mixture_chain_data])

plt.scatter(samples[:,0],samples[:,1],c=colors[0],s=20,label='Markov chain samples')



plt.savefig('examples/figures/images/nis_pwl_chain.pdf',bbox_inches='tight')

plt.show()

plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
plt.grid(False)
plt.xlim(-4,6)
plt.ylim(-4,6)

contour_plot((-4,6), (-4,6), 0.1, c_pwl, color='black',levels=[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.scatter(4,0,s=200,c='black',marker='s',zorder=3)

samples = nis.importance_sampler.importance_samples

plt.scatter(samples[:,0],samples[:,1],c=colors[0],s=20)




plt.savefig('examples/figures/images/nis_pwl_import.pdf',bbox_inches='tight')

plt.show()