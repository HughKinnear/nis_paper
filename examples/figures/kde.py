import pickle
import matplotlib.pyplot as plt
from examples.figures.utils import colors
import seaborn as sns

filename = 'examples/numerical_examples/pwl/sis_pwl.pkl'
with open(filename, "rb") as file:
    results = pickle.load(file)

sis_probs = results[0]['estimates']
ref_prob = 3.2e-05

plt.style.use('ggplot')

plt.figure(figsize=(10, 7))

sns.kdeplot(data=sis_probs,
            fill=True,
            linewidth=1,
            log_scale=True,
            color=colors[3],
            )

plt.axvline(ref_prob,
            color='black',
            ymax=1,
            alpha=1,
            linestyle='dashed',
            linewidth=1,
            label='Ref Probability')

plt.xlabel('Failure Probability Estimate',fontsize=15)
plt.ylabel('Denstiy',fontsize=15)
plt.minorticks_off()
plt.tick_params(labelsize=15)
plt.legend(fontsize=15,frameon=True,facecolor='white',loc='upper right')
plt.savefig('examples/figures/images/sis_kde.pdf',bbox_inches='tight')
plt.show()

