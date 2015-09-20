import json
import pandas as pd
import seaborn as sns
import scipy as sp

def load_results(experiment_name):
    results = json.load(open(str.format('results/{}.json', experiment_name), 'r'))
    timesteps = sorted(list(set([r['n_timesteps'] for r in results])))
    neurons = sorted(list(set([r['n_neurons'] for r in results])))
    df = pd.DataFrame(index=neurons, columns=timesteps, dtype=float)
    for r in results:
        seq_length = (r['n_timesteps'] - 1)/2
        n_unpredictable = seq_length - 1
        n_predicted = r['n_timesteps'] - 1
        loss_floor = sp.log(2)*n_unpredictable/n_predicted
        loss_ceil = sp.log(2)
        df.loc[r['n_neurons'], r['n_timesteps']] = (r['loss'] - loss_floor)/(loss_ceil - loss_floor)

    ax = sns.heatmap(df, vmin=0, vmax=1)

    print(df.loc[1])

    return ax

ax = load_results('saturation%-1')
ax.figure.savefig('results/tmp.png', bbox_inches='tight')
