import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})
sns.axes_style()
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def plot_Scatter(x, y, output_dir='', name='Normal_pred_age'):
    x = np.squeeze(x)
    y = np.squeeze(y)
    plt.figure()
    plt.scatter(x, y, alpha=0.2)
    parameter = np.polyfit(x, y, 1)
    f = np.poly1d(parameter)
    plt.plot(x, f(x), color='red', linestyle='--', alpha=0.5)
    plt.plot((0, 1), (0, 1), ls=':', c='black', alpha=0.5)

    plt.xlim([10, 90])
    plt.ylim([10, 90])
    plt.xlabel('Chronological Age')
    plt.ylabel('Predicted')
    # plt.show()
    plt.title(name)
    plt.show()
    plt.close()


def pred2pd(pkl_path, group=None):
    result_ = pickle.load(open(pkl_path, 'rb'))

    try:
        real_age = np.concatenate(result_['age']).reshape(-1, 1)
    except:
        real_age = result_['pred_age']

    try:
        pred_age = np.concatenate(result_['pred_age']).reshape(-1, 1)
    except:
        pred_age = result_['pred_age']

    real_age = list(np.squeeze(np.array(real_age)))
    pred_age = list(np.squeeze(np.array(pred_age)))
    MAE = list(abs(np.squeeze(np.array(pred_age))-np.squeeze(np.array(real_age))))
    GAP = list(np.squeeze(np.array(pred_age))-np.squeeze(np.array(real_age)))

    data = {
        'Chronological age': real_age,
        'Predicted Age': pred_age,
        'MAE': MAE,
        'GAP': GAP
    }
    data = pd.DataFrame(data=data)
    data['group'] = group
    return data


def save_scatter(data, y='Predicted Age', output_path=None):
    x = 'Chronological age'
    xlim = [10, 90]
    if y == 'Predicted Age':
        ylim = [10, 90]
    elif y == 'MAE':
        ylim = [0, 25]
    elif y == 'GAP':
        ylim = [-25, 25]
    else:
        raise f'Y: {y} is not defined'

    graph = sns.jointplot(
        x=x,
        y=y,
        # hue='group',
        data=data,
        xlim=xlim,
        ylim=ylim,
        kind="reg",
        # color='red',
        # kws=dict(stat_func=stats.pearsonr),
        joint_kws={
            'scatter_kws': dict(alpha=0.15),
        },
        marginal_kws={'hist_kws': {'alpha': 0.1}}
    )
    # r, p = stats.pearsonr(data[x], data[y])
    r, p = stats.spearmanr(data[x], data[y])
    phantom, = graph.ax_joint.plot([], [], alpha=0)
    # graph.ax_joint.legend([phantom], ['r={:.3f}, p={:.3f}'.format(r, p)])
    graph.ax_joint.legend([phantom], ['r={:.3f}'.format(r)])

    # plt.show()
    if output_path is not None:
        graph.figure.savefig('{}{}.png'.format(output_path, y),
                    # bbox_inches='tight',
                    # transparent=True,
                    dpi=600)
    plt.close()


def plot3images(pkl_dir, output):
    data = pred2pd(pkl_dir, group='test')
    save_scatter(data, y='Predicted Age', output_path=output)
    save_scatter(data, y='MAE', output_path=output)
    save_scatter(data, y='GAP', output_path=output)