import numpy as np
import os
import matplotlib.pyplot as pl
from cell_fitting.optimization.evaluation.plot_hyper_depo.plot_hyper_depo_summary import get_star_from_ttest
pl.style.use('paper')


def plot_current_threshold(diff_current_data, diff_current_model, save_dir):
    h0 = 0
    star = get_star_from_ttest(diff_current_threshold_data, h0)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fig, ax = pl.subplots(figsize=(2.5, 4.8))
    ax.errorbar(0.2, np.mean(diff_current_data), yerr=np.std(diff_current_data), color='k', marker='o', capsize=3)
    ax.plot(np.zeros(len(diff_current_data)), diff_current_data, 'ok', alpha=0.5)
    #ax.plot(np.zeros(len(diff_current_model)), diff_current_model, 'or', alpha=0.5)
    #for i, model_id in enumerate(model_ids):
    #    ax.annotate(str(model_id), xy=(0.05, diff_current_model[i]), color='r', fontsize=8)
    vertical_square_bracket(ax, star, x1=0.35, x2=0.4, y1=np.mean(diff_current_data), y2=h0)
    ax.set_xticks([])
    ax.set_ylabel('Current Threshold: Rest - DAP (nA)')
    ax.set_xlim([-1, 1])
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'current_threshold.png'))
    pl.show()


def vertical_square_bracket(ax, star, x1, x2, y1, y2):
    ax.plot([x1, x2, x2, x2 + 0.1, x2, x2, x1], [y1, y1, (y1 + y2) * 0.5, (y1 + y2) * 0.5, (y1 + y2) * 0.5, y2, y2],
            lw=1.5, c='k')
    ax.text(x2 + 0.2, (y1 + y2) * 0.5, star, va='center', color='k', fontsize=14)


if __name__ == '__main__':

    cell_ids = ['2014_07_10b', '2014_07_03a', '2014_07_08d', '2014_07_09c', '2014_07_10d', '2014_07_09f']
    model_ids = range(1, 7)
    save_dir_data = '../../../data/plots'
    save_dir_model = '../../../results/best_models'
    save_dir_img = os.path.join(save_dir_model, 'img', 'PP', 'summary')

    # load
    diff_current_threshold_data = np.zeros(len(cell_ids))
    for i, cell_id in enumerate(cell_ids):
        diff_current_threshold_data[i] = float(np.loadtxt(os.path.join(save_dir_data, 'PP', cell_id, 'diff_current_threshold.txt')))

    diff_current_threshold_model = np.zeros(len(cell_ids))
    for i, model_id in enumerate(model_ids):
        diff_current_threshold_model[i] = float(np.loadtxt(os.path.join(save_dir_model, str(model_id), 'img', 'rampIV', 'diff_current_threshold.txt')))

    plot_current_threshold(diff_current_threshold_data, diff_current_threshold_model, save_dir_img)