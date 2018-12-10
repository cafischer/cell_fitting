import numpy as np
import os
import matplotlib.pyplot as pl
from cell_fitting.optimization.evaluation.plot_hyper_depo.plot_hyper_depo_summary import get_star_from_ttest#, get_star_from_p_val
pl.style.use('paper')


def get_star_from_p_val(p):
    star_idx = np.where([p < 0.01, p < 0.001, p < 0.0001])[0]
    if len(star_idx) == 0:
        star_idx = 0
    else:
        star_idx = star_idx[-1] + 1
    stars = ['n.s.', '*', '**', '***']
    star = stars[star_idx]
    return star


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


def plot_current_threshold_all_cells_on_ax(ax, current_thresholds_DAP, current_thresholds_rest, step_amps,
                                           color=('k', 'k', 'k'), plot_sig=True, p_groups=None):
    percentage_difference = 100 - (current_thresholds_DAP / np.repeat(np.array([current_thresholds_rest]).T, 3, 1) * 100)

    if plot_sig:
        ax.errorbar([0], np.mean(percentage_difference, 0)[0], yerr=np.std(percentage_difference, 0)[0],
                    color=color[0], marker='o', capsize=3, linestyle='')
        ax.errorbar([2.0], np.mean(percentage_difference, 0)[1], yerr=np.std(percentage_difference, 0)[1],
                    color=color[1], marker='o', capsize=3, linestyle='')
        ax.errorbar([4.0], np.mean(percentage_difference, 0)[2], yerr=np.std(percentage_difference, 0)[2],
                    color=color[2], marker='o', capsize=3, linestyle='')
        ax.plot(np.zeros(len(percentage_difference)) - 0.4, percentage_difference[:, 0], 'o', color=color[0], alpha=0.5)
        ax.plot(np.zeros(len(percentage_difference)) + 1.6, percentage_difference[:, 1], 'o', color=color[1], alpha=0.5)
        ax.plot(np.zeros(len(percentage_difference)) + 3.6, percentage_difference[:, 2], 'o', color=color[2], alpha=0.5)

        h0 = 0
        star = get_star_from_ttest(percentage_difference[:, 0], h0)
        vertical_square_bracket(ax, star, x1=0.35, x2=0.4, y1=np.mean(percentage_difference[:, 0]), y2=h0, dtext=0.1)
        star = get_star_from_ttest(percentage_difference[:, 1], h0)
        vertical_square_bracket(ax, star, x1=2.35, x2=2.4, y1=np.mean(percentage_difference[:, 1]), y2=h0, dtext=0.1)
        star = get_star_from_ttest(percentage_difference[:, 2], h0)
        vertical_square_bracket(ax, star, x1=4.35, x2=4.4, y1=np.mean(percentage_difference[:, 2]), y2=h0, dtext=0.1)
        ax.set_xlim([-1, 5.0])
        ax.set_xticks(np.array([0, 2, 4]) - 0.4)

        # group comparisons
        if p_groups is not None:
            star = get_star_from_p_val(p_groups[0])
            horizontal_square_bracket(ax, star, x1=0, x2=1.9, y1=76, y2=77, dtext=1.0)
            star =  get_star_from_p_val(p_groups[1])
            horizontal_square_bracket(ax, star, x1=2.1, x2=4, y1=76, y2=77, dtext=1.0)
            star =  get_star_from_p_val(p_groups[2])
            horizontal_square_bracket(ax, star, x1=0, x2=4, y1=82, y2=84, dtext=0.1)
    else:
        ax.errorbar([0], np.mean(percentage_difference, 0)[0], yerr=np.std(percentage_difference, 0)[0],
                    color=color[0], marker='o', capsize=3, linestyle='')
        ax.errorbar([1.0], np.mean(percentage_difference, 0)[1], yerr=np.std(percentage_difference, 0)[1],
                    color=color[1], marker='o', capsize=3, linestyle='')
        ax.errorbar([2.0], np.mean(percentage_difference, 0)[2], yerr=np.std(percentage_difference, 0)[2],
                    color=color[2], marker='o', capsize=3, linestyle='')
        ax.plot(np.zeros(len(percentage_difference)) - 0.4, percentage_difference[:, 0], 'o', color=color[0], alpha=0.5)
        ax.plot(np.zeros(len(percentage_difference)) + 0.6, percentage_difference[:, 1], 'o', color=color[1], alpha=0.5)
        ax.plot(np.zeros(len(percentage_difference)) + 1.6, percentage_difference[:, 2], 'o', color=color[2], alpha=0.5)
        ax.set_xlim([-1.0, 2.2])
        ax.set_xticks(np.array([0, 1, 2])-0.4)
    ax.set_xticklabels(step_amps)
    ax.set_ylim([0, 100])
    #ax.set_ylabel('Decrease current \nthresh. (%)')
    ax.set_xlabel('Amp. (nA)')
    return np.mean(percentage_difference, 0)


def vertical_square_bracket(ax, star, x1, x2, y1, y2, dtext):
    ax.plot([x1, x2, x2, x1], [y1, y1, y2, y2], lw=1.5, c='k')

    if star == 'n.s.':
        fontsize = 10
    else:
        fontsize = 12
    ax.text(x2 + dtext, (y1 + y2) * 0.5, star, va='center', color='k', fontsize=fontsize)


def horizontal_square_bracket(ax, star, x1, x2, y1, y2, dtext):
    ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], lw=1.5, c='k')
    if star == 'n.s.':
        fontsize = 10
    else:
        fontsize = 12
    ax.text((x1 + x2) * 0.5, y2 + dtext, star, ha='center', color='k', fontsize=fontsize)


if __name__ == '__main__':

    cell_ids = ['2014_07_10b', '2014_07_03a', '2014_07_08d', '2014_07_09c', '2014_07_10d', '2014_07_09f']
    model_ids = range(1, 7)
    save_dir_data = '../../../data/plots'
    save_dir_model = '../../../results/best_models'
    save_dir_img = os.path.join(save_dir_model, 'img', 'PP', 'summary')

    # load
    diff_current_threshold_data = np.zeros(len(cell_ids))
    for i, cell_id in enumerate(cell_ids):
        diff_current_threshold_data[i] = float(np.loadtxt(os.path.join(save_dir_data, 'PP', cell_id,
                                                                       'diff_current_threshold.txt')))

    diff_current_threshold_model = np.zeros(len(cell_ids))
    for i, model_id in enumerate(model_ids):
        diff_current_threshold_model[i] = float(np.loadtxt(os.path.join(save_dir_model, str(model_id), 'img', 'rampIV',
                                                                        'diff_current_threshold.txt')))

    plot_current_threshold(diff_current_threshold_data, diff_current_threshold_model, save_dir_img)