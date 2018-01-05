import os
from nrn_wrapper import Cell
from matplotlib.backends.backend_pdf import PdfPages
from cell_fitting.optimization.evaluation.plot_rampIV import evaluate_rampIV
from cell_fitting.optimization.evaluation.plot_IV import evaluate_IV
from cell_fitting.optimization.evaluation.plot_zap import evaluate_zap
from cell_fitting.optimization.evaluation.plot_hyper_depo import evaluate_hyper_depo
from cell_fitting.optimization.evaluation.plot_double_ramp import evaluate_double_ramp
from cell_fitting.optimization.evaluation.plot_sine_stimulus import evaluate_sine_stimulus


def evaluate_model(model_dir, mechanism_dir, save_dir):

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    with PdfPages(os.path.join(save_dir, 'summary_model.pdf')) as pdf:

        data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'
        data_dir_characteristics = '../../data/plots/spike_characteristics/distributions/rat'
        evaluate_rampIV(pdf, cell, data_dir, data_dir_characteristics, save_dir)

        data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'
        data_dir_FI_fit = '../../data/plots/IV/fi_curve/rat/summary'
        data_dir_sag = '../../data/plots/IV/sag_hist/rat'
        evaluate_IV(pdf, cell, data_dir, data_dir_FI_fit, data_dir_sag, save_dir)

        data_dir_resonance = '../../data/plots/Zap20/rat/summary'
        evaluate_zap(pdf, cell, data_dir_resonance, save_dir)

        data_dir_slopes = '../../data/plots/hyper_depo/summary'
        evaluate_hyper_depo(pdf, cell, data_dir_slopes, save_dir)

        evaluate_double_ramp(pdf, cell, save_dir)

        evaluate_sine_stimulus(pdf, cell, save_dir)


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    #save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/server_17_12_04/2017-12-26_08:14:12/6/L-BFGS-B'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    evaluate_model(model_dir, mechanism_dir, save_dir)


# 2017-12-26_08:14:12: 185, 61, 105, 446