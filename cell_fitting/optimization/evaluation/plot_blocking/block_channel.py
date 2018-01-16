import matplotlib.pyplot as pl
import os
from cell_fitting.optimization.evaluation.rampIV import simulate_rampIV
from nrn_wrapper import Cell, load_mechanism_dir
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    model_ids = range(1, 7)
    mechanism_dir = '../../../model/channels/vavoulis'
    ramp_amp = 3.5
    load_mechanism_dir(mechanism_dir)
    channel_names = [['nat', 'nap']]  # ['hcn_slow', 'nat', 'nap', 'kdr']
    percent_blocks = [[20, 20]]  # [5, 10, 20, 50, 100]

    for model_id in model_ids:
        for channel_name in channel_names:
            for percent_block in percent_blocks:
                # load model
                cell = Cell.from_modeldir(os.path.join(save_dir, str(model_id), 'cell.json'))

                # simulation
                v_before, t_before, _ = simulate_rampIV(cell, ramp_amp, v_init=-75)

                # blocking
                if isinstance(channel_name, list):
                    for p_b, c_n in zip(percent_block, channel_name):
                        old_gbar = cell.get_attr(['soma', '0.5', c_n, 'gbar'])
                        new_gbar = old_gbar * (100 - p_b) / 100
                        cell.update_attr(['soma', '0.5', c_n, 'gbar'], new_gbar)
                else:
                    old_gbar = cell.get_attr(['soma', '0.5', channel_name, 'gbar'])
                    new_gbar = old_gbar * (100 - percent_block) / 100
                    cell.update_attr(['soma', '0.5', channel_name, 'gbar'], new_gbar)

                # simulation
                v_after, t_after, _ = simulate_rampIV(cell, ramp_amp, v_init=-75)

                # plot
                save_dir_img = os.path.join(save_dir, str(model_id), 'img', 'blocking', 'rampIV', str(channel_name),
                                            str(percent_block))
                if not os.path.exists(save_dir_img):
                    os.makedirs(save_dir_img)
                print model_id
                pl.figure()
                pl.plot(t_before, v_before, 'r', label='before block')
                pl.plot(t_after, v_after, 'b', label='after block')
                pl.xlabel('Time (ms)')
                pl.ylabel('Membrane potential (mV)')
                pl.legend(loc='upper right')
                pl.tight_layout()
                pl.savefig(os.path.join(save_dir_img, str(ramp_amp) + '(nA).png'))
                #pl.show()

                pl.figure()
                pl.plot(t_before, v_before, 'r', label='before block')
                pl.plot(t_after, v_after, 'b', label='after block')
                pl.xlabel('Time (ms)')
                pl.ylabel('Membrane potential (mV)')
                pl.legend(loc='upper right')
                pl.xlim(5, 80)
                pl.tight_layout()
                pl.savefig(os.path.join(save_dir_img, str(ramp_amp) + '(nA)_zoom.png'))
                pl.show()