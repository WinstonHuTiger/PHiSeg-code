# Get classification metrics for a trained classifier model
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse

import config.system as sys_config
from phiseg.phiseg_model import phiseg
import utils
from scipy.stats import wilcoxon

import logging
from data.data_switch import data_switch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}


def main(model_path, exp_config, do_plots=False):
    n_samples = 50
    model_selection = 'best_ged'

    # Get Data
    phiseg_model = phiseg(exp_config=exp_config)
    phiseg_model.load_weights(model_path, type=model_selection)

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    N = data.test.images.shape[0]

    qubiq_list = []
    ncc_list = []
    ged_list = []
    sample_accuracy = []
    sample_diversity = []

    for ii in range(N):

        if ii % 10 == 0:
            logging.info("Progress: %d" % ii)

        x_b = data.test.images[ii, ...].reshape([1] + list(exp_config.image_size))
        s_b = data.test.labels[ii, ...]

        x_b_stacked = np.tile(x_b, [n_samples, 1, 1, 1])

        feed_dict = {}
        feed_dict[phiseg_model.training_pl] = False
        feed_dict[phiseg_model.x_inp] = x_b_stacked

        s_arr_sm = phiseg_model.sess.run(phiseg_model.s_out_eval_sm, feed_dict=feed_dict)
        s_arr = np.argmax(s_arr_sm, axis=-1)

        # s_arr = np.squeeze(np.asarray(s_list)) # num samples x X x Y
        s_b_r = s_b.transpose((2, 0, 1))  # num gts x X x Y
        s_b_r_sm = utils.convert_batch_to_onehot(s_b_r, exp_config.nlabels)  # num gts x X x Y x nlabels

        ged = utils.generalised_energy_distance(s_arr, s_b_r, nlabels=exp_config.nlabels - 1,
                                                label_range=range(1, exp_config.nlabels))
        ged_list.append(ged)
        qubiq_list.append(utils.QUBIQ(np.mean(s_arr_sm, axis=-1), s_b_r))
        ncc = utils.variance_ncc_dist(s_arr_sm, s_b_r_sm)
        ncc_list.append(ncc)

        sample_diversity.append(utils.sample_diversity(s_arr))
        sample_accuracy.append(utils.sample_accuracy(sample_arr=s_arr, gt_arr= s_b_r))

    ged_arr = np.asarray(ged_list)
    ncc_arr = np.asarray(ncc_list)
    qubiq_arr = np.asarray(qubiq_list)
    sa_arr = np.asarray(sample_accuracy)
    sd_arr = np.asarray(sample_diversity)

    logging.info('-- GED: --')
    logging.info(np.mean(ged_arr))
    logging.info(np.std(ged_arr))
    logging.info("25 percentile: %.4f" % (np.percentile(ged_arr, 25)))
    logging.info("75 percentile: %.4f" % (np.percentile(ged_arr, 75)))
    w, p = wilcoxon(ged_arr)
    logging.info("p-value: %.4f" % (p))
    logging.info("p-value (w): %.4f" % (w))

    logging.info('-- NCC: --')
    logging.info(np.mean(ncc_arr))
    logging.info(np.std(ncc_arr))

    logging.info("--QUBIQ: --")
    logging.info(np.mean(qubiq_arr))
    logging.info(np.std(qubiq_arr))
    logging.info("25 percentile: %.4f" % (np.percentile(qubiq_arr, 25)))
    logging.info("75 percentile: %.4f" % (np.percentile(qubiq_arr, 75)))
    w, p = wilcoxon(qubiq_arr[0])
    logging.info("p-value: %.4f" % (p))
    logging.info("p-value (w): %.4f" % (w))

    np.savez(os.path.join(model_path, 'ged%s_%s.npz' % (str(n_samples), model_selection)), ged_arr)
    np.savez(os.path.join(model_path, 'ncc%s_%s.npz' % (str(n_samples), model_selection)), ncc_arr)
    np.savez(os.path.join(model_path, "qubiq%s_%s.npz" % (str(n_samples), model_selection)), qubiq_arr)

    logging.info("--Sample Diversity: --")
    logging.info(np.mean(sd_arr))
    logging.info(np.std(sd_arr))

    logging.info("--Sample Accuracy: --")
    logging.info(np.mean(sa_arr))
    logging.info(np.std(sd_arr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a network on the test dataset")
    parser.add_argument("EXP_PATH", type=str,
                        help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = args.EXP_PATH
    print(model_path)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config, do_plots=False)
