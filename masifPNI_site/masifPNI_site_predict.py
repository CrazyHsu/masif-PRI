#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: masifPNI_site_predict.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-12 16:42:58
Last modified: 2022-09-12 16:42:58
'''

import time, os, sys, importlib
import numpy as np
from parseConfig import DefaultConfig
from masifPNI_site.masifPNI_site_train import pad_indices
from masifPNI_site.masifPNI_site_nn import MasifPNI_site_nn


# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


# Run masif site on a protein, on a previously trained network.
def run_masif_site(params, learning_obj, rho_wrt_center, theta_wrt_center, input_feat, mask, indices):
    indices = pad_indices(indices, mask.shape[1])
    mask = np.expand_dims(mask, 2)
    feed_dict = {
        learning_obj.rho_coords: rho_wrt_center,
        learning_obj.theta_coords: theta_wrt_center,
        learning_obj.input_feat: input_feat,
        learning_obj.mask: mask,
        learning_obj.indices_tensor: indices,
    }

    score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
    return score


"""
masif_site_predict: Evaluate one or multiple proteins on MaSIF-site. 
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""
def masifPRI_site_predict(argv):
    masifOpts = DefaultConfig().masifpniOpts
    params = masifOpts["masifPNI_site"]

    if argv.config:
        custom_params_file = argv.config
        custom_params = importlib.import_module(custom_params_file, package=None)
        custom_params = custom_params.custom_params

        for key in custom_params:
            print("Setting {} to {} ".format(key, custom_params[key]))
            params[key] = custom_params[key]

    # Set precomputation dir.
    parent_in_dir = params["masif_precomputation_dir"]
    eval_list = []


    if len(argv) == 3:
        ppi_pair_ids = [argv[2]]
    # Read a list of pdb_chain entries to evaluate.
    elif len(argv) == 4 and argv[2] == "-l":
        listfile = open(argv[3])
        ppi_pair_ids = []
        for line in listfile:
            eval_list.append(line.rstrip())
        for mydir in os.listdir(parent_in_dir):
            ppi_pair_ids.append(mydir)
    else:
        sys.exit(1)

    # Build the neural network model

    learning_obj = MasifPRI_site_nn(
        params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        idx_gpu="/gpu:0",
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )
    print("Restoring model from: " + params["model_dir"] + "model")
    learning_obj.saver.restore(learning_obj.session, params["model_dir"] + "model")

    if not os.path.exists(params["out_pred_dir"]):
        os.makedirs(params["out_pred_dir"])

    for ppi_pair_id in ppi_pair_ids:
        print(ppi_pair_id)
        in_dir = parent_in_dir + ppi_pair_id + "/"

        fields = ppi_pair_id.split('_')
        if len(fields) < 2:
            continue
        pdbid = ppi_pair_id.split("_")[0]
        chain1 = ppi_pair_id.split("_")[1]
        pids = ["p1"]
        chains = [chain1]
        if len(fields) == 3 and fields[2] != "":
            chain2 = fields[2]
            pids = ["p1", "p2"]
            chains = [chain1, chain2]

        for ix, pid in enumerate(pids):
            pdb_chain_id = str(pdbid) + "_" + str(chains[ix])
            if len(eval_list) > 0 and pdb_chain_id not in eval_list and pdb_chain_id + "_" not in eval_list:
                continue

            print("Evaluating {}".format(pdb_chain_id))

            try:
                rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
            except:
                print("File not found: {}".format(in_dir + pid + "_rho_wrt_center.npy"))
                continue
            theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
            input_feat = np.load(in_dir + pid + "_input_feat.npy")
            input_feat = mask_input_feat(input_feat, params["feat_mask"])
            mask = np.load(in_dir + pid + "_mask.npy")
            indices = np.load(in_dir + pid + "_list_indices.npy", encoding="latin1", allow_pickle=True)
            labels = np.zeros((len(mask)))

            print("Total number of patches:{} \n".format(len(mask)))

            tic = time.time()
            scores = run_masif_site(
                params,
                learning_obj,
                rho_wrt_center,
                theta_wrt_center,
                input_feat,
                mask,
                indices,
            )
            toc = time.time()
            print("Total number of patches for which scores were computed: {}\n".format(len(scores[0])))
            print("GPU time (real time, not actual GPU time): {:.3f}s".format(toc - tic))
            np.save(params["out_pred_dir"] + "/pred_" + pdbid + "_" + chains[ix] + ".npy", scores, )


if __name__ == '__main__':
    masifPRI_site_predict(sys.argv)
