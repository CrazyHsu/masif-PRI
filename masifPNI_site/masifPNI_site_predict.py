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

from Bio.PDB import PDBList
from collections import namedtuple
from multiprocessing import Pool, JoinableQueue

from commonFuncs import *
from parseConfig import DefaultConfig
from pdbDownload import targetPdbDownload
from masifPNI_site.masifPNI_site_train import pad_indices
from masifPNI_site.masifPNI_site_nn import MasifPNI_site_nn


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
def masifPNI_site_predict(argv):
    masifpniOpts = mergeParams(argv)
    params = masifpniOpts["masifpni_site"]

    if masifpniOpts["use_gpu"]:
        idx_xpu = masifpniOpts["gpu_dev"] if masifpniOpts["gpu_dev"] else "/gpu:0"
    else:
        idx_xpu = masifpniOpts["cpu_dev"] if masifpniOpts["cpu_dev"] else "/cpu:0"

    # Set precomputation dir.
    parent_in_dir = params["masif_precomputation_dir"]
    precomputatedIds = os.listdir(parent_in_dir)
    eval_list = []

    if argv.list:
        eval_list = getIdChainPairs(masifpniOpts, fromList=argv.list.split(","))
    if argv.file:
        eval_list = getIdChainPairs(masifpniOpts, fromFile=argv.file)
    if argv.custom_pdb:
        eval_list = getIdChainPairs(masifpniOpts, fromCustomPDB=argv.custom_pdb)

    if len(eval_list) == 0:
        print("Please input the PDB ids or PDB files that you want to evaluate.")
        return

    # Build the neural network model

    learning_obj = MasifPNI_site_nn(
        params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        idx_xpu=idx_xpu,
        feat_mask=params["n_feat"] * [1.0],
        n_conv_layers=params["n_conv_layers"],
    )
    print("Restoring model from: " + params["model_dir"] + "model")
    # learning_obj.saver.restore(learning_obj.session, params["model_dir"] + "model")

    if not os.path.exists(params["out_pred_dir"]):
        os.makedirs(params["out_pred_dir"])

    idToDownload = [r.PDB_id for r in eval_list if r.PDB_id not in precomputatedIds]
    idToDownload = list(set(idToDownload))
    if len(idToDownload):
        pdbl = PDBList(server='http://ftp.wwpdb.org')
        q = JoinableQueue(5)
        pool = Pool(processes=masifpniOpts["n_threads"])
        for pid in idToDownload:
            pool.apply_async(targetPdbDownload, (masifpniOpts, pid, pdbl))
        pool.close()
        pool.join()
        q.join()

    uniquePairs = []
    for pid in eval_list:
        if (pid.PDB_id, pid.pChain) in uniquePairs: continue
        uniquePairs.append((pid.PDB_id, pid.pChain))
        print("Evaluating chain {} in protein {}".format(pid.pChain, pid.PDB_id))

        in_dir = os.path.join(parent_in_dir, pid.PDB_id)
        try:
            rho_wrt_center = np.load(os.path.join(in_dir, pid.pChain + "_rho_wrt_center.npy"))
        except:
            print("File not found: {}".format(os.path.join(in_dir, pid.pChain + "_rho_wrt_center.npy")))
            continue
        theta_wrt_center = np.load(os.path.join(in_dir, pid.pChain + "_theta_wrt_center.npy"))
        input_feat = np.load(os.path.join(in_dir, pid.pChain + "_input_feat.npy"))
        input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
        mask = np.load(os.path.join(in_dir, pid.pChain + "_mask.npy"))
        indices = np.load(os.path.join(in_dir, pid.pChain + "_list_indices.npy"), encoding="latin1", allow_pickle=True)
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
        np.save(os.path.join(params["out_pred_dir"], "pred_" + pid.PDB_id + "_" + pid.pChain + ".npy"), scores)


if __name__ == '__main__':
    masifPNI_site_predict(sys.argv)
