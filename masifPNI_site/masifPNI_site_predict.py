#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: masifPNI_site_predict.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-12 16:42:58
Last modified: 2022-09-12 16:42:58
'''

import time, os, sys, importlib, glob
import numpy as np
import pandas as pd
import pymesh

from Bio.PDB import PDBList
from collections import namedtuple
from multiprocessing import Pool, JoinableQueue
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from commonFuncs import *
from parseConfig import DefaultConfig
from pdbDownload import targetPdbDownload
from inputOutputProcess import getPdbChainLength
from dataPreparation import dataprepFromList1
from masifPNI_site.masifPNI_site_train import pad_indices
from masifPNI_site.masifPNI_site_nn import MasifPNI_site_nn


def filterPDBs(pdb_list, data_dirs, masifpniOpts, batchRunFlag=False, filterChainByLen=False):
    raw_pdbs = [i.split(".")[0] for i in os.listdir(masifpniOpts['raw_pdb_dir'])]
    idToDownload = [r.PDB_id for r in pdb_list if r.PDB_id not in set(data_dirs + raw_pdbs)]
    idToDownload = list(set(idToDownload))
    if len(idToDownload) > 0:
        dataprepFromList1(idToDownload, masifpniOpts, batchRunFlag=batchRunFlag)
        precompute_dir = masifpniOpts["masifpni_site"]["masif_precomputation_dir"]
        processedIds = [i for i in idToDownload if os.path.exists(os.path.join(precompute_dir, i))]
        data_dirs = list(set(data_dirs + processedIds))

    my_df = pd.DataFrame(pdb_list)
    if filterChainByLen:
        pdbIds = list(my_df.PDB_id.unique())
        commonKeys = ["PDB_id", "pChain", "naChain"]
        chainLenDf = getPdbChainLength(pdbIds, masifpniOpts['raw_pdb_dir'])
        chainLenDf = chainLenDf[(chainLenDf.pChainLen >= 50) & (chainLenDf.naChainLen >= 15)]
        i1 = my_df.set_index(commonKeys).index
        i2 = chainLenDf.set_index(commonKeys).index
        my_df = my_df[i1.isin(i2)]

    filtered_training_list = list(my_df.PDB_id.unique())
    return filtered_training_list, data_dirs


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
    print("Restoring model from: " + os.path.join(params["model_dir"], "model"))
    learning_obj.saver.restore(learning_obj.session, os.path.join(params["model_dir"], "model"))

    if not os.path.exists(params["out_pred_dir"]):
        os.makedirs(params["out_pred_dir"])

    # raw_pdbs = [i.split(".")[0] for i in os.listdir(masifpniOpts['raw_pdb_dir'])]
    # idToDownload = [r.PDB_id for r in eval_list if r.PDB_id not in set(precomputatedIds + raw_pdbs)]
    # # idToDownload = [r.PDB_id for r in eval_list if r.PDB_id not in precomputatedIds]
    # idToDownload = list(set(idToDownload))
    # if len(idToDownload) > 0:
    #     dataprepFromList1(idToDownload, masifpniOpts)

    eval_list, data_dirs = filterPDBs(eval_list, precomputatedIds, masifpniOpts,
                                                   filterChainByLen=argv.filterChainByLen,
                                                   batchRunFlag=argv.preprocessNobatchRun)

    # eval_df = pd.DataFrame(eval_list)
    uniquePairs = []
    for pdb_id in eval_list:
        mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
        tmpList = glob.glob(os.path.join(mydir, "*_iface_labels.npy"))
        if len(tmpList) == 0: continue
        pChains = [os.path.basename(i).split("_")[0] for i in tmpList]
        for chain in pChains:
            if (pdb_id, chain) in uniquePairs: continue
            uniquePairs.append((pdb_id, chain))
            print("Evaluating chain {} in protein {}".format(chain, pdb_id))

            rho_wrt_center = np.load(os.path.join(mydir, chain + "_rho_wrt_center.npy"))

            theta_wrt_center = np.load(os.path.join(mydir, chain + "_theta_wrt_center.npy"))
            input_feat = np.load(os.path.join(mydir, chain + "_input_feat.npy"))
            input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
            mask = np.load(os.path.join(mydir, chain + "_mask.npy"))
            indices = np.load(os.path.join(mydir, chain + "_list_indices.npy"), encoding="latin1", allow_pickle=True)
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
            np.save(os.path.join(params["out_pred_dir"], "pred_" + pdb_id + "_" + chain + ".npy"), scores)

            ply_file = masifpniOpts["ply_file_template"].format(pdb_id, chain)
            mymesh = pymesh.load_mesh(ply_file)
            ground_truth = mymesh.get_attribute('vertex_iface')

            roc_auc = roc_auc_score(ground_truth, scores[0])
            print("ROC AUC score for protein {} : {:.2f} ".format(pdb_id + '_' + chain, roc_auc))

            pred_acc = np.round(scores[0])
            tn, fp, fn, tp = confusion_matrix(ground_truth, pred_acc).ravel()
            sn = tp/(tp + fn)
            print("SN score for protein {} : {:.2f} ".format(pdb_id + '_' + chain, sn))

            sp = tn/(tn + fp)
            print("SP score for protein {} : {:.2f} ".format(pdb_id + '_' + chain, sp))

            acc = accuracy_score(ground_truth, pred_acc)
            print("ACC score for protein {} : {:.2f} ".format(pdb_id + '_' + chain, acc))

            mcc = matthews_corrcoef(ground_truth, pred_acc)
            print("MCC score for protein {} : {:.2f} ".format(pdb_id + '_' + chain, mcc))

            ppv = tp / (tp + fp)
            print("Precision score for protein {} : {:.2f} ".format(pdb_id + '_' + chain, ppv))

            # F1 score - harmonic mean of precision and recall [2*tp/(2*tp + fp + fn)]
            f1 = 2 * ppv * sn / (ppv + sp)
            print("F1 score for protein {} : {:.2f} ".format(pdb_id + '_' + chain, f1))


    # if (pid.PDB_id, pid.pChain) in uniquePairs: continue
    #     uniquePairs.append((pid.PDB_id, pid.pChain))
    #     print("Evaluating chain {} in protein {}".format(pid.pChain, pid.PDB_id))
    #
    #     in_dir = os.path.join(parent_in_dir, pid.PDB_id)
    #     try:
    #         rho_wrt_center = np.load(os.path.join(in_dir, pid.pChain + "_rho_wrt_center.npy"))
    #     except:
    #         print("File not found: {}".format(os.path.join(in_dir, pid.pChain + "_rho_wrt_center.npy")))
    #         continue
    #     theta_wrt_center = np.load(os.path.join(in_dir, pid.pChain + "_theta_wrt_center.npy"))
    #     input_feat = np.load(os.path.join(in_dir, pid.pChain + "_input_feat.npy"))
    #     input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
    #     mask = np.load(os.path.join(in_dir, pid.pChain + "_mask.npy"))
    #     indices = np.load(os.path.join(in_dir, pid.pChain + "_list_indices.npy"), encoding="latin1", allow_pickle=True)
    #     labels = np.zeros((len(mask)))
    #
    #     print("Total number of patches:{} \n".format(len(mask)))
    #
    #     tic = time.time()
    #     scores = run_masif_site(
    #         params,
    #         learning_obj,
    #         rho_wrt_center,
    #         theta_wrt_center,
    #         input_feat,
    #         mask,
    #         indices,
    #     )
    #     toc = time.time()
    #     print("Total number of patches for which scores were computed: {}\n".format(len(scores[0])))
    #     print("GPU time (real time, not actual GPU time): {:.3f}s".format(toc - tic))
    #     np.save(os.path.join(params["out_pred_dir"], "pred_" + pid.PDB_id + "_" + pid.pChain + ".npy"), scores)


if __name__ == '__main__':
    masifPNI_site_predict(sys.argv)
