#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: pdbDownload.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:47:44
Last modified: 2022-09-11 20:47:44
'''

# import os, sys, importlib
from Bio.PDB import *

from commonFuncs import *
from inputOutputProcess import protonate
# from multiprocessing import Pool, JoinableQueue

def targetPdbDownload(masifpniOpts, pdb_id, pdbl, overwrite=True):
    resolveDirs([masifpniOpts['raw_pdb_dir'], masifpniOpts['tmp_dir']])

    protonated_file = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
    # print(masifpniOpts['tmp_dir'])
    pdb_filename = ""
    if overwrite:
        pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masifpniOpts['tmp_dir'], file_format='pdb', overwrite=overwrite)
    else:
        if os.path.exists(protonated_file): return

    ##### Protonate with reduce, if hydrogens included.
    # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
    unDownload = []
    if os.path.exists(pdb_filename):
        protonate(pdb_filename, protonated_file)
    else:
        unDownload.append(pdb_id)
    return unDownload
    # pdb_filename = protonated_file


def pdbDownload(argv):
    masifpniOpts = mergeParams(argv)

    pdbIds = []
    if argv.list:
        pdbIds = [j.split("_")[0] for j in [i for i in argv.list.split(",")]]
    if argv.file:
        with open(argv.file) as f:
            for i in f.readlines():
                if i.startswith("#"): continue
                pdbIds.append(i.strip().split("_")[0])
    if not pdbIds and not argv.all:
        pdbIds = ["4un3"]
    pdbIds = list(set(pdbIds))

    pdbl = PDBList(server='http://ftp.wwpdb.org')
    targetPdbDownloadBatchRun = []
    for pdb_id in pdbIds:
        pdbFile = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
        if os.path.exists(pdbFile): continue
        targetPdbDownloadBatchRun.append((masifpniOpts, pdb_id, pdbl, argv.overwrite))
    resultList = batchRun1(targetPdbDownload, targetPdbDownloadBatchRun, n_threads=masifpniOpts["n_threads"])

    # unDownload = list(itertools.chain.from_iterable([i.get() for i in resultList]))
    unDownload = list(itertools.chain.from_iterable(resultList))
    with open(os.path.join(masifpniOpts["log_dir"], "unable_download.txt"), "w") as f:
        for i in unDownload:
            print(i, file=f)

    removeDirs([masifpniOpts["tmp_dir"]], empty=True)
