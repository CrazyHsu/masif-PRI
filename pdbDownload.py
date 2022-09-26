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
# from parseConfig import DefaultConfig, ParseConfig
from inputOutputProcess import protonate
from multiprocessing import Pool, JoinableQueue

def targetPdbDownload(masifpniOpts, pdb_id, pdbl):
    # for pdb_id in pdbIds:
    resolveDirs([masifpniOpts['raw_pdb_dir'], masifpniOpts['tmp_dir']])

    protonated_file = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
    if os.path.exists(protonated_file): return
    pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masifpniOpts['tmp_dir'], file_format='pdb', overwrite=True)

    ##### Protonate with reduce, if hydrogens included.
    # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
    # protonated_file = os.path.join(masifOpts['raw_pdb_dir'], pdb_id + ".pdb")
    protonate(pdb_filename, protonated_file)
    # pdb_filename = protonated_file


# def targetPdbDownload1(masifpniOpts, pdbIds=None):
#     # Download pdb
#     pdbl = PDBList(server='http://ftp.wwpdb.org')
#
#     if not pdbIds:
#         pdbIds= pdbl.get_all_entries()
#
#     for pdb_id in pdbIds:
#         resolveDirs([masifpniOpts['raw_pdb_dir'], masifpniOpts['tmp_dir']])
#
#         protonated_file = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
#         if os.path.exists(protonated_file): continue
#         pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masifpniOpts['tmp_dir'], file_format='pdb')
#
#         ##### Protonate with reduce, if hydrogens included.
#         # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
#         # protonated_file = os.path.join(masifOpts['raw_pdb_dir'], pdb_id + ".pdb")
#         protonate(pdb_filename, protonated_file)
#         # pdb_filename = protonated_file


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

    q = JoinableQueue(5)
    pool = Pool(processes=masifpniOpts["n_threads"])
    for pdb_id in pdbIds:
        pool.apply_async(targetPdbDownload, (masifpniOpts, pdb_id, pdbl))
    pool.close()
    pool.join()

    q.join()

    removeDirs([masifpniOpts["tmp_dir"]], empty=True)
