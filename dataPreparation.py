#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: dataPreparation.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-13 00:16:34
Last modified: 2022-09-13 00:16:34
'''

import sys, itertools
import numpy as np
from Bio.PDB import PDBList

from commonFuncs import *
from parseConfig import DefaultConfig, GlobalVars
from pdbDownload import targetPdbDownload
from precompute import precomputeProteinPlyInfo, precomputeNaPlyInfo
from inputOutputProcess import extractPniPDB
from readDataFromSurface import extractProteinTriangulate, extractNaTriangulate


def dataprepFromList(pdbIdChains, masifpniOpts):
    pdbIds = [i.split("_")[0] for i in set(pdbIdChains)]
    pdbl = PDBList(server='http://ftp.wwpdb.org')
    targetPdbDownloadBatchRun = []
    for pdb_id in pdbIds:
        pdbFile = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
        if os.path.exists(pdbFile): continue
        targetPdbDownloadBatchRun.append((masifpniOpts, pdb_id, pdbl, True))

    resultList = batchRun(targetPdbDownload, targetPdbDownloadBatchRun, n_threads=masifpniOpts["n_threads"])
    unDownload = list(itertools.chain.from_iterable([i.get() for i in resultList]))
    with open(os.path.join(masifpniOpts["log_dir"], "unable_download.txt"), "w") as f:
        for i in unDownload:
            print(i, file=f)

    removeDirs([masifpniOpts["tmp_dir"]], empty=True)

    # selectPniChainPairs = []
    extractPniPDBbatchRun = []
    pdbId2field = {}
    for i in set(pdbIdChains) - set(unDownload):
        in_fields = i.split("_")
        pdb_id = in_fields[0]
        chainIds = list(itertools.chain.from_iterable([list(map(str, j)) for j in in_fields[1:]]))
        pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
        extractPniPDBbatchRun.append((pdbFile, masifpniOpts["extract_pdb"], chainIds))
        pdbId2field[pdb_id] = in_fields

    resultList = batchRun(extractPniPDB, extractPniPDBbatchRun, n_threads=masifpniOpts["n_threads"])
    pniChainPairs = list(itertools.chain.from_iterable([i.get() for i in resultList]))

    selectPniChainPairs = []
    for i in pniChainPairs:
        fields = pdbId2field[i.PDB_id]
        if len(fields) == 1:
            selectPniChainPairs.append(i)
        elif len(fields) == 2:
            if i.pChain in fields[1]:
                selectPniChainPairs.append(i)
        else:
            if i.pChain in fields[1] and i.naChain in fields[2]:
                selectPniChainPairs.append(i)
    np.save(masifpniOpts["pni_pairs_file"], selectPniChainPairs)

    GlobalVars().setEnviron()

    extractProteinTriangulateBatchRun = []
    extractNaTriangulateBatchRun = []
    precomputeProteinPlyInfoBatchRun = []
    precomputeNaPlyInfoBatchRun = []

    for pairs in selectPniChainPairs:
        pdb_id = pairs.PDB_id
        pChain = pairs.pChain
        naChain = pairs.naChain
        pPdbFile = os.path.join(masifpniOpts["extract_pdb"], "protein", pdb_id + "_" + pChain + ".pdb")
        naPdbFile = os.path.join(masifpniOpts["extract_pdb"], pairs.naType, pdb_id + "_" + naChain + ".pdb")
        # print(pPdbFile, naPdbFile)

        rawPdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
        extractProteinTriangulateBatchRun.append((pPdbFile, rawPdbFile))
        extractNaTriangulateBatchRun.append((naPdbFile, pairs.naType, rawPdbFile))

        precomputeProteinPlyInfoBatchRun.append((pdb_id, pChain))
        precomputeNaPlyInfoBatchRun.append((pdb_id, pChain))

    extractProteinTriangulateBatchRun = [(masifpniOpts,) + i for i in set(extractProteinTriangulateBatchRun)]
    extractNaTriangulateBatchRun = [(masifpniOpts,) + i for i in set(extractNaTriangulateBatchRun)]
    precomputeProteinPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeProteinPlyInfoBatchRun)]
    precomputeNaPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeNaPlyInfoBatchRun)]

    batchRun(extractProteinTriangulate, extractProteinTriangulateBatchRun, n_threads=masifpniOpts["n_threads"])
    batchRun(extractNaTriangulate, extractNaTriangulateBatchRun, n_threads=masifpniOpts["n_threads"])
    batchRun(precomputeProteinPlyInfo, precomputeProteinPlyInfoBatchRun, n_threads=masifpniOpts["n_threads"])
    batchRun(precomputeNaPlyInfo, precomputeNaPlyInfoBatchRun, n_threads=masifpniOpts["n_threads"])


def dataprep(argv):
    masifpniOpts = mergeParams(argv)

    pdbIdChains = []
    if argv.add_default:
        pdbIdChains = DefaultConfig().getListFromFile(masifpniOpts["default_pdb_file"])
    if "file" in vars(argv) and argv.file:
        with open(argv.file) as f:
            for line in f.readlines():
                if line.startswith("#"): continue
                pdbIdChains.append(line.strip())
    if "list" in vars(argv) and argv.list:
        pdbIdChains.extend(argv.list.split(","))

    pdbIdChains = set(pdbIdChains)
    if len(pdbIdChains) > 0:
        dataprepFromList(pdbIdChains, masifpniOpts)


if __name__ == '__main__':

    dataprep(sys.argv)

