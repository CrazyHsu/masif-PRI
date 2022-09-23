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

def batchRun(myFunc, argList, n_threads=1):
    from multiprocessing import Pool, JoinableQueue
    q = JoinableQueue(5)
    pool = Pool(processes=n_threads)
    for arg in argList:
        if len(arg) == 1:
            pool.apply_async(myFunc, (arg[0], ))
        else:
            pool.apply_async(myFunc, arg)
    pool.close()
    pool.join()
    q.join()

def dataprep(argv):
    masifpniOpts = mergeParams(argv)

    pdbIdChains = DefaultConfig().getListFromFile(masifpniOpts["default_pdb_file"])
    if "file" in vars(argv) and argv.file:
        with open(argv.file) as f:
            pdbIdChains.append(f.readline().strip())
    if "list" in vars(argv) and argv.list:
        pdbIdChains.extend(argv.list.split(","))
    pdbIdChains = set(pdbIdChains)
    pdbIds = [i.split("_")[0] for i in set(pdbIdChains)]
    pdbl = PDBList(server='http://ftp.wwpdb.org')
    for pdb_id in pdbIds:
        pdbFile = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
        if os.path.exists(pdbFile): continue
        targetPdbDownload(masifpniOpts, pdb_id, pdbl)

    removeDirs([masifpniOpts["tmp_dir"]], empty=True)

    selectPniChainPairs = []
    for i in pdbIdChains:
        in_fields = i.split("_")
        pdb_id = in_fields[0]
        chainIds = list(itertools.chain.from_iterable([list(map(str, j)) for j in in_fields[1:]]))
        pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
        # print(pdbFile, chainIds)
        pniChainPairs = extractPniPDB(pdbFile, masifpniOpts["extract_pdb"], chainIds=chainIds)
        # print(i, pniChainPairs)

        if len(chainIds) == 1:
            tmpPairs = [i for i in pniChainPairs if i.pChain == chainIds[0]]
        elif len(chainIds) == 2:
            tmpPairs = [i for i in pniChainPairs if i.pChain == chainIds[0] and i.naChain == chainIds[1]]
        else:
            tmpPairs = pniChainPairs

        selectPniChainPairs.extend(tmpPairs)
    np.save(masifpniOpts["pni_pairs_file"], selectPniChainPairs)

    GlobalVars().setEnviron()
    # pairs = selectPniChainPairs[0]
    # pdb_id = pairs.PDB_id
    # pChain = pairs.pChain
    # naChain = pairs.naChain
    # pPdbFile = os.path.join(masifpniOpts["extract_pdb"], "protein", pdb_id + "_" + pChain + ".pdb")
    # naPdbFile = os.path.join(masifpniOpts["extract_pdb"], pairs.naType, pdb_id + "_" + naChain + ".pdb")
    # print(pPdbFile, naPdbFile)
    # rawPdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
    # # extractProteinTriangulate(masifpniOpts, pPdbFile, rawPdbFile)
    # extractNaTriangulate(masifpniOpts, naPdbFile, pairs.naType, rawPdbFile)
    #
    # precomputeProteinPlyInfo(masifpniOpts, pdb_id, pChain)

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

        # extractProteinTriangulate(masifpniOpts, pPdbFile, rawPdbFile)
        # extractNaTriangulate(masifpniOpts, naPdbFile, pairs.naType, rawPdbFile)
        #
        # precomputeProteinPlyInfo(masifpniOpts, pdb_id, pChain)
        # precomputeNaPlyInfo(masifpniOpts, pdb_id, naChain)

    extractProteinTriangulateBatchRun = [(masifpniOpts, ) + i for i in set(extractProteinTriangulateBatchRun)]
    extractNaTriangulateBatchRun = [(masifpniOpts,) + i for i in set(extractNaTriangulateBatchRun)]
    precomputeProteinPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeProteinPlyInfoBatchRun)]
    precomputeNaPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeNaPlyInfoBatchRun)]

    # print(extractProteinTriangulateBatchRun)
    # print(set(extractProteinTriangulateBatchRun))
    batchRun(extractProteinTriangulate, extractProteinTriangulateBatchRun, n_threads=masifpniOpts["n_threads"])
    batchRun(extractNaTriangulate, extractNaTriangulateBatchRun, n_threads=masifpniOpts["n_threads"])
    batchRun(precomputeProteinPlyInfo, precomputeProteinPlyInfoBatchRun, n_threads=masifpniOpts["n_threads"])
    batchRun(precomputeNaPlyInfo, precomputeNaPlyInfoBatchRun, n_threads=masifpniOpts["n_threads"])


if __name__ == '__main__':

    dataprep(sys.argv)

