#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: dataPreparation.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-13 00:16:34
Last modified: 2022-09-13 00:16:34
'''

import sys, itertools, tempfile
import numpy as np
import pandas as pd
from Bio.PDB import PDBList

from commonFuncs import *
from parseConfig import DefaultConfig, GlobalVars
from pdbDownload import targetPdbDownload
from precompute import precomputeProteinPlyInfo, precomputeNaPlyInfo
from inputOutputProcess import extractPniPDB, extractPDB, findProteinChainBoundNA
from readDataFromSurface import extractProteinTriangulate, extractNaTriangulate


def dataprepFromList1(pdbIdChains, masifpniOpts, runAll=False, resumeDownload=False, resumeFindBound=False,
                      resumeExtractPDB=False, resumeExtractTriangulate=False, resumePrecomputePly=False, batchRunFlag=True):
    pdbIds = [i.split("_")[0].upper() for i in set(pdbIdChains)]
    pdbl = PDBList(server='http://ftp.wwpdb.org')
    targetPdbDownloadBatchRun = []
    for pdb_id in pdbIds:
        pdbFile = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
        if os.path.exists(pdbFile): continue
        targetPdbDownloadBatchRun.append((masifpniOpts, pdb_id, pdbl, True))

    downloadDesc = "Download PDBs"
    resultList = batchRun1(targetPdbDownload, targetPdbDownloadBatchRun, n_threads=masifpniOpts["n_threads"],
                           desc=downloadDesc, batchRunFlag=batchRunFlag)
    unDownload = list(itertools.chain.from_iterable(resultList))
    with open(os.path.join(masifpniOpts["log_dir"], "unable_download.txt"), "w") as f:
        for i in unDownload:
            print(i, file=f)

    resumeFromChainPairs = False
    selectPniChainPairs = []
    findProteinChainBoundNABatchRun = []
    if not resumeFromChainPairs:
        pdbId2field = {}
        pdbIdChains = list(set([i for i in pdbIdChains if i.split("_")[0] not in unDownload]))
        for i in pdbIdChains:
            fields = i.split("_")
            pid = fields[0]
            pdbId2field[pid] = fields
            pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pid + ".pdb")
            if not os.path.exists(pdbFile): continue
            findProteinChainBoundNABatchRun.append((pdbFile,))

        findProteinChainBoundNADesc = "Find protein-NA bounding chains"
        resultList = batchRun1(findProteinChainBoundNA, findProteinChainBoundNABatchRun,
                               n_threads=masifpniOpts["n_threads"], desc=findProteinChainBoundNADesc,
                               batchRunFlag=batchRunFlag)
        pniChainPairs = list(itertools.chain.from_iterable(resultList))
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
        tempfileBase = os.path.basename(tempfile.mkdtemp())
        pniPairFile = os.path.join(masifpniOpts['tmp_dir'], tempfileBase + ".pni_pairs_file")
        np.save(pniPairFile, selectPniChainPairs)
    else:
        try:
            tmp = np.load(masifpniOpts["pni_pairs_file"])
        except:
            print("Please specify the right pni_pairs_file!")
            return
        for pair in tmp:
            selectPniChainPairs.append(BoundTuple(*pair.tolist()))

    GlobalVars().setEnviron()

    selectPniChainPairsDF = pd.DataFrame(selectPniChainPairs, columns=["PDB_id", "pChain", "naChain", "naType"])
    selectPniChainPairsGP = selectPniChainPairsDF.groupby(["PDB_id", "naChain", "naType"])

    pPdbDir = os.path.join(masifpniOpts["extract_pdb"], "protein")
    rnaPdbDir = os.path.join(masifpniOpts["extract_pdb"], "RNA")
    dnaPdbDir = os.path.join(masifpniOpts["extract_pdb"], "DNA")
    resolveDirs([pPdbDir, rnaPdbDir, dnaPdbDir])

    extractPchainPDBbatchRun = []
    extractNAchainPDBbatchRun = []
    extractProteinTriangulateBatchRun = []
    extractNaTriangulateBatchRun = []
    precomputeProteinPlyInfoBatchRun = []
    precomputeNaPlyInfoBatchRun = []
    for gp in selectPniChainPairsGP.groups:
        tmpGroup = selectPniChainPairsGP.get_group(gp)
        pid = tmpGroup.PDB_id.unique()[0]
        pChains = "".join(tmpGroup.pChain.unique())
        naChain = tmpGroup.naChain.unique()[0]
        naType = tmpGroup.naType.unique()[0]

        rawPdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pid + ".pdb")
        pPdbFile = os.path.join(masifpniOpts["extract_pdb"], "protein", pid + "_" + "".join(pChains) + ".pdb")
        naPdbFile = os.path.join(masifpniOpts["extract_pdb"], naType, pid + "_" + naChain + ".pdb")

        extractPchainPDBbatchRun.append((rawPdbFile, pPdbFile, pChains))
        extractNAchainPDBbatchRun.append((rawPdbFile, naPdbFile, naChain))
        extractProteinTriangulateBatchRun.append((pPdbFile, rawPdbFile))
        extractNaTriangulateBatchRun.append((naPdbFile, naType, rawPdbFile))
        precomputeProteinPlyInfoBatchRun.append((pid, "".join(pChains)))
        precomputeNaPlyInfoBatchRun.append((pid, naChain))

    extractPchainPDBbatchRun = set(extractPchainPDBbatchRun)
    extractNAchainPDBbatchRun = set(extractNAchainPDBbatchRun)
    extractProteinTriangulateBatchRun = [(masifpniOpts,) + i for i in set(extractProteinTriangulateBatchRun)]
    extractNaTriangulateBatchRun = [(masifpniOpts,) + i for i in set(extractNaTriangulateBatchRun)]
    precomputeProteinPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeProteinPlyInfoBatchRun)]
    precomputeNaPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeNaPlyInfoBatchRun)]

    extractPchainPDBDesc = "Extract protein chains"
    batchRun1(extractPDB, extractPchainPDBbatchRun, n_threads=masifpniOpts["n_threads"], desc=extractPchainPDBDesc,
              batchRunFlag=batchRunFlag)
    extractNAchainPDBDesc = "Extract nucleic acid chains"
    batchRun1(extractPDB, extractNAchainPDBbatchRun, n_threads=masifpniOpts["n_threads"], desc=extractNAchainPDBDesc,
              batchRunFlag=batchRunFlag)
    extractProteinTriangulateDesc = "Extract protein triangulate"
    batchRun1(extractProteinTriangulate, extractProteinTriangulateBatchRun, n_threads=masifpniOpts["n_threads"],
              desc=extractProteinTriangulateDesc, batchRunFlag=batchRunFlag)
    extractNaTriangulateDesc = "Extract nucleic acid triangulate"
    batchRun1(extractNaTriangulate, extractNaTriangulateBatchRun, n_threads=masifpniOpts["n_threads"],
              desc=extractNaTriangulateDesc, batchRunFlag=batchRunFlag)

    plySize = 5000000
    precomputeProteinPlyInfoBatchRun = [i for i in precomputeProteinPlyInfoBatchRun if os.path.exists(masifpniOpts['ply_file_template'].format(i[1], i[2]))]
    precomputeProteinPlyInfoBatchRun1 = [i for i in precomputeProteinPlyInfoBatchRun if os.path.getsize(masifpniOpts['ply_file_template'].format(i[1], i[2])) <= plySize]
    precomputeProteinPlyInfoBatchRun2 = [i for i in precomputeProteinPlyInfoBatchRun if os.path.getsize(masifpniOpts['ply_file_template'].format(i[1], i[2])) > plySize]

    precomputeProteinPlyInfoDesc1 = "Precompute protein ply information for size less than {}".format(plySize)
    batchRun1(precomputeProteinPlyInfo, precomputeProteinPlyInfoBatchRun1, n_threads=masifpniOpts["n_threads"]+10,
              desc=precomputeProteinPlyInfoDesc1, batchRunFlag=batchRunFlag)
    precomputeProteinPlyInfoDesc2 = "Precompute protein ply information for size large than {}".format(plySize)
    batchRun1(precomputeProteinPlyInfo, precomputeProteinPlyInfoBatchRun2, n_threads=masifpniOpts["n_threads"]//2,
              desc=precomputeProteinPlyInfoDesc2, batchRunFlag=batchRunFlag)
    precomputeNaPlyInfoDesc = "Precompute nucleic acid ply information"
    batchRun1(precomputeNaPlyInfo, precomputeNaPlyInfoBatchRun, n_threads=masifpniOpts["n_threads"],
              desc=precomputeNaPlyInfoDesc, batchRunFlag=batchRunFlag)

    # removeDirs([masifpniOpts["tmp_dir"]], empty=True)

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

    resumeFromChainPairs = False
    selectPniChainPairs = []
    if not resumeFromChainPairs:
        extractPniPDBbatchRun = []
        pdbId2field = {}
        for i in set(pdbIdChains) - set(unDownload):
            fields = i.split("_")
            pdb_id = fields[0]
            chainIds = list(itertools.chain.from_iterable([list(map(str, j)) for j in fields[1:]]))
            pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
            if not os.path.exists(pdbFile): continue
            extractPniPDBbatchRun.append((pdbFile, masifpniOpts["extract_pdb"], chainIds))
            pdbId2field[pdb_id] = fields

        resultList = batchRun(extractPniPDB, extractPniPDBbatchRun, n_threads=masifpniOpts["n_threads"])
        pniChainPairs = list(itertools.chain.from_iterable([i.get() for i in resultList]))
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
    else:
        tmp = np.load(masifpniOpts["pni_pairs_file"])
        for pair in tmp:
            selectPniChainPairs.append(BoundTuple(*pair.tolist()))

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
        dataprepFromList1(pdbIdChains, masifpniOpts, batchRunFlag=argv.nobatchRun)


if __name__ == '__main__':

    dataprep(sys.argv)

