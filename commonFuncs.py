#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: commonFuncs.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-18 21:56:11
Last modified: 2022-09-18 21:56:11
'''

import os, shutil, itertools
import numpy as np

from tqdm import tqdm
from collections import namedtuple
from parseConfig import DefaultConfig, ParseConfig

BoundTuple = namedtuple("BoundTuple", ["PDB_id", "pChain", "naChain", "naType"])
BoundTuple.__new__.__defaults__ = ("",) * len(BoundTuple._fields)

# Apply mask to input_feat
def batchRun(myFunc, argList, n_threads=1, chunk=False, chunkSize=500):
    from multiprocessing import Pool, JoinableQueue
    resultList = []
    if chunk:
        for i in range(0, len(argList), chunkSize):
            chunkList = argList[i:i+chunkSize]
            pool = Pool(processes=n_threads)
            for arg in chunkList:
                if len(arg) == 1:
                    res = pool.apply_async(myFunc, (arg[0],))
                else:
                    res = pool.apply_async(myFunc, arg)
                resultList.append(res)
            pool.close()
            pool.join()
    else:
        pool = Pool(processes=n_threads)
        for arg in argList:
            if len(arg) == 1:
                res = pool.apply_async(myFunc, (arg[0], ))
            else:
                res = pool.apply_async(myFunc, arg)
            resultList.append(res)
        pool.close()
        pool.join()
    return resultList

def batchRun1(myFunc, argList, n_threads=1, chunk=False, chunkSize=500, desc="", batchRunFlag=True):
    resultList = []
    if batchRunFlag:
        from multiprocessing import Pool, JoinableQueue
        jobs = []
        if chunk:
            for i in range(0, len(argList), chunkSize):
                chunkList = argList[i:i+chunkSize]
                pool = Pool(processes=n_threads, maxtasksperchild=10)
                for arg in chunkList:
                    if len(arg) == 1:
                        jobs.append(pool.apply_async(myFunc, (arg[0],)))
                    else:
                        jobs.append(pool.apply_async(myFunc, arg))
                pool.close()
                for job in tqdm(jobs, desc=desc):
                    resultList.append(job.get())
                pool.join()
        else:
            pool = Pool(processes=n_threads, maxtasksperchild=10)
            for arg in argList:
                if len(arg) == 1:
                    jobs.append(pool.apply_async(myFunc, (arg[0],)))
                else:
                    jobs.append(pool.apply_async(myFunc, arg))
            pool.close()
            for job in tqdm(jobs, desc=desc):
                resultList.append(job.get())
            pool.join()
    else:
        print(desc)
        if chunk:
            for i in range(0, len(argList), chunkSize):
                chunkList = argList[i:i + chunkSize]
                for arg in chunkList:
                    resultList.append(myFunc(*arg))
        else:
            for arg in argList:
                resultList.append(myFunc(*arg))
    return resultList


def makeLink(sourcePath, targetPath):
    if os.path.islink(targetPath) or os.path.exists(targetPath):
        os.remove(targetPath)
    os.symlink(sourcePath, targetPath)


def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def getIdChainPairs(masifpniOpts, fromFile=None, fromList=None, fromCustomPDB=None):
    from Bio.SeqUtils import IUPACData
    from biopandas.pdb import PandasPdb
    from inputOutputProcess import findProteinChainBoundNA
    PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

    # BoundTuple = namedtuple("BoundTuple", ["PDB_id", "pChain", "naChain", "naType"])
    # BoundTuple.__new__.__defaults__ = ("",) * len(BoundTuple._fields)
    myList = []
    if fromFile:
        with open(fromFile) as f:
            for line in f.readlines():
                if line.startswith("#"): continue
                fields = line.strip().split("_")
                pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], fields[0] + ".pdb")
                if len(fields) == 1:
                    myList.extend(findProteinChainBoundNA(pdbFile))
                elif len(fields) == 2:
                    pChains = list(map(str, fields[1]))
                    for pChain in pChains:
                        myList.extend(findProteinChainBoundNA(pdbFile, pChainId=pChain))
                else:
                    x, y, z = line.strip().split("_")
                    for i in list(itertools.product(list(map(str, y)), list(map(str, z)))):
                        myList.append(BoundTuple(x, i[0], i[1], ""))

    if fromList:
        for l in fromList:
            fields = l.strip().split("_")
            pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], fields[0] + ".pdb")
            if len(fields) == 1:
                myList.extend(findProteinChainBoundNA(pdbFile))
            elif len(fields) == 2:
                pChains = list(map(str, fields[1]))
                for pChain in pChains:
                    myList.extend(findProteinChainBoundNA(pdbFile, pChainId=pChain))
            else:
                x, y, z = fields
                for i in list(itertools.product(list(map(str, y)), list(map(str, z)))):
                    myList.append(BoundTuple(x, i[0], i[1], ""))

    if fromCustomPDB:
        with open(fromCustomPDB) as f:
            for pdbFile in f.readlines():
                if pdbFile.startswith("#"): continue
                pdbFile = pdbFile.strip()
                tmp = findProteinChainBoundNA(pdbFile)
                fields = os.path.basename(pdbFile).split("_")
                pdbId = fields[0]
                if len(tmp) == 0:
                    ppdb = PandasPdb()
                    pdbStruc = ppdb.read_pdb(pdbFile)
                    atomDf = pdbStruc.df["ATOM"]
                    pChains = list(set(atomDf[atomDf["residue_name"].isin(PROTEIN_LETTERS)]["chain_id"].tolist()))
                    myList.extend([BoundTuple(pdbId, i, "", "") for i in pChains])
                else:
                    if len(fields) == 1:
                        myList.extend(tmp)
                    elif len(fields) == 2:
                        for i in fields[1]:
                            myList.extend([j for j in tmp if j.pChain == i])
                    else:
                        x, y, z = fields
                        for i in list(itertools.product(list(map(str, y)), list(map(str, z)))):
                            for j in tmp:
                                if i[0] == j.pChain and i[1] == j.naChain:
                                    myList.append(BoundTuple(x, i[0], i[1], j.naType))

    return myList


def mergeParams(argv):
    masifpniOpts = DefaultConfig().masifpniOpts
    # params = masifOpts["masifpni_site"]

    outSetting = ""
    if argv.config:
        custom_params_file = argv.config
        custom_params = ParseConfig(custom_params_file).params

        for key in custom_params:
            if key not in ["masifpni_site", "masifpni_search", "masifpni_ligand"]:
                outSetting += "Setting {} to {} \n".format(key, custom_params[key])
                # print("Setting {} to {} ".format(key, custom_params[key]), file=logfile)
                masifpniOpts[key] = custom_params[key]
            else:
                for key2 in custom_params[key]:
                    outSetting += "Setting {} to {} \n".format(key2, custom_params[key][key2])
                    # print("Setting {} to {} ".format(key2, custom_params[key][key2]), file=logfile)
                    masifpniOpts[key][key2] = custom_params[key][key2]
    else:
        for key in masifpniOpts:
            outSetting += "Setting {} to {} \n".format(key, masifpniOpts[key])
            # print("Setting {} to {} ".format(key, masifpniOpts[key]), file=logfile)

    masifpniOpts["n_threads"] = argv.n_threads
    resolveDirs([masifpniOpts["out_base_dir"], masifpniOpts["log_dir"], masifpniOpts["tmp_dir"]])
    logfile = open(masifpniOpts["setting_log"], "w")
    logfile.write(outSetting)
    logfile.close()

    return masifpniOpts


def resolveDir(dirName, chdir=False):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    if chdir:
        os.chdir(dirName)


def resolveDirs(dirList):
    for d in dirList:
        resolveDir(d, chdir=False)


def removeFile(myFile=None, fileList=None, myDir=None):
    if fileList:
        for f in fileList:
            if myDir:
                if os.path.exists(os.path.join(myDir, f.strip("\n"))):
                    os.remove(os.path.join(myDir, f.strip("\n")))
            else:
                if os.path.exists(f):
                    os.remove(f.strip("\n"))
    else:
        if myFile:
            if myDir:
                if os.path.exists(os.path.join(myDir, myFile.strip("\n"))):
                    os.remove(os.path.join(myDir, myFile.strip("\n")))
            else:
                if os.path.exists(myFile):
                    os.remove(myFile)


def removeFiles(fileList=None, myDir=None):
    for f in fileList:
        if myDir:
            if os.path.exists(os.path.join(myDir, f.strip("\n"))):
                os.remove(os.path.join(myDir, f.strip("\n")))
        else:
            if os.path.exists(f):
                os.remove(f.strip("\n"))

def removeDirs(myDirs, empty=True):
    for i in myDirs:
        if empty:
            for filename in os.listdir(i):
                file_path = os.path.join(i, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        else:
            shutil.rmtree(i)

