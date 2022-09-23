#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: commonFuncs.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-18 21:56:11
Last modified: 2022-09-18 21:56:11
'''

import os, shutil
from parseConfig import DefaultConfig, ParseConfig

def mergeParams(argv):
    masifpniOpts = DefaultConfig().masifpniOpts
    # masifpniOpts["n_threads"] = argv.n_threads
    # params = masifOpts["masifpni_site"]

    outSetting = ""
    if argv.config:
        custom_params_file = argv.config
        custom_params = ParseConfig(custom_params_file).params
        # custom_params = importlib.import_module(custom_params_file, package=None)
        # custom_params = custom_params.custom_params

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


def removeFiles(myDir=None, fileList=None):
    for f in fileList:
        if myDir:
            os.remove(os.path.join(myDir, f.strip("\n")))
        else:
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

