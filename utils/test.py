#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: test.py
Author: CrazyHsu @ crazyhsu9527@gmail.com 
Created on: 2022-09-20 14:54:33
Last modified: 2022-09-20 15:07:24
'''

import os
import subprocess

# from test2 import ParseConfig
#
# utilPath = os.path.dirname(os.path.abspath(__file__))
# os.environ["LD_LIBRARY_PATH"] += os.pathsep + os.path.join(utilPath, "lib")
#
# print(os.environ["LD_LIBRARY_PATH"])
#
# pdbfile = "/home/xufeng/xufeng/Projects/lailab/denovo_protein_design/masif-PRI/test_data/1a0g.ppi.pdb"
# outBase = "test"
# # cmd = "pdb2pqr_pack/pdb2pqr --ff=parse --whitespace --noopt --apbs-input {} {}".format(pdbfile, outBase)
# #subprocess.call(cmd, shell=True, executable="/bin/bash")
#
# cmd = "/home/xufeng/xufeng/Projects/lailab/denovo_protein_design/masif-PRI/utils/apbs {}.in".format(outBase)
# subprocess.call(cmd, shell=True, executable="/bin/bash")
import time,random
def test3(a1):
    print(11)
    # t = random.randint(1, int(a1)//10+1)
    # time.sleep(t)
    # print(a1)

def test4(a1, a2):
    print(11)


def test2(myFunc, argList):
    j = 0
    for i in argList:
        # j += 1
        # print(j)
        myFunc(*i)


def batchRun(myFunc, argList, n_threads=1):
    from multiprocessing import Pool, JoinableQueue
    q = JoinableQueue(5)
    pool = Pool(processes=n_threads)
    for arg in argList:
        # if len(arg) == 1:
        #     pool.apply_async(myFunc, args=(arg[0],))
        # else:
        #     pool.apply_async(myFunc, args=(arg[0],))
        pool.apply_async(myFunc, arg)
    pool.close()
    pool.join()
    q.join()

def test1():
    # l = [[1,2,3], [4,5,6]] * 10
    l = [(i,) for i in range(100)]
    print(l)
    # batchRun(test3, l, n_threads=5)
    batchRun(test3, l, n_threads=5)


def main():
    test1()

if __name__ == '__main__':
    main()