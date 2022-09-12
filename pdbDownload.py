#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: pdbDownload.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:47:44
Last modified: 2022-09-11 20:47:44
'''

import os, sys
from Bio.PDB import * 
from defaultConfig import DefaultConfig
from inputOutputProcess import protonate

masifOpts = DefaultConfig().masifOpts

if len(sys.argv) <= 1: 
    print("Usage: "+sys.argv[0]+" PDBID_A_B")
    print("A or B are the chains to include in this pdb.")
    sys.exit(1)

if not os.path.exists(masifOpts['raw_pdb_dir']):
    os.makedirs(masifOpts['raw_pdb_dir'])

if not os.path.exists(masifOpts['tmp_dir']):
    os.mkdir(masifOpts['tmp_dir'])

in_fields = sys.argv[1].split('_')
pdb_id = in_fields[0]

def pdbDownload(masifOpts, pdb_id):
    # Download pdb
    pdbl = PDBList(server='http://ftp.wwpdb.org')
    pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masifOpts['tmp_dir'], file_format='pdb')

    ##### Protonate with reduce, if hydrogens included.
    # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
    protonated_file = masifOpts['raw_pdb_dir']+"/"+pdb_id+".pdb"
    protonate(pdb_filename, protonated_file)
    pdb_filename = protonated_file


pdbDownload(masifOpts, pdb_id)
