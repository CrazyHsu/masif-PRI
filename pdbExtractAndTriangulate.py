#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: pdbExtractAndTriangulate.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:44:16
Last modified: 2022-09-11 20:44:16
'''

import os, shutil, sys
import pymesh
import numpy as np

# import Bio
# from Bio.PDB import *

from IPython.core.debugger import set_trace
from defaultConfig import DefaultConfig
from triangulation import (computeMSMS, fixMesh, computeHydrophobicity, computeCharges, assignChargesToNewMesh,
                           computeAPBS, compute_normal)
from inputOutputProcess import (extractPDB, protonate, read_ply, save_ply)
from sklearn.neighbors import KDTree

masifOpts = DefaultConfig().masifOpts


if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)


# Save the chains as separate files. 
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masifOpts["ligand"]["assembly_dir"],pdb_id+".pdb")
else:
    pdb_filename = masifOpts['raw_pdb_dir']+pdb_id+".pdb"
tmp_dir = masifOpts['tmp_dir']
protonated_file = tmp_dir+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# Extract chains of interest.
out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

# Compute MSMS of surface w/hydrogens,
vertices1, faces1, normals1, names1, areas1 = None, None, None, None, None
try:
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+".pdb", protonate=True)
except:
    set_trace()

# Compute "charged" vertices
vertex_hbond = None
if masifOpts['use_hbond']:
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)

# For each surface residue, assign the hydrophobicity of its amino acid.
vertex_hphobicity = None
if masifOpts['use_hphob']:
    vertex_hphobicity = computeHydrophobicity(names1)

# If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
vertices2 = vertices1
faces2 = faces1

# Fix the mesh.
mesh = pymesh.form_mesh(vertices2, faces2)
regular_mesh = fixMesh(mesh, masifOpts['mesh_res'])

# Compute the normals
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
# Assign charges on new vertices based on charges of old vertices (nearest
# neighbor)

vertex_hbond = None
if masifOpts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masifOpts)

vertex_hphobicity = None
if masifOpts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masifOpts)

vertex_charges = None
if masifOpts['use_apbs']:
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)

iface = np.zeros(len(regular_mesh.vertices))
if 'compute_iface' in masifOpts and masifOpts['compute_iface']:
    # Compute the surface of the entire complex and from that compute the interface.
    v3, f3, _, _, _ = computeMSMS(pdb_filename, protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d) # Square d, because this is how it was in the pyflann version.
    assert(len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal,
             charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond,
             hphob=vertex_hphobicity, iface=iface)

else:
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal,
             charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)

if not os.path.exists(masifOpts['ply_chain_dir']):
    os.makedirs(masifOpts['ply_chain_dir'])
if not os.path.exists(masifOpts['pdb_chain_dir']):
    os.makedirs(masifOpts['pdb_chain_dir'])
shutil.copy(out_filename1+'.ply', masifOpts['ply_chain_dir']) 
shutil.copy(out_filename1+'.pdb', masifOpts['pdb_chain_dir']) 



