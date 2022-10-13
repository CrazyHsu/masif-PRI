#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: precompute.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:49:01
Last modified: 2022-09-11 20:49:01
'''

import os
import numpy as np
# import warnings

# from defaultConfig import DefaultConfig
from commonFuncs import resolveDir
from readDataFromSurface import read_data_from_surface

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)


def precomputeProteinPlyInfo(masifpniOpts, pdb_id, pChain):
    ply_file = masifpniOpts['ply_file_template'].format(pdb_id, pChain)
    if not os.path.exists(ply_file): return

    params = masifpniOpts['masifpni_site']

    try:
        input_feat, rho, theta, mask, neigh_indices, iface_labels, verts = read_data_from_surface(ply_file, params)
    except:
        return

    my_precomp_dir = os.path.join(params['masif_precomputation_dir'], pdb_id)
    resolveDir(my_precomp_dir, chdir=False)
    np.save(os.path.join(my_precomp_dir, pChain + '_rho_wrt_center.npy'), rho)
    np.save(os.path.join(my_precomp_dir, pChain + '_theta_wrt_center.npy'), theta)
    np.save(os.path.join(my_precomp_dir, pChain + '_input_feat.npy'), input_feat)
    np.save(os.path.join(my_precomp_dir, pChain + '_mask.npy'), mask)
    np.save(os.path.join(my_precomp_dir, pChain + '_list_indices.npy'), neigh_indices)
    np.save(os.path.join(my_precomp_dir, pChain + '_iface_labels.npy'), iface_labels)
    # Save x, y, z
    np.save(os.path.join(my_precomp_dir, pChain + '_X.npy'), verts[:, 0])
    np.save(os.path.join(my_precomp_dir, pChain + '_Y.npy'), verts[:, 1])
    np.save(os.path.join(my_precomp_dir, pChain + '_Z.npy'), verts[:, 2])


def precomputeNaPlyInfo(masifpniOpts, pdb_id, naChain):
    pass
