#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: defaultConfig.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:44:43
Last modified: 2022-09-11 20:44:43
'''

import tempfile, sys, os
import numpy as np
from IPython.core.debugger import set_trace


class DefaultConfig(object):
    def __init__(self):
        self.masifOpts = self.getDefaultConfig()
        self.radii = self.getRadii()
        self.polarHydrogens = self.getPolarHydrogens()
        self.hbond_std_dev = np.pi / 3
        self.acceptorAngleAtom = self.getAcceptorAngleAtom()
        self.acceptorPlaneAtom = self.getAcceptorPlaneAtom()
        self.donorAtom = self.getDonorAtom()

    def getDefaultConfig(self):
        masifOpts = {}
        # Default directories
        masifOpts["raw_pdb_dir"] = "data_preparation/00-raw_pdbs/"
        masifOpts["pdb_chain_dir"] = "data_preparation/01-benchmark_pdbs/"
        masifOpts["ply_chain_dir"] = "data_preparation/01-benchmark_surfaces/"
        masifOpts["tmp_dir"] = tempfile.gettempdir()
        masifOpts["ply_file_template"] = masifOpts["ply_chain_dir"] + "/{}_{}.ply"

        # Surface features
        masifOpts["use_hbond"] = True
        masifOpts["use_hphob"] = True
        masifOpts["use_apbs"] = True
        masifOpts["compute_iface"] = True
        # Mesh resolution. Everything gets very slow if it is lower than 1.0
        masifOpts["mesh_res"] = 1.0
        masifOpts["feature_interpolation"] = True


        # Coords params
        masifOpts["radius"] = 12.0

        # # Neural network patch application specific parameters.
        # masifOpts["ppi_search"] = {}
        # masifOpts["ppi_search"]["training_list"] = "lists/training.txt"
        # masifOpts["ppi_search"]["testing_list"] = "lists/testing.txt"
        # masifOpts["ppi_search"]["max_shape_size"] = 200
        # masifOpts["ppi_search"]["max_distance"] = 12.0  # Radius for the neural network.
        # masifOpts["ppi_search"][
        #     "masif_precomputation_dir"
        # ] = "data_preparation/04b-precomputation_12A/precomputation/"
        # masifOpts["ppi_search"]["feat_mask"] = [1.0] * 5
        # masifOpts["ppi_search"]["max_sc_filt"] = 1.0
        # masifOpts["ppi_search"]["min_sc_filt"] = 0.5
        # masifOpts["ppi_search"]["pos_surf_accept_probability"] = 1.0
        # masifOpts["ppi_search"]["pos_interface_cutoff"] = 1.0
        # masifOpts["ppi_search"]["range_val_samples"] = 0.9  # 0.9 to 1.0
        # masifOpts["ppi_search"]["cache_dir"] = "nn_models/sc05/cache/"
        # masifOpts["ppi_search"]["model_dir"] = "nn_models/sc05/all_feat/model_data/"
        # masifOpts["ppi_search"]["desc_dir"] = "descriptors/sc05/all_feat/"
        # masifOpts["ppi_search"]["gif_descriptors_out"] = "gif_descriptors/"
        # # Parameters for shape complementarity calculations.
        # masifOpts["ppi_search"]["sc_radius"] = 12.0
        # masifOpts["ppi_search"]["sc_interaction_cutoff"] = 1.5
        # masifOpts["ppi_search"]["sc_w"] = 0.25

        # Neural network patch application specific parameters.
        masifOpts["site"] = {}
        masifOpts["site"]["training_list"] = "lists/training.txt"
        masifOpts["site"]["testing_list"] = "lists/testing.txt"
        masifOpts["site"]["max_shape_size"] = 100
        masifOpts["site"]["n_conv_layers"] = 3
        masifOpts["site"]["max_distance"] = 9.0  # Radius for the neural network.
        masifOpts["site"][
            "masif_precomputation_dir"
        ] = "data_preparation/04a-precomputation_9A/precomputation/"
        masifOpts["site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
        masifOpts["site"]["model_dir"] = "nn_models/all_feat_3l/model_data/"
        masifOpts["site"]["out_pred_dir"] = "output/all_feat_3l/pred_data/"
        masifOpts["site"]["out_surf_dir"] = "output/all_feat_3l/pred_surfaces/"
        masifOpts["site"]["feat_mask"] = [1.0] * 5

        # # Neural network ligand application specific parameters.
        # masifOpts["ligand"] = {}
        # masifOpts["ligand"]["assembly_dir"] = "data_preparation/00b-pdbs_assembly"
        # masifOpts["ligand"]["ligand_coords_dir"] = "data_preparation/00c-ligand_coords"
        # masifOpts["ligand"][
        #     "masif_precomputation_dir"
        # ] = "data_preparation/04a-precomputation_12A/precomputation/"
        # masifOpts["ligand"]["max_shape_size"] = 200
        # masifOpts["ligand"]["feat_mask"] = [1.0] * 5
        # masifOpts["ligand"]["train_fract"] = 0.9 * 0.8
        # masifOpts["ligand"]["val_fract"] = 0.1 * 0.8
        # masifOpts["ligand"]["test_fract"] = 0.2
        # masifOpts["ligand"]["tfrecords_dir"] = "data_preparation/tfrecords"
        # masifOpts["ligand"]["max_distance"] = 12.0
        # masifOpts["ligand"]["n_classes"] = 7
        # masifOpts["ligand"]["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
        # masifOpts["ligand"]["costfun"] = "dprime"
        # masifOpts["ligand"]["model_dir"] = "nn_models/all_feat/"
        # masifOpts["ligand"]["test_set_out_dir"] = "test_set_predictions/"

        return masifOpts

    def getRadii(self):
        radii = {}
        radii["N"] = "1.540000"
        radii["N"] = "1.540000"
        radii["O"] = "1.400000"
        radii["C"] = "1.740000"
        radii["H"] = "1.200000"
        radii["S"] = "1.800000"
        radii["P"] = "1.800000"
        radii["Z"] = "1.39"
        radii["X"] = "0.770000"  ## Radii of CB or CA in disembodied case.

        return radii

    def getPolarHydrogens(self):
        # This  polar hydrogen's names correspond to that of the program Reduce.
        polarHydrogens = {}
        polarHydrogens["ALA"] = ["H"]
        polarHydrogens["GLY"] = ["H"]
        polarHydrogens["SER"] = ["H", "HG"]
        polarHydrogens["THR"] = ["H", "HG1"]
        polarHydrogens["LEU"] = ["H"]
        polarHydrogens["ILE"] = ["H"]
        polarHydrogens["VAL"] = ["H"]
        polarHydrogens["ASN"] = ["H", "HD21", "HD22"]
        polarHydrogens["GLN"] = ["H", "HE21", "HE22"]
        polarHydrogens["ARG"] = ["H", "HH11", "HH12", "HH21", "HH22", "HE"]
        polarHydrogens["HIS"] = ["H", "HD1", "HE2"]
        polarHydrogens["TRP"] = ["H", "HE1"]
        polarHydrogens["PHE"] = ["H"]
        polarHydrogens["TYR"] = ["H", "HH"]
        polarHydrogens["GLU"] = ["H"]
        polarHydrogens["ASP"] = ["H"]
        polarHydrogens["LYS"] = ["H", "HZ1", "HZ2", "HZ3"]
        polarHydrogens["PRO"] = []
        polarHydrogens["CYS"] = ["H"]
        polarHydrogens["MET"] = ["H"]

        return polarHydrogens

    def getAcceptorAngleAtom(self):
        # Dictionary from an acceptor atom to its directly bonded atom on which to
        # compute the angle.
        acceptorAngleAtom = {}
        acceptorAngleAtom["O"] = "C"
        acceptorAngleAtom["O1"] = "C"
        acceptorAngleAtom["O2"] = "C"
        acceptorAngleAtom["OXT"] = "C"
        acceptorAngleAtom["OT1"] = "C"
        acceptorAngleAtom["OT2"] = "C"

        # ASN Acceptor
        acceptorAngleAtom["OD1"] = "CG"

        # ASP
        # Plane: CB-CG-OD1
        # Angle CG-ODX-point: 120
        acceptorAngleAtom["OD2"] = "CG"

        acceptorAngleAtom["OE1"] = "CD"
        acceptorAngleAtom["OE2"] = "CD"

        # HIS Acceptors: ND1, NE2
        # Plane ND1-CE1-NE2
        # Angle: ND1-CE1 : 125.5
        # Angle: NE2-CE1 : 125.5
        acceptorAngleAtom["ND1"] = "CE1"
        acceptorAngleAtom["NE2"] = "CE1"

        # TYR acceptor OH
        # Plane: CE1-CZ-OH
        # Angle: CZ-OH 120
        acceptorAngleAtom["OH"] = "CZ"

        # SER acceptor:
        # Angle CB-OG-X: 120
        acceptorAngleAtom["OG"] = "CB"

        # THR acceptor:
        # Angle: CB-OG1-X: 120
        acceptorAngleAtom["OG1"] = "CB"

        return acceptorAngleAtom

    def getAcceptorPlaneAtom(self):
        # Dictionary from acceptor atom to a third atom on which to compute the plane.
        acceptorPlaneAtom = {}
        acceptorPlaneAtom["O"] = "CA"

        acceptorPlaneAtom["OD1"] = "CB"
        acceptorPlaneAtom["OD2"] = "CB"

        acceptorPlaneAtom["OE1"] = "CG"
        acceptorPlaneAtom["OE2"] = "CG"

        acceptorPlaneAtom["ND1"] = "NE2"
        acceptorPlaneAtom["NE2"] = "ND1"

        acceptorPlaneAtom["OH"] = "CE1"

        return acceptorPlaneAtom

    def getDonorAtom(self):
        # Dictionary from an H atom to its donor atom.
        donorAtom = {}
        donorAtom["H"] = "N"
        # Hydrogen bond information.
        # ARG
        # ARG NHX
        # Angle: NH1, HH1X, point and NH2, HH2X, point 180 degrees.
        # radii from HH: radii[H]
        # ARG NE
        # Angle: ~ 120 NE, HE, point, 180 degrees
        donorAtom["HH11"] = "NH1"
        donorAtom["HH12"] = "NH1"
        donorAtom["HH21"] = "NH2"
        donorAtom["HH22"] = "NH2"
        donorAtom["HE"] = "NE"

        # ASN
        # Angle ND2,HD2X: 180
        # Plane: CG,ND2,OD1
        # Angle CG-OD1-X: 120
        donorAtom["HD21"] = "ND2"
        donorAtom["HD22"] = "ND2"

        # GLU
        # PLANE: CD-OE1-OE2
        # ANGLE: CD-OEX: 120
        # GLN
        # PLANE: CD-OE1-NE2
        # Angle NE2,HE2X: 180
        # ANGLE: CD-OE1: 120
        donorAtom["HE21"] = "NE2"
        donorAtom["HE22"] = "NE2"

        # HIS Donors: ND1, NE2
        # Angle ND1-HD1 : 180
        # Angle NE2-HE2 : 180
        donorAtom["HD1"] = "ND1"
        donorAtom["HE2"] = "NE2"

        # TRP Donor: NE1-HE1
        # Angle NE1-HE1 : 180
        donorAtom["HE1"] = "NE1"

        # LYS Donor NZ-HZX
        # Angle NZ-HZX : 180
        donorAtom["HZ1"] = "NZ"
        donorAtom["HZ2"] = "NZ"
        donorAtom["HZ3"] = "NZ"

        # TYR donor: OH-HH
        # Angle: OH-HH 180
        donorAtom["HH"] = "OH"

        # SER donor:
        # Angle: OG-HG-X: 180
        donorAtom["HG"] = "OG"

        # THR donor:
        # Angle: OG1-HG1-X: 180
        donorAtom["HG1"] = "OG1"

        return donorAtom


class GlobalVars(object):
    def __init__(self):
        self.msms_bin = ""
        self.pdb2pqr_bin = ""
        self.apbs_bin = ""
        self.multivalue_bin = ""
        self.epsilon = 1.0e-6

    def initation(self):
        if 'MSMS_BIN' in os.environ:
            self.msms_bin = os.environ['MSMS_BIN']
        else:
            set_trace()
            print("ERROR: MSMS_BIN not set. Variable should point to MSMS program.")
            sys.exit(1)

        if 'PDB2PQR_BIN' in os.environ:
            self.pdb2pqr_bin = os.environ['PDB2PQR_BIN']
        else:
            print("ERROR: PDB2PQR_BIN not set. Variable should point to PDB2PQR_BIN program.")
            sys.exit(1)

        if 'APBS_BIN' in os.environ:
            self.apbs_bin = os.environ['APBS_BIN']
        else:
            print("ERROR: APBS_BIN not set. Variable should point to APBS program.")
            sys.exit(1)

        if 'MULTIVALUE_BIN' in os.environ:
            self.multivalue_bin = os.environ['MULTIVALUE_BIN']
        else:
            print("ERROR: MULTIVALUE_BIN not set. Variable should point to MULTIVALUE program.")
            sys.exit(1)
