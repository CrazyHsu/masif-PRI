#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: parseConfig.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:44:43
Last modified: 2022-09-11 20:44:43
'''

import tempfile, sys, os
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
from IPython.core.debugger import set_trace

DIR_SECTION = "default_dirs"
FILE_SECTION = "default_files"
SYS_SECTION = "default_sys_params"
SURFEAT_SECTION = "surface_features"
MASIFPNISITE_SECTION = "masifpni_site"
MASIFPNISEARCH_SECTION = "masifpni_search"
MASIFPNILIGAND_SECTION = "masifpni_ligand"

SECTION_TYPE_LIST_ORIGIN = [DIR_SECTION, FILE_SECTION, SYS_SECTION, SURFEAT_SECTION, MASIFPNISITE_SECTION,
                            MASIFPNISEARCH_SECTION, MASIFPNILIGAND_SECTION]
SECTION_TYPE_LIST_LOWER = [i.lower() for i in SECTION_TYPE_LIST_ORIGIN]

DIR_TAGS = [OUT_BASE_DIR, RAW_PDB_DIR, PDB_CHAIN_DIR, PLY_CHAIN_DIR, TMP_DIR, PLY_FILE_TEMPLATE, EXTRACT_PDB] = \
    ["out_base_dir", "raw_pdb_dir", "pdb_chain_dir", "ply_chain_dir", "tmp_dir", "ply_file_template", "extract_pdb"]

FILE_TAGS = [DEFAULT_PDB_FILE, PNI_PAIRS_FILE, SETTING_LOG] = ["default_pdb_file", "pni_pairs_file", "setting_log"]

SYS_TAGS = [N_THREADS, USE_GPU, USE_CPU, GPU_DEV, CPU_DEV] = ["n_threads", "use_gpu", "use_cpu", "gpu_dev", "cpu_dev"]

SURFEAT_TAGS = [USE_HBOND, USE_HPHOB, USE_APBS, COMPUTE_IFACE, MESH_RES, FEATURE_INTERPOLATION, RADIUS] = \
    ["use_hbond", "use_hphob", "use_apbs", "compute_iface", "mesh_res", "feature_interpolation", "radius"]

COMMON_TAGS = [MAX_SHAPE_SIZE, MAX_DISTANCE, MASIF_PRECOMPUTATION_DIR, MODEL_DIR, N_FEAT] = \
    ["max_shape_size", "max_distance", "masif_precomputation_dir", "model_dir", "n_feat"]

MASIFPNISITE_TAGS = [TRAINING_LIST, TESTING_LIST, N_CONV_LAYERS, RANGE_VAL_SAMPLES, OUT_PRED_DIR, OUT_SURF_DIR] = \
    ["training_list", "testing_list", "n_conv_layers", "range_val_samples", "out_pred_dir", "out_surf_dir"]

MASIFPNISEARCH_TAGS = [TRAINING_LIST, TESTING_LIST, MAX_SC_FILT, MIN_SC_FILT, POS_SURF_ACCEPT_PROBABILITY,
                       POS_INTERFACE_CUTOFF, RANGE_VAL_SAMPLES, CACHE_DIR, DESC_DIR, GIF_DESCRIPTORS_OUT, SC_RADIUS,
                       SC_INTERACTION_CUTOFF, SC_W] = \
    ["training_list", "testing_list", "max_sc_filt", "min_sc_filt", "pos_surf_accept_probability",
     "pos_interface_cutoff", "range_val_samples", "cache_dir", "desc_dir", "gif_descriptors_out", "sc_radius",
     "sc_interaction_cutoff", "sc_w"]

MASIFPNILIGAND_TAGS = [ASSEMBLY_DIR, LIGAND_COORDS_DIR, TRAIN_FRACT, VAL_FRACT, TEST_FRACT, TFRECORDS_DIR, N_CLASSES,
                       COSTFUN, TEST_SET_OUT_DIR] = \
    ["assembly_dir", "ligand_coords_dir", "train_fract", "val_fract", "test_fract", "tfrecords_dir",
     "n_classes", "costfun", "test_set_out_dir"]

MASIFPNISITE_TAGS = COMMON_TAGS + MASIFPNISITE_TAGS
MASIFPNISEARCH_TAGS = COMMON_TAGS + MASIFPNISEARCH_TAGS
MASIFPNILIGAND_TAGS = COMMON_TAGS + MASIFPNILIGAND_TAGS

INTEGER_TAGS = [N_THREADS, MAX_SHAPE_SIZE, N_CLASSES, N_CONV_LAYERS, N_FEAT]
FLOAT_TAGS = [MESH_RES, RADIUS, MAX_DISTANCE, MAX_SC_FILT, MIN_SC_FILT, POS_SURF_ACCEPT_PROBABILITY,
              POS_INTERFACE_CUTOFF, SC_RADIUS, SC_INTERACTION_CUTOFF, SC_W, RANGE_VAL_SAMPLES, TRAIN_FRACT, VAL_FRACT,
              TEST_FRACT]
BOOLEAN_TAGS = [USE_CPU, USE_GPU, USE_APBS, USE_HBOND, USE_HPHOB, COMPUTE_IFACE, FEATURE_INTERPOLATION]

VALID_TAGS = list(set(DIR_TAGS + FILE_TAGS + SYS_TAGS + SURFEAT_TAGS + MASIFPNISITE_TAGS + MASIFPNISEARCH_TAGS + MASIFPNILIGAND_TAGS))

class ParseConfig(object):
    def __init__(self, cfgFile=None, **args):
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        if cfgFile:
            self.config.read(cfgFile)
            self.validate()
            self.params = self.instantiate()

    def validate(self):
        validSet = set(VALID_TAGS)
        for secId in self.getIds():
            configSet = set(self.config.options(secId))
            badOptions = configSet - validSet
            if badOptions:
                raise ValueError('Unrecognized options found in %s section: %s\nValid options are: %s' % (
                    secId, ', '.join(badOptions), ', '.join(validSet)))

    def instantiate(self):
        params = {}
        tmpMapping = dict(zip([MASIFPNISITE_SECTION.lower(), MASIFPNISEARCH_SECTION.lower(), MASIFPNILIGAND_SECTION.lower()],
                              [MASIFPNISITE_SECTION, MASIFPNISEARCH_SECTION, MASIFPNILIGAND_SECTION]))
        for sec in self.getIds():
            if sec.lower() not in SECTION_TYPE_LIST_LOWER:
                raise ValueError("The section name of %s isn't in %s" % (sec, ",".join(SECTION_TYPE_LIST_ORIGIN)))

            if sec.lower() in [DIR_SECTION.lower(), FILE_SECTION.lower(), SYS_SECTION.lower(), SURFEAT_SECTION.lower()]:
                for opt in self.config.options(sec):
                    params[opt] = self.getValue(sec, opt)
            else:
                params[tmpMapping[sec.lower()]] = {}
                for opt in self.config.options(sec):
                    params[tmpMapping[sec.lower()]][opt] = self.getValue(sec, opt)
        return params

    def getIds(self):
        """Returns a list of all plot sections found in the config file."""
        return [s for s in self.config.sections()]

    def getValue(self, section, name, default=None):
        """Returns a value from the configuration."""
        tag = name.lower()

        try:
            if tag in FLOAT_TAGS:
                value = self.config.getfloat(section, name)
            elif tag in INTEGER_TAGS:
                value = self.config.getint(section, name)
            elif tag in BOOLEAN_TAGS:
                value = self.config.getboolean(section, name)
            else:
                value = self.config.get(section, name)
        except Exception:
            return default

        return value


class DefaultConfig(object):
    def __init__(self):
        self.masifpniOpts = self.getDefaultConfig()
        self.radii = self.getRadii()
        self.polarHydrogens = self.getPolarHydrogens()
        self.hbond_std_dev = np.pi / 3
        self.acceptorAngleAtom = self.getAcceptorAngleAtom()
        self.acceptorPlaneAtom = self.getAcceptorPlaneAtom()
        self.donorAtom = self.getDonorAtom()

    def getDefaultConfig(self):
        masifpniOpts = {}
        # Default directories
        masifpniOpts["out_base_dir"] = "test_data"
        masifpniOpts["raw_pdb_dir"] = os.path.join(masifpniOpts["out_base_dir"], "data_preparation", "00-raw_pdbs")
        masifpniOpts["pdb_chain_dir"] = os.path.join(masifpniOpts["out_base_dir"], "data_preparation", "01-benchmark_pdbs")
        masifpniOpts["ply_chain_dir"] = os.path.join(masifpniOpts["out_base_dir"], "data_preparation", "01-benchmark_surfaces")
        masifpniOpts["tmp_dir"] = tempfile.gettempdir()
        masifpniOpts["ply_file_template"] = os.path.join(os.path.join(masifpniOpts["ply_chain_dir"], "{}_{}.ply"))
        masifpniOpts["extract_pdb"] = os.path.join(masifpniOpts["out_base_dir"], "data_preparation", "extract_pdb")

        # Default files
        masifpniOpts["default_pdb_file"] = os.path.join(masifpniOpts["out_base_dir"], "default_pdb_file")
        masifpniOpts["pni_pairs_file"] = os.path.join(masifpniOpts["out_base_dir"], "pni_pairs_file.npy")
        masifpniOpts["setting_log"] = os.path.join(masifpniOpts["out_base_dir"], "setting_log.txt")

        # Default system params
        masifpniOpts["n_threads"] = 1
        masifpniOpts["use_gpu"] = False
        masifpniOpts["use_cpu"] = True
        masifpniOpts["gpu_dev"] = "/gpu:0"
        masifpniOpts["cpu_dev"] = "/cpu:0"

        # Surface features
        masifpniOpts["use_hbond"] = True
        masifpniOpts["use_hphob"] = True
        masifpniOpts["use_apbs"] = True
        masifpniOpts["compute_iface"] = True
        # Mesh resolution. Everything gets very slow if it is lower than 1.0
        masifpniOpts["mesh_res"] = 1.0
        masifpniOpts["feature_interpolation"] = True
        # Coords params
        masifpniOpts["radius"] = 12.0

        # Neural network patch application specific parameters.
        masifpniOpts["masifpni_search"] = {}
        masifpniOpts["masifpni_search"]["training_list"] = "lists/training.txt"
        masifpniOpts["masifpni_search"]["testing_list"] = "lists/testing.txt"
        masifpniOpts["masifpni_search"]["max_shape_size"] = 200
        masifpniOpts["masifpni_search"]["max_distance"] = 12.0  # Radius for the neural network.
        masifpniOpts["masifpni_search"]["masif_precomputation_dir"] = os.path.join(masifpniOpts["out_base_dir"],
                                                                                   "data_preparation", "precomputation", "search",
                                                                                   str(masifpniOpts["masifpni_search"]["max_distance"]))
        masifpniOpts["masifpni_search"]["n_feat"] = 5
        masifpniOpts["masifpni_search"]["max_sc_filt"] = 1.0
        masifpniOpts["masifpni_search"]["min_sc_filt"] = 0.5
        masifpniOpts["masifpni_search"]["pos_surf_accept_probability"] = 1.0
        masifpniOpts["masifpni_search"]["pos_interface_cutoff"] = 1.0
        masifpniOpts["masifpni_search"]["range_val_samples"] = 0.9  # 0.9 to 1.0
        masifpniOpts["masifpni_search"]["cache_dir"] = "nn_models/sc05/cache/"
        masifpniOpts["masifpni_search"]["model_dir"] = "nn_models/sc05/all_feat/model_data/"
        masifpniOpts["masifpni_search"]["desc_dir"] = "descriptors/sc05/all_feat/"
        masifpniOpts["masifpni_search"]["gif_descriptors_out"] = "gif_descriptors/"
        # Parameters for shape complementarity calculations.
        masifpniOpts["masifpni_search"]["sc_radius"] = 12.0
        masifpniOpts["masifpni_search"]["sc_interaction_cutoff"] = 1.5
        masifpniOpts["masifpni_search"]["sc_w"] = 0.25

        # Neural network patch application specific parameters.
        masifpniOpts["masifpni_site"] = {}
        masifpniOpts["masifpni_site"]["training_list"] = "lists/training.txt"
        masifpniOpts["masifpni_site"]["testing_list"] = "lists/testing.txt"
        masifpniOpts["masifpni_site"]["max_shape_size"] = 100
        masifpniOpts["masifpni_site"]["n_conv_layers"] = 3
        masifpniOpts["masifpni_site"]["max_distance"] = 9.0  # Radius for the neural network.
        masifpniOpts["masifpni_site"]["masif_precomputation_dir"] = os.path.join(masifpniOpts["out_base_dir"],
                                                                                 "data_preparation", "precomputation", "site",
                                                                                 str(masifpniOpts["masifpni_site"]["max_distance"]))
        masifpniOpts["masifpni_site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
        masifpniOpts["masifpni_site"]["model_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "nn_models")
        masifpniOpts["masifpni_site"]["out_pred_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred_data")
        masifpniOpts["masifpni_site"]["out_surf_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred_surfaces")
        masifpniOpts["masifpni_site"]["n_feat"] = 5

        # Neural network ligand application specific parameters.
        masifpniOpts["masifpni_ligand"] = {}
        masifpniOpts["masifpni_ligand"]["assembly_dir"] = os.path.join(masifpniOpts["out_base_dir"], "data_preparation", "pdbs_assembly")
        masifpniOpts["masifpni_ligand"]["ligand_coords_dir"] = os.path.join(masifpniOpts["out_base_dir"], "data_preparation", "ligand_coords")
        masifpniOpts["masifpni_ligand"]["max_distance"] = 12.0
        masifpniOpts["masifpni_ligand"]["masif_precomputation_dir"] = os.path.join(masifpniOpts["out_base_dir"],
                                                                                   "data_preparation", "precomputation", "ligand",
                                                                                   str(masifpniOpts["masifpni_ligand"]["max_distance"]))
        masifpniOpts["masifpni_ligand"]["max_shape_size"] = 200
        masifpniOpts["masifpni_ligand"]["n_feat"] = 5
        masifpniOpts["masifpni_ligand"]["train_fract"] = 0.72
        masifpniOpts["masifpni_ligand"]["val_fract"] = 0.08
        masifpniOpts["masifpni_ligand"]["test_fract"] = 0.2
        masifpniOpts["masifpni_ligand"]["tfrecords_dir"] = "data_preparation/tfrecords"
        masifpniOpts["masifpni_ligand"]["n_classes"] = 7
        masifpniOpts["masifpni_ligand"]["costfun"] = "dprime"
        masifpniOpts["masifpni_ligand"]["model_dir"] = "nn_models/all_feat/"
        masifpniOpts["masifpni_ligand"]["test_set_out_dir"] = "test_set_predictions/"

        return masifpniOpts

    def getListFromFile(self, myFile):
        myList = []
        with open(myFile) as f:
            for i in f.readlines():
                if i.startswith("#"): continue
                myList.append(i.strip())
        return myList

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
        self.reduce_bin = ""
        self.msms_bin = ""
        self.pdb2pqr_bin = ""
        self.apbs_bin = ""
        self.multivalue_bin = ""
        self.epsilon = 1.0e-6

    def initation(self):
        utilPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
        reduce_bin = os.path.join(utilPath, "reduce")
        msms_bin = os.path.join(utilPath, "msms")
        pdb2pqr_bin = os.path.join(utilPath, "pdb2pqr_pack", "pdb2pqr")
        apbs_bin = os.path.join(utilPath, "apbs")
        multivalue_bin = os.path.join(utilPath, "multivalue")

        if os.path.exists(reduce_bin):
        # if 'MSMS_BIN' in os.environ:
            self.reduce_bin = reduce_bin
        else:
            set_trace()
            print("ERROR: reduce_bin not set. Variable should point to MSMS program.")
            sys.exit(1)

        if os.path.exists(msms_bin):
        # if 'MSMS_BIN' in os.environ:
            self.msms_bin = msms_bin
        else:
            set_trace()
            print("ERROR: MSMS_BIN not set. Variable should point to MSMS program.")
            sys.exit(1)

        if os.path.exists(pdb2pqr_bin):
        # if 'PDB2PQR_BIN' in os.environ:
            self.pdb2pqr_bin = pdb2pqr_bin
        else:
            print("ERROR: PDB2PQR_BIN not set. Variable should point to PDB2PQR_BIN program.")
            sys.exit(1)

        if os.path.exists(apbs_bin):
        # if 'APBS_BIN' in os.environ:
            self.apbs_bin = apbs_bin
        else:
            print("ERROR: APBS_BIN not set. Variable should point to APBS program.")
            sys.exit(1)

        if os.path.exists(multivalue_bin):
        # if 'MULTIVALUE_BIN' in os.environ:
            self.multivalue_bin = multivalue_bin
        else:
            print("ERROR: MULTIVALUE_BIN not set. Variable should point to MULTIVALUE program.")
            sys.exit(1)

    def setEnviron(self):
        utilPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
        os.environ["LD_LIBRARY_PATH"] = os.path.join(utilPath, "lib")
