#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: masifPNI.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:31:49
Last modified: 2022-09-11 20:31:49
'''

import sys, warnings
import multiprocessing
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

__version__ = "1.0.0"

defaultArguments = ["masifPNI.py", "masifPNI-site", "masifPNI-search", "masifPNI-ligand", "dataprep", "train", "predict"]


def dataprep(argv):
    print(vars(argv))
    print(len(vars(argv)))
    print(11)


def pdbDownload(argv):
    print(vars(argv))
    print(len(vars(argv)))
    print(11)


def train_masifPNI_site1(argv):
    print(22)


def masifPNI_site_predict(argv):
    print(33)


def test4(argv):
    # from Bio.PDB import PDBList
    # pdbl = PDBList(server='http://ftp.wwpdb.org')
    #
    # from multiprocessing import Pool, JoinableQueue
    # q = JoinableQueue(5)
    # pool = Pool(processes=8)
    #
    # tmp = argv.file
    # resultList = []
    # with open(tmp) as f:
    #     for line in f.readlines():
    #         if line.startswith("#"): continue
    #         pdb_id = line.strip().split("_")[0]
    #         res = pool.apply_async(pdbl.retrieve_pdb_file, kwds={"pdb_code":pdb_id, "pdir":"test_data", "file_format":'pdb', "overwrite":True})
    #         resultList.append(res)
    # pool.close()
    # pool.join()
    #
    # q.join()
    # for i in resultList:
    #     tmp = i.get()
    #     print(tmp)
    print(44)


def parseArgsDownload(parser, argv):
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    parser.add_argument('--config', dest='config', help='Config file contains the parameters to run masifPNI.', type=str)
    parser.add_argument('-l', '--list', type=str, default=None, help="Lists of PDB ids, separated by comma")
    parser.add_argument('-f', '--file', type=str, default=None, help="File contains lists of PDB ids. One per separate line.")
    parser.add_argument('-a', "--all", action="store_true", default=False, help="Download all PDB entries.")
    parser.add_argument('-overwrite', action="store_true", default=False, help="Overwrite existing PDB files.")
    parser.add_argument('-n', "--n_threads", type=int, default=1, help="The number of threads to download pdb files.")
    parser.add_argument('--nobatchRun', action="store_false", default=True, help="Don't batch run the program.")
    from pdbDownload import pdbDownload
    parser.set_defaults(func=pdbDownload)


def parseArgsSite(parser, argv):
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    subparsers = parser.add_subparsers(help="Modes to run masifPNI-site", metavar='{dataprep, train, predict}')

    parser_dataprep = subparsers.add_parser("dataprep", help="Prepare the data used in next steps")
    optional_dataprep = parser_dataprep._action_groups.pop()
    required_dataprep = parser_dataprep.add_argument_group('required arguments')
    optional_dataprep.add_argument('--config', dest='config', help='Config file contains the parameters to run masifPNI.', type=str)
    optional_dataprep.add_argument('-l', '--list', type=str, default=None, help="Lists of PDB ids, separated by comma")
    optional_dataprep.add_argument('-f', '--file', type=str, default=None, help="File contains lists of PDB ids. One per separate line.")
    optional_dataprep.add_argument('-n', '--n_threads', type=int, default=1, help="Threads used to prepare files.")
    optional_dataprep.add_argument('-overwrite', action="store_true", default=False, help="Overwrite existing PDB files.")
    optional_dataprep.add_argument('-add_default', "--add_default", action="store_true", default=False, help="Add default PDB ids to list used to data preparation.")
    optional_dataprep.add_argument('--nobatchRun', action="store_false", default=True, help="Don't batch run the program.")
    optional_dataprep.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    parser_dataprep._action_groups.append(optional_dataprep)
    from dataPreparation import dataprep
    parser_dataprep.set_defaults(func=dataprep)

    parser_train = subparsers.add_parser("train", help="Train the neural network model with protein-RNA interaction complex files")
    parser_train.add_argument('--config', dest='config', help='Config file contains the parameters to run masifPNI.', type=str)
    parser_train.add_argument('--training_list', dest='training_list', type=str, default=False, help='The list of PDB ids used to train nucleic network model.')
    parser_train.add_argument('--testing_list', dest='testing_list', type=str, default=False, help='The list of PDB ids used to test nucleic network model.')
    parser_train.add_argument('--draw_roc', dest='draw_roc', action="store_true", default=False, help='Whether to draw ROC plot.')
    parser_train.add_argument('-n', '--n_threads', type=int, default=1, help="Threads used to prepare files.")
    parser_train.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    from masifPNI_site.masifPNI_site_train import train_masifPNI_site1
    parser_train.set_defaults(func=train_masifPNI_site1)

    parser_prection = subparsers.add_parser("predict", help="Predict the protein-RNA complex")
    parser_prection.add_argument('--config', dest='config', help='Config file contains the parameters to run masifPNI.', type=str)
    parser_prection.add_argument('-l', '--list', type=str, default=None, help="Lists of PDB ids, separated by comma")
    parser_prection.add_argument('-f', '--file', type=str, default=None, help="File contains lists of PDB ids. One per separate line.")
    parser_prection.add_argument('-custom_pdb', '--custom_pdb', type=str, default=None, help="File contain the path of custom PDB files or the directory contains the target PDB files")
    parser_prection.add_argument('-n', '--n_threads', type=int, default=1, help="Threads used to prepare files.")
    parser_prection.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    from masifPNI_site.masifPNI_site_predict import masifPNI_site_predict
    parser_prection.set_defaults(func=masifPNI_site_predict)

    opt2parser = {"dataprep": parser_dataprep, "train": parser_train, "predict": parser_prection}
    if len(set(argv) - set(defaultArguments)) == 0:
        if "masifPNI-site" in argv:
            tmp = list(set(argv) & set(["dataprep", "train", "predict"]))
            if len(tmp) == 1:
                tmpParser = opt2parser[tmp[0]]
                tmpParser.print_help()
                tmpParser.exit()
            else:
                parser.print_help()
                parser.exit()


def parseArgsSearch(parser, argv):
    subparsers = parser.add_subparsers(help="Modes to run masifPNI-search", metavar='{dataprep, train, predict}')
    parser_dataprep = subparsers.add_parser("dataprep", help="Prepare the data used in next steps")
    parser_dataprep.add_argument('--gtf_or_gff', dest='gtf_or_gff', help='GTF or GFF file path.', type=str)


def parseArgsligand(parser, argv):
    subparsers = parser.add_subparsers(help="Modes to run masifPNI-ligand", metavar='{dataprep, train, predict}')
    parser_dataprep = subparsers.add_parser("dataprep", help="Prepare the data used in next steps")
    parser_dataprep.add_argument('--gtf_or_gff', dest='gtf_or_gff', help='GTF or GFF file path.', type=str)


def parseOptions1(argv):
    parser = ArgumentParser(prog='masifPNI')

    subparsers = parser.add_subparsers(help='Running modes', metavar='{download, masifPNI-site, masifPNI-search, masifPNI-ligand}')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    parser_download = subparsers.add_parser('download', help='Download target PDB files')
    parseArgsDownload(parser_download, argv)


    ### RUN MODE "masifPNI-site"
    parser_masifPNI_site = subparsers.add_parser('masifPNI-site', help='Run masifPNI-site')
    parseArgsSite(parser_masifPNI_site, argv)

    ### RUN MODE "masifPNI-search"
    parser_masifPNI_search = subparsers.add_parser('masifPNI-search', help='Run masifPNI-search')
    parseArgsSearch(parser_masifPNI_search, argv)

    ### RUN MODE "masifPNI-ligand"
    parser_masifPNI_ligand = subparsers.add_parser('masifPNI-ligand', help='Run masifPNI-ligand')
    parseArgsligand(parser_masifPNI_ligand, argv)

    if len(argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv[1:])


# def parseOptions(argv):
#     parser = ArgumentParser(prog='masifPNI')
#     subparsers = parser.add_subparsers(help='Running modes', metavar='{masifPNI-site, masifPNI-search, masifPNI-ligand}')
#     parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
#
#     ### RUN MODE "DATAPREP"
#     parser_dataprep = subparsers.add_parser('masifPNI-site', help='run mode for preprocessing nanopolish eventalign.txt before differential modification analysis')
#     optional_dataprep = parser_dataprep._action_groups.pop()
#     required_dataprep = parser_dataprep.add_argument_group('required arguments')
#     # Required arguments
#     required_dataprep.add_argument('--eventalign', dest='eventalign', help='eventalign filepath, the output from nanopolish.',required=True)
#     # #required.add_argument('--summary', dest='summary', help='eventalign summary filepath, the output from nanopolish.',required=True)
#     # required_dataprep.add_argument('--out_dir', dest='out_dir', help='output directory.',required=True)
#     # optional_dataprep.add_argument('--gtf_or_gff', dest='gtf_or_gff', help='GTF or GFF file path.',type=str)
#     # optional_dataprep.add_argument('--transcript_fasta', dest='transcript_fasta', help='transcript FASTA path.',type=str)
#     # # Optional arguments
#     # # optional_dataprep.add_argument('--skip_eventalign_indexing', dest='skip_eventalign_indexing', help='skip indexing the eventalign nanopolish output.',default=False,action='store_true')
#     # # parser.add_argument('--features', dest='features', help='Signal features to extract.',type=list,default=['norm_mean'])
#     # optional_dataprep.add_argument('--genome', dest='genome', help='to run on Genomic coordinates. Without this argument, the program will run on transcriptomic coordinates',default=False,action='store_true')
#     # optional_dataprep.add_argument('--n_processes', dest='n_processes', help='number of processes to run.',type=int, default=1)
#     # optional_dataprep.add_argument('--chunk_size', dest='chunk_size', help='number of lines from nanopolish eventalign.txt for processing.',type=int, default=1000000)
#     # optional_dataprep.add_argument('--readcount_min', dest='readcount_min', help='minimum read counts per gene.',type=int, default=1)
#     # optional_dataprep.add_argument('--readcount_max', dest='readcount_max', help='maximum read counts per gene.',type=int, default=1000)
#     # optional_dataprep.add_argument('--resume', dest='resume', help='with this argument, the program will resume from the previous run.',default=False,action='store_true') #todo
#     # parser_dataprep._action_groups.append(optional_dataprep)
#     # parser_dataprep.set_defaults(func=dataprep)
#     #
#     # ### RUN MODE "DIFFMOD"
#     # parser_diffmod = subparsers.add_parser('diffmod', help='run mode for performing differential modification analysis')
#     # optional_diffmod = parser_diffmod._action_groups.pop()
#     # required_diffmod = parser_diffmod.add_argument_group('required arguments')
#     # # Required arguments
#     # required_diffmod.add_argument('--config', dest='config', help='YAML configuraion filepath.',required=True)
#     # # Optional arguments
#     # optional_diffmod.add_argument('--n_processes', dest='n_processes', help='number of processes to run.',type=int,default=1)
#     # optional_diffmod.add_argument('--save_models', dest='save_models', help='with this argument, the program will save the model parameters for each id.',default=False,action='store_true') # todo
#     # optional_diffmod.add_argument('--resume', dest='resume', help='with this argument, the program will resume from the previous run.',default=False,action='store_true')
#     # optional_diffmod.add_argument('--ids', dest='ids', help='gene or transcript ids to model.',default=[],nargs='*')
#     # parser_diffmod._action_groups.append(optional_diffmod)
#     # parser_diffmod.set_defaults(func=diffmod)
#     #
#     # ### RUN MODE "POSTPROCESSING"
#     # parser_postprocessing = subparsers.add_parser('postprocessing', help='run mode for post-processing diffmod.table, the result table from differential modification analysis.')
#     # required_postprocessing = parser_postprocessing.add_argument_group('required arguments')
#     # # Required arguments
#     # required_postprocessing.add_argument('--diffmod_dir', dest='diffmod_dir', help='diffmod directory path, the output from xpore-diffmod.',required=True)
#     # parser_postprocessing.set_defaults(func=postprocessing)
#
#
#     if len(argv) == 1:
#         print(111)
#         parser.print_usage()
#         parser.exit()
#
#     return parser.parse_args(argv[1:])

def main(argv=sys.argv):
    options = parseOptions1(argv)
    options.func(options)

if __name__ == '__main__':
    main(sys.argv)
