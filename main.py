
import sys
import argparse
import os
import pickle

import global_var
import graph_analysis

#**** imports from our codebase *****
import src.graph_init

def run(args):
  if args.net != None:
    global_var.network= args.net
    global_var.refresh(args.net)

  analysis_obj= graph_analysis.graph_analysis_c() 
  analysis_obj.test(args)
  exit(0)

def main(argv=None):
  parser = argparse.ArgumentParser(description='GraphOpt superlayer generation')
  parser.add_argument('--net', type=str, choices=[\
    'mnist' , \
    'mnist_2', \
    'diabetes', \
    'mnist_985', \
    'uci_har_nb' , \
    'bnetflix', \
    'bbc', \
    'book', \
    'kdd', \
    'msnbc', \
    'baudio', \
    'cpu', \
    'eeg_eye', \
    'bank_note', \
    'bio_response', \
    'sbn' , \
    'adult', \
    'ad', \
    'audio', \
    'baudio', \
    'bbc_0-21', \
    'bbc', \
    'bnetflix', \
    'book', \
    'c20ng', \
    'cr52', \
    'cwebkb', \
    'dna-500', \
    'dna', \
    'elevators', \
    'exp-D15-N1000-C4', \
    'insurance', \
    'jester', \
    'kdd-6k', \
    'kdd', \
    'little_4var', \
    'mnist-antonio', \
    'mnist', \
    'msnbc_0-10', \
    'msnbc_0-115', \
    'msnbc_0-25', \
    'msnbc_0-50', \
    'msnbc_0-95', \
    'msnbc', \
    'msnbc-yitao-a', \
    'msnbc-yitao-b', \
    'msnbc-yitao-c', \
    'msnbc-yitao-d', \
    'msnbc-yitao-e', \
    'msnc_0-5', \
    'msweb', \
    'nltcs.10split', \
    'nltcs.clt', \
    'nltcs', \
    'plants', \
    'pumsb_star', \
    'simple2.1', \
    'simple2.2', \
    'simple2.3', \
    'simple2.4', \
    'simple2.5', \
    'simple2.6', \
    'tmovie', \
    'tretail', \
    'wilt', \
    'dot_for_paper', \
    'HB/bcspwr01', \
    'HB/bcspwr01', \
    'HB/bcsstm02' , \
    'HB/bcsstm05' , \
    'HB/bcsstm22' , \
    'HB/can_24'   , \
    'HB/can_62'   , \
    'HB/ibm32'    , \
    'test_net'], \
    help='Enter the name of the network to be used as an input')

  parser.add_argument('--cir_type', type=str, choices=['psdd', 'sptrsv', 'none'], default='none', help='Specify the type of circuit to be read. Default= ac')
  
  parser.add_argument('--threads', type=int, default=2, help='Specify the number of target parallel threads. Default= 2')

  parser.add_argument('--tmode', type=str, \
      choices= [\
        'try', \
        'null', \
        'full', \
        'verif_helper', \
        'ac_eval', \
        'openmp', \
        'batched_cuda', \
        'async_partition', \
        'sparse_tr_solve_statistics', \
        'sparse_tr_solve_full', \
        'psdd_full', \
        'sparse_tr_solve_low_precision' \
        ] , \
        help='mode')

  parser.add_argument('--targs', nargs= '*', help= 'Some tests may need additional arguments')

  args = parser.parse_args(argv)

  run(args)

if __name__ == "__main__":
  sys.exit(main())
