#=======================================================================
# Created by         : KU Leuven
# Filename           : graph_analysis.py
# Author             : Nimish Shah
# Created On         : 2019-10-21 16:59
# Last Modified      : 
# Update Count       : 2019-10-21 16:59
# Description        : 
#                      
#=======================================================================

import global_var
import pickle
import queue
import copy
import time
import os
import random
import math
import sys
from termcolor import colored
from collections import defaultdict
import re
import logging
import numpy as np
import networkx as nx
import cProfile
from scipy.sparse import linalg
from statistics import mean

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#**** imports from our codebase *****
import src.common_classes
import src.reporting_tools.reporting_tools
import src.graph_init
import src.useful_methods
import src.ac_eval
import src.FixedPointImplementation as FixedPointImplementation
import src.files_parser
import src.hw_struct_methods
from src.useful_methods import printcol
from src.useful_methods import printlog
import src.verif_helper
import src.super_layer_gen.partition
import src.psdd
import src.openmp.gen_code
import src.sparse_linear_algebra.main
import src.sparse_linear_algebra.matrix_names_list

#********
# Contains methods to analysze the DAG
# Main function is : full_run()
#********
class graph_analysis_c():  
  """ Main class for AC analysis.
  Important attributes of the AC are stored in the object that are initialized in constructor
  """
  def __init__(self):
    self.global_var= global_var
    
    self.name= "graph_analysis_c"
    
    # Graph that captures the AC structure. KEY= AC node_id =Line number of the operation in (binarized) AC, VAL: Node data structure
    # Created in graph_init.py
    self.graph= {}
    
    # A networkx DiGraph to represent AC structure
    # This object should be used when networkx features are to be used on the AC
    # Created in graph_init.py
    self.graph_nx= None

    self.ac_node_list= []
    self.leaf_list= [] # Lists all leaf node ids
    self.head_node= 0  # ID of the final node of the AC

  def test(self, args):
    mode= args.tmode
    print("total_nodes: ", len(self.graph), ", leaf_nodes: ", len(self.leaf_list), ", arith_nodes: ", len(self.graph) - len(self.leaf_list))
    
    if mode == "null":
      pass
      exit(0)

    if mode == 'sparse_tr_solve_statistics':
      logging.basicConfig(filename='./no_backup/run_hybrid_statistics.log', filemode='a', level=logging.INFO)
      logging.basicConfig(level=logging.INFO)
      # name_list= src.sparse_linear_algebra.matrix_names_list.matrices_path(self.global_var.SPARSE_MATRIX_MATLAB_PATH, 
          # self.global_var,
      #     mode= 'only_category', exclude_category_ls= ['HB', 'Schenk_AFE', 'ML_Graph', 'VDOL'])
      name_list= src.sparse_linear_algebra.matrix_names_list.matrices_path(self.global_var.SPARSE_MATRIX_MARKET_FACTORS_PATH,
          self.global_var,
          mode= 'with_LU_factors', exclude_category_ls= ['HB', 'Bai', 'Schenk_AFE'])
      # name= src.sparse_linear_algebra.matrix_names_list.matrix_names[0]
      print(name_list)

      file_path= './no_backup/matrix_statistics_3.csv'
      with open(file_path, 'a') as f:
        f.write('Start\n')
        f.close()
      for name in name_list:
        tr_solve_obj= src.sparse_linear_algebra.main.SparseTriangularSolve(self.global_var, name, write_files= False, verify=False, read_files= True)
        tr_solve_obj.statistics(write_files= True, file_path= file_path)
      
      exit(1)

    if mode == 'sparse_tr_solve_full':
      # sys.setrecursionlimit(1800)
      # logging.basicConfig(filename='./no_backup/run_hybrid_two_way_part_only_local.log', filemode='w', level=logging.INFO)
      logging.basicConfig(level=logging.INFO)
      
      # create graph from matrix
      # name_list= src.sparse_linear_algebra.matrix_names_list.matrices_path(self.global_var.SPARSE_MATRIX_MATLAB_PATH, 
          # self.global_var,
      #     mode= 'only_category', exclude_category_ls= ['HB', 'Schenk_AFE', 'ML_Graph', 'VDOL'])
      # name_list= src.sparse_linear_algebra.matrix_names_list.matrices_path(self.global_var.SPARSE_MATRIX_MARKET_FACTORS_PATH,
      #     self.global_var,
      #     mode= 'with_LU_factors', exclude_category_ls= ['Schenk_AFE', 'ML_Graph', 'VDOL'])
      # name= src.sparse_linear_algebra.matrix_names_list.matrix_names[0]

      name_list = [
          # 'HB/orani678',
          'HB/bcsstk08',
          'HB/bcsstk21',
          'HB/orsreg_1',
          'Bai/rdb3200l',
          'HB/sherman2',
          'HB/lshp3025',
          'Bai/dw8192',
          'Bai/dw4096',
          'Bai/cryg10000',
          'Bai/rdb5000',
        ]
      # big matrices
      name_list = [
          # 'Bai/rdb5000',
          # 'HB/bcsstk17',
          # 'Bai/dw8192',          
          # 'HB/bcsstk23',
          # 'HB/bcsstk15',
          # 'HB/bcsstk25',
          # 'HB/bcsstk18',
          # 'Bai/af23560',
          # 'HB/psmigr_1',
          'Boeing/bcsstk38'
        ]
      # name_list= ['HB/bcsstk28']
      # name_list= ['HB/bcsstk23']
      # name_list= ['HB/bcsstk16']
      # name_list= ['HB/orani678']
      # name_list= ['Bai/dw8192']
      # name_list= ['HB/494_bus']
      # name_list= ['GHS_psdef/torsion1']

      # name_list= ['HB/orsreg_1']

      # selected matrices
      # name_list= ['HB/can_24', 'HB/can_62', 'HB/west0156', 'Pothen/mesh1e1', 'Bai/tols340', 'Bai/mhdb416', 'Bai/olm1000', 'HB/saylr1', 'HB/steam1', 'MathWorks/Pd', 'Bai/mhd4800b', 'Nasa/barth4', 'MathWorks/Kaufhold', 'Bindel/ted_B_unscaled', 'Nasa/barth5', 'Norris/lung2', 'Pothen/barth5', 'Oberwolfach/gyro_m', 'GHS_psdef/jnlbrng1', 'Boeing/crystm02', 'Oberwolfach/rail_79841', 'Boeing/crystm03']
      name_list = ["Bai/tols340","HB/steam1","Bindel/ted_B_unscaled","Pothen/barth5","Oberwolfach/gyro_m","Boeing/crystm02"]
      # name_list = ["Oberwolfach/gyro_m","Boeing/crystm02"]
      # name_list = ["Bai/tols340","HB/steam1"]
      # name_list = ["Bai/tols340","HB/steam1","Bindel/ted_B_unscaled","Pothen/barth5"]
      # name_list = ["Bai/tols340"]
      # name_list = ["Pothen/barth5"]

      log_path= "./src/openmp/run_log_sparse_tr_solve_two_way_Ofast_eridani"
      # name_list = src.sparse_linear_algebra.matrix_names_list.matrices_based_on_size(log_path, 5_000, 25_000)
      name_list = src.sparse_linear_algebra.matrix_names_list.matrices_based_on_size(log_path, 25_000, 50_000)
      if "Bai/olm2000" in name_list: name_list.remove("Bai/olm2000")
      #Done_matrices: ['HB/west0479', 'HB/lns_511', 'HB/lshp_577', 'HB/sherman4', 'HB/west1505', 'HB/fs_541_3', 'HB/west2021', 'Pothen/mesh2e1', 'HB/662_bus', 'Bai/bfwa398', 'Bai/mhd3200b', 'HB/west0655', 'Bai/qh882', 'HB/gre_343', 'Bai/qh1484', 'HB/gre_512', 'MathWorks/Sieber', 'Bai/dw256A', 'HB/fs_541_4', 'HB/bcsstk19', 'HB/west0381', 'Bai/bfwb398', 'Bai/dw256B', 'Bai/bfwa782', 'HB/mahindas', 'Bai/rw496', 'HB/gre_216a', 'Bai/olm5000', 'HB/fs_541_2', 'HB/bp_200', 'FIDAP/ex1', 'Pothen/mesh3em5', 'HB/steam1', 'HB/lshp_406', 'HB/bp_1200', 'Bai/bfwb782', 'HB/bp_1600', 'HB/bcsstm25', 'HB/lund_a', 'Bai/bwm2000', 'Pothen/mesh2em5', 'HB/bp_600', 'Bai/rdb450l', 'Bai/rdb200l', 'HB/shl_400', 'HB/dwt_361', 'HB/bp_1000', 'Bai/dwa512', 'HB/bp_400', 'HB/nos6', 'HB/nnc666', 'HB/685_bus', 'HB/young3c', 'HB/lshp_265', 'HB/plat362', 'Bai/rdb450', 'Bai/dwb512', 'HB/pores_3', 'Pothen/sphere3', 'HB/jagmesh5', 'Bai/tols4000', 'Norris/lung1', 'HB/lnsp_511', 'MathWorks/Pd', 'HB/fs_541_1', 'HB/1138_bus', 'HB/bp_1400', 'HB/lund_b', 'Bai/rdb200', 'HB/plskz362', 'Pothen/mesh3e1', 'HB/bp_800', 'HB/dwt_503', 'Oberwolfach/t3dl_e', 'HB/gre_185', 'HB/bcsstk04']
      #out_of_mem_ls: []

      # Done matirces: 5000 to 25000 size
      # Done with threads 1,2,4,8,...
      # name_list= ['Pothen/mesh3e1', 'HB/plskz362', 'Bai/dwa512', 'HB/bp_1200', 'Pothen/sphere3', 'HB/bcsstm25', 'HB/gre_216a', 'HB/fs_541_4', 'HB/dwt_503', 'Bai/bfwb398', 'HB/bp_200', 'HB/west2021', 'HB/bp_800', 'Bai/dwb512', 'Bai/olm5000', 'HB/gre_185', 'Bai/bfwa398', 'Pothen/mesh3em5', 'Bai/dw256A', 'HB/lnsp_511', 'Bai/rw496', 'HB/lshp_406', 'HB/bp_600', 'Pothen/mesh2e1', 'Norris/lung1', 'HB/685_bus', 'Bai/dw256B', 'HB/bcsstk04', 'Bai/rdb200', 'HB/1138_bus', 'HB/bp_1400', 'HB/west0655', 'Oberwolfach/t3dl_e', 'HB/nos6', 'HB/fs_541_2', 'Bai/bfwb782', 'HB/plat362', 'HB/662_bus', 'HB/lshp_265', 'HB/nnc666', 'HB/bcsstk19', 'Bai/tols4000', 'Pothen/mesh2em5', 'HB/lshp_577', 'Bai/rdb200l', 'Bai/bwm2000', 'HB/shl_400', 'HB/bp_1000', 'HB/west1505', 'HB/jagmesh5', 'MathWorks/Sieber', 'MathWorks/Pd', 'HB/bp_1600', 'Bai/mhd3200b', 'HB/lns_511', 'HB/west0479', 'Bai/rdb450l', 'HB/pores_3', 'HB/sherman4', 'HB/gre_343', 'HB/mahindas', 'HB/dwt_361', 'HB/lund_a', 'Bai/bfwa782', 'HB/steam1', 'FIDAP/ex1', 'HB/fs_541_3', 'HB/gre_512', 'HB/young3c', 'HB/west0381', 'HB/bp_400', 'Bai/rdb450', 'Bai/qh882', 'Bai/qh1484', 'HB/lund_b', 'HB/fs_541_1']

      # Done matrices: 25000 to 50000
      name_list= [ 'HB/jagmesh7', 'HB/jagmesh3', 'HB/fs_760_1', 'HB/can_445', 'HB/nos5', 'Bai/cdde3', 'Oberwolfach/rail_1357', 'HB/mcfe', 'HB/fs_760_3', 'Bai/pde900', 'HB/gr_30_30', 'Bai/mhda416', 'HB/fs_760_2', 'Bai/rdb800l', 'Oberwolfach/spiral', 'HB/bcsstk07', 'FIDAP/ex2', 'HB/bcsstk06', 'HB/bcsstm07', 'HB/jagmesh8', 'Bai/cdde1', 'Bai/cdde5', 'HB/sherman1', 'HB/jagmesh4', 'Bai/cdde4', 'HB/can_838', 'HB/jagmesh1', 'HB/steam2', 'FIDAP/ex32', 'HB/plsk1919', 'HB/lshp_778', 'HB/lshp1009', 'Bai/cdde6', 'HB/jagmesh2', 'Bai/cdde2', 'HB/hor_131', 'FIDAP/ex22', 'Bai/mhd4800b', 'HB/jagmesh6', 'HB/bcsstk10', 'HB/bcsstm10', 'HB/pores_2']
      
      # matrices with high parallelism and reasonable size
      name_list= [
          "Bai/pde2961",
          "Bai/cryg2500",
          "HB/gemat12",
          "HB/blckhole",
          "HB/orani678",
          "Bai/tols4000",
          "HB/west2021",
          "MathWorks/Sieber",
          "HB/lshp2614",
          "HB/gre_1107",
          "Bai/dw2048",
          "Bai/dw1024",
          "HB/lshp2233",
          "HB/bp_1200",
          "HB/bp_1400",
          "HB/gemat11",
          "HB/jpwh_991",
          "HB/lshp1882",
          "HB/orsirr_2",
          "HB/bp_1600",
          "HB/watt_1",
          "HB/plat1919",
          "HB/nos7",
          "HB/bp_800",
          "Bai/rdb1250",
          "HB/sherman5",
          "HB/lshp1561",
          "Bai/rdb1250l",
          "HB/bp_400",
          "HB/jagmesh4",
          "Bai/mhd3200a",
          "HB/bp_600",
          "HB/bp_1000",
          "HB/bcsstk09",
          "HB/bp_200"
        ]

      # JSSC list
      name_list= [\
        'Bai/tols4000',
        'HB/bp_200',
        'HB/west2021',
        'Bai/qh1484',
        'MathWorks/Sieber',
        'HB/gemat12',
        'Bai/dw2048',
        'HB/orani678',
        'Bai/pde2961',
        'HB/blckhole',
      ]
      # name_list= ['HB/494_bus']
      # name_list= ['Pothen/mesh3e1']

      PLOT_CHARTS= False
      # PLOT_CHARTS= True
      if PLOT_CHARTS:
        plot_d= {}

      # Ofast
      # name_list= ['MathWorks/Kaufhold', 'Boeing/crystm03', 'Boeing/crystm02', 'Oberwolfach/gyro_m', 'Nasa/barth5', 'HB/bcsstm25', 'MathWorks/Pd', 'Boeing/bcsstm39', 'HB/gemat11', 'HB/gemat12', 'Bates/Chem97ZtZ', 'GHS_psdef/gridgena', 'Bindel/ted_B', 'Bindel/ted_B_unscaled', 'Oberwolfach/rail_20209', 'Pothen/barth5']
      # name_list= ['MathWorks/Kaufhold', 'MathWorks/Pd', 'HB/gemat11', 'HB/gemat12']

      # name= args.targs[0]
      # name_list= [name]

      # n_threads_ls= [6,8,10,12]
      # n_threads_ls= [2,4,8,16,32,64]
      n_threads_ls= [64]
      # n_threads_ls= [int(args.targs[1])]

      done_list= []
      # workloads that do not fit in PRU memory
      out_of_mem_ls= []

      print(len(name_list))
      print(name_list)

      for name in name_list:
        logger.info(f'matrix name: {name}')

        # partition the graph
        graph_mode = 'FINE'
        # graph_mode = 'COARSE'
        partition_mode= 'TWO_WAY_PARTITION'
        # partition_mode= 'LAYER_WISE'
        # partition_mode= 'HEURISTIC'
        # sub_partition_mode= 'ALAP'
        sub_partition_mode= 'TWO_WAY_LIMIT_LAYERS'
        # sub_partition_mode= 'TWO_WAY_FULL'
        # sub_partition_mode= None
        # run_mode= 'FULL'
        run_mode= 'RESUME'
        # n_threads_ls= [2, 4, 8, 16, 32]
        # n_threads_ls= [32, 16, 8, 12, 20, 28]
        # n_threads_ls= [1,2, 4,8,16]
        # n_threads_ls= [8, 4, 16, 2]
        # n_threads_ls= [8, 10]
        target_device= 'PRU'
        target_app= 'SPARSE_TR_SOLVE'

        # HW details
        # GLOBAL_MEM_DEPTH= 1024
        # LOCAL_MEM_DEPTH= 512
        GLOBAL_MEM_DEPTH= 65536
        LOCAL_MEM_DEPTH= 65536
        STREAM_LD_BANK_DEPTH= 65536 # words
        STREAM_ST_BANK_DEPTH= 65536 # words
        STREAM_INSTR_BANK_DEPTH= 65536 # words

        # COMBINE_LAYERS_THRESHOLD = 2000 is used for GRAPHOPT experiments
        COMBINE_LAYERS_THRESHOLD= 2000

        # default values to False
        write_files= False
        GEN_PARTITIONS= False
        READ_PARTITIONS= False
        COMBINE_SMALL_LAYERS= False
        GEN_OPENMP_CODE= False

        # partition generation
        # write_files= True
        # GEN_PARTITIONS= True

        # openmp generation
        READ_PARTITIONS= True
        # COMBINE_SMALL_LAYERS= True
        # GEN_OPENMP_CODE= True

        # if partition_mode == 'TWO_WAY_PARTITION':
        #   self.graph= tr_solve_obj.L_coarse_graph_obj.graph
        #   self.graph_nx= tr_solve_obj.L_coarse_graph_obj.graph_nx
        # else:
        #   self.graph= tr_solve_obj.L_graph_obj.graph
        #   self.graph_nx= tr_solve_obj.L_graph_obj.graph_nx
        
        # self.graph= tr_solve_obj.L_coarse_graph_obj.graph
        # self.graph_nx= tr_solve_obj.L_coarse_graph_obj.graph_nx
        
        tr_solve_obj= src.sparse_linear_algebra.main.SparseTriangularSolve(self.global_var, name, write_files= False, verify=False, read_files= True, graph_mode= graph_mode)
        if graph_mode == 'FINE':
          final_output_nodes= set(tr_solve_obj.L_map_x_to_node.values())
          
          # initialize values of b
          tr_solve_obj.instantiate_b('L')
        elif graph_mode == 'COARSE':
          pass
        else:
          assert 0

        for n_threads in n_threads_ls:
          config_obj = src.super_layer_gen.partition.CompileConfig(name= name.replace('/','_'), N_PE= n_threads, \
              GLOBAL_MEM_DEPTH=GLOBAL_MEM_DEPTH, LOCAL_MEM_DEPTH=LOCAL_MEM_DEPTH, \
              STREAM_LD_BANK_DEPTH= STREAM_LD_BANK_DEPTH, STREAM_ST_BANK_DEPTH= STREAM_ST_BANK_DEPTH, STREAM_INSTR_BANK_DEPTH= STREAM_INSTR_BANK_DEPTH,
              graph_mode= graph_mode,
              partition_mode= partition_mode, sub_partition_mode=sub_partition_mode, run_mode= run_mode, \
              target_app = target_app, target_device= target_device, \
              write_files= write_files, global_var= self.global_var)

          if config_obj.graph_mode == config_obj.graph_mode_enum.FINE:
            self.graph= tr_solve_obj.L_graph_obj.graph
            self.graph_nx= tr_solve_obj.L_graph_obj.graph_nx
          elif config_obj.graph_mode == config_obj.graph_mode_enum.COARSE:
            self.graph= tr_solve_obj.L_coarse_graph_obj.graph
            self.graph_nx= tr_solve_obj.L_coarse_graph_obj.graph_nx
          else:
            assert 0

          if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
            node_w= defaultdict(lambda:1)
          elif config_obj.graph_mode== config_obj.graph_mode_enum.COARSE:
            node_w= {n: len(list(self.graph_nx.predecessors(n))) + 1 for n in self.graph_nx.nodes()}
          else:
            assert 0

          if GEN_PARTITIONS:
            list_of_partitions, status_dict= self.async_partition(name, self.graph, self.graph_nx, node_w, config_obj)
          
          elif READ_PARTITIONS:
            path= config_obj.get_partitions_file_name()
            logger.info(f"Resuming by reading partitions from {path}")
            with open(path, 'rb') as fp:
              list_of_partitions, status_dict = pickle.load(fp)
              logger.info(f"name: {name}, threads: {n_threads}, number of synchronization barriers: {len(list_of_partitions[0])}")

              self.count_edges(self.graph_nx, list_of_partitions, skip_leafs= True)
          
          if COMBINE_SMALL_LAYERS:
            list_of_partitions_combined= src.super_layer_gen.partition.combine_small_layers(self.graph_nx, list_of_partitions, COMBINE_LAYERS_THRESHOLD, node_w, config_obj)
            if config_obj.graph_mode == config_obj.graph_mode_enum.FINE:
              list_of_partitions_coarse, semi_coarse_g, map_n_to_semi_coarse, map_r_to_nodes_info, map_n_to_r, map_semi_coarse_to_tup = \
                tr_solve_obj.coarsen_partitions(self.graph, self.graph_nx, list_of_partitions_combined, tr_solve_obj.L_map_r_to_nodes_info)

          if GEN_OPENMP_CODE:
            if config_obj.graph_mode == config_obj.graph_mode_enum.FINE:
              head_node= tr_solve_obj.L_map_x_to_node[tr_solve_obj.ncols - 1]
              golden_val= src.ac_eval.ac_eval_non_recurse(self.graph, self.graph_nx, head_node)
              if not COMBINE_SMALL_LAYERS:
                src.openmp.gen_code.par_for_sparse_tr_solve(self.graph, self.graph_nx, list_of_partitions, config_obj, head_node, golden_val)
              else:
                src.openmp.gen_code.par_for_sparse_tr_solve_semi_coarse(self.graph, self.graph_nx, list_of_partitions_coarse, semi_coarse_g,
                    map_semi_coarse_to_tup, map_n_to_r, map_n_to_semi_coarse, map_r_to_nodes_info,
                    config_obj, head_node, golden_val)
            elif config_obj.graph_mode == config_obj.graph_mode_enum.COARSE:
              matrix= tr_solve_obj.L
              b= np.array([1.0 for _ in range(len(self.graph_nx))], dtype= 'double')
              x_golden= linalg.spsolve_triangular(matrix, b, lower= True)
              head_node= len(b) - 1
              golden_val = x_golden[-1]
              src.openmp.gen_code.par_for_sparse_tr_solve_full_coarse(self.graph, self.graph_nx, b, list_of_partitions_combined, matrix, config_obj, head_node, golden_val)
            else:
              assert 0

          if PLOT_CHARTS:
            plot_d[(name,n_threads)] = (list_of_partitions, node_w, config_obj, self.graph_nx, self.graph)

          done_list.append(name)
          print(f"Done_matrices: {done_list}")
          print(f"out_of_mem_ls: {out_of_mem_ls}")

      if PLOT_CHARTS:
        # src.reporting_tools.reporting_tools.plot_superlayers_ops_vs_layers(name_list, n_threads_ls, plot_d)
        src.reporting_tools.reporting_tools.plot_workload_balancing_info(name_list, n_threads_ls, plot_d)

      print(f"Done_matrices: {done_list}")
      print(f"out_of_mem_ls: {out_of_mem_ls}")

      exit(1)

    if mode == 'async_partition':
      logging.basicConfig(level=logging.INFO)
      partition_mode= args.targs[0]
      write_files= bool(int(args.targs[1]))
#      n_threads_ls= [1,2,4,8,16,32,64,128,256,512,1024]
#      n_threads_ls= [1024, 2048]
      n_threads_ls= [64]
      sub_partition_mode= None
      run_mode= None

      node_w= defaultdict(lambda:1)

      for n_threads in n_threads_ls:
        config_obj = src.super_layer_gen.partition.CompileConfig(name= self.net, N_PE= n_threads, partition_mode= partition_mode, sub_partition_mode=sub_partition_mode, run_mode= run_mode, write_files= write_files, global_var= global_var)

        self.async_partition(name, self.graph, self.graph_nx, node_w, config_obj)

      exit(0)

    if mode == 'psdd_full':
      # name_list= ['bnetflix']
      name_list = [\
        # 'ad', \
        # 'baudio', \
        # 'bbc', \
        # 'bnetflix', \
        # 'book', \
        # 'c20ng', \
        # 'cr52', \
        # 'cwebkb', \
        # 'jester', \
        # 'kdd', \
        # 'mnist', \
        'msnbc', \
        'msweb', \
        'nltcs', \
        'pumsb_star', \
        'tretail', \
      ]

      # JSSC list
      name_list= [
        'mnist',
        'nltcs',
        'msnbc',
        'bnetflix',
        'ad',
        'bbc',
        'c20ng',
        'kdd',
        'baudio',
        'pumsb_star',
      ]

      # n_threads_ls= [2,4,8,16,32,64]
      n_threads_ls= [64]
      for name in name_list:
        logger.info(f'matrix name: {name}')

        # partition the graph
        graph_mode = 'FINE'
        partition_mode= 'TWO_WAY_PARTITION'
        # partition_mode= 'LAYER_WISE'
        # partition_mode= 'HEURISTIC'
        # sub_partition_mode= 'ALAP'
        sub_partition_mode= 'TWO_WAY_LIMIT_LAYERS'
        # sub_partition_mode= 'TWO_WAY_FULL'
        # sub_partition_mode= None
        # run_mode= 'FULL'
        run_mode= 'RESUME'
        target_device= 'PRU'
        # target_device= 'CPU'
        target_app= 'SPN'

        # HW details
        # GLOBAL_MEM_DEPTH= 1024
        # LOCAL_MEM_DEPTH= 512
        GLOBAL_MEM_DEPTH= 65536
        LOCAL_MEM_DEPTH= 65536
        STREAM_LD_BANK_DEPTH= 65536 # words
        STREAM_ST_BANK_DEPTH= 65536 # words
        STREAM_INSTR_BANK_DEPTH= 65536 # words

        COMBINE_LAYERS_THRESHOLD= 2000

        # default values to False
        write_files= False
        GEN_PARTITIONS= False
        READ_PARTITIONS= False
        COMBINE_SMALL_LAYERS= False
        GEN_OPENMP_CODE= False

        # partition generation
        # write_files= True
        # GEN_PARTITIONS= True

        # openmp generation
        READ_PARTITIONS= True
        # COMBINE_SMALL_LAYERS= True
        # GEN_OPENMP_CODE= True

        path= global_var.PSDD_PATH_PREFIX + name + '.psdd'
        self.graph, self.graph_nx, self.head_node, self.leaf_list, _ = src.psdd.main(path)
        logger.info(f"name, critical path length: {name, nx.algorithms.dag.dag_longest_path_length(self.graph_nx)}")

        # partitioning assumes that leaf are all computed
        for n in self.leaf_list:
          self.graph[n].computed= True

        for n_threads in n_threads_ls:
          config_obj = src.super_layer_gen.partition.CompileConfig(name= name.replace('/','_'), N_PE= n_threads, \
              GLOBAL_MEM_DEPTH=GLOBAL_MEM_DEPTH, LOCAL_MEM_DEPTH=LOCAL_MEM_DEPTH, \
              STREAM_LD_BANK_DEPTH= STREAM_LD_BANK_DEPTH, STREAM_ST_BANK_DEPTH= STREAM_ST_BANK_DEPTH, STREAM_INSTR_BANK_DEPTH= STREAM_INSTR_BANK_DEPTH,
              graph_mode= graph_mode,
              partition_mode= partition_mode, sub_partition_mode=sub_partition_mode, run_mode= run_mode, \
              target_app = target_app, target_device= target_device, \
              write_files= write_files, global_var= self.global_var)

          if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
            node_w= defaultdict(lambda:1)
          elif config_obj.graph_mode== config_obj.graph_mode_enum.COARSE:
            assert 0
          else:
            assert 0

          if GEN_PARTITIONS:
            list_of_partitions, status_dict= self.async_partition(name, self.graph, self.graph_nx, node_w, config_obj)
          elif READ_PARTITIONS:
            path= config_obj.get_partitions_file_name()
            logger.info(f"Resuming by reading partitions from {path}")
            with open(path, 'rb') as fp:
              list_of_partitions, status_dict = pickle.load(fp)
              logger.info(f"name: {name}, threads: {n_threads}, number of synchronization barriers: {len(list_of_partitions[0])}")

              self.count_edges(self.graph_nx, list_of_partitions, skip_leafs= True)
          
          if COMBINE_SMALL_LAYERS:
            list_of_partitions= src.super_layer_gen.partition.combine_small_layers(self.graph_nx, list_of_partitions, COMBINE_LAYERS_THRESHOLD, node_w, config_obj)

          if GEN_OPENMP_CODE:
            dataset= src.files_parser.read_dataset(global_var, name, 'test')
            src.psdd.instanciate_literals(self.graph, dataset[0])
            golden_val= src.ac_eval.ac_eval_non_recurse(self.graph, self.graph_nx, self.head_node)
            print(golden_val)

            outpath= config_obj.get_openmp_file_name()
            batch_sz= 1
            src.openmp.gen_code.par_for(outpath,self.graph, self.graph_nx, list_of_partitions, golden_val, batch_sz)

      exit(1)
          
    if mode == 'openmp':
      # Compiler for new arch
      store_prefix= '/esat/puck1/users/nshah/cpu_openmp/'
      ld_prefix= '/esat/puck1/users/nshah/partitions/'
      dataset= src.files_parser.read_dataset(global_var, self.net, 'test')
      src.psdd.instanciate_literals(self.graph, dataset[0])
      golden_val= src.ac_eval.ac_eval_non_recurse(self.graph, self.graph_nx, self.head_node)
      print(golden_val)

      # batch_sz_ls= [1,2,4,8,16,32,64,128,256,512]
      batch_sz_ls= [1]
      # n_threads_ls= [1,2,4,8,16,32,64,128,256,512,1024]
      n_threads_ls= [2]
#      batch_sz_ls= [1]
#      n_threads_ls= [1]
      for n_threads in n_threads_ls:
        for batch_sz in batch_sz_ls:
          #batch_sz = 2**j
          #n_threads= 2**i
          in_path= ld_prefix + self.net + '_' + str(n_threads) + '.p'
          with open(in_path, 'rb+') as fp:
            list_of_partitions= pickle.load(fp)
          outpath= store_prefix + '{}_{}threads_{}batch.c'.format(self.net, n_threads, batch_sz)
          src.openmp.gen_code.par_for(outpath,self.graph, self.graph_nx, list_of_partitions, golden_val, batch_sz)
      exit(0)

    if mode == 'batched_cuda':
      store_prefix= '/esat/puck1/users/nshah/gpu_cuda/'
      ld_prefix= '/esat/puck1/users/nshah/partitions/'
      dataset= src.files_parser.read_dataset(global_var, self.net, 'test')
      src.psdd.instanciate_literals(self.graph, dataset[0])
      golden_val= src.ac_eval.ac_eval_non_recurse(self.graph, self.graph_nx, self.head_node)
      print(golden_val)

      for i in range(10):
        n_threads= 2**i
      #hw_details= src.super_layer_gen.partition.hw_details_class(N_PE= n_threads)
        in_path= ld_prefix + self.net + '_' + str(n_threads) + '.p'
        with open(in_path, 'rb+') as fp:
          list_of_partitions= pickle.load(fp)
        outpath= store_prefix + '{}_{}threads.cu'.format(self.net, n_threads)
        # src.cuda.gen_code.main(outpath, self.graph, self.graph_nx, list_of_partitions, golden_val)
      exit(0)

    if mode=="ac_eval":
      src.verif_helper.init_leaf_val(self.graph, mode='all_1s')
      return_val= src.ac_eval.ac_eval(self.graph, self.head_node, elimop= 'PROD_BECOMES_SUM')
      
      print('AC eval:', return_val)
      exit(0)

  def count_edges(self, graph_nx, list_of_partitions, skip_leafs= False):
    map_n_to_pe= {}

    for pe, partitions in enumerate(list_of_partitions):
      for partition in partitions:
        for n in partition:
          assert n not in map_n_to_pe
          map_n_to_pe[n]= pe

    global_edges= 0
    local_edges= 0
    for src, dst in graph_nx.edges():
      if src in map_n_to_pe: # not a leaf
        if map_n_to_pe[src] == map_n_to_pe[dst]:
          local_edges += 1
        else:
          global_edges += 1
      else: # leaf
        if not skip_leafs:
          local_edges += 1

    logger.info(f"total_edges: {graph_nx.number_of_edges()}, global_edges: {global_edges}, local_edges: {local_edges}")


  def async_partition(self, name, graph, graph_nx, node_w, config_obj):
    if not config_obj.write_files:
      logger.warning("Not writing partitions to files")
    else:
      assert config_obj.write_files == True
      logger.warning("Writing partitions to files")

    if config_obj.hw_details.N_PE > 1:
      list_of_partitions , status_dict = src.super_layer_gen.partition.global_barriers(name, graph, graph_nx, node_w, config_obj)
      edge_crossing= 0
      for node, obj in graph.items():
        if not obj.is_leaf():
          pe_set= set([status_dict[parent].pe_id for parent in obj.parent_key_list])
          pe_set.add(status_dict[node].pe_id)
          edge_crossing += len(pe_set) - 1

      logger.info(f"edge_crossing : {edge_crossing}")
    else:
      if config_obj.graph_mode == config_obj.graph_mode_enum.FINE:
        nodes_to_map= src.useful_methods.get_non_leaves(graph_nx)
      elif config_obj.graph_mode == config_obj.graph_mode_enum.COARSE:
        nodes_to_map= list(graph_nx.nodes())
      else:
        assert 0
      list_of_partitions= [[set(nodes_to_map)]]
      status_dict= set()

    if config_obj.write_files:
      with open(config_obj.get_partitions_file_name(), 'wb+') as fp:
        logger.warning(f"Writing partitions to files at {config_obj.get_partitions_file_name()}")
        pickle.dump((list_of_partitions, status_dict), fp)


    return list_of_partitions, status_dict

  def process_data(self, verbose=0):
    #self.process_depth_data()
    self.avg_node_reuse = src.analysis_node_reuse._node_reuse_profile(self.graph, self.head_node)
    if (verbose):
      print("Avg. node reuse: ", self.avg_node_reuse)
  
  

  def full_run(self, args):
    verbose= args.v
    src.graph_init.construct_graph(self, args, global_var)

    src.analysis_depth_comm_child._depth_profile_for_common_child(self.graph, self.head_node, self.depth_list, 16)
    
    if (verbose):
      print("Num of Leaves: " , self.n_leaf)
    
    self.process_data()
    #self.avg_node_reuse = src.analysis_node_reuse._node_reuse_profile(self.graph, self.head_node)
    print("Avg. node reuse: ", self.avg_node_reuse)
#    self.test(args)
    
  def _add_node(self, graph, node_key, child_key_list, op_type):
    # Add the node and update it's child list
    node_obj= src.common_classes.node(node_key)
    for item in child_key_list:
      node_obj.add_child(item)
    node_obj.set_operation(op_type)
    
    graph[node_obj.get_self_key()]= node_obj
    
    # Update parent list of the child
    for item in child_key_list:
      graph[item].add_parent(node_key)
  
  def create_file_name(self, hw_depth, max_depth, min_depth, out_mode, fitness_wt_in, fitness_wt_out, fitness_wt_distance):
    hw_details= src.hw_struct_methods.hw_nx_graph_structs(hw_depth, max_depth, min_depth)
    list_of_depth_lists= hw_details.list_of_depth_lists

    hw_details_str= out_mode + '_' + str(fitness_wt_in) + str(fitness_wt_out) + str(fitness_wt_distance) + '_'.join([''.join(str(y) for y in x) for x in list_of_depth_lists])

    return hw_details_str  


  def create_file_name_full(self, args):
    assert len(args.targs) == 5, len(args.targs) 
    hw_depth= int(args.targs[0])
    self.hw_depth= hw_depth
    max_depth= int(args.targs[1])
    min_depth= int(args.targs[2])
    fitness_wt_distance= float(args.targs[3])
    out_mode= args.targs[4] 
    
    print("hw_depth", hw_depth)
    print("max_depth", max_depth)
    print("min_depth", min_depth)
    print("out_mode", out_mode)

    assert out_mode in ['ALL','VECT' , 'TOP_1', 'TOP_2']

    fitness_wt_in= 0.0
    fitness_wt_out= 0.0

    print('fitness_wt_distance: ', fitness_wt_distance)

    return self.create_file_name(hw_depth, max_depth, min_depth, out_mode, fitness_wt_in, fitness_wt_out, fitness_wt_distance)


class schedule_param():
  def __init__(self, SCHEDULING_SEARCH_WINDOW, RANDOM_BANK_ALLOCATE):
    self.SCHEDULING_SEARCH_WINDOW=SCHEDULING_SEARCH_WINDOW
    self.RANDOM_BANK_ALLOCATE= RANDOM_BANK_ALLOCATE

  def str_fname(self):
    fname= ""
    fname += str(self.SCHEDULING_SEARCH_WINDOW)
    fname += str(self.RANDOM_BANK_ALLOCATE)
    return fname

class decompose_param():
  def __init__(self, max_depth, min_depth, fitness_wt_distance, fitness_wt_in, fitness_wt_out):
    self.max_depth= max_depth
    self.min_depth= min_depth
    self.fitness_wt_distance= fitness_wt_distance
    self.fitness_wt_in= fitness_wt_in
    self.fitness_wt_out= fitness_wt_out

  def str_fname(self):
    fname= ""
    fname += str(self.max_depth)
    fname += str(self.min_depth)
    fname += str(self.fitness_wt_distance)
    fname += str(self.fitness_wt_in)
    fname += str(self.fitness_wt_out)
    return fname

def str_with_hw_details(hw_details):
  fname= ""
  fname += "D" + str(hw_details.tree_depth)
  fname += "_N" + str(hw_details.n_tree)
  fname += "_" + str(hw_details.n_banks)
  fname += "_" + str(hw_details.reg_bank_depth)
  fname += "_" + str(hw_details.mem_bank_depth)
  fname += "_" + str(hw_details.mem_addr_bits)
  fname += "_" + str(hw_details.n_bits)
  fname += "_" + str(hw_details.n_pipe_stages)

  return fname

def create_schedule_name(hw_details, schedule_param_obj, decompose_param_obj):
  """
    Creates file name for schedule (or instr_ls) using hw_details_class obj and schedule_param obj
  """

  fname= str_with_hw_details(hw_details)
  fname += decompose_param_obj.str_fname()
  fname += schedule_param_obj.str_fname()

  return fname

def create_decompose_file_name(hw_details, decompose_param_obj):
  """
    Creates file name for output of decompose process using hw_details_class obj and schedule_param obj
  """
 
  fname = ""
  fname += str(hw_details.tree_depth)
  fname += str(hw_details.n_tree)
  fname += decompose_param_obj.str_fname()

  return fname

