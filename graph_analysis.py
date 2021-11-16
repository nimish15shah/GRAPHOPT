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
import src.hw_struct_methods
import src.reporting_tools.reporting_tools
import src.scheduling
import src.reporting_tools.write_binary
import src.graph_init
import src.analysis_depth_comm_child
import src.analysis_node_reuse
import src.useful_methods
import src.ac_eval
import src.evidence_analysis
import src.decompose
import src.FixedPointImplementation as FixedPointImplementation
import src.files_parser
import src.energy
import src.bank_allocate
import src.scheduling_gather
import src.partition
import src.hw_struct_methods
import src.logistic_circuits
import src.explore
from src.useful_methods import printcol
from src.useful_methods import printlog
import src.gui
import src.verif_helper
import src.new_arch.partition
import src.psdd
import src.milp_optimization
import src.optimization.pipeline_scheduling_or
import src.optimization.write_to_file
import src.optimization.pe_bank_allocate
import src.openmp.gen_code
# import src.cuda.gen_code
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
    self.net= global_var.network
    self.print_attr()
    
    # BN Graph
    self.BN= {} # key= 'BN_var_name', val= list of possible state names 
    self.BN_evidence={} # Contains evidence on the BN in the format { Key='BN_var_name' : Val: 'BN_state_name'}

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

    # Objects to capture the LMAP mappings
    # Contains the mapping (node_id of leaf (int) -> literal_id (int))
    # Contains the mapping (node_id of indicator leaf -> Variable_name)
    # Contains the mapping (node_id of weight leaf -> Value of weight)
    self.map_Literal_To_LeafKey= {} # Inverse mapping where you want to key with LeafKey is not required as the Literal_val is stored in node's attribute

    #----- HW description and mapping details--------
    self.hw_struct= {} # Graph that captures the HW structure. KEY= Node_id, VAL: Node data structure
    self.hw_depth= 0   # HW depth
    self.hw_dict= {}   # Dictionary that capture crude details about HW struct, eg. how many hw units are there at a certain depth. KEY: Depth value, VAL: Node_id at that depth
    self.hw_misc_dict={} # Contains miscellenious details about each depth. KEY: Depth Value, VAL: Dictionary keyed with attribute strings
    
    #Build block details
    # Key= Build_block key
    # Val= Obj of build_blk class
    self.BB_graph= None
    
    # Bulding blok graph in networkx fromat
    # A MultiDiGraph with multiple edges between BBs. This is required as there could be multiple outputs in a BB
    self.BB_graph_nx= None

    #-----Analytical data-----
    #--------------------------
    self.depth_list= [] 
    self.avg_node_reuse= 0 
    
    # Variables to capture args passed from command line
    self.use_ac_file= 0
  
  def command_list(self):
    """ Contains list of important command-line commands and also class function sequences
    """
    
    # To visualize the AC using zgrviewer
    # In test(): 
    #src.evidence_analysis.populate_BN_list(self.graph, self.BN, self.head_node, dict.fromkeys(self.ac_node_list, False) ) 
    #src.reporting_tools.reporting_tools.create_dot_file(self, global_var.GRAPHVIZ_BN_VAR_FILE, 'BN_var_details', self.head_node)
    # From command line
    # python -test alarm_2 | zgrviewer

  def test(self, args):
    mode= args.tmode
    print("total_nodes: ", len(self.graph), ", leaf_nodes: ", len(self.leaf_list), ", arith_nodes: ", len(self.graph) - len(self.leaf_list))
    
    if mode == 'try':
      EXP_LEN=4
      MNT_LEN= 10
      y= FixedPointImplementation.flt_to_custom_flt(0.75, EXP_LEN, MNT_LEN, False, True)
      print(y)
      y= FixedPointImplementation.custom_flt_to_flt(y, EXP_LEN, MNT_LEN, False, True)
      print(y)
      y= (2**-21)*-0.25
      y= 1.21
      y= FixedPointImplementation.flt_to_custom_flt(y, EXP_LEN, MNT_LEN, False, True)
      print(y)
      y= FixedPointImplementation.custom_flt_to_flt(y, EXP_LEN, MNT_LEN, False, True)
      print(y)
      y= -1.22
      y= FixedPointImplementation.flt_to_custom_flt(y, EXP_LEN, MNT_LEN, False, True)
      y2= 0.74
      y2= FixedPointImplementation.flt_to_custom_flt(y2, EXP_LEN, MNT_LEN, False, True)
      y_o= FixedPointImplementation.flt_add_signed(y, y2, EXP_LEN, MNT_LEN, False, False)
      y_o= FixedPointImplementation.custom_flt_to_flt(y_o, EXP_LEN, MNT_LEN, False, True)
      print(y_o)
      y_o= FixedPointImplementation.flt_add(y, y2, EXP_LEN, MNT_LEN, False, False)
      y_o= FixedPointImplementation.custom_flt_to_flt(y_o, EXP_LEN, MNT_LEN, False, True)
      print(y_o)
      exit(1)

      G= nx.Graph()
      G.add_nodes_from([0,1,2,3])  
      G.add_edge(0,1)
      G.add_edge(1,2)
      path_len_dict= nx.algorithms.shortest_paths.shortest_path_length(G,0)
      print(path_len_dict)
      
      exit(1)

      src.sparse_linear_algebra.main.main(self.global_var)
      exit(1)

      prefix= '/esat/puck1/users/nshah/two_way_partition/'
      fp= open(prefix + self.net +'_two_way_partition_64.p', 'rb')
      list_of_partitions, status_dict = pickle.load(fp)
      count= [0 for _ in range(len(list_of_partitions[0]))]
      for pe, partition in enumerate(list_of_partitions):
        for l, part in enumerate(partition):
          count[l] += len(part)
          if l==0:
            print(len(part), end= " ")
      print(count)
      print(len(self.graph))
      
      edge_crossing= 0
      for node, obj in self.graph.items():
        if not obj.is_leaf():
          pe_set= set([status_dict[parent].pe_id for parent in obj.parent_key_list])
          pe_set.add(status_dict[node].pe_id)
          edge_crossing += len(pe_set) - 1

      print(f"edge_crossing : {edge_crossing}")
      exit(1)

      src.reporting_tools.reporting_tools.write_psdd_dot(self.graph)
      exit(1)

      # code for CUDA  
      fname= './LOG/'+ self.net + '_gpu_cuda_4.cu'
      src.reporting_tools.reporting_tools.write_c_for_gpu_cuda_4(fname, self.graph, self.graph_nx, self.head_node)
      exit(1)

      N_THREADS= 512
      SHARED=False
      print(N_THREADS)
      if SHARED:
        fname= './LOG/'+ self.net + '_thread' + str(N_THREADS) + '_no_bankcnf_gpu_cuda_3.cu'
      else:
        fname= './LOG/'+ self.net + '_thread' + str(N_THREADS) + '_no_bankcnf_global_gpu_cuda_3.cu'

      src.reporting_tools.reporting_tools.write_c_for_gpu_cuda_3(fname, self.graph, self.graph_nx, self.head_node, N_THREADS= N_THREADS, SHARED= SHARED)
      exit(1)

      N_THREADS= 96
      print(N_THREADS)
      src.reporting_tools.reporting_tools.write_c_for_gpu_cuda_2('./LOG/'+ self.net + '_thread' + str(N_THREADS) + '_gpu_cuda_2.cu', self.graph, self.graph_nx, self.head_node, N_THREADS= N_THREADS)
      exit(1)
      src.reporting_tools.reporting_tools.write_c_for_gpu_cuda_1('./LOG/'+ self.net + '_gpu_cuda_1.c', self.graph)
      exit(1)
      hw_details_str= self.create_file_name_full(args)
      print(hw_details_str)
#      self.BB_graph, self.graph, self.BB_graph_nx, self.graph_nx = src.files_parser.read_BB_graph_and_main_graph(self.global_var, self.hw_depth, hw_details_str)

      #src.explore.reuse_factor(self.graph, self.leaf_list)
      #src.explore.nice_bb(self.graph, self.BB_graph)
      
      instr_ls_obj= src.files_parser.read_schedule(self.global_var, self.hw_depth, hw_details_str)
      gui= src.gui.GUI(instr_ls_obj)

      exit(1)


      # Scheduling based on a DFS, without worrying about pipelining hazards
      head_bb= src.scheduling.create_bb_dependecny_graph(self.graph, self.BB_graph) 
      src.scheduling.compute_reverse_lvl(self.BB_graph, head_bb)
#      src.scheduling.assign_sch_lvl(self.BB_graph, head_bb, src.common_classes.Mutable_level(), {}) 
      src.scheduling_gather.assign_sch_reverse_lvl_wise(self.BB_graph, head_bb, {})

      src.partition.greedy_partitioning(self.BB_graph, self.BB_graph_nx, 2)
      
#      printcol("Paritioning as undirected graph", 'red')
#      src.partition.partition(self.graph_nx, self.BB_graph_nx, self.BB_graph)
#      src.files_parser.modify_logistic_circuit()
#      src.reporting_tools.reporting_tools.write_c_for_asip('./LOG/profile_'+ str(self.net) +'.c', self.graph)
      
      exit(0) 
      
    if mode == "null":
      pass
      exit(0)

    if mode == "sparse_tr_solve_low_precision":

      filter_names = None
      filter_names = set([\
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
          ])

      # plots
      log_path= './no_backup/log/sparse_tr_solve_low_precision_signed_nz_norm_recursive.log'
      # log_path= './no_backup/log/sparse_tr_solve_low_precision.log'
      exit(1)

      name_list = [
          'HB/orani678',
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

      log_path= "./src/openmp/run_log_sparse_tr_solve_two_way_Ofast_eridani"
      name_list = src.sparse_linear_algebra.matrix_names_list.matrices_based_on_size(log_path, 20_000, 100_000)
      # name_list.remove("Bai/olm2000")
      # name_list= ['HB/494_bus']
      # name_list= ['HB/orani678']
      name_list = [\
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
      
      graph_mode = 'FINE'
      
      # arith_type, int_bits, exp_bits, frac_bits, total_bits
      precision_list= [\
        # ('POSIT', None, 6, None, 32),
        # ('POSIT', None, 4, None, 16),
        # ('POSIT', None, 2, None, 8),
        # ('POSIT', None, 2, None, 32),
        # ('POSIT', None, 1, None, 16),
        # ('POSIT', None, 0, None, 8),
        ('FLOAT', None, 8, 23, 32),
        ('FLOAT', None, 5, 10, 16),
        ('FLOAT', None, 4, 3, 8)
      ]
      # arith_type= 'FLOAT'
      # int_bits= None
      # exp_bits = 11
      # frac_bits= 52
      # total_bits= 16

      log_path= './no_backup/log/sparse_tr_solve_low_precision_signed_nz_norm_recursive.log'
      fp= open(log_path, 'a+')
      print("Start", file= fp, flush=True)

      logger.setLevel(logging.WARNING)
      n_iter= 50

      for name in name_list:
        tr_solve_obj= src.sparse_linear_algebra.main.SparseTriangularSolve(self.global_var, name, write_files= False, verify=False, read_files= True, graph_mode= graph_mode)
        self.graph= tr_solve_obj.L_graph_obj.graph
        self.graph_nx= tr_solve_obj.L_graph_obj.graph_nx

        # take a backup of original values
        L_map_nz_node_to_val= {}
        for n in tr_solve_obj.L_map_nz_idx_to_node.values():
          obj= self.graph[n]
          assert obj.is_leaf()
          val= obj.curr_val
          L_map_nz_node_to_val[n] = val

        nz_norm= np.linalg.norm(list(L_map_nz_node_to_val.values()))

        for arith_type, int_bits, exp_bits, frac_bits, total_bits in precision_list:

          precision_obj= src.ac_eval.PrecisionConfig()
          precision_obj.int_bits= int_bits
          precision_obj.exp_bits= exp_bits
          precision_obj.frac_bits= frac_bits
          precision_obj.total_bits= total_bits
          precision_obj.set_arith_type(arith_type)

          # convert nz to custom
          # NOTE: this is a destroying conversion, the original nz value will be lost
          # L_map_nz_node_to_val= {}
          # for n in tr_solve_obj.L_map_nz_idx_to_node.values():
          #   obj= self.graph[n]
          #   assert obj.is_leaf()
          #   val= obj.curr_val
          #   L_map_nz_node_to_val[n] = val
          #   obj.curr_val = precision_obj.to_custom(float(val))
          for n in tr_solve_obj.L_map_nz_idx_to_node.values():
            obj= self.graph[n]
            obj.curr_val = precision_obj.to_custom(float(L_map_nz_node_to_val[n]))

          full_error_vect = []
          full_golden_vect = []
          b_golden= np.array([random.uniform(0, nz_norm) for _ in range(tr_solve_obj.nrows)], dtype= 'double')
          b_custom= [precision_obj.to_custom(float(val)) for val in b_golden]
          for it in range(n_iter):
            print(name, arith_type, exp_bits, frac_bits, total_bits, it)
            
            # instantiate_b and get golden double value
            # b= np.array([1.0 for _ in range(tr_solve_obj.nrows)], dtype= 'double')
            x_golden= linalg.spsolve_triangular(tr_solve_obj.L, b_golden, lower= True)

            # val= 1.1234 * (2**-1000)
            # val= 1
            # val_2= 2.3
            # print(val)
            # val= precision_obj.to_custom(val)
            # val_2= precision_obj.to_custom(val_2)
            # val= precision_obj.add(val, val_2)
            # val= precision_obj.from_custom(val)
            # print(val)
            # FixedPointImplementation.test_flt_mul(1000, exp_bits, frac_bits)
            # FixedPointImplementation.test_flt_add(1000, exp_bits, frac_bits)
            # exit(1)

            tr_solve_obj.instantiate_b("L", b_custom)
            
            src.ac_eval.ac_eval_non_recurse(self.graph, self.graph_nx, final_node= None, precision_obj= precision_obj)
            
            x_custom = []
            x_custom_original = []
            for r in range(tr_solve_obj.nrows):
              node= tr_solve_obj.L_map_x_to_node[r]
              val= self.graph[node].curr_val
              x_custom_original.append(val)
              val= precision_obj.from_custom(val)
              x_custom.append(float(val))
            x_custom= np.array(x_custom)

            # b for next iteration is x_golden of this iteration,
            # recursive estimation of b, acculumates more error, used in iterative algorithms
            b_golden = np.array(x_golden)
            b_custom = x_custom_original

            logger.info(f"top elements of x_golden: {[val for val in sorted(x_golden, reverse = True)[:10]]}")
            error_vect=np.subtract(x_golden, x_custom)
            error_vect= np.absolute(error_vect)
            full_error_vect += list(error_vect)
            full_golden_vect += list(x_golden)

            abs_error= np.linalg.norm(error_vect)
            norm= np.linalg.norm(x_golden)
            rel_error= abs_error/norm

            logger.info(f"few elements of x_golden: {[val for val in x_golden[:20]]}")
            logger.info(f"few elements of x_custom: {[val for val in x_custom[:20]]}")
            logger.info(f"top elements of error_vect: {[val for val in sorted(error_vect, reverse = True)[:10]]}")
            logger.info(f"top elements of x_golden: {[val for val in sorted(x_golden, reverse = True)[:10]]}")
            logger.info(f"top elements of x_custom: {[val for val in sorted(x_custom, reverse = True)[:10]]}")
            
            logger.info(f"precision: {arith_type}, exp_bits: {exp_bits}, frac_bits: {frac_bits}, total_bits: {total_bits}")
            logger.info(f"norm: {norm}")
            logger.info(f"abs. error: {abs_error}")
            logger.info(f"rel. error: {rel_error}")

          full_error_vect_norm= np.linalg.norm(full_error_vect)
          full_golden_vect_norm= np.linalg.norm(full_golden_vect)
          full_error_vect_norm_relative= full_error_vect_norm/full_golden_vect_norm

          element_wise_mean_abs_error = np.mean(full_error_vect)
          element_wise_mean_rel_error = np.mean(np.divide(full_error_vect,full_golden_vect))
          log_str= f"name, {name},\
arith_type, {arith_type},\
int_bits, {int_bits},\
exp_bits, {exp_bits},\
frac_bits, {frac_bits},\
total_bits, {total_bits},\
full_error_vect_norm, {full_error_vect_norm},\
full_golden_vect_norm, {full_golden_vect_norm},\
full_error_vect_norm_relative, {full_error_vect_norm_relative},\
element_wise_mean_abs_error, {element_wise_mean_abs_error},\
element_wise_mean_rel_error, {element_wise_mean_rel_error},\
last_iter_abs_error, {abs_error},\
last_iter_rel_error, {rel_error},\
n_rows, {tr_solve_obj.nrows},\
nnz, {tr_solve_obj.nnz},\
n_iter, {n_iter},\
"
          print(log_str)
          print(log_str, file= fp, flush=True)

      exit(1)
          
    if mode == 'milp_optimization':
#      src.milp_optimization.write_graph(self.global_var.GRAPH_NX_FILES_PATH, self.graph, self.graph_nx)
#      src.optimization.pipeline_scheduling_or.main(self.graph_nx, PIPE_STAGES= 3)
#      src.optimization.write_to_file.minizinc_decompose('/users/micas/nshah/Downloads/no_backup/Setups/minizinc_code/no_backup/' + str(self.global_var.network) + ".dzn",self.graph_nx)

      src.optimization.write_to_file.global_partition('/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/optimization/minizinc_code/code/decompose/no_backup/' + str(self.global_var.network) + "_global_partition.dzn", self.graph_nx, D= 3)
#      src.optimization.write_to_file.custom_dot_file(self.graph_nx)
      
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
        GEN_SV_VERIF_FILES= False
        GEN_OPENMP_CODE= False

        # partition generation
        # write_files= True
        # GEN_PARTITIONS= True

        # openmp generation
        READ_PARTITIONS= True
        # COMBINE_SMALL_LAYERS= True
        # GEN_OPENMP_CODE= True

        # GEN_SV_VERIF_FILES= True

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
          config_obj = src.new_arch.partition.CompileConfig(name= name.replace('/','_'), N_PE= n_threads, \
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
          
          elif GEN_SV_VERIF_FILES:
            try:
              status_dict, list_of_schedules, hw_details, _ = src.new_arch.partition.main(self.global_var, name, self.graph, self.graph_nx, node_w, config_obj, final_output_nodes, verbose=False)
              # head_node= tr_solve_obj.L_map_x_to_node[tr_solve_obj.ncols - 1]
              # print(head_node, max(list(self.graph.keys())))
              # assert not self.graph[head_node].is_leaf()
              src.verif_helper.pru_async(self.global_var, self.graph, self.graph_nx, final_output_nodes, status_dict, list_of_schedules, config_obj)
            except OverflowError:
              print("Out of memory!!")
              out_of_mem_ls.append(name)
            except KeyError: # remove this later
              print("ERROR: Some random key error")

          # COMBINE_LAYERS_THRESHOLD *= n_threads
          if COMBINE_SMALL_LAYERS:
            list_of_partitions_combined= src.new_arch.partition.combine_small_layers(self.graph_nx, list_of_partitions, COMBINE_LAYERS_THRESHOLD, node_w, config_obj)
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
        config_obj = src.new_arch.partition.CompileConfig(name= self.net, N_PE= n_threads, partition_mode= partition_mode, sub_partition_mode=sub_partition_mode, run_mode= run_mode, write_files= write_files, global_var= global_var)

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
        GEN_SV_VERIF_FILES= False
        GEN_OPENMP_CODE= False

        # partition generation
        # write_files= True
        # GEN_PARTITIONS= True

        # openmp generation
        READ_PARTITIONS= True
        # COMBINE_SMALL_LAYERS= True
        # GEN_OPENMP_CODE= True

        # GEN_SV_VERIF_FILES= True

        path= global_var.PSDD_PATH_PREFIX + name + '.psdd'
        self.graph, self.graph_nx, self.head_node, self.leaf_list, _ = src.psdd.main(path)
        logger.info(f"name, critical path length: {name, nx.algorithms.dag.dag_longest_path_length(self.graph_nx)}")

        # partitioning assumes that leaf are all computed
        for n in self.leaf_list:
          self.graph[n].computed= True

        for n_threads in n_threads_ls:
          config_obj = src.new_arch.partition.CompileConfig(name= name.replace('/','_'), N_PE= n_threads, \
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
            list_of_partitions= src.new_arch.partition.combine_small_layers(self.graph_nx, list_of_partitions, COMBINE_LAYERS_THRESHOLD, node_w, config_obj)

          if GEN_OPENMP_CODE:
            dataset= src.files_parser.read_dataset(global_var, name, 'test')
            src.psdd.instanciate_literals(self.graph, dataset[0])
            golden_val= src.ac_eval.ac_eval_non_recurse(self.graph, self.graph_nx, self.head_node)
            print(golden_val)

            outpath= config_obj.get_openmp_file_name()
            batch_sz= 1
            src.openmp.gen_code.par_for(outpath,self.graph, self.graph_nx, list_of_partitions, golden_val, batch_sz)

          if GEN_SV_VERIF_FILES:
            try:
              status_dict, list_of_schedules, hw_details, _ = src.new_arch.partition.main(self.global_var, name, self.graph, self.graph_nx, node_w, config_obj, set([self.head_node]), verbose=False)
              # head_node= tr_solve_obj.L_map_x_to_node[tr_solve_obj.ncols - 1]
              # print(head_node, max(list(self.graph.keys())))
              # assert not self.graph[head_node].is_leaf()
              src.verif_helper.pru_async(self.global_var, self.graph, self.graph_nx, set([self.head_node]), status_dict, list_of_schedules, config_obj)
            except OverflowError:
              print("Out of memory!!")
              out_of_mem_ls.append(name)
            except KeyError: # remove this later
              print("ERROR: Some random key error")

      exit(1)
          
    if mode == 'compile_for_async_arch':
      # Compiler for new arch
      N_PE= int(args.targs[0])

      partition_mode= args.targs[1]
      assert partition_mode in ['HEURISTIC', 'TWO_WAY_PARTITION', 'LAYER_WISE']

      run_mode= args.targs[2]
      assert run_mode in ['full', 'resume'], run_mode
      
      if partition_mode == 'LAYER_WISE':
        assert run_mode != 'resume'
      
      GLOBAL_MEM_DEPTH= 8192
      LOCAL_MEM_DEPTH= 4096

      config_obj = src.new_arch.partition.CompileConfig(name= self.net, N_PE= N_PE, GLOBAL_MEM_DEPTH= GLOBAL_MEM_DEPTH, LOCAL_MEM_DEPTH= LOCAL_MEM_DEPTH, partition_mode= partition_mode, run_mode= run_mode)
      node_w= defaultdict(lambda:1)
      status_dict, list_of_schedules, hw_details, _ = src.new_arch.partition.main(self.global_var, self.net, self.graph, self.graph_nx, node_w, config_obj, nodes_to_store= set([self.head_node]))

      self.global_var.PRU_ASYNC_VERIF_PATH += f'tb_data_{partition_mode}/{self.net}'
      src.verif_helper.pru_async(self.global_var, self.graph, self.head_node, status_dict, list_of_schedules, hw_details, write_files= True)

      exit(0)

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
      #hw_details= src.new_arch.partition.hw_details_class(N_PE= n_threads)
        in_path= ld_prefix + self.net + '_' + str(n_threads) + '.p'
        with open(in_path, 'rb+') as fp:
          list_of_partitions= pickle.load(fp)
        outpath= store_prefix + '{}_{}threads.cu'.format(self.net, n_threads)
        # src.cuda.gen_code.main(outpath, self.graph, self.graph_nx, list_of_partitions, golden_val)
      exit(0)

    if mode == 'generate_binary_executable':

      #===========================================
      #       HW details
      #===========================================
      assert len(args.targs) == 5 
      self.hw_depth= int(args.targs[0])
      n_mem_banks= 32
      reg_bank_depth= 64 
      n_pipe_stages= 5
      N_BITS= 32
      MEM_ADDR_BITS= 16

      SCRATCH_PAD_SIZE= 16 # in kB
      PARAM_MEM_SIZE= 128 #kB
      base_scratch_pad_addr= 0
      last_scratch_pad_addr= base_scratch_pad_addr + int(SCRATCH_PAD_SIZE* 1024/(n_mem_banks*(N_BITS/8))) + 1
      base_param_addr= last_scratch_pad_addr + 1
      last_param_addr= base_param_addr + int(PARAM_MEM_SIZE*1024/(n_mem_banks*(N_BITS/8))) + 1
      printlog('base_scratch_pad_addr: ' + str(base_scratch_pad_addr))
      printlog('base_param_addr: ' + str(base_param_addr))

      print('n_mem_banks:', n_mem_banks)
      print('reg_bank_size:', reg_bank_depth)
      print('pipestages:', n_pipe_stages)
      
      # -- decompose_param
      max_depth= int(args.targs[1])
      min_depth= int(args.targs[2])
      fitness_wt_distance= float(args.targs[3])
      fitness_wt_in= 0.0
      fitness_wt_out= 0.0
      decompose_param_obj= decompose_param(max_depth, min_depth, fitness_wt_distance, fitness_wt_in, fitness_wt_out)

      # -- hw_details
      hw_details= src.hw_struct_methods.hw_nx_graph_structs(self.hw_depth, max_depth, min_depth)
      hw_details.n_banks= n_mem_banks
      hw_details.reg_bank_depth= reg_bank_depth
      hw_details.n_bits= N_BITS
      hw_details.mem_bank_depth= last_param_addr
      hw_details.n_pipe_stages= n_pipe_stages
      hw_details.mem_addr_bits= MEM_ADDR_BITS
      
      # -- schedule param
      SCHEDULING_SEARCH_WINDOW= 300
      RANDOM_BANK_ALLOCATE= False
      schedule_param_obj= schedule_param(SCHEDULING_SEARCH_WINDOW, RANDOM_BANK_ALLOCATE)

      schedule_fname= create_schedule_name(hw_details, schedule_param_obj, decompose_param_obj)
      instr_ls= src.files_parser.read_schedule(self.global_var, self.hw_depth, schedule_fname)

      #===========================================
      #       Binary generation
      #===========================================
      fname= './LOG/'+ self.net + schedule_fname + '.bin.txt'

      src.reporting_tools.write_binary.write_binary(fname, self.global_var, instr_ls, hw_details)

      exit(0)


    if mode == 'vectorize_inputs_of_building_blocks':
      self.hw_depth= 4
      self.BB_graph, self.graph = src.files_parser.read_BB_graph_and_main_graph(self.global_var, self.hw_depth)
      
      assert len(self.BB_graph), "BB_graph not generated yet! Run the code with hw_tree_blocks mode" 
      
      src.scheduling.trial(self.graph, self.BB_graph, self.head_node, self.leaf_list)
      #exit(0) 

      src.scheduling.BB_scheduling(self.graph, self.BB_graph, self.head_node, self.global_var, self.leaf_list, write_asm= True, make_vid= False)  
      exit(0)

    if mode == 'float_add_optimization_exhaustive':
      src.evidence_analysis.fload_add_opt_exhaust(self.graph, self.BN, self.BN_evidence, self.head_node, self.global_var, self.leaf_list, self.ac_node_list, self.net)
      exit(0)

    if mode == 'float_add_opt':
      #--- MIN-MAX of every node with the constraints of evidence
      #self.BN_evidence= {'HRBP': 'HIGH'}
      #self.BN_evidence= {'HRBP': 'HIGH', 'PAP': 'NORMAL', 'HRSAT': 'HIGH', 'EXPCO2': 'HIGH', 'MINVOL': 'HIGH', 'HYPOVOLEMIA': 'FALSE', 'HREKG': 'HIGH', 'CVP': 'NORMAL', 'BP': 'HIGH', 'PRESS': 'HIGH', 'PCWP': 'NORMAL', 'HISTORY': 'FALSE'}
      src.evidence_analysis.set_BN_evidence(self.BN, self.BN_evidence, self.net, self.global_var , 'MAP_30')
      obs_nodes= list(self.BN_evidence.keys())

      if len(obs_nodes)>25:
        obs_nodes= obs_nodes[0:25]

      use_MIN_MAX_file= True
      if not use_MIN_MAX_file:
        for key, obj in list(self.graph.items()):
          relevant_obs_nodes= []

#          if len(obs_nodes) < len(obj.BN_node_list):
          for node in obs_nodes:
            for rel_nodes in obj.BN_node_list:
              var_name= rel_nodes.split('$')
              if node == var_name[0]: 
                relevant_obs_nodes.append(node)
                break

          #ret_dict= src.evidence_analysis.find_inst_worst_case_error(self.graph, self.head_node, self.BN, self.leaf_list, root_nodes, leaf_inst, m, lb_arr, self.ac_node_list, mode='max')
          m= [{}]
          lb_arr= [0.0]
          ret_dict= src.evidence_analysis.find_inst_worst_case_error(self.graph, key, self.BN, self.leaf_list, relevant_obs_nodes, {}, m, lb_arr, self.ac_node_list, mode='max')
          print('Key:',key,', MAX: ', lb_arr[0], 'end=',' ')
          #print ret_dict['recur_count'],
          obj.max_val = lb_arr[0]
          
          m= [{}]
          lb_arr= [1000.0]
          ret_dict= src.evidence_analysis.find_inst_worst_case_error(self.graph, key, self.BN, self.leaf_list, relevant_obs_nodes, {}, m, lb_arr, self.ac_node_list, mode='min')
          print('MIN: ', lb_arr[0])
          obj.min_val = lb_arr[0]

      elif use_MIN_MAX_file:
        fp= open('./LOG/alarm.net.min_max', 'r')
        MAX_re= re.compile('MAX:[^M]*')
        MIN_re= re.compile('MIN:[^\n]*')
        KEY_re= re.compile('Key:[^,]*')
        
        lines= fp.readlines()

        for line in lines:
          min_val= float(MIN_re.findall(line)[0].split(':')[1])
          max_val= float(MAX_re.findall(line)[0].split(':')[1])
          key_val= int(KEY_re.findall(line)[0].split(':')[1])
          
          self.graph[key_val].max_val= max_val
          self.graph[key_val].min_val= min_val
      
      bits= 30
      src.ac_eval.set_ac_node_bits(self, bits)
      src.evidence_analysis.modify_bits_in_tree_below(self.graph, self.head_node, bits, mode='SET')
      possible_savings= 0

      open_set= queue.Queue()
      open_set.put(self.head_node)

      done_nodes= set()
      graph= self.graph
      
      opt_node_track= dict.fromkeys(self.ac_node_list, False)
      OPT_MODE= 'best_case' # 'worst_case' or 'best_case' 
      while not open_set.empty():
        curr_node= open_set.get()
        if curr_node in done_nodes:
          continue
        
        curr_obj= graph[curr_node]

        for childs in curr_obj.child_key_list:
          open_set.put(childs)

        done_nodes.add(curr_node)
        key= curr_node
        obj= curr_obj
        
        # Check if sum node
        BITS_SET= False
        if obj.operation_type == src.common_classes.OPERATOR.SUM:
          # check if this sum node is not elimiating for a obs_node
          continue_flag= False
          for elim_node in obj.elim_BN_var:
            if elim_node in obs_nodes:
              continue_flag= True
              break
          
          if not continue_flag:
            # Analyze range of children
            assert len(obj.child_key_list) <= 2, "More than 2 children encoutered"
            if len(obj.child_key_list) != 0:
              child_0= self.graph[obj.child_key_list[0]]
              child_1= self.graph[obj.child_key_list[1]]
              
              if OPT_MODE== 'best_case':
                if child_0.max_val > child_1.max_val:
                  if child_0.min_val > child_1.min_val:
                    
                    if child_1.max_val != 0:
                      max_diff= child_0.max_val/child_1.max_val
                    else:
                      max_diff= 2**30
                    if child_1.min_val != 0:
                      min_diff= child_0.min_val/child_1.min_val
                    else:
                      min_diff= 2**30
                    
                    if max_diff < min_diff:
                      diff_bits= math.log(max_diff,2)
                    else:
                      diff_bits= math.log(min_diff,2)
                    
                    possible_savings= possible_savings + diff_bits
                    child_1.bits= obj.bits - diff_bits
                    child_0.bits= obj.bits
                    if child_1.bits < 0:
                      child_1.bits= 0
                    BITS_SET= True
                    opt_node_track[child_1.key]= True
                    #src.evidence_analysis.modify_bits_in_tree_below(self.graph, child_1.key, diff_bits, mode= 'REMOVE_SATURATE')
                    #src.evidence_analysis.modify_bits_in_tree_below(self.graph, child_0.key, child_0.bits, mode= 'SET')
                
                elif child_0.max_val < child_1.max_val:
                  if child_0.min_val < child_1.min_val:
                    #max_diff= child_1.max_val/child_0.max_val
                    #min_diff= child_1.min_val/child_0.min_val
                    
                    if child_0.max_val != 0:
                      max_diff= child_1.max_val/child_0.max_val
                    else:
                      max_diff= 2**30
                    if child_0.min_val != 0:
                      min_diff= child_1.min_val/child_0.min_val
                    else:
                      min_diff= 2**30
                    
                    if max_diff < min_diff:
                      diff_bits= math.log(max_diff,2)
                    else:
                      diff_bits= math.log(min_diff,2)
                    
                    possible_savings= possible_savings + diff_bits
                    child_1.bits= obj.bits
                    child_0.bits= obj.bits - diff_bits
                    if child_0.bits < 0:
                      child_0.bits= 0
                    BITS_SET= True
                    opt_node_track[child_0.key]= True
                    #src.evidence_analysis.modify_bits_in_tree_below(self.graph, child_0.key, diff_bits, mode= 'REMOVE_SATURATE')
                    #src.evidence_analysis.modify_bits_in_tree_below(self.graph, child_1.key, child_1.bits, mode= 'SET')
              
              elif OPT_MODE== 'worst_case':
                if child_0.min_val > child_1.max_val:
                  diff_bits= math.log( child_0.min_val/ child_1.max_val,2)
                  child_1.bits= obj.bits - diff_bits
                  child_0.bits= obj.bits
                  if child_1.bits < 0:
                    child_1.bits= 0
                  BITS_SET= True
                  opt_node_track[child_1.key]= True

                if child_1.min_val > child_0.max_val:
                  diff_bits= math.log( child_1.min_val/ child_0.max_val,2)
                  child_0.bits= obj.bits - diff_bits
                  child_1.bits= obj.bits
                  if child_0.bits < 0:
                    child_0.bits= 0
                  BITS_SET= True
                  opt_node_track[child_0.key]= True
                  
        if not BITS_SET:
          for child in obj.child_key_list:
            graph[child].bits= obj.bits
      
      ''' make sure the child always has as many bits as the most important parent 
      '''
      open_set= queue.Queue()
      open_set.put(self.head_node)

      done_nodes= set()
      
      for curr_node in range(self.head_node, -1, -1):
      #while not open_set.empty():
      #   curr_node= open_set.get()
        if curr_node in done_nodes:
          continue

        curr_obj= graph[curr_node]
        for childs in curr_obj.child_key_list:
          open_set.put(childs)
        
        # Make sure current node is not optimized
        if not opt_node_track[curr_node]:
          for parent in curr_obj.parent_key_list:
            if graph[parent].bits > curr_obj.bits:
              curr_obj.bits= graph[parent].bits
        
        done_nodes.add(curr_node)
       
      print('AVG bits before optimization: ', bits)
      avg_bits= src.evidence_analysis.avg_ac_bits(self.graph)
      print('AVG bits after optimization: ', avg_bits)
      print('Reduction: ', bits- avg_bits)
      # Avg leaf bits
      tot_leaf_bits= 0
      weight_leaf_cnt=0
      for key in self.leaf_list:
        obj= self.graph[key]
        if obj.leaf_type == obj.LEAF_TYPE_WEIGHT:
          weight_leaf_cnt += 1
          tot_leaf_bits += obj.bits

      print('Avg CPT bits:', tot_leaf_bits/weight_leaf_cnt)

      #src.reporting_tools.reporting_tools.create_dot_file(self.graph, self.BN, self.ac_node_list,\
          #'./LOG/alarm_bit_color.dot', option='color_according_to_bits', ac_node= self.head_node)
      exit(0)
    #---

    if mode == 'post_scheduling':
      hw_details_str= self.create_file_name_full(args)
      print(hw_details_str)
      SCHEDULING_SEARCH_WINDOW= 300
      RANDOM_BANK_ALLOCATE= False

      instr_ls= src.files_parser.read_schedule(self.global_var, self.hw_depth, hw_details_str + str(SCHEDULING_SEARCH_WINDOW) + str(RANDOM_BANK_ALLOCATE))
    
      src.scheduling_gather.print_instr_breakup_1(instr_ls)
      exit(0)

    if mode == 'scheduling_for_gather':
      sys.setrecursionlimit(1800)
      
      #n_mem_banks= 2**(self.hw_depth)
      assert len(args.targs) == 5 
      self.hw_depth= int(args.targs[0])
      max_depth= int(args.targs[1])
#      n_mem_banks= 32
      n_mem_banks= 2**self.hw_depth
      reg_bank_depth= 32 
#      n_outputs= 32
      n_outputs= 2**self.hw_depth
#      n_pipe_stages= 5
      n_pipe_stages= max_depth
      w_conflict= 1
      MEM_ADDR_BITS= 16
      
      SCHEDULING_SEARCH_WINDOW= 300
      RANDOM_BANK_ALLOCATE= False
      schedule_param_obj= schedule_param(SCHEDULING_SEARCH_WINDOW, RANDOM_BANK_ALLOCATE)

      MEM_LOAD_CONST= True
      SCRATCH_PAD_SIZE= 16 # in kB
      PARAM_MEM_SIZE= 128 #kB
      N_BITS= 32
      base_scratch_pad_addr= 0
      last_scratch_pad_addr= base_scratch_pad_addr + int(SCRATCH_PAD_SIZE* 1024/(n_mem_banks*(N_BITS/8))) + 1
      base_param_addr= last_scratch_pad_addr + 1
      last_param_addr= base_param_addr + int(PARAM_MEM_SIZE*1024/(n_mem_banks*(N_BITS/8))) + 1
      printlog('base_scratch_pad_addr: ' + str(base_scratch_pad_addr))
      printlog('base_param_addr: ' + str(base_param_addr))

      print('w_conflict:', w_conflict)
      print('n_mem_banks:', n_mem_banks)
      print('reg_bank_depth:', reg_bank_depth)
      print('pipestages:', n_pipe_stages)

      # -- decompose_param
      max_depth= int(args.targs[1])
      min_depth= int(args.targs[2])
      fitness_wt_distance= float(args.targs[3])
      fitness_wt_in= 0.0
      fitness_wt_out= 0.0
      decompose_param_obj= decompose_param(max_depth, min_depth, fitness_wt_distance, fitness_wt_in, fitness_wt_out)

      # -- hw_details
      hw_details= src.hw_struct_methods.hw_nx_graph_structs(self.hw_depth, max_depth, min_depth)
      hw_details.n_banks= n_mem_banks
      hw_details.reg_bank_depth= reg_bank_depth
      hw_details.n_bits= N_BITS
      hw_details.mem_bank_depth= last_param_addr
      hw_details.n_pipe_stages= n_pipe_stages
      hw_details.mem_addr_bits= MEM_ADDR_BITS

      decompose_fname= create_decompose_file_name(hw_details, decompose_param_obj)
  
      # -- read decompose outputs
      self.BB_graph, self.graph, self.BB_graph_nx, self.graph_nx, misc = \
          src.files_parser.read_BB_graph_and_main_graph(self.global_var, self.hw_depth, decompose_fname)
      
      # -- perform scheduling
      instr_ls_obj, map_param_to_addr= src.scheduling_gather.instruction_gen(self.net, self.graph, self.BB_graph, self.global_var,\
          self.leaf_list, misc, hw_details,\
          w_conflict, MEM_LOAD_CONST, base_scratch_pad_addr, last_scratch_pad_addr,\
          base_param_addr, last_param_addr, SCHEDULING_SEARCH_WINDOW, RANDOM_BANK_ALLOCATE, write_asm= False, make_vid= False)  

      # Construct initialization for inputs, for verification
      #param_init= src.verif_helper.param_init(self.graph, instr_ls_obj, map_param_to_addr, n_pipe_stages)
      #with open(global_var.LOG_PATH + self.net + '_param_init', 'w+') as fp:
      #  pickle.dump(param_init, fp)

      tot_reg_rd=0
      tot_reg_wr=0
      for bb in list(self.BB_graph.values()):
        #assert len(bb.out_list) <= 4
        tot_reg_rd += len(bb.in_list_unique)
        tot_reg_wr += len(bb.out_list)
      
      print('Reg_rd:', tot_reg_rd, 'Reg_wr:', tot_reg_wr)

      #src.reporting_tools.reporting_tools.write_c_for_asip_2('./LOG/asip.c', self.BB_graph, self.graph, 16, self.leaf_list)
      #src.reporting_tools.reporting_tools.write_c_for_asip('./LOG/vect_' + str(self.net)+'_gcc.c', self.graph)
      #src.reporting_tools.reporting_tools.write_vect_c('./LOG/vect_'+ str(self.net) +'.c', self.graph, self.BB_graph)
      
      fname= create_schedule_name(hw_details, schedule_param_obj, decompose_param_obj)
      src.files_parser.write_schedule(self.global_var, self.hw_depth, fname, list(instr_ls_obj.instr_ls) )
      exit(0) 

    if mode== 'hw_tree_blocks':
      # Break AC into HW tree blocks
      #self.hw_depth= 5
      #max_depth= 3
      #min_depth= 2
      
      assert len(args.targs) == 5 
      self.hw_depth= int(args.targs[0])
      max_depth= int(args.targs[1])
      min_depth= int(args.targs[2])
      fitness_wt_distance= float(args.targs[3])
      out_mode= args.targs[4] #'TOP_2'
      
      #n_outputs= 32
      fitness_wt_in= 0.0
      fitness_wt_out= 0.0
      
      decompose_param_obj= decompose_param(max_depth, min_depth, fitness_wt_distance, fitness_wt_in, fitness_wt_out)

      assert out_mode in ['ALL','VECT' , 'TOP_1', 'TOP_2']
      
      print('fitness_wt_distance: ', fitness_wt_distance)
      #print 'n_outputs', n_outputs
      print('output mode:', out_mode)

      hw_details= src.hw_struct_methods.hw_nx_graph_structs(self.hw_depth, max_depth, min_depth)

      n_outputs= None
      return_dict= src.decompose.decompose(self.graph, self.hw_depth, self.head_node, self.leaf_list, n_outputs, decompose_param_obj, hw_details, out_mode)
      self.BB_graph= return_dict['BB_graph'] 
      self.BB_graph_nx= return_dict['BB_graph_nx'] 
      
      list_of_depth_lists= hw_details.list_of_depth_lists
      #hw_details_str= out_mode + '_' + str(fitness_wt_in) + str(fitness_wt_out) + str(fitness_wt_distance) + '_'.join([''.join(str(y) for y in x) for x in list_of_depth_lists])
      #print hw_details_str 
      fname= create_decompose_file_name(hw_details, decompose_param_obj)
      

      # Misc. information to store for next steps
      misc= {}
#      misc['best_hw_set']= return_dict['best_hw_set']

      src.files_parser.write_BB_graph_and_main_graph(self.global_var, self.hw_depth, fname, self.graph, self.graph_nx, self.BB_graph, self.BB_graph_nx, misc)

      exit(0)
      # Decome AC into hw-related blocks
      #self.hw_struct, self.hw_depth, self.hw_dict, self.hw_misc_dict = src.hw_struct_methods._read_hw_struct(global_var.HW_STRUCT_MAC_FILE, verbose=True)
      #src.decompose._map_AC_to_hw(global_var, self.graph, self.n_operations, self.hw_depth, self.leaf_list, self.head_node, False, MAC=False, ref_hw_struct= self.hw_struct, ref_hw_depth= self.hw_depth, ref_hw_dict= self.hw_dict, ref_hw_misc_dict= self.hw_misc_dict)
      
    if mode=="verif_helper":
      exit(1)

    if mode=="ac_eval":
      src.verif_helper.init_leaf_val(self.graph, mode='all_1s')
      return_val= src.ac_eval.ac_eval(self.graph, self.head_node, elimop= 'PROD_BECOMES_SUM')
      
      print('AC eval:', return_val)
      exit(0)

    if mode=="generate_ASIP_cfile":
      # main file for ASIP
      src.reporting_tools.reporting_tools.write_c_for_asip(global_var.ASIP_CFILE, self.graph)
      exit(0)
    
    if mode== 'max_FixPt_err_query':
      # Search for the instantiation that produces high error
      #root_nodes= 'Smoker'
      #root_nodes= 'HYPOVOLEMIA,LVFAILURE,KINKEDTUBE,INSUFFANESTH,ERRCAUTER,PULMEMBOLUS,MINVOLSET,DISCONNECT,ERRLOWOUTPUT,FIO2,INTUBATION,ANAPHYLAXIS'
      root_nodes= 'PrntngArOK,EPSGrphc,DrvOK,FntInstlltn,PrtQueue,NtwrkCnfg,PrtSel,PrntrAccptsTrtyp,DSApplctn,PrtPaper,PrtTimeOut,FllCrrptdBffr,PrtSpool,PrtMem,CblPrtHrdwrOK,PrtThread,PrtMpTPth,DskLocal,PTROFFLINE,AppOK,ScrnFntNtPrntrFnt,TrTypFnts,PrtOn,DataFile,GrphcsRltdDrvrSttngs,PrtPath,PrtPort,TnrSpply,PrtDriver,DrvSet,NetPrint,PrtPScript,PgOrnttnOK,PrtCbl'
      root_nodes= root_nodes.split(',')


      #leaf_nodes= 'Xray,Dyspnoea,Pollution'
      #leaf_nodes= 'PAP,MINVOL,HREKG,CVP,PRESS,EXPCO2,HRSAT,HRBP,BP,HISTORY,PCWP'
      leaf_nodes= 'PrtIcon,PrtFile,PrtStatMem,TstpsTxt,PrtStatPaper,REPEAT,PrtStatOff,Problem5,PrtStatToner,Problem4,Problem3,Problem2,Problem1,PSERRMEM,HrglssDrtnAftrPrnt,Problem6'
      leaf_nodes= leaf_nodes.split(',')
  
      m = [{}]
      lb_arr = [1000.0]
      ret_dict= src.evidence_analysis.find_inst_worst_case_error(self.graph, self.head_node, self.BN, self.leaf_list, leaf_nodes, {}, m, lb_arr, self.ac_node_list, mode='min')
      print('MIN setting:', m[0], lb_arr[0]) 
      denom= src.ac_eval.ac_eval_with_evidence(self.graph, self.BN, self.head_node, m[0], self.leaf_list)
      print('Denom:', denom)

      leaf_inst= copy.deepcopy(m[0])
      m = [{}]
      lb_arr = [0]
      ret_dict= src.evidence_analysis.find_inst_worst_case_error(self.graph, self.head_node, self.BN, self.leaf_list, root_nodes, leaf_inst, m, lb_arr, self.ac_node_list, mode='max')
      root_inst= copy.deepcopy(m[0])
      
      print('MAX setting:', m[0], lb_arr[0]) 
      print('Cond_query: ', src.ac_eval.ac_eval_with_evidence(self.graph, self.BN, self.head_node, root_inst, self.leaf_list)/denom)
      
      print('Leaf:')
      for var in list(leaf_inst.keys()):
        sys.stdout.write(var.strip('"\'')+ ',')
        sys.stdout.flush()
        root_inst.pop(var)
      print('')
      for var in list(leaf_inst.values()):
        sys.stdout.write(var.strip('"\'')+ ',')
        sys.stdout.flush()
      print('')
      print('Query:')
      for var in list(root_inst.keys()):
        sys.stdout.write(var.strip('"\'')+ ',')
        sys.stdout.flush()
      print('')
      for var in list(root_inst.values()):
        sys.stdout.write(var.strip('"\'')+ ',')
        sys.stdout.flush()
      print('')

      exit(0)
    
    if mode== 'output_min_max':
      # Find MAX- and MIN value
      print("AC length", len(self.graph))
      print("No evidence AC eval: ", src.ac_eval.ac_eval(self.graph, self.head_node))
      max_val= 0
      for node in self.graph:
        if self.graph[node].curr_val > max_val:
          max_val = self.graph[node].curr_val
      print('Max_val: ', max_val)
      
      print("Min AC eval: ", src.ac_eval.ac_eval(self.graph, self.head_node, 'MIN'))
      min_val= 100
      for node in self.graph:
        if self.graph[node].curr_val < min_val and self.graph[node].curr_val != 0:
          min_val = self.graph[node].curr_val
      print('Min_val: ', min_val)
      exit(0)

    if mode== 'munin_single_query_verify':
      # Munin signle query check
      var_lst= open('/users/micas/nshah/Temp/temp_munin', 'r').readlines()
      var_lst= var_lst[0].split(',')
      var_lst[-1]= var_lst[-1][:-1]
      state_lst= open('/users/micas/nshah/Temp/temp_munin_st', 'r').readlines()
      state_lst= state_lst[0].split(',')
      state_lst[-1]= state_lst[-1][:-1]
      self.BN_evidence= {}
      for idx, var in enumerate(var_lst):
        self.BN_evidence[var]= state_lst[idx]
      x_1= src.ac_eval.ac_eval_with_evidence(self.graph, self.BN, self.head_node, self.BN_evidence, self.leaf_list)
      print('AC_query with root:', x_1)
      self.BN_evidence.pop('L_LNLW_MED_TIME', None)
      x_2= src.ac_eval.ac_eval_with_evidence(self.graph, self.BN, self.head_node, self.BN_evidence, self.leaf_list)
      print('AC_query without root:', x_2) 
      if x_1 == x_2:
        print('Equal')
      else:
        print('Not equal:', x_2 - x_1)
      exit(0)
    
    if mode== 'eval_float_error':
      print("AC length", len(self.graph))
      ## Float error
      bits= 23
      for bits in range(7,20):
        src.ac_eval.set_ac_node_bits(self, bits)
        error= src.ac_eval.error_eval(self, 'float', self.head_node, custom_bitwidth= False)
        c= math.log(error+1)/math.log(1+2**(-bits-1))
        low_error= 1 - (1-2**(-bits-1))**c
        
        kwargs= {'precision': 'CUSTOM', 'arith_type' : 'FLOAT', 'exp': 9, 'mant': bits}
        energy= src.energy.energy_est(self.graph, **kwargs)
        
        print('for ', bits,' bits mantt the rel. error bound is: ', error, ' for negative c, error could be: ', low_error, ', energy in fJ: ', energy)
        #print 'Number of bits:', abs(math.log(error, 2))
        #print 'Reduction', bits - abs(math.log(error,2))
      exit(0)
 
    if mode== 'eval_fixed_error':
      for bits in range(7, 70):
        print('for ', bits, end=' ')
        #src.ac_eval.set_ac_node_bits(self, bits)
        #error= src.ac_eval.error_eval(self, 'fixed', self.head_node, custom_bitwidth= False)
        #print ' bits fraction, the abs. error bound is: ', error , 
        
        kwargs= {'precision': 'CUSTOM', 'arith_type' : 'FIXED', 'int': 1, 'frac': bits}
        energy= src.energy.energy_est(self.graph, **kwargs)
        print("energy in fJ: ", energy)
      
      exit(0)
    
    if mode== 'ac_eval_testset_error':
      arith_type= 'FIXED'
      bit_0= 1
      bit_1= 14
      test_cases= 1000
      run_tests= 0
      
      lp_file_num= global_var.EVIDENCE_WMC_LP_PREFIX + '_' + arith_type + '_' + str(bit_0) + str(bit_1) + '_num'
      lp_file_denom= global_var.EVIDENCE_WMC_LP_PREFIX + '_' + arith_type + '_' + str(bit_0) + str(bit_1) + '_denom'
      
      if run_tests:
        dat_fp= open(global_var.EVIDENCE_DB_FILE, 'r')
        dat_content= dat_fp.readlines()
        
        dat_content= dat_content[:test_cases]

        evid_db= src.files_parser.read_evid_db(dat_content, self.BN)

        # HAR_NaiveBayes26F
        # All: ['V505', 'V42', 'V370', 'V80', 'V451', 'V452', 'V121', 'V160', 'Class', 'V538', 'V519', 'V557', 'V108', 'V105', 'V558', 'V559', 'V412', 'V54', 'V52', 'V53', 'V303', 'V38', 'V70', 'V158', 'V159', 'V449']
        #BN_nodes_num= ['V505', 'V42', 'V370', 'V80', 'V451', 'V452', 'V121', 'V160', 'Class', 'V538', 'V519'] 
        #BN_nodes_denom= ['V505', 'V42', 'V370', 'V80', 'V451', 'V452', 'V121', 'V160', 'V538', 'V519']
       
        #UNIMIB_NB
        # All: ['F20', 'F1', 'F3', 'F5', 'F6', 'F7', 'F8', 'F9', 'F17', 'F19', 'F12', 'F16', 'Class', 'F14', 'F15']
        #BN_nodes_num= ['Class', 'F1']
        #BN_nodes_denom= ['F1']
        #BN_nodes_num= ['F20', 'F1', 'F8', 'F9', 'F17', 'F19', 'F12', 'F16', 'Class', 'F14', 'F15']
        #BN_nodes_denom= ['F20', 'F1', 'F8', 'F9', 'F17', 'F19', 'F12', 'F16', 'F14', 'F15']
        
        #UIWADS
        #All: ['V1', 'V2', 'V3', 'Class']
        #BN_nodes_num= ['V3', 'Class']
        #BN_nodes_denom= ['V3']
        #BN_nodes_num= ['V1', 'V2', 'V3', 'Class']
        #BN_nodes_denom= ['V1', 'V2', 'V3']
        
        # Alarm:
        #All: ['PAP', 'VENTLUNG', 'SAO2', 'SHUNT', 'HR', 'HREKG', 'CVP', 'KINKEDTUBE', 'BP', 'ERRCAUTER', 'PULMEMBOLUS', 'EXPCO2', 'MINVOLSET', 'CATECHOL', 'ERRLOWOUTPUT', 'FIO2', 'LVEDVOLUME', 'INTUBATION', 'STROKEVOLUME', 'PRESS', 'HRBP', 'VENTMACH', 'CO', 'MINVOL', 'HYPOVOLEMIA', 'LVFAILURE', 'HRSAT', 'INSUFFANESTH', 'TPR', 'HISTORY', 'VENTALV', 'DISCONNECT', 'VENTTUBE', 'ARTCO2', 'ANAPHYLAXIS', 'PVSAT', 'PCWP']
        
        BN_nodes_num= ['KINKEDTUBE', 'BP', 'ERRCAUTER', 'PULMEMBOLUS', 'EXPCO2', 'MINVOLSET', 'CATECHOL', 'ERRLOWOUTPUT', 'FIO2', 'LVEDVOLUME', 'INTUBATION', 'STROKEVOLUME', 'PRESS', 'HRBP', 'VENTMACH', 'CO', 'MINVOL', 'HYPOVOLEMIA', 'LVFAILURE', 'HRSAT', 'INSUFFANESTH', 'TPR', 'HISTORY', 'VENTALV', 'DISCONNECT', 'VENTTUBE', 'ARTCO2', 'ANAPHYLAXIS', 'PVSAT', 'PCWP']
        BN_nodes_denom= ['BP', 'ERRCAUTER', 'PULMEMBOLUS', 'EXPCO2', 'MINVOLSET', 'CATECHOL', 'ERRLOWOUTPUT', 'FIO2', 'LVEDVOLUME', 'INTUBATION', 'STROKEVOLUME', 'PRESS', 'HRBP', 'VENTMACH', 'CO', 'MINVOL', 'HYPOVOLEMIA', 'LVFAILURE', 'HRSAT', 'INSUFFANESTH', 'TPR', 'HISTORY', 'VENTALV', 'DISCONNECT', 'VENTTUBE', 'ARTCO2', 'ANAPHYLAXIS', 'PVSAT', 'PCWP']

        print("Starting Golden simulations")
        # Numerator
        gld_sim_lst_num= src.evidence_analysis.cust_precision_db_sim(self.graph, self.BN, self.head_node,BN_nodes_num, evid_db, self.leaf_list)
        src.reporting_tools.reporting_tools._write_csv(global_var.EVIDENCE_WMC_GLD_PREFIX + '_num', gld_sim_lst_num)
        
        # Denominator
        gld_sim_lst_denom= src.evidence_analysis.cust_precision_db_sim(self.graph, self.BN, self.head_node,BN_nodes_denom, evid_db, self.leaf_list)
        src.reporting_tools.reporting_tools._write_csv(global_var.EVIDENCE_WMC_GLD_PREFIX + '_denom', gld_sim_lst_denom)

        if arith_type == 'FIXED':
          kwargs= {'precision': 'CUSTOM', 'arith_type' : arith_type, 'int': bit_0, 'frac': bit_1}
        elif arith_type == 'FLOAT':
          kwargs= {'precision': 'CUSTOM', 'arith_type' : arith_type, 'exp': bit_0, 'mant': bit_1}
        
        print(kwargs)
        print("Starting Low precision simulations")
        # Numerator
        lp_sim_lst_num= src.evidence_analysis.cust_precision_db_sim(self.graph, self.BN, self.head_node, BN_nodes_num, evid_db, self.leaf_list, **kwargs)
        src.reporting_tools.reporting_tools._write_csv(lp_file_num, lp_sim_lst_num)
        
        # Denominator
        lp_sim_lst_denom= src.evidence_analysis.cust_precision_db_sim(self.graph, self.BN, self.head_node, BN_nodes_denom, evid_db, self.leaf_list, **kwargs)
        src.reporting_tools.reporting_tools._write_csv(lp_file_denom, lp_sim_lst_denom)
      
      else:
        gld_sim_lst_num= src.reporting_tools.reporting_tools._read_csv(global_var.EVIDENCE_WMC_GLD_PREFIX + '_num')[0]
        lp_sim_lst_num= src.reporting_tools.reporting_tools._read_csv(lp_file_num)[0]
        gld_sim_lst_denom= src.reporting_tools.reporting_tools._read_csv(global_var.EVIDENCE_WMC_GLD_PREFIX + '_denom')[0]
        lp_sim_lst_denom= src.reporting_tools.reporting_tools._read_csv(lp_file_denom)[0]
        
        gld_sim_lst_num= [float(i) for i in gld_sim_lst_num]
        lp_sim_lst_num= [float(i) for i in lp_sim_lst_num]
        gld_sim_lst_denom= [float(i) for i in gld_sim_lst_denom]
        lp_sim_lst_denom= [float(i) for i in lp_sim_lst_denom]

      abs_err_lst=[]
      rel_err_lst=[]
      abs_err_lst_cnd=[]
      rel_err_lst_cnd=[]
      EXCEPTION_MAR= False
      EXCEPTION_CND= True
      for idx,gld_val in enumerate(gld_sim_lst_num):
        lp_val= lp_sim_lst_num[idx]
        err= abs(lp_val -gld_val)
        abs_err_lst.append(err)

        if not gld_val==0:
          rel_err_lst.append(err/gld_val)
        
        if not gld_sim_lst_denom[idx]==0:
          gld_cnd= gld_val/ gld_sim_lst_denom[idx]
          if not lp_sim_lst_denom[idx]==0:
            lp_cnd= lp_val/ lp_sim_lst_denom[idx]
            err_cnd= abs(lp_cnd - gld_cnd)
            abs_err_lst_cnd.append(err_cnd)
            
            if not gld_cnd==0:
              rel_err_lst_cnd.append(err_cnd/gld_cnd)
          else:
            rel_err_lst_cnd.append(abs(1.0 - gld_cnd))  
            EXCEPTION_CND= True

        print(gld_cnd, lp_cnd, rel_err_lst_cnd[-1])
      
      print('Marginal_Q Max_abs_error:', max(abs_err_lst))
      print('Marginal_Q Max_rel_error:', max(rel_err_lst))
      print('Conditional_Q Max_abs_error:', max(abs_err_lst_cnd))
      print('Conditional_Q Max_rel_error:', max(rel_err_lst_cnd))
      exit(0)

    if mode== 'adaptively_optimize_for_Fxpt':
      ## Optimize non-linear bits
      bits= 23
      src.ac_eval.set_ac_node_bits(self, bits)
      
      #print "------- Error without taking evidence into consideration--------"
      error= src.ac_eval.error_eval(self, 'fixed', self.head_node, custom_bitwidth= False)
      print('Error: ', error)
      print('Number of bits:', abs(math.log(error, 2)))
      print('Reduction', bits - abs(math.log(error,2)))
      #

      exit(1)
      src.evidence_analysis.set_BN_evidence(self.BN, self.BN_evidence, self.net, self.global_var , 'MAP')
      #self.BN_evidence= {'HRBP': 'HIGH', 'PAP': 'NORMAL', 'HRSAT': 'HIGH', 'EXPCO2': 'HIGH', 'MINVOL': 'HIGH', 'HYPOVOLEMIA': 'FALSE', 'HREKG': 'HIGH', 'CVP': 'NORMAL', 'BP': 'HIGH', 'PRESS': 'HIGH', 'PCWP': 'NORMAL', 'HISTORY': 'FALSE'}
      src.evidence_analysis.set_evidence_in_AC(self.graph, self.BN, self.BN_evidence, self.leaf_list)
      #
      print("------- Error with evidence 30% nodes observed--------")
      error= src.ac_eval.error_eval(self, 'fixed', self.head_node, custom_bitwidth= False)
      print('Error: ', error)
      print('Number of bits:', abs(math.log(error, 2)))
      print('Reduction', bits - abs(math.log(error,2)))
      

      print("------- Error with custom width --------")
      # Find worst case instantiation for non-uniform bits
      bit_content= open(self.global_var.BITWIDTH_FILE, 'r').readlines()
      src.graph_init.read_custom_bits_file(self, bit_content)
      src.evidence_analysis.set_BN_evidence(self.BN, self.BN_evidence, self.net, self.global_var , 'MAP')
      # Make a copy of original AC
      self.BN_evidence= {'HRBP': 'HIGH', 'PAP': 'NORMAL', 'HRSAT': 'HIGH', 'EXPCO2': 'HIGH', 'MINVOL': 'HIGH', 'HYPOVOLEMIA': 'FALSE', 'HREKG': 'HIGH', 'CVP': 'NORMAL', 'BP': 'HIGH', 'PRESS': 'HIGH', 'PCWP': 'NORMAL', 'HISTORY': 'FALSE'}
      src.evidence_analysis.set_evidence_in_AC(self.graph, self.BN, self.BN_evidence, self.leaf_list)
      error= src.ac_eval.error_eval(self, 'fixed', self.head_node, custom_bitwidth= True, verb=False)
      print('Error: ', error)
      print('Number of bits:', abs(math.log(error, 2)))
      print('Reduction', bits - abs(math.log(error,2)))
      
      m = [{}]
      lb_arr = [0.0]
      #ret_dict= src.evidence_analysis.find_inst_worst_case_error(self.graph, self.head_node, self.BN, self.leaf_list, list(self.BN_evidence.keys()), {}, m, lb_arr, self.ac_node_list, mode='max')
      print('MAX setting:', m[0], lb_arr[0])
      #print 'recurse_cost:', ret_dict['recur_count']
      #self.BN_evidence= m[0]
      src.evidence_analysis.set_evidence_in_AC(self.graph, self.BN, self.BN_evidence, self.leaf_list)
      print(src.ac_eval.ac_eval(self.graph,self.head_node))
      
      m = [{}]
      lb_arr = [1000.0]
      ret_dict= src.evidence_analysis.find_inst_worst_case_error(self.graph, self.head_node, self.BN, self.leaf_list, list(self.BN_evidence.keys()), {}, m, lb_arr, self.ac_node_list, mode='min')
      print('MIN setting:', m[0], lb_arr[0]) 
      print('recurse_cost:', ret_dict['recur_count'])
      self.BN_evidence= m[0]
      src.evidence_analysis.set_evidence_in_AC(self.graph, self.BN, self.BN_evidence, self.leaf_list)
      print(src.ac_eval.ac_eval(self.graph,self.head_node))
      
      # Find and optimize for culprit nodes
      total_bits= 0
      for iteration in range(0,50):
        m = [{}]
        lb_arr = [0.0]
        ret_dict= src.evidence_analysis.find_inst_worst_case_error(self.graph, self.head_node, self.BN, self.leaf_list, list(self.BN_evidence.keys()), {}, m, lb_arr, self.ac_node_list, 'error')
        print(m[0], lb_arr[0])
        print('recurse_cost:', ret_dict['recur_count'])
        self.BN_evidence= m[0]
        src.evidence_analysis.set_evidence_in_AC(self.graph, self.BN, self.BN_evidence, self.leaf_list)
        error= src.ac_eval.error_eval(self, 'fixed', self.head_node, custom_bitwidth= False, verb=False)
        print('Error: ', error)
        print('Reduction', bits - abs(math.log(error,2)))
        print('overall_reduction:', float((len(self.graph)*bits + total_bits))/len(self.graph) - abs(math.log(error,2)))
        
        for opt_iter in range(0,75):
          culprit_nodes=src.evidence_analysis.find_OverOptimized_nodes(self, self.graph, self.BN, self.leaf_list, self.head_node, self.map_BinaryOpListKey_To_OriginalKey, self.BN_evidence)
          #print len(culprit_nodes), culprit_nodes
          for idx, node in enumerate(culprit_nodes):
            if idx < 4:
              self.graph[node].bits = self.graph[node].bits + 1
              total_bits= total_bits + 1
        print(total_bits)
        if total_bits > 15000:
          break
      print('tolal_bits:', total_bits)
      print('overall_reduction:', float((len(self.graph)*bits + total_bits))/len(self.graph) - abs(math.log(error,2)))
      
      # Document the change in bits
      new_NonUniform_bits= [0] * len(self.graph)
      orig_AC_len= 0
      for key, obj in list(self.graph.items()):
        orig_AC_key= self.map_BinaryOpListKey_To_OriginalKey[key]
        if orig_AC_key > (orig_AC_len-1):
          orig_AC_len= orig_AC_key + 1

        if obj.bits > new_NonUniform_bits[orig_AC_key]:
          new_NonUniform_bits[orig_AC_key]= obj.bits

      new_NonUniform_bits= new_NonUniform_bits[0:orig_AC_len]
      print(new_NonUniform_bits)
      for idx, val in enumerate(new_NonUniform_bits):
        print(idx, val)
      exit(0)
    
    if mode== 'exhaust_search_for_max_error':
      #src.ac_eval.set_ac_node_bits(self, bits)
      src.evidence_analysis.set_BN_evidence(self.BN, self.BN_evidence, self.net, self.global_var , 'rand')
      max_error= 0
      for iter_idx in range(0,500000):
        src.evidence_analysis.set_BN_evidence(self.BN, self.BN_evidence, self.net, self.global_var , 'exhaustive')

        src.evidence_analysis.set_evidence_in_AC(self.graph, self.BN, self.BN_evidence, self.leaf_list)
        
        curr_error= src.ac_eval.error_eval(self, 'fixed', self.head_node, custom_bitwidth= False)
        if curr_error > max_error:
          max_error= curr_error
          print('New max-error found at:', self.BN_evidence)

        if iter_idx % 1000 == 0:
          print(max_error, abs(math.log(max_error, 2))) #self.BN_evidence
      
      print("Maximum error in random evidence instantiaition:", max_error)
      print('Number of bits:', abs(math.log(max_error, 2)))
      exit(0)
   
 
    if mode== 'dot_file_to_visualize_ac':
      #--- Create dot file to visualize graphs
      src.evidence_analysis.populate_BN_list(self.graph, self.BN, self.head_node, dict.fromkeys(self.ac_node_list, False) ) 
      src.reporting_tools.reporting_tools.create_dot_file(self, global_var.GRAPHVIZ_BN_VAR_FILE, 'BN_var_details', self.head_node)
      #src.evidence_analysis.print_BN_list(self)
      exit(0)

    print("No test mode selected!! Exiting without doing anything.") 
    assert 0

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
      list_of_partitions , status_dict = src.new_arch.partition.global_barriers(name, graph, graph_nx, node_w, config_obj)
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


  def process_depth_data(self):
    occur_list= []
    for i in range(16):
      occur_list.append(self.depth_list.count(i))
    
    occur_list.append(occur_list[0])
    occur_list[0]=0
    src.reporting_tools.reporting_tools._write_csv(global_var.COMMON_CHILD_DEPTH_CSV, occur_list)
  
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
  
  def print_BN(self):
    for node in self.BN:
      print('Name:', self.BN[node].node_name, ' $ Par_lst:', self.BN[node].parent_list, ' $ Ch_lst:', self.BN[node].child_list, ' $ potent:', self.BN[node].potential)

  def print_attr(self):
    print("Network: ", self.net)
    
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

