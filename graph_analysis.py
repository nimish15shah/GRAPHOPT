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
import subprocess

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
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
import src.super_layer_generation.partition
import src.psdd
import src.openmp.gen_code
import src.openmp.compile
import src.sparse_linear_algebra.main
import src.sparse_linear_algebra.matrix_names_list

#********
# Calls methods to generate superlayers
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

    if mode == 'async_partition':
      partition_mode= args.targs[0]
      write_files= bool(int(args.targs[1]))
#      n_threads_ls= [1,2,4,8,16,32,64,128,256,512,1024]
#      n_threads_ls= [1024, 2048]
      n_threads_ls= [64]
      sub_partition_mode= None
      run_mode= None

      node_w= defaultdict(lambda:1)

      for n_threads in n_threads_ls:
        config_obj = src.super_layer_generation.partition.CompileConfig(name= self.net, N_PE= n_threads, partition_mode= partition_mode, sub_partition_mode=sub_partition_mode, run_mode= run_mode, write_files= write_files, global_var= global_var)

        self.async_partition(name, self.graph, self.graph_nx, node_w, config_obj)

      exit(0)

    if mode == 'full':
      logger.info(f"###### Target workload: {args.net}, Target threads: {args.threads} ######")
      if args.cir_type == 'psdd':
        graph_mode = 'FINE'
        target_app= 'SPN'
      elif args.cir_type == 'sptrsv':
        graph_mode = 'COARSE'
        target_app= 'SPARSE_TR_SOLVE'
      else:
        assert 0

      config_obj = src.common_classes.ConfigObjTemplates()
      config_obj.graph_init(name= args.net, cir_type= args.cir_type, graph_mode= graph_mode)
      self.graph, self.graph_nx, self.head_node, self.leaf_list, other_obj = src.graph_init.get_graph(global_var, config_obj)

      # partition the graph
      partition_mode= 'TWO_WAY_PARTITION'
      # partition_mode= 'LAYER_WISE'
      # partition_mode= 'HEURISTIC'
      # sub_partition_mode= 'ALAP'
      sub_partition_mode= 'TWO_WAY_LIMIT_LAYERS'
      # sub_partition_mode= 'TWO_WAY_FULL'
      # sub_partition_mode= None
      run_mode= 'FULL'
      # run_mode= 'RESUME'
      # target_device= 'PRU'
      target_device= 'CPU'
      COMBINE_LAYERS_THRESHOLD= 2000

      # partition generation
      write_files= True
      COMBINE_SMALL_LAYERS= True

      # partitioning assumes that leaf are all computed
      for n in self.leaf_list:
        self.graph[n].computed= True

      name= global_var.network
      threads= args.threads
      config_obj = src.super_layer_generation.partition.CompileConfig(name= name.replace('/','_'), N_PE= threads, \
          graph_mode= graph_mode,
          partition_mode= partition_mode, sub_partition_mode=sub_partition_mode, run_mode= run_mode, \
          target_app = target_app, target_device= target_device, \
          write_files= write_files, global_var= self.global_var)

      if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
        node_w= defaultdict(lambda:1)
      elif config_obj.graph_mode== config_obj.graph_mode_enum.COARSE:
        node_w= {n: len(list(self.graph_nx.predecessors(n))) + 1 for n in self.graph_nx.nodes()}
      else:
        assert 0

      logger.info("####### Generating superlayers #######")
      list_of_partitions, status_dict= self.async_partition(name, self.graph, self.graph_nx, node_w, config_obj)
      
      if COMBINE_SMALL_LAYERS:
        list_of_partitions= src.super_layer_generation.partition.combine_small_layers(self.graph_nx, list_of_partitions, COMBINE_LAYERS_THRESHOLD, node_w, config_obj)

      logger.info("####### Generating OpenMP code #######")
      if args.cir_type == 'psdd':
        dataset= src.files_parser.read_dataset(global_var, name, 'test')
        src.psdd.instanciate_literals(self.graph, dataset[0])
        golden_val= src.ac_eval.ac_eval_non_recurse(self.graph, self.graph_nx, self.head_node)

        outpath= config_obj.get_openmp_file_name()
        batch_sz= 1
        src.openmp.gen_code.par_for(outpath,self.graph, self.graph_nx, list_of_partitions, golden_val, batch_sz)
      elif args.cir_type == 'sptrsv':
        tr_solve_obj = other_obj['tr_solve_obj']
        matrix= tr_solve_obj.L
        b= np.array([1.0 for _ in range(len(self.graph_nx))], dtype= 'double')
        x_golden= linalg.spsolve_triangular(matrix, b, lower= True)
        head_node= len(b) - 1
        golden_val = x_golden[-1]
        src.openmp.gen_code.par_for_sparse_tr_solve_full_coarse(self.graph, self.graph_nx, b, list_of_partitions, matrix, config_obj, head_node, golden_val)
      else:
        assert 0

      logger.info("####### Executing parallelized DAG #######")
      log_path = self.global_var.LOG_PATH + 'run_log_openmp'
      logger.info(f"Logging results in {log_path} file")
      suffix = f"{partition_mode}"
      suffix += f"_{sub_partition_mode}"
      suffix += f"_{target_device}"
      suffix += f"_{graph_mode}"

      if args.cir_type == 'psdd':
        par_for_psdd([args.net], [args.threads], log_path, "../../" + self.global_var.OPENMP_PATH, suffix)
      elif args.cir_type == 'sptrsv':
        par_for_sptrsv([args.net], [args.threads], log_path, "../../" + self.global_var.OPENMP_PATH, suffix)
      else:
        assert 0

      logger.info("####### Done! #######")

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
      #hw_details= src.super_layer_generation.partition.hw_details_class(N_PE= n_threads)
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
      list_of_partitions , status_dict = src.super_layer_generation.partition.global_barriers(name, graph, graph_nx, node_w, config_obj)
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

def par_for_sptrsv(name_ls, thread_ls, log_path, openmp_prefix, suffix):
  line_number= 49
  run_log= open(log_path, 'a+')

  cmd= "cd src/openmp/; make set_env"
  os.system(cmd)

  for mat in name_ls:
    for th in thread_ls:
      mat = mat.replace('/', '_')
      data_path= openmp_prefix + f"{mat}_{suffix}_{th}.c"
      data_path = data_path.replace('/', '\/')
      openmp_main_file= "./src/openmp/par_for_sparse_tr_solve_coarse.cpp"
      cmd= f"sed -i '{line_number}s/.*/#include \"{data_path}\"/' {openmp_main_file}"
      os.system(cmd)
      cmd= "cd src/openmp; make normal_cpp"
      err= os.system(cmd)
      if err:
        print(f"Error in compilation {mat}, {th}")
        print(f"{mat},{th},Error compilation", file= run_log, flush= True)
      else:
        logger.info("Excuting 1k iterations of parallel code...")
        cmd= "cd src/openmp; make run"
        output= subprocess.check_output(cmd, shell=True)
        # os.system(cmd)
        output = str(output)
        output = output[:-3]
        output= output[output.find('N_layers'):]
        msg= f"{mat},{th},{output}"
        print(msg, file= run_log, flush= True)
        logger.info(f"Run statistics: {msg}")
        logger.info(f"Adding result to log file: {log_path}")
    

def par_for_psdd(name_ls, thread_ls, log_path, openmp_prefix, suffix):
  line_number= 8
  run_log= open(log_path, 'a+')

  cmd= "cd src/openmp/; make set_env"
  os.system(cmd)
  for net in name_ls:
    for th in thread_ls:
      data_path= f"{openmp_prefix}{net}_{suffix}_{th}.c" 
      data_path = data_path.replace('/', '\/')
      openmp_main_file= "./src/openmp/par_for_v2.cpp"
      cmd= "sed -i '8s/.*/#include \"" + data_path + f"\"/' {openmp_main_file}"
      logger.info(f"Modifying main openmp file: {openmp_main_file} to include the header file {data_path}")
      print(cmd)
      os.system(cmd)
      cmd= "cd src/openmp; make normal_cpp_psdd"
      err= os.system(cmd)
      if err:
        print(f"Error in compilation {net}, {th}")
        print(f"{net},{th},Error compilation", file= run_log, flush= True)
      else:
        logger.info("Excuting 10k iterations of parallel code...")
        cmd= "cd src/openmp; make run_psdd"
        output= subprocess.check_output(cmd, shell=True)
        # os.system(cmd)
        output = str(output)
        output = output[:-3]
        output= output[output.find('N_layers'):]
        msg= f"{net},{th},{output}"
        print(msg, file= run_log, flush= True)
        logger.info(f"Run statistics: {msg}")
        logger.info(f"Adding result to log file: {log_path}")

