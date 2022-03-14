
import pickle
import queue
import networkx as nx
print(nx.__file__)
import math
from statistics import mean, stdev, mode, median
import statistics
import logging
from enum import Enum
import time

from ..useful_methods import printlog, printcol
from .. import useful_methods
from . import instr_types
from . import minizinc_top
from . import gurobi_top


logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CompileConfig():
  def __init__(self, name= None, N_PE= None, GLOBAL_MEM_DEPTH= None, LOCAL_MEM_DEPTH= None, 
                STREAM_LD_BANK_DEPTH=None, STREAM_ST_BANK_DEPTH= None, STREAM_INSTR_BANK_DEPTH= None,
                partition_mode= None, sub_partition_mode= None, partition_len_heuristic= None, 
                graph_mode= None,
                target_app= None, target_device= None, 
                run_mode= None, write_files= None, global_var= None):
    
    self.global_var= global_var
    self.name= name
    self.hw_details= hw_details_class(N_PE, GLOBAL_MEM_DEPTH, LOCAL_MEM_DEPTH, \
          STREAM_LD_BANK_DEPTH, STREAM_ST_BANK_DEPTH, STREAM_INSTR_BANK_DEPTH,
        )

    self.graph_mode_enum= Enum('GRAPH_MODE', 'COARSE, FINE', module=__name__)
    if graph_mode == None:
      self.graph_mode= None
    else:
      self.graph_mode= self.graph_mode_enum[graph_mode]

    self.partition_mode_enum= Enum('PARTITION_MODE', 'HEURISTIC, TWO_WAY_PARTITION, LAYER_WISE', module=__name__)
    if partition_mode == None:
      self.partition_mode= None
    else:
      self.partition_mode= self.partition_mode_enum[partition_mode]
    
    self.sub_partition_mode_enum= Enum('SUB_PARTITION_MODE', 'TWO_WAY_FULL, TWO_WAY_LIMIT_LAYERS, ALAP, ASAP',  module=__name__)
    if sub_partition_mode == None:
      self.sub_partition_mode= None
    else:
      self.sub_partition_mode= self.sub_partition_mode_enum[sub_partition_mode]
    
    self.partition_len_heuristic= self.partition_len_heuristic_1

    self.run_mode_enum= Enum('RUN_MODE', 'FULL, RESUME', module=__name__)
    if run_mode == None:
      self.run_mode= None
    else:
      self.run_mode= self.run_mode_enum[run_mode]

    self.target_device_enum= Enum('TARGET_DEVICE', 'CPU, GPU, PRU', module= __name__)
    if target_device == None:
      self.target_device= None
    else:
      self.target_device= self.target_device_enum[target_device]

    self.target_app_enum= Enum('TARGET_APP', 'SPARSE_TR_SOLVE, SPN', module= __name__)
    self.targe_app= self.target_app_enum.SPARSE_TR_SOLVE

    self.write_files= write_files

    self.global_time_out= 36000
    
  def partition_len_heuristic_1(self, lengths):
    l_mean= int(mean(lengths))
    if len(lengths) > 2:
      l_median= int(median(lengths))
    else:
      l_median= min(lengths)

    part_len= min(l_mean, l_median)
    if self.target_device == self.target_device_enum.PRU:
      part_len= max(10, part_len)
    else:
      part_len= max(500, part_len)

    # logger.info(f'part_len: {part_len}, mean: {l_mean}, median: {l_median}')
    
    return part_len

  def partition_len_heuristic_2(self, lengths):
    return max(lengths)

  def get_partitions_file_name(self):
    prefix= self.global_var.PARTITIONS_PATH
    path = prefix + self.common_suffix() + ".p"
    return path

  def get_openmp_file_name(self):
    prefix= self.global_var.OPENMP_PATH
    path = prefix + self.common_suffix() + ".c"
    return path

  def common_suffix(self):
    path = self.name.replace('/','_') 
    path += f"_{self.partition_mode.name}"
    path += f"_{self.sub_partition_mode.name}"
    path += f"_{self.target_device.name}"
    if self.graph_mode != None:
      path += f"_{self.graph_mode.name}"
    path += f"_{self.hw_details.N_PE}"

    return path



def output_file_name_spn(global_var, net, partition_mode, n_threads):
  prefix= global_var.NO_BACKUP_PATH
  if partition_mode == 'TWO_WAY_PARTITION':
    prefix += 'two_way_partition_equal_parts/'
  elif partition_mode == 'HEURISTIC':
    prefix += 'heuristic_partitions/'
  elif partition_mode == 'LAYER_WISE':
    prefix += 'layer_wise_partition/'
  else:
    assert 0

  path= prefix + net + f'_{partition_mode}_{n_threads}.p'

  return path

class hw_details_class():
  def __init__(self, N_PE, GLOBAL_MEM_DEPTH, LOCAL_MEM_DEPTH, STREAM_LD_BANK_DEPTH=1024, STREAM_ST_BANK_DEPTH=512, STREAM_INSTR_BANK_DEPTH=1024):
    self.N_PE         = N_PE 
    self.REGBANK_L    =32 # words
    self.LD_PORT_SIZE =2  
    
    self.RESERVED_REG_IN_0 = 0
    self.RESERVED_REG_IN_1 = 1
    self.RESERVED_REG_OUT  = 2

    self.LOCAL_MEM_DEPTH= LOCAL_MEM_DEPTH # words
    self.GLOBAL_MEM_DEPTH= GLOBAL_MEM_DEPTH
    
    # self.STREAM_MAX_ADDR_L_PER_BANK= 16 # bits , indicates the size of each (instr, ld, or store) stream bank
    # self.STREAM_LD_BANK_DEPTH= 65536 # words
    # self.STREAM_ST_BANK_DEPTH= 65536 # words
    # self.STREAM_INSTR_BANK_DEPTH= 65536 # words
    self.STREAM_LD_BANK_DEPTH= STREAM_LD_BANK_DEPTH # words
    self.STREAM_ST_BANK_DEPTH= STREAM_ST_BANK_DEPTH # words
    self.STREAM_INSTR_BANK_DEPTH= STREAM_INSTR_BANK_DEPTH # words
    if STREAM_LD_BANK_DEPTH != None:
      self.STREAM_MAX_ADDR_L_PER_BANK= useful_methods.clog2(max(STREAM_LD_BANK_DEPTH, STREAM_ST_BANK_DEPTH, STREAM_INSTR_BANK_DEPTH)) # bits , indicates the size of each (instr, ld, or store) stream bank

    self.DTYPE= 'posit' # in ['default', 'posit', 'flt']

    # flt
    self.EXP_L= 7
    self.MANT_L= 22

    # posit
    self.POSIT_L= 32
    self.POSIT_ES= 6
#    self.POSIT_L= 16
#    self.POSIT_ES= 4
#    self.POSIT_L= 8
#    self.POSIT_ES= 2
    
    
class status_node():
  def __init__(self, node_key):
    self.node = node_key
    self.leaf= False

    self.pe_id= None

    self.to_be_stored= False
    self.store_in_local_mem= None
    self.store_in_global_mem= None
    
    self.bank= None
    self.pos= None

    # Using duirng partitioning only
    self.local_committed= False
    self.global_committed= False

  def is_committed(self, pe_id):
    return (self.is_local_committed(pe_id) or self.global_committed)

  def is_local_committed(self, pe_id):
    if self.local_committed:
      assert self.pe_id != None
      if self.pe_id == pe_id:
        return True
    
    return False

  def commit_locally(self, pe_id):
    self.local_committed= True
    self.pe_id= pe_id
  
  def commit_globally(self):
    self.global_committed = True   
  
  def create_copy(self):
    obj= status_node(self.node)
    obj.local_committed = self.local_committed
    obj.global_committed = self.global_committed
    obj.pe_id = self.pe_id

    return obj

  def is_leaf(self):
    return self.leaf
  
def is_schedulable(graph, node):
  obj= graph[node]

  SCHEDULABLE= True
  
  if obj.computed== False:
    for child in obj.child_key_list:
      if graph[child].computed== False:
        SCHEDULABLE= False
        break
  else:
    SCHEDULABLE= False

  return SCHEDULABLE


def print_statistics(graph, list_of_schedules, status_dict):
  print('Local barriers:')
  list_of_local_barriers= []
  for schedules in list_of_schedules:
    list_of_local_barriers.append([])
    for schedule in schedules:
      list_of_local_barriers[-1].append(len([instr for instr in schedule if instr.is_local_barrier()]))
  
  for i in range(len(list_of_schedules[0])):
    nodes_cnt= [len(schedules[i]) for schedules in list_of_schedules]
#    printcol('# nodes:           ' + str(nodes_cnt)[1:-1], 'blue')
    print(' # nodes:           ' + str(nodes_cnt)[1:-1])

    local_barrier= [barrier[i] for barrier in list_of_local_barriers]
    printcol('# local barriers : ' + str(local_barrier)[1:-1], 'green')
    
    printcol(' global barrier', 'red')
  
  all_instr= [instr for schedules in list_of_schedules for schedule in schedules for instr in schedule]
  ld_cnt= 0
  local_ld_cnt =0
  global_ld_cnt= 0
  loaded_nodes= []
  global_ld_that_can_be_avoided= 0
  leaf_ld= 0
  for instr in all_instr:
    if instr.to_load_0:
      ld_cnt += 1
      loaded_nodes.append(instr.load_0_node)
      if status_dict[instr.load_0_node].is_leaf():
        leaf_ld += 1
      if status_dict[instr.load_0_node].store_in_local_mem:
        local_ld_cnt += 1
      elif status_dict[instr.load_0_node].store_in_global_mem:
        global_ld_cnt += 1
        if instr.node != None:
          if (status_dict[instr.load_0_node].pe_id == status_dict[instr.node].pe_id):
            global_ld_that_can_be_avoided += 1
      else:
        assert 0

    if instr.to_load_1:
      loaded_nodes.append(instr.load_1_node)
      ld_cnt += 1
      if status_dict[instr.load_1_node].is_leaf():
        leaf_ld += 1
      if status_dict[instr.load_1_node].store_in_local_mem:
        local_ld_cnt += 1
      elif status_dict[instr.load_1_node].store_in_global_mem:
        global_ld_cnt += 1
        if instr.node != None:
          if (status_dict[instr.load_1_node].pe_id == status_dict[instr.node].pe_id):
            global_ld_that_can_be_avoided += 1
      else:
        assert 0

  store_global_mem_cnt= 0
  store_local_mem_cnt = 0
  for node, obj in list(status_dict.items()):
    if not graph[node].is_leaf():
      if obj.store_in_global_mem == True:
        store_global_mem_cnt += 1
      if obj.store_in_local_mem == True:
        store_local_mem_cnt += 1

  tot_store_cnt= len([node for node, obj in list(status_dict.items()) if obj.to_be_stored])

  printlog('Ld cnt: ' + str(ld_cnt))
  printlog('local ld cnt: ' + str(local_ld_cnt))
  printlog('leaf ld cnt: ' + str(leaf_ld))
  printlog('global ld cnt: ' + str(global_ld_cnt))
  printlog('global ld to avoid: ' + str(global_ld_that_can_be_avoided))

  fpath= './ld_st_stat_log.csv'
  len_arith_nodes= len([node for node, obj in graph.items() if not obj.is_leaf()])
  with open(fpath, "a+") as fp:
    print("{},{},{},{},{},{},{}".format(len_arith_nodes, ld_cnt, local_ld_cnt, global_ld_cnt,\
        tot_store_cnt, store_local_mem_cnt, store_global_mem_cnt) ,file=fp)
#  print sorted(loaded_nodes)

def init_status_dict(graph):
  status_dict= {}
  for node, obj in list(graph.items()):
    status_obj= status_node(node)
    
    if obj.is_leaf():
      status_obj.global_committed = True
      status_obj.leaf= True
      assert obj.computed == True
    
    status_dict[node]= status_obj

  return status_dict

def first_partition(graph, graph_nx, hw_details, status_dict):
  N_PE= hw_details.N_PE

  # Mark leaf nodes computed
  for obj in list(graph.values()):
    obj.computed= obj.is_leaf()
  
#  leaves= set([node for node, obj in graph.items() if obj.is_leaf()])
#  first_leaf= leaves.pop()
  
  schedulable_nodes= set([node for node, obj in list(graph.items()) if is_schedulable(graph, node)])
#  N_NODE_FOR_1_PE= math.ceil(float(len(schedulable_nodes))/N_PE)
  N_NODE_FOR_1_PE= (len(schedulable_nodes) + N_PE) // N_PE

  list_of_chosen_sets= [set() for _ in range(N_PE)]
  graph_nx_undirected= graph_nx.to_undirected()

  for set_idx in range(N_PE):
    # pick any node  and compute distance of the rest from this node
    first_node= list(schedulable_nodes)[0]

    path_len_dict= nx.algorithms.shortest_paths.shortest_path_length(graph_nx_undirected,first_node)
    
    sorted_list= sorted(list(schedulable_nodes), key= lambda x: path_len_dict[x])
    chosen_set= sorted_list[:N_NODE_FOR_1_PE]

#    print chosen_set

    schedulable_nodes -= set(chosen_set)
    
    list_of_chosen_sets[set_idx] = chosen_set

    if len(schedulable_nodes) == 0:
      break

#  print list_of_chosen_sets
  return list_of_chosen_sets, status_dict


def schedule_check(graph, status_dict, curr_node, target_pe):
  """
    Checks if immediate children are committed
  """

  for child in graph[curr_node].child_key_list:
    if not status_dict[child].is_committed(target_pe):
      return False
  
  return True
    
def trial_allocation(graph, status_dict, list_of_reachable_nodes, done_nodes, hw_details, mode= 'trial'):
  assert mode in ['trial']
  N_PE= hw_details.N_PE

  status_dict_copy= {key: obj.create_copy() for key, obj in list(status_dict.items())}
  
#  list_of_reachable_nodes_copy= [set(x) for x in list_of_reachable_nodes]
  done_nodes_copy= set(done_nodes)
  
  count= []

  for pe_id in range(N_PE):
    total_nodes= 0
    query_node_set= set(list_of_reachable_nodes[pe_id])
    
    while len(query_node_set) != 0:
      curr_node= query_node_set.pop()
      if schedule_check(graph, status_dict_copy, curr_node, pe_id) == True:
        status_dict_copy[curr_node].commit_locally(pe_id)
        query_node_set |= set(graph[curr_node].parent_key_list) - done_nodes_copy

        done_nodes_copy.add(curr_node)
        
        total_nodes += 1
    
    count.append(total_nodes)
  
  mean_val= mean(count)
  stdev_val =0
  if len(count) > 1:
    stdev_val= stdev(count)

  print('trial count: ', count)
  print('mean: ', mean_val)
#  return max(8, mean_val - stdev_val)
#  return max(8, mean_val + 10)
  return max(100, mean_val - 30)
#  return max(10, mode(count))
#  return min(count) + 5
#  return mean(count) + 3

def layer_wise_partition_ALAP(graph_nx, status_dict, config_obj, partition_leaves= False):
  hw_details= config_obj.hw_details
  N_PE= hw_details.N_PE

  leaf_nodes= useful_methods.get_leaves(graph_nx)

  if config_obj.graph_mode == config_obj.graph_mode_enum.COARSE:
    logger.warning("partition_leaves is being overriden, is set to True now")
    partition_leaves = True

  # compute_lvl statistics
  if partition_leaves:
    map_v_to_lvl= useful_methods.compute_lvl(graph_nx)
  else:
    internal_graph_nx= useful_methods.get_non_leaves_subgraph(graph_nx)
    map_v_to_lvl= useful_methods.compute_lvl(internal_graph_nx)

  max_lvl= max(list(map_v_to_lvl.values()))
  min_lvl= min(list(map_v_to_lvl.values()))
  assert min_lvl == 0

  map_lvl_to_v= {l:set() for l in range(max_lvl + 1)}
  for v, l in map_v_to_lvl.items():
    map_lvl_to_v[l].add(v)
  
  layer_sets= []    
  for l in sorted(list(map_lvl_to_v.keys())):
    layer_sets.append(map_lvl_to_v[l])
    assert len(map_lvl_to_v[l]) != 0, "Atleas one node per level, otherwise there are gaps in assigning levels"

  list_of_partitions= [[] for _ in range(N_PE)]
  mapped_nodes= set(leaf_nodes)

  # generate partitions from layer sets
  for curr_candidates in layer_sets:
    curr_partitions= [set() for _ in range(N_PE)]

    # limit on number of nodes that can be mapped to a PE
    assert len(curr_candidates) != 0
    per_pe_limit= (len(curr_candidates) + N_PE - 1) // N_PE

    # a histogram of predecessors that will help load balance during allocation
    predecessor_pe_list= [status_dict[p].pe_id for n in curr_candidates for p in graph_nx.predecessors(n) if status_dict[p].pe_id != None]
    predecessor_pe_histogram= [0 for _ in range(N_PE)]
    for pe in predecessor_pe_list:
      predecessor_pe_histogram[pe] += 1
    preferable_pe= sorted(list(range(N_PE)), key= lambda x : predecessor_pe_histogram[x])

    for n in curr_candidates:
      allocation_to_pe_in_a_layer(n, curr_partitions, graph_nx, status_dict, per_pe_limit, preferable_pe)
      mapped_nodes.add(n)
    
    for pe in range(N_PE):
      list_of_partitions[pe].append(curr_partitions[pe])
    
  logger.info(f"Max level: {max_lvl}")
  logger.info(f"Total nodes: {len(graph_nx)}")
  logger.info(f"Total number of layers: {len(list_of_partitions[0])}")
  logger.info(f"nodes_on_each_layer: {[len(s) for s in layer_sets]}")

  return list_of_partitions, layer_sets

def layer_wise_partition_ASAP(graph_nx, status_dict, config_obj, partition_leaves= False):
  hw_details= config_obj.hw_details
  N_PE= hw_details.N_PE
  
  if config_obj.graph_mode == config_obj.graph_mode_enum.COARSE:
    logger.warning("partition_leaves is being overriden, is set to True now")
    partition_leaves = True

  leaf_nodes= useful_methods.get_leaves(graph_nx)
  internal_nodes= useful_methods.get_non_leaves(graph_nx)

  mapped_nodes= set(leaf_nodes)
  list_of_partitions= [[] for _ in range(N_PE)]

  # assign leaves to a PE in staus dict
  # NOTE: remove this assignment at the end if partition_leaves== False
  chunk_size= (len(leaf_nodes) + N_PE -1 ) // N_PE
  chunked_leaves= [leaf_nodes[i:i + chunk_size] for i in range(0, len(leaf_nodes), chunk_size)]
  for pe, leaf_ls in enumerate(chunked_leaves):
    for n in leaf_ls:
      assert pe < N_PE
      status_dict[n].pe_id= pe
  
  if partition_leaves:
    for pe in range(N_PE):
      if pe < len(chunked_leaves):
        list_of_partitions[pe].append(set(chunked_leaves[pe]))
      else:
        list_of_partitions[pe].append(set())

  # statistics
  if partition_leaves:
    nodes_on_each_layer= [len(leaf_nodes)]
  else:
    nodes_on_each_layer= []

  curr_candidates= set(useful_methods.get_leaves(graph_nx.subgraph(internal_nodes)))
  while len(mapped_nodes) != len(graph_nx):
    curr_partitions= [set() for _ in range(N_PE)]
    next_candidates= set()

    # limit on number of nodes that can be mapped to a PE
    assert len(curr_candidates) != 0
    per_pe_limit= (len(curr_candidates) + N_PE - 1) // N_PE
    print(len(curr_candidates), per_pe_limit)

    # a histogram of predecessors that will help load balance during allocation
    predecessor_pe_list= [status_dict[p].pe_id for n in curr_candidates for p in graph_nx.predecessors(n)]
    predecessor_pe_histogram= [0 for _ in range(N_PE)]
    for pe in predecessor_pe_list:
      predecessor_pe_histogram[pe] += 1
    preferable_pe= sorted(list(range(N_PE)), key= lambda x : predecessor_pe_histogram[x])

    for n in curr_candidates:
      allocation_to_pe_in_a_layer(n, curr_partitions, graph_nx, status_dict, per_pe_limit, preferable_pe)
      next_candidates |= set(graph_nx.successors(n))
      mapped_nodes.add(n)
    
    nodes_per_pe= [len(part) for part in curr_partitions]
    nodes_on_each_layer.append(sum(nodes_per_pe))
    logger.info(f"Total nodes mapped: {sum(nodes_per_pe)}")
    logger.info(f"PE-wise distribution: {nodes_per_pe}")

    for pe in range(N_PE):
      list_of_partitions[pe].append(curr_partitions[pe])
    
    # chosse schedulable candidates for the next iteration
    curr_candidates= set()
    for n in next_candidates:
      predecessors= set(graph_nx.predecessors(n))
      mapped_predecessors= predecessors & mapped_nodes
      if len(mapped_predecessors) == len(predecessors):
        curr_candidates.add(n)
  
  # Erase PEs from leaf_nodes
  if not partition_leaves:
    for n in leaf_nodes:
      status_dict[n].pe_id= None

  logger.info(f"Total nodes: {len(graph_nx)}")
  logger.info(f"Total number of layers: {len(list_of_partitions[0])}")
  logger.info(f"nodes_on_each_layer: {nodes_on_each_layer}")

  layer_sets= []
  for l in range(len(list_of_partitions[0])):
    curr_set= set()
    for pe in range(hw_details.N_PE):
      curr_set |= list_of_partitions[pe][l]
    layer_sets.append(curr_set)        

  # macs= [sum([len(list(graph_nx.in_edges(v))) + 1 for v in l]) for l in layer_sets]
  # logger.info(f"macs on each layer: {macs}")

  return list_of_partitions, layer_sets

def allocation_to_pe_in_a_layer(curr_node, curr_partitions, graph_nx, status_dict, per_pe_limit, preferable_pe):    
  predecessor_pe_list= [status_dict[p].pe_id for p in graph_nx.predecessors(curr_node) if status_dict[p].pe_id != None]

  pe_candidates= sorted(predecessor_pe_list, key= lambda x: predecessor_pe_list.count(x))
  pe_candidates= [pe for pe in pe_candidates if len(curr_partitions[pe]) <= per_pe_limit]

  if not pe_candidates: # all the interesting PEs are full
    pe_candidates= [pe for pe in preferable_pe if len(curr_partitions[pe]) < per_pe_limit]
    # print(preferable_pe)
    # print([len(curr_partitions[pe]) for pe in range(64)])
  
  # print(pe_candidates)
  chosen_pe= pe_candidates[0] # there should always be atlead one pe in this list
  
  curr_partitions[chosen_pe].add(curr_node)
  status_dict[curr_node].pe_id= chosen_pe

def heuristic_partition(graph, graph_nx, first_partition_list, status_dict, config_obj):
  hw_details= config_obj.hw_details
  N_PE= hw_details.N_PE
  BARRIER_PENALTY= 10
  CONFLICT_PROB= 0.1

  graph_nx_undirected= graph_nx.to_undirected()
  list_of_partitions= [[] for _ in range(N_PE)]
  
  list_of_reachable_nodes= [set(partition) for partition in first_partition_list]

  all_done= False
  
  done_nodes= set()
  
  total_nodes= 0
  counter= 0
  total_cycles= 0

  while all_done== False:
#    counter += 1
#    if counter == 6:
#      break

    # Do a trial_allocation for load balancing.
    # This makes sure that one PE is not keeping everyone else waiting
#    threshold_partition_len = len(graph)
    threshold_partition_len= trial_allocation(graph, status_dict, list_of_reachable_nodes, done_nodes, hw_details)

    for pe_id in range(N_PE):
      curr_partition= set()

      query_node_set= set(list_of_reachable_nodes[pe_id])
      
#      for node in partition:
#        status_dict[node].commit_locally(pe_id)
#        query_node_set |= set(graph[node].parent_key_list)
      
      while len(query_node_set) != 0:
        if len(curr_partition) > 1* threshold_partition_len:
          break
        curr_node= query_node_set.pop()
        if schedule_check(graph, status_dict, curr_node, pe_id) == True:
          if config_obj.partition_mode == config_obj.partition_mode_enum.HEURISTIC:
            status_dict[curr_node].commit_locally(pe_id)
            query_node_set |= set(graph[curr_node].parent_key_list) - done_nodes
          elif config_obj.partition_mode == config_obj.partition_mode_enum.LAYER_WISE:
            status_dict[curr_node].pe_id = pe_id
          else:
            assert 0

          list_of_reachable_nodes[pe_id] |= set(graph[curr_node].parent_key_list) - done_nodes

          assert not graph[curr_node].is_leaf()
          curr_partition.add(curr_node)
          list_of_reachable_nodes[pe_id].remove(curr_node) 
          
          assert curr_node not in done_nodes

          done_nodes.add(curr_node)
          
          total_nodes += 1
      
#      print curr_partition
      print(len(curr_partition), end=' ')
      list_of_partitions[pe_id].append(curr_partition)

      # remove curr_partition from every PE's reachable set, 
      # because some PEs might think that a node is reachable from it,
      # but it might actually be scheduled by other pE
      list_of_reachable_nodes = [reachable_nodes-curr_partition for reachable_nodes in list_of_reachable_nodes]
    
    print("")
    nodes_cnt_list= [len(x[-1]) for x in list_of_partitions]
    print("tot_nodes count: ", sum(nodes_cnt_list))
    cycle_cnt_for_this_barrier= max(nodes_cnt_list) + BARRIER_PENALTY
    cycle_cnt_for_this_barrier *= (1 + CONFLICT_PROB)
    total_cycles += cycle_cnt_for_this_barrier

    # for the final node
    for last_partition in list_of_partitions:
      last_partition = last_partition[-1]
      for node in last_partition:
        status_dict[node].commit_globally()
    
    ## For equipartition of nodes
    all_schedulable_nodes= set()
    for reachable_nodes in list_of_reachable_nodes:
      all_schedulable_nodes |= set([node for node in reachable_nodes if schedule_check(graph, status_dict, node, None)])
    
    len_each_partition= (len(all_schedulable_nodes)//N_PE) + 1

    printcol("len_each_partition: " + str(len_each_partition), "blue")
#    print all_schedulable_nodes
    
    alloted_nodes= set()
    non_alloted_nodes= set(all_schedulable_nodes)

    new_list_of_reachable_nodes= []
    for reachable_nodes in list_of_reachable_nodes:
      schedulable_nodes= set([node for node in reachable_nodes if schedule_check(graph, status_dict, node, None)])
#      print "schedulable_nodes pre", schedulable_nodes
      
      discarded_set = schedulable_nodes & alloted_nodes
      schedulable_nodes -= alloted_nodes
#      print "schedulable_nodes post", schedulable_nodes

      if len(schedulable_nodes) > len_each_partition:
        random_node= list(schedulable_nodes)[0]

        path_len_dict= nx.algorithms.shortest_paths.shortest_path_length(graph_nx_undirected, random_node)
        
        sorted_list= sorted(list(schedulable_nodes), key= lambda x: path_len_dict[x])
        chosen_set= set(sorted_list[:len_each_partition])

      else:
        new_nodes= non_alloted_nodes - schedulable_nodes

        if len(schedulable_nodes) > 0:
          random_node= list(schedulable_nodes)[0]
          path_len_dict= nx.algorithms.shortest_paths.shortest_path_length(graph_nx_undirected, random_node)
          sorted_list= sorted(list(new_nodes), key= lambda x: path_len_dict[x])
        else:
          sorted_list= list(new_nodes)

        chosen_set= set(sorted_list[: len_each_partition - len(schedulable_nodes)])
        chosen_set |= schedulable_nodes

#      print "chosen_set", chosen_set
#      print "discarded_set", discarded_set
      
#      printcol(len(chosen_set), "green")

      discarded_set |= schedulable_nodes - chosen_set
      alloted_nodes |= chosen_set
      non_alloted_nodes -= chosen_set
      
      new_list_of_reachable_nodes.append((reachable_nodes | chosen_set )- discarded_set)
      
    list_of_reachable_nodes= new_list_of_reachable_nodes
#    print alloted_nodes
    assert len(alloted_nodes) == len(all_schedulable_nodes), [len(alloted_nodes), len(all_schedulable_nodes)]

    ## Check if all nodes are done
    all_done= True
    for reachable_nodes in list_of_reachable_nodes:
      if len(reachable_nodes) != 0:
        all_done= False
        break
    
    ## All reachable nodes
    all_reachable_nodes= [node for reachable_nodes in list_of_reachable_nodes for node in reachable_nodes]
#    print all_reachable_nodes
    assert len(all_reachable_nodes) == len(set(all_reachable_nodes))

    printcol("Global barrier", "red")

  print("Total nodes computed: ", total_nodes)
  print("Total cycles: ", total_cycles)
  acceleration_factor= total_nodes/total_cycles
  print("Acceleration factor: " , acceleration_factor)
  print("Effective utilization: ", acceleration_factor/N_PE)
  
  assert total_nodes == len([x for x, obj in list(graph.items()) if not obj.is_leaf()])

  return list_of_partitions

def global_barriers(net, graph, graph_nx, node_w, config_obj):
  hw_details= config_obj.hw_details

  status_dict= init_status_dict(graph)

  mode= config_obj.partition_mode
  if mode == config_obj.partition_mode_enum.HEURISTIC:
    first_partition_list = first_partition(graph,graph_nx, hw_details, status_dict)
    list_of_partitions = heuristic_partition(graph, graph_nx, first_partition_list, status_dict, config_obj)

  elif mode == config_obj.partition_mode_enum.TWO_WAY_PARTITION:
    """
      The main optimization function that generates superlayer with recursive two=way partitions
    """
    # S1 technique: layer sets to help reduce graph size during two_way_partition
    status_dict_dummy= init_status_dict(graph)
    _, layer_sets= layer_wise_partition_ALAP(graph_nx, status_dict, config_obj)
    list_of_partitions, run_time = minizinc_top.two_way_partition_all_layers(net, graph_nx, node_w, status_dict, layer_sets, config_obj)
    # NOTE: CHANGE for disabling optimizations
    with open('./log/superlayer_gen_time_log', 'a+') as fp:
      print(f"network, {config_obj.name}, threads, {config_obj.hw_details.N_PE}, compile time (s), {run_time}, timeout, {config_obj.global_time_out}", file=fp, flush= True)

  elif mode == config_obj.partition_mode_enum.LAYER_WISE:
    # first_partition_list = first_partition(graph,graph_nx, hw_details, status_dict)
    # list_of_partitions = heuristic_partition(graph, graph_nx, first_partition_list, status_dict, config_obj)
    if config_obj.sub_partition_mode == config_obj.sub_partition_mode_enum.ALAP:
      list_of_partitions, _ = layer_wise_partition_ALAP(graph_nx, status_dict, config_obj)
    elif config_obj.sub_partition_mode == config_obj.sub_partition_mode_enum.ASAP:
      list_of_partitions, _ = layer_wise_partition_ASAP(graph_nx, status_dict, config_obj)
    else:
      assert 0

  else:
    assert 0

  return list_of_partitions, status_dict

def insert_set_ld_stream_len_instr(list_of_schedules):

  # list of lists
  # Each list per PE
  # Records the number of loads in between barriers (of any type)
  list_of_ld_stream_len= []
  
  for pe_id, pe_schedule in enumerate(list_of_schedules):
    list_of_ld_stream_len.append([])
    for schedule in pe_schedule:
      curr_ld_cnt= 0
      for instr in schedule:
        if instr.is_local_barrier():
          list_of_ld_stream_len[pe_id].append(curr_ld_cnt)
          curr_ld_cnt= 0
        
        if instr.to_load_0:
          curr_ld_cnt += 1

        if instr.to_load_1:
          curr_ld_cnt += 1

      list_of_ld_stream_len[pe_id].append(curr_ld_cnt) # global_barrier
      curr_ld_cnt= 0
  
  new_list_of_schedules= []

  for pe_id, pe_schedule in enumerate(list_of_schedules):
    new_list_of_schedules.append([])
    for schedule_id, schedule in enumerate(pe_schedule):
      new_list_of_schedules[pe_id].append([])
      
      # after global_barrier
      instr_obj = instr_types.full_instr()
      instr_obj.set_op('set_ld_stream_len')
      instr_obj.ld_stream_len= list_of_ld_stream_len[pe_id].pop(0)
      new_list_of_schedules[pe_id][schedule_id].append(instr_obj)

      for instr in schedule:
        new_list_of_schedules [pe_id] [schedule_id].append( instr )

        # after local barriers
        if instr.is_local_barrier():
          instr_obj = instr_types.full_instr()
          instr_obj.set_op('set_ld_stream_len')
          instr_obj.ld_stream_len= list_of_ld_stream_len[pe_id].pop(0)
          new_list_of_schedules[pe_id][schedule_id].append(instr_obj)
  
  for ld_stream_len in list_of_ld_stream_len:
    assert len(ld_stream_len) == 0

  return new_list_of_schedules

def combine_small_layers(graph_nx, list_of_partitions, thresh, node_w, config_obj):
  N_PE= len(list_of_partitions)
  assert N_PE != 0

  n_layers= len(list_of_partitions[0])
  assert n_layers != 0
  logger.info(f"layers before combining: {n_layers}")

  n_total_nodes= sum([len(list_of_partitions[pe][l]) for pe in range(N_PE) for l in range(len(list_of_partitions[0]))])

  list_of_combined_layer_list= []
  curr_layer_list= []
  for l in range(n_layers):
    # do combination of coarse nodes for LAYER_WISE
    if config_obj.partition_mode == config_obj.partition_mode_enum.LAYER_WISE:
      n_nodes= sum([sum([1 for n in list_of_partitions[pe][l]]) for pe in range(N_PE)])
    else:
      n_nodes= sum([sum([node_w[n] for n in list_of_partitions[pe][l]]) for pe in range(N_PE)])

    assert n_nodes != 0
    if n_nodes <= thresh: # combine
      curr_layer_list.append(l)
    else: # do not combine
      if len(curr_layer_list) > 0: # curr n_nodes exceed thresh but there are some layer from previous combining
        list_of_combined_layer_list.append(curr_layer_list)
        curr_layer_list= []
      curr_layer_list.append(l)
      list_of_combined_layer_list.append(curr_layer_list)
      curr_layer_list= []
  
  if len(curr_layer_list) != 0:
    list_of_combined_layer_list.append(curr_layer_list)

  logger.info(f"list_of_combined_layer_list: {list_of_combined_layer_list}")
  
  new_list_of_partitions= [[] for _ in range(N_PE)]
  for curr_layer_list in list_of_combined_layer_list:
    if len(curr_layer_list) == 1: # use original partitions as is
      for pe in range(N_PE):
        new_list_of_partitions[pe].append(list_of_partitions[pe][curr_layer_list[0]])
    else:
      # combine nodes in all the layers
      combined_nodes= set()
      for l in curr_layer_list:
        for pe in range(N_PE):
          combined_nodes |= list_of_partitions[pe][l]
      
      assert len(combined_nodes) != 0
      # just assign all nodes to pe=0
      for pe in range(N_PE):
        if pe == 0:
          new_list_of_partitions[0].append(combined_nodes)
        else:
          new_list_of_partitions[pe].append(set())
  
  n_total_nodes_new= sum([len(new_list_of_partitions[pe][l]) for pe in range(N_PE) for l in range(len(new_list_of_partitions[0]))])
  assert n_total_nodes_new == n_total_nodes, f"{n_total_nodes} {n_total_nodes_new}"

  n_layers_new= set([len(new_list_of_partitions[pe]) for pe in range(N_PE)])
  assert len(n_layers_new) == 1
  n_layers_new = n_layers_new.pop()

  assert n_layers_new <= n_layers
  logger.info(f"layers after combining: {n_layers_new}")

  return new_list_of_partitions

  
        

  

  
  

