from minizinc import Instance, Model, Solver
import time
import minizinc
import multiprocessing
import pickle
from collections import deque

from . import local_optimization
from . import partition
from ..useful_methods import get_leaves, printlog, printcol
from collections import defaultdict
from ..optimization.write_to_file import relabel_nodes_with_contiguous_numbers
import datetime
import random
from .. import useful_methods 
import itertools
import asyncio
import time
import logging
import statistics
from statistics import mean, stdev, mode, median

import networkx as nx
from ..reporting_tools import reporting_tools
from typing import Mapping, MutableMapping, MutableSequence, Sequence, Iterable, List, Set, Dict

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)

def dfs_traversal_recurse(graph_nx, curr_node, 
    node_list: List, diff_list: List, 
    curr_diff: MutableSequence, 
    done_node: MutableMapping,
    map_v_to_reverse_lvl: Mapping):
  
  if done_node[curr_node]:
    return
  
  curr_diff[0] += 1
  
  sorted_preds= sorted(graph_nx.predecessors(curr_node), key= lambda x: map_v_to_reverse_lvl[x], reverse= True)
  
  for pred in sorted_preds:
    dfs_traversal_recurse(graph_nx, pred, node_list, diff_list, curr_diff, done_node, map_v_to_reverse_lvl)

  node_list.append(curr_node)
  done_node[curr_node] = True
  diff_list.append(curr_diff[0])

  curr_diff[0] = 0
  return

def dfs_traversal(graph_nx, map_v_to_reverse_lvl: Mapping):

  curr_diff= 0
  done_nodes= set()
  node_list= []
  diff_list= []

  head_list= [n for n in graph_nx if len(list(graph_nx.successors(n))) == 0]
  head_list_copy = list(head_list) 

  # the head with the highest reverse level at the end of the list
  head_list= sorted(head_list, key= lambda x: map_v_to_reverse_lvl[x])
  assert len(head_list) != 0

  while len(head_list):    
    stack= []
    stack.append(head_list.pop()) # pops from the end of the list
    while stack:
      curr_node= stack[-1] # do not remove yet

      if curr_node in done_nodes:
        stack.pop()
        continue
        
      ch_ls= [ch for ch in graph_nx.predecessors(curr_node) if not ch in done_nodes]

      curr_diff += 1
      if len(ch_ls) == 0:
        # All ch are done already, so add to node_list
        # and pop from stack
        node_list.append(curr_node)
        assert curr_node not in done_nodes, f"{curr_node}"
        done_nodes.add(curr_node)
        diff_list.append(curr_diff)
        curr_diff = 0
        stack.pop()
      else: 
        # the ch with the highest reverse level added last to the stack
        ch_ls= sorted(ch_ls, key= lambda x: map_v_to_reverse_lvl[x])
        for ch in ch_ls:
          assert ch not in done_nodes
          stack.append(ch)
  
  assert len(node_list) == len(diff_list)
  assert len(node_list) == len(graph_nx), f"{len(node_list)}, {len(graph_nx)}"
  assert len(done_nodes)== len(graph_nx), f"{len(done_nodes)} ,{len(graph_nx)}"

  return node_list, diff_list

def create_chunks(node_list, diff_list, graph_nx, diff_threshold, chunk_len_threshold, out_degree_threshold):
  logger.info("Coarsening the graph")
  assert len(node_list) == len(diff_list)
  leaf_ls= useful_methods.get_leaves(graph_nx)

  chunks= []
  chunk= set([node_list[0]])
  for idx in range(1, len(node_list)):
    n= node_list[idx]
    d= diff_list[idx]
    new_chunk_cond = (d >= diff_threshold) 
    new_chunk_cond |= (len(chunk) >= chunk_len_threshold)
    new_chunk_cond |= (graph_nx.out_degree(n) >= out_degree_threshold)
    # new_chunk_cond |= (n in leaf_ls)
    if new_chunk_cond:
      chunks.append(chunk)
      chunk= set()

    chunk.add(n)
  chunks.append(chunk)
  
  return chunks

def create_coarse_graph(graph_nx, diff_threshold, chunk_len_threshold, out_degree_threshold, config_obj, start_idx):
#  graph_nx= useful_methods.get_non_leaves_subgraph(graph_nx)

#  dfs_topological_list= useful_methods.dfs_topological_sort(graph_nx, source_node= None, depth_limit= None)
#  print(dfs_topological_list)
#  chunks = [set(dfs_topological_list[i:i + chunk_len_threshold]) for i in range(0, len(dfs_topological_list), chunk_len_threshold)]
    
#  head= useful_methods.check_if_only_one_root(graph_nx)
  map_v_to_reverse_lvl= useful_methods.compute_reverse_lvl(graph_nx)
  # head_ls= useful_methods.get_head_ls(graph_nx)
  # node_list= []
  # diff_list= []
  # curr_diff= [0]
  # done_node= defaultdict(lambda: False)
  # for head in head_ls:
  #   dfs_traversal_recurse(graph_nx, head, node_list, diff_list, curr_diff, done_node, map_v_to_reverse_lvl)

  start= time.time()
  node_list, diff_list= dfs_traversal(graph_nx, map_v_to_reverse_lvl)
  logger.info(f"A: {time.time() - start}")

  start= time.time()

  chunks= create_chunks(node_list, diff_list, graph_nx, diff_threshold, chunk_len_threshold, out_degree_threshold)
  logger.info(f"B: {time.time() - start}")
  
  logger.info(f"Number of chunks: {len(chunks)}, number of nodes: {len(graph_nx)}")
  start= time.time()
  # nodes
  coarse_graph_nx= nx.DiGraph()
  map_coarse_node_to_set= defaultdict(set)
  map_node_to_coarse_node= {}
  node_attr_container= []
  for i, chunk in enumerate(chunks):
    chunk_id = i + start_idx
    # coarse_graph_nx.add_node(chunk_id, weight= len(chunk))
    node_attr_container.append((chunk_id, dict(weight = len(chunk))))
    map_coarse_node_to_set[chunk_id]= set(chunk)
    for n in chunk:
      map_node_to_coarse_node[n] = chunk_id
  coarse_graph_nx.add_nodes_from(node_attr_container)
    
  # edges
  edge_ls= []
  for e in graph_nx.edges():
    src=map_node_to_coarse_node[e[0]]
    dst=map_node_to_coarse_node[e[1]]
    if src != dst:
      edge_ls.append((src, dst))

  coarse_graph_nx.add_edges_from(edge_ls)
    
  # assertions
  assert min(list(coarse_graph_nx.nodes())) == start_idx
  assert max(list(coarse_graph_nx.nodes())) == start_idx + len(chunks) - 1
  assert sum([len(chunk) for chunk in map_coarse_node_to_set.values()]) == len(graph_nx)
  COSTLY_ASSERTION= False
  if COSTLY_ASSERTION:
    assert nx.algorithms.dag.is_directed_acyclic_graph(coarse_graph_nx)
    assert nx.algorithms.components.is_weakly_connected(coarse_graph_nx) == nx.algorithms.components.is_weakly_connected(graph_nx)

  # statistics
  if len(graph_nx) < 10000 and COSTLY_ASSERTION:
    before_len= nx.algorithms.dag.dag_longest_path_length(graph_nx)
    after_len= nx.algorithms.dag.dag_longest_path_length(coarse_graph_nx)
    logger.info(f"longest path len before: {before_len} after: {after_len}")

  logger.info(f"C: {time.time() - start}")
  logger.info(f"number of edges before: {graph_nx.number_of_edges()} after: {coarse_graph_nx.number_of_edges()}")

  return coarse_graph_nx, map_coarse_node_to_set, map_node_to_coarse_node

class Limit_layers_handling():
  def __init__(self, layer_sets):
    self.layer_sets= layer_sets
    self.mean_nodes_per_layer= mean([len(l) for l in self.layer_sets])
    self.l_ptr= 0
    
    self.MAX_NODES_LIMIT= 10_000
    self.MIN_NODES_LIMIT= 3*self.mean_nodes_per_layer
    self.MAX_LAYERS= 50
    self.limit= max(self.MIN_NODES_LIMIT, self.MAX_LAYERS*self.mean_nodes_per_layer)
    self.limit= min(self.MAX_NODES_LIMIT, self.limit)
    # self.limit= self.MAX_NODES_LIMIT
    # self.limit= 40_000

    logger.info(f"limit: {self.limit},  mean: {self.mean_nodes_per_layer}")

    # assert self.mean_nodes_per_layer <= self.MAX_NODES_LIMIT, f" {self.mean_nodes_per_layer}, {self.MAX_NODES_LIMIT}, Use TWO_WAY_FULL with coarsening instead of TWO_WAY_LIMIT_LAYERS mode because layers are too big"
  
  def append_new_layers(self, nodes_to_map, done_nodes):
    assert self.l_ptr <= len(self.layer_sets)

    if len(nodes_to_map) < self.limit:
      while len(nodes_to_map) < self.limit:
        if self.l_ptr == len(self.layer_sets):
          return
        nodes_to_map |= (self.layer_sets[self.l_ptr] - done_nodes)
        self.l_ptr += 1
    else:
      while len(nodes_to_map) > self.limit:
        l_ptr_minus_1= self.l_ptr - 1
        if len(nodes_to_map) > len(self.layer_sets[l_ptr_minus_1]):
          nodes_to_map -= self.layer_sets[l_ptr_minus_1]
          self.l_ptr -= 1
        else:
          break

    
def two_way_partition_all_layers(net, graph_nx, node_w, status_dict, layer_sets, config_obj):

  assert nx.algorithms.is_directed_acyclic_graph(graph_nx)
  hw_details= config_obj.hw_details
  N_PE= hw_details.N_PE
  done_set_0 = set()
  done_set_1 = set()
  leaf_set= set(get_leaves(graph_nx))
  internal_nodes= set(graph_nx.nodes()) - leaf_set
  
  if config_obj.sub_partition_mode == config_obj.sub_partition_mode_enum.TWO_WAY_FULL:
    if config_obj.graph_mode == config_obj.graph_mode_enum.FINE:
      nodes_to_map= set(internal_nodes)
    elif config_obj.graph_mode == config_obj.graph_mode_enum.COARSE:
      nodes_to_map= set(graph_nx.nodes())
    else:
      assert 0

  elif config_obj.sub_partition_mode == config_obj.sub_partition_mode_enum.TWO_WAY_LIMIT_LAYERS:
    assert layer_sets !=  None
    limit_layers_obj=  Limit_layers_handling(layer_sets)
    nodes_to_map= set()
    limit_layers_obj.append_new_layers(nodes_to_map, done_nodes= set())
  else:
    assert 0

  done_sets= [set() for _ in range(N_PE)]
  mapped_count = 1

  # create model
  prefix= "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/optimization/minizinc_code/code/async/two_way_partition/"
  model_path= prefix + "two_way_partition.mzn"
  two_way_partition_fine = Model(model_path)

  model_path= prefix + "two_way_partition_coarse.mzn"
  two_way_partition_coarse = Model(model_path)

  
  list_of_partitions= [[] for _ in range(N_PE)]

  mapped_count_list= []

  if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
    done_nodes= set(leaf_set)
    how_many_to_map= len(internal_nodes)
  elif config_obj.graph_mode== config_obj.graph_mode_enum.COARSE:
    done_nodes= set()
    how_many_to_map= len(graph_nx)
  else:
    assert 0


  start_time = time.time()
  for _ in range(100000): # just a very large number of iterations
    if sum(mapped_count_list) >= how_many_to_map:
      break

    do_equalize = True
    pe_tup= tuple(range(N_PE))
    # mapped_count, mapped_nodes, curr_partition= two_way_partition_one_layer(net, done_sets, nodes_to_map, graph_nx, hw_details, two_way_partition_fine, two_way_partition_coarse, config_obj)
    # mapped_count, mapped_nodes, curr_partition= two_way_partition_one_layer_non_binary(net, done_sets, tuple(range(N_PE)),nodes_to_map, graph_nx, hw_details, two_way_partition_fine, two_way_partition_coarse, config_obj)

    curr_pred= set([p for n in nodes_to_map for p in graph_nx.predecessors(n)])
    curr_done_sets= [curr_set & curr_pred for curr_set in done_sets]

    running_avg_obj = RunningAverage(get_w(nodes_to_map, node_w)/N_PE, N_PE)
    layer_parts= {pe:set() for pe in pe_tup}
    tried_partitioning= set()
    mapped_count, mapped_nodes, curr_partition= partition_considering_connectivity(net, curr_done_sets, done_nodes, pe_tup, nodes_to_map, graph_nx, node_w, two_way_partition_fine, two_way_partition_coarse, do_equalize, running_avg_obj, layer_parts, tried_partitioning, config_obj)

    done_sets= [curr_set | curr_partition[i] for i, curr_set in enumerate(done_sets)]
    done_nodes |= mapped_nodes

    # remove nodes from done_sets whose all the successors are computed
    new_done_sets= []
    for curr_set in done_sets:
      new_curr_set= set([])
      for n in curr_set:
        unmapped_succ= [s for s in graph_nx.successors(n) if s not in done_nodes]
        if len(unmapped_succ) != 0:
          new_curr_set.add(n)
      new_done_sets.append(new_curr_set)
    done_sets = new_done_sets

    logger.info("Total nodes mapped in curr layer: {}".format(mapped_count))
    logger.info("Total operations in curr layer: {}".format(get_w(mapped_nodes, node_w)))
    nodes_to_map -= mapped_nodes
    if config_obj.sub_partition_mode == config_obj.sub_partition_mode_enum.TWO_WAY_LIMIT_LAYERS:
      # update limit depending on the last allocation
      # TODO: could use a running average here
      # limit_layers_obj.limit = max(limit_layers_obj.MIN_NODES_LIMIT, 4*mapped_count)
      limit_layers_obj.limit = max(500, 4*mapped_count)
      limit_layers_obj.append_new_layers(nodes_to_map, done_nodes)
    mapped_count_list.append(mapped_count)

    logger.info(f"Total mapped: {mapped_count_list}")
    logger.info(f"To be mapped: {sum(mapped_count_list) - how_many_to_map}")
    for pe in range(N_PE):
      list_of_partitions[pe].append(curr_partition[pe])
      for n in curr_partition[pe]:
        status_dict[n].pe_id = pe

    run_time= time.time() - start_time
    if run_time > config_obj.global_time_out:
      logger.warning(f"global_time_out exceeded, {run_time}")
      print(f"global_time_out exceeded, {run_time}")
      with open('./no_backup/run_time_log', 'a+') as fp:
        print(f"network, {config_obj.name}, threads, {config_obj.hw_details.N_PE}, run_time (s), {run_time} TIMEOUT, timeout, {config_obj.global_time_out}", file=fp, flush= True)
      exit(1)

  print(mapped_count_list)
  assert sum(mapped_count_list) == how_many_to_map, "{} {}".format(sum(mapped_count_list), how_many_to_map)
  assert len(done_nodes) == len(graph_nx)

  return list_of_partitions, run_time

def map_one_node_per_pe(nodes_to_map, schedulable_leaves, n_parallel_pe, config_obj):
  """
    n_parallel_pe may be < N_PE, if we are mapping to a subset of PEs
  """

  assert len(schedulable_leaves) <= n_parallel_pe

def distribution_heuristic(n_total_nodes, leaves, component, map_done_n_to_pe, ):
  inputs= set([i for n in target_part for i in graph_nx.predecessors(n)])
  chosen_pe= max(target_indices, key= lambda x: len(inputs & done_sets[x]))
  pass

class RunningAverage():
  def __init__(self, expected_avg, N_PE):
    self.running_avg= expected_avg
    self.N_PE= N_PE
  def update_avg(self, n_curr_pes, tot_nodes_mapped):
    logger.info(f"Updating running average: old: {self.running_avg}, n_curr_pes: {n_curr_pes}, tot_nodes_mapped: {tot_nodes_mapped}")
    curr_avg= tot_nodes_mapped/n_curr_pes
    a= 1.0/self.N_PE * n_curr_pes
    b= 1.0/self.N_PE * (self.N_PE - n_curr_pes)
    self.running_avg = b * self.running_avg + a *curr_avg
    logger.info(f"new average: {self.running_avg}")
  
  def lower_threshold(self):
    return 0.8 * self.running_avg

  def upper_threshold(self):
    return 1.2 * self.running_avg

def partition_leaves(pe_tup, leaves, graph_nx, node_w, done_sets):
  layer_parts= {pe:set() for pe in pe_tup}
  avg_nodes_per_pe= (get_w(leaves, node_w) + len(pe_tup) - 1) // len(pe_tup)

  remaining_pes= set(pe_tup)
  for l in leaves:
    inputs= set(graph_nx.predecessors(l))
    chosen_pe= max(remaining_pes, key= lambda x: len(inputs & done_sets[x]))

    layer_parts[chosen_pe].add(l)
    if get_w(layer_parts[chosen_pe], node_w) >= avg_nodes_per_pe:
      remaining_pes.remove(chosen_pe)

  assert sum([len(p) for p in layer_parts.values()]) == len(leaves)
  return layer_parts

def get_w(node_s,  node_w):
  """
    node_s can be a set, list or a single node
  """
  if isinstance(node_s, int):
    return node_w[node_s]
  else:
    return sum([node_w[n] for n in node_s])

def partition_considering_connectivity(net, done_sets, done_nodes, pe_tup, nodes_to_map, graph_nx, node_w, model_fine, model_coarse, do_equalize, running_avg_obj, layer_parts, tried_partitioning, config_obj):
  n_parallel_pe= len(pe_tup)

  # NOTE: CHANGE for disabling optimizations
  DO_NOT_CONSIDER_COMPONENTS = False
  
  leaves= set([])
  for n in nodes_to_map:
    unmapped_pred= [s for s in graph_nx.predecessors(n) if s not in done_nodes]
    if len(unmapped_pred) == 0:
      leaves.add(n)

  local_layer_parts= {pe:set() for pe in pe_tup}

  # partition based on weakly connected components
  components= list(nx.weakly_connected_components(graph_nx.subgraph(nodes_to_map)))

  components= sorted(components, key = lambda x: get_w(x, node_w), reverse= True)
  logger.info(f"component len: {[len(component) for component in components]}")

  remaining_pes= set(pe_tup)
  curr_nodes_to_map= set()
  for idx, component in enumerate(components):
    curr_nodes_to_map |= component
    curr_w= get_w(curr_nodes_to_map, node_w)

    if (curr_w >= running_avg_obj.lower_threshold()) or \
       (idx == (len(components) - 1)):
      inputs= set([i for n in curr_nodes_to_map for i in graph_nx.predecessors(n)])

      # sort according to two criteria: 1) number of edges coming from that PE, 2) curr nodes mapped to that PE
      sorted_pes= sorted(remaining_pes, key= lambda x: len(inputs & done_sets[x]) - 2*len(layer_parts[x]), reverse= True)

      # map to a single PE
      if curr_w <= running_avg_obj.upper_threshold():
        chosen_pes = sorted_pes[:1]

      # map to multiple PEs via two_way_partition
      else:
        n_chosen_pes= max(1, 1 +  int(curr_w// running_avg_obj.running_avg))
        chosen_pes = sorted_pes[ : n_chosen_pes]

      # NOTE: CHANGE for disabling optimizations
      if DO_NOT_CONSIDER_COMPONENTS:
        chosen_pes = list(pe_tup)
        n_chosen_pes= len(chosen_pes)
        curr_nodes_to_map= nodes_to_map

      if len(chosen_pes) == 0: # only 0 pe chosen
        logger.info(f"No PEs available to map {len(curr_nodes_to_map)} nodes. Unbalanced distribution")

      elif len(chosen_pes) == 1: # only 1 pe chosen
        logger.info(f"Mapping {len(curr_nodes_to_map)} nodes to a single PE: {chosen_pes[0]}")
        layer_parts[chosen_pes[0]] |= curr_nodes_to_map
        local_layer_parts[chosen_pes[0]] |= curr_nodes_to_map

      else: # multiple pes remaining
        logger.info(f"Mapping {len(curr_nodes_to_map)} nodes to a multiple PEs: {chosen_pes}")
        _ , _ , curr_partition= two_way_partition_one_layer_non_binary(net, done_sets, done_nodes, tuple(chosen_pes), 
            curr_nodes_to_map, graph_nx, node_w, model_fine, model_coarse, False, running_avg_obj, layer_parts, tried_partitioning, config_obj, DO_NOT_CONSIDER_COMPONENTS)

        for pe in chosen_pes:
          layer_parts[pe] |= curr_partition[pe]
          local_layer_parts[pe] |= curr_partition[pe]
      
      # every recursive call of this function creates a new layer_parts, 
      # hence we may get smaller values for tot_part_len,
      # to avoid that , we only update running_avg for the top call
      # if do_equalize:
      tot_part_len= sum([get_w(layer_parts[pe], node_w) for pe in chosen_pes])
      running_avg_obj.update_avg(len(chosen_pes), tot_part_len)

      # Do not remove any PE with the new heurisitc
      # for pe in chosen_pes:
      #   remaining_pes.remove(pe)
      curr_nodes_to_map = set()

      # NOTE: CHANGE for disabling optimizations
      if DO_NOT_CONSIDER_COMPONENTS:
        break


  # equalize_all parts
  if do_equalize:
    # shuffle to reduce global edges      
    layer_parts= reshuffle_to_increase_local_edges(graph_nx, layer_parts, done_sets, done_nodes)

    lengths= [len(layer_parts[pe]) for pe in pe_tup]
    print(f"lenghts before equalizing: {lengths}")
    logger.info(f"lenghts before equalizing: {lengths}")
    layer_parts= equalize_parts_redistribute(graph_nx, node_w, layer_parts, done_nodes, done_sets, model_fine, model_coarse, tried_partitioning, config_obj)

    # check if simple leaves partitioning is better
    all_len= sum([get_w(p, node_w) for p in layer_parts.values()]) 
    if all_len < 0.8 * get_w(leaves, node_w):
      logger.info(f"Resorting to simple leaves partition instead of two_way_partition: {all_len}, {len(leaves)}")
      layer_parts = partition_leaves(pe_tup, leaves, graph_nx, node_w, done_sets)

    # shuffle to reduce global edges      
    layer_parts= reshuffle_to_increase_local_edges(graph_nx, layer_parts, done_sets, done_nodes)

  all_union= set()
  all_len= 0
  for part in layer_parts.values():
    assert part != None
    all_union |= part
    all_len += len(part)
  assert len(all_union) == all_len

  lengths= [len(layer_parts[pe]) for pe in pe_tup]
  print(f"lenghts: {lengths}")
  logger.info(f"lenghts: {lengths}")
  print(f"tot_nodes mapped: {all_len}")
  logger.info(f"tot_nodes mapped: {all_len}")

  op_lengths= [get_w(layer_parts[pe], node_w) for pe in pe_tup]
  print(f"operation lenghts: {op_lengths}")
  logger.info(f"operation lenghts: {op_lengths}")

  return all_len, all_union, layer_parts

def reshuffle_to_increase_local_edges(graph_nx, layer_parts, done_sets, done_nodes):
  """
    runtime quadratic in number of PEs
  """
  pe_tup= tuple(list(layer_parts.keys()))

  map_part_to_inputs= {}
  for pe, part in layer_parts.items():
    inputs= set([i for n in part for i in graph_nx.predecessors(n) if i in done_nodes])
    map_part_to_inputs[pe] = inputs

  # earlier pe is used as a proxy to "part" here
  map_part_to_local_edge_cnt= {}
  for pe, inputs in map_part_to_inputs.items():
    for new_pe in pe_tup:
      map_part_to_local_edge_cnt[(pe, new_pe)]=  len(inputs & done_sets[new_pe]) 

  map_part_to_local_edge_cnt_copy= dict(map_part_to_local_edge_cnt)
  # graph matching
  G= nx.Graph()
  edge_weights= {}
  for pe in pe_tup:
    G.add_node(f"o{pe}")
    G.add_node(f"n{pe}")
  for o_pe in pe_tup:
    o_name= f"o{o_pe}"
    for n_pe in pe_tup:
      n_name= f"n{n_pe}"
      G.add_edge(o_name, n_name, weight= -1 * map_part_to_local_edge_cnt[(o_pe, n_pe)])

  assert G.number_of_nodes() == 2*len(pe_tup)
  assert G.number_of_edges() == len(pe_tup) ** 2
  map_old_to_new_pe_graph= nx.bipartite.minimum_weight_full_matching(G, weight= 'weight')
  map_old_to_new_pe_graph_new= {}
  for name_0, name_1 in map_old_to_new_pe_graph.items():
    if name_0[0] == 'o':
      o_name= name_0
      n_name= name_1
    elif name_1[0] == 'o':
      o_name= name_1
      n_name= name_0
    else:
      assert 0
    map_old_to_new_pe_graph_new[int(o_name[1:])] = int(n_name[1:])
  map_old_to_new_pe_graph= map_old_to_new_pe_graph_new

  matching_weight_graph= sum([map_part_to_local_edge_cnt[o_pe, n_pe] for o_pe, n_pe in map_old_to_new_pe_graph.items()])
  assert len(map_old_to_new_pe_graph) == len(pe_tup), f"{len(pe_tup)}, {len(map_old_to_new_pe_graph)}, pe_tup"
  assert len(set(list(map_old_to_new_pe_graph.values()))) == len(pe_tup)

  tot_inputs= sum([len(inputs) for inputs in map_part_to_inputs.values()])
  logger.info(f"graph reshuffling: local edges:{matching_weight_graph} out of total incomping edges {tot_inputs}")

  new_layer_parts= {}
  for old_pe, new_pe in map_old_to_new_pe_graph.items():
    new_layer_parts[new_pe] = layer_parts[old_pe]

  assert len(new_layer_parts) == len(pe_tup)
  return new_layer_parts
  
def two_way_partition_one_layer_non_binary(net, done_sets, done_nodes, pe_tup_full, nodes_to_map_original, graph_nx, node_w, model_fine, model_coarse, do_equalize, running_avg_obj, layer_parts, tried_partitioning, config_obj, DO_NOT_CONSIDER_COMPONENTS= False):
  """
    two partitioning but the number of PEs are decided dynamically based on the ration of leaves
  """
  map_pe_list_to_nodes= {pe_tup_full : nodes_to_map_original}
  # layer_parts= {pe:None for pe in pe_tup_full}

  while len(map_pe_list_to_nodes) != 0:
    curr_map_pe_list_to_nodes= {}
    for pe_tup, nodes_to_map in map_pe_list_to_nodes.items():
      if len(pe_tup) == 1:
        pe= pe_tup[0]
        layer_parts[pe] |= nodes_to_map
        # if layer_parts[pe] != None:
        #   assert layer_parts[pe] == nodes_to_map
        # else:
        #   layer_parts[pe] |= nodes_to_map
        continue

      pe_indices_0 = pe_tup[: len(pe_tup)//2]
      pe_indices_1 = pe_tup[len(pe_tup)//2 : ]

      done_set_0 = [done_sets[pe] for pe in pe_indices_0]
      done_set_0 = set().union(*done_set_0)

      done_set_1 = [done_sets[pe] for pe in pe_indices_1]
      done_set_1 = set().union(*done_set_1)

      assert len(done_set_0 & done_set_1) == 0

      printcol(f'tot_nodes: {len(nodes_to_map)} among {len(pe_tup)} PEs', 'green')

      leaves= set([])
      for n in nodes_to_map:
        unmapped_pred= [s for s in graph_nx.predecessors(n) if s not in done_nodes]
        if len(unmapped_pred) == 0:
          leaves.add(n)

      print(f'tot leaves: {len(leaves)}')

      number_weakly_connected_components= nx.number_weakly_connected_components(graph_nx.subgraph(nodes_to_map))
      logger.info(f"number_weakly_connected_components : {number_weakly_connected_components}")

      # if only one connected component, go for optimization 
      if number_weakly_connected_components <= 1 or pe_tup == pe_tup_full or DO_NOT_CONSIDER_COMPONENTS:
        # curr_part_0, curr_part_1, result = two_way_partition_get_best_result(net, done_set_0, done_set_1, done_nodes, nodes_to_map, graph_nx, config_obj, model_fine, model_coarse)
        frozen= frozenset(nodes_to_map)
        tried_partitioning.add(frozen)
        
        if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
          local_opt_threshold= 700
        elif config_obj.graph_mode== config_obj.graph_mode_enum.COARSE:
          local_opt_threshold= 200
        else:
          assert 0

        # NOTE: CHANGE for disabling optimizations
        # local_opt_threshold = 0

        if len(nodes_to_map) > local_opt_threshold or len(leaves) <= 2:
          curr_part_0, curr_part_1, result = two_way_partition_get_best_result(net, done_set_0, done_set_1, done_nodes, nodes_to_map, graph_nx, node_w, config_obj, model_fine, model_coarse)
        else:
          logger.info("local_optimization")
          loc_opt_obj= local_optimization.Local_optimization_partition(nodes_to_map, graph_nx, done_set_0, done_set_1, done_nodes, node_w, config_obj)
          curr_part_0, curr_part_1 = loc_opt_obj.get_results()

        # reset pe_indices_0 and pe_indices_1 based on number of size of parts
        curr_part_0_w= get_w(curr_part_0, node_w)
        curr_part_1_w= get_w(curr_part_1, node_w)
        part_diff= abs( curr_part_0_w - curr_part_1_w)
        part_tot= curr_part_0_w + curr_part_1_w
        NODES_THRESHOLD= 6_00
        if ( 
          get_w(nodes_to_map, node_w) > NODES_THRESHOLD  and 
          part_tot > 20 and 
          part_diff > max(1, part_tot/len(pe_tup)) and 
          part_tot > 3*len(pe_tup)
          ):
          n_pe_0 = int(len(pe_tup) * curr_part_0_w/part_tot)
          n_pe_1 = len(pe_tup) - n_pe_0

          if n_pe_0 <= 0:
            n_pe_0 = 1
            n_pe_1 = len(pe_tup) - 1
          elif n_pe_1 <= 0:
            n_pe_1 = 1
            n_pe_0 = len(pe_tup) - 1

          pe_indices_0 = pe_tup[ : n_pe_0]
          pe_indices_1 = pe_tup[ n_pe_0 : ]

        # reset pe_indices_0 and pe_indices_1 based on number of leaf nodes
        leaves_0 = curr_part_0 & leaves
        leaves_1 = curr_part_1 & leaves
        logger.info(f'leaves distribution : {len(leaves_0)}, {len(leaves_1)}')

        if len(leaves) <= len(pe_tup):
          n_pe_0 = int(len(pe_tup) * len(leaves_0)/len(leaves))
          n_pe_1 = len(pe_tup) - n_pe_0
        elif len(leaves_0) < len(pe_indices_0):
          n_pe_0 = len(leaves_0)
          n_pe_1 = len(pe_tup) - n_pe_0
        elif len(leaves_1) < len(pe_indices_1):
          n_pe_1 = len(leaves_1)
          n_pe_0 = len(pe_tup) - n_pe_1
        else:
          n_pe_0 = len(pe_indices_0)
          n_pe_1 = len(pe_indices_1)

        # at least 1 pe in each partition
        if n_pe_0 <= 0:
          n_pe_0 = 1
          n_pe_1 = len(pe_tup) - 1
        elif n_pe_1 <= 0:
          n_pe_1 = 1
          n_pe_0 = len(pe_tup) - 1

        pe_indices_0 = pe_tup[ : n_pe_0]
        pe_indices_1 = pe_tup[ n_pe_0 : ]

        logger.info(f"n_pe_0 : {n_pe_0}, n_pe_1 : {n_pe_1}")
        assert len(pe_indices_0) != 0
        assert len(pe_indices_1) != 0

        running_avg_obj.update_avg(n_pe_0, curr_part_0_w)
        running_avg_obj.update_avg(n_pe_1, curr_part_1_w)

        process_output(graph_nx, curr_part_0, curr_part_1, pe_indices_0, pe_indices_1, layer_parts, curr_map_pe_list_to_nodes, done_sets, mode= 'non_binary')
      else: # more than one weakly connected component
        _, _, curr_partition= partition_considering_connectivity(net, done_sets, done_nodes, pe_tup, nodes_to_map, graph_nx, node_w,
                                  model_fine, model_coarse, False, running_avg_obj, layer_parts, tried_partitioning, config_obj)
        assert len(layer_parts) >= config_obj.hw_details.N_PE, "layer_parts has been tempered with during recursive call"
        for pe, part in curr_partition.items():
          curr_map_pe_list_to_nodes[tuple([pe])] = part

    printcol("Done iteration", 'red')
    map_pe_list_to_nodes = curr_map_pe_list_to_nodes
    
  # equalize_all parts
  if do_equalize:
    layer_parts= equalize_parts_redistribute(graph_nx, node_w, layer_parts, done_nodes, done_sets, model_fine, model_coarse, tried_partitioning, config_obj)

  all_union= set()
  all_len= 0
  for part in layer_parts.values():
    assert part != None
    all_union |= part
    all_len += len(part)
  assert len(all_union) == all_len

  lengths= [(pe,len(layer_parts[pe])) for pe in pe_tup_full]
  print(f"lenghts: {lengths}")
  logger.info(f"lenghts: {lengths}")
  print(f"tot_nodes mapped: {all_len}")
  logger.info(f"tot_nodes mapped: {all_len}")

  op_lengths= [get_w(layer_parts[pe], node_w) for pe in pe_tup]
  print(f"operation lenghts: {op_lengths}")
  logger.info(f"operation lenghts: {op_lengths}")

  return all_len, all_union, layer_parts

  

def two_way_partition_one_layer(net, done_sets, nodes_not_mapped_until_this_layer, graph_nx, node_w, hw_details, model_fine, model_coarse, config_obj):
  N_PE= hw_details.N_PE

  assert N_PE > 1
  assert useful_methods.isPowerOfTwo(N_PE)
  assert len(done_sets) == N_PE

  n_iter= useful_methods.clog2(N_PE)

  last_parts= [set(nodes_not_mapped_until_this_layer)]
  
  layer_parts= {pe:None for pe in range(N_PE)}

  for it in reversed(range(n_iter)): # log2 number of iterations
    curr_parts= []
    for inst, nodes_to_map in enumerate(last_parts): 
      done_set_len= pow(2, it)
      start_offset= inst * 2 * done_set_len
      start_0= start_offset
      end_0 = start_0 + done_set_len
      done_set_0= done_sets[start_0:end_0]
      done_set_0= set().union(*done_set_0)
      pe_indices_0= list(range(start_0, end_0))

      start_1= end_0
      end_1 = start_1 + done_set_len
      done_set_1= done_sets[start_1:end_1]
      done_set_1= set().union(*done_set_1)
      pe_indices_1= list(range(start_1, end_1))

      # early decision
      # layer_parts are already defined
      if layer_parts[start_0] != None:
        for pe in pe_indices_0 + pe_indices_1:
          assert layer_parts[pe] != None
        curr_parts.append(None)
        curr_parts.append(None)
        continue

      assert len(done_set_0 & done_set_1) == 0
      print('tot_nodes:', len(nodes_to_map))
      leaves= [n for n in nodes_to_map if len(set(graph_nx.predecessors(n)) & nodes_to_map) == 0]
      print(f'tot leaves: {len(leaves)}')
      curr_part_0, curr_part_1, result = two_way_partition_get_best_result(net, done_set_0, done_set_1, done_nodes, nodes_to_map, graph_nx, node_w, config_obj, model_fine, model_coarse)

      process_output(graph_nx, curr_part_0, curr_part_1, pe_indices_0, pe_indices_1, layer_parts, curr_parts, done_sets, mode= 'binary')

    printcol("Done iteration", 'red')
    last_parts = curr_parts

  for pe in range(N_PE):
    if layer_parts[pe] == None:
      assert last_parts[pe] != None 
      layer_parts[pe] = last_parts[pe]
    else:
      assert last_parts[pe] == None

  # equalize_all parts
  layer_parts= equalize_parts_truncate(graph_nx, node_w, layer_parts, config_obj.partition_len_heuristic)
  # layer_parts= equalize_parts_redistribute(graph_nx, node_w, layer_parts, done_nodes, done_sets, model_fine, model_coarse, config_obj)

  assert len(layer_parts) == N_PE
  all_union= set()
  all_len= 0
  for part in layer_parts.values():
    all_union |= part
    all_len += len(part)
  assert len(all_union) == all_len

  lengths= [len(layer_parts[pe]) for pe in range(N_PE)]
  print(f"lenghts: {lengths}")
  logger.info(f"lenghts: {lengths}")
  print(f"tot_nodes mapped: {all_len}")
  logger.info(f"tot_nodes mapped: {all_len}")

  return all_len, all_union, layer_parts

def process_output(graph_nx, curr_part_0, curr_part_1, pe_indices_0, pe_indices_1, layer_parts, curr_parts, done_sets, mode):
  """
    pe_indices_0 and pe_indices_1 contains list of indices that are target for curr_part_0 and curr_part_1
  """
  assert mode in ['binary', 'non_binary']

  assert curr_part_0 != None
  assert curr_part_1 != None

  # check if one of the part has 0 elements, 
  # if true the other part cannot be broken further,
  # and hence can be allocated to a single PE while assigning 0 nodes to all other PEs
  if len(curr_part_0) == 0:
    target_part = curr_part_1
    target_indices= pe_indices_1
    zero_indices= pe_indices_0
  elif len(curr_part_1) == 0:
    target_part = curr_part_0
    target_indices= pe_indices_0
    zero_indices= pe_indices_1
  else:
    target_part = None

  if target_part != None:
    # leaves= [n for n in target_part if len(set(graph_nx.predecessors(n)) & target_part) == 0]
    # print(leaves)
    # useful_methods.plot_graph(graph_nx.subgraph(target_part))
    # assert len(leaves) == 1, leaves

    # Use the pe that has the most inputs
    inputs= set([i for n in target_part for i in graph_nx.predecessors(n)])
    chosen_pe= max(target_indices, key= lambda x: len(inputs & done_sets[x]))
    if mode == 'binary':
      layer_parts[chosen_pe] = target_part
      for other_pe in target_indices + zero_indices:
        if other_pe != chosen_pe:
          layer_parts[other_pe] = set([])

      curr_parts.append(None)
      curr_parts.append(None)
    elif mode =='non_binary':
      layer_parts[chosen_pe] |= target_part
      curr_parts[tuple([chosen_pe])] = target_part
      for other_pe in target_indices + zero_indices:
        if other_pe != chosen_pe:
          layer_parts[other_pe] |= set([])
          curr_parts[tuple([other_pe])] = set()
    else:
      assert 0
  else:
    if mode == 'binary':
      curr_parts.append(curr_part_0)
      curr_parts.append(curr_part_1)
    elif mode =='non_binary':
      curr_parts[pe_indices_0] = curr_part_0
      curr_parts[pe_indices_1] = curr_part_1
    else:
      assert 0
    
def equalize_parts_truncate(graph_nx, node_w, all_parts, part_len_heuristic):
  equal_parts= []

  critical_path_len= 1
  tot_nodes= 0
  for part in all_parts.values():
    tot_nodes += len(part)
    sub_graph_nx= graph_nx.subgraph(part)
    curr_len= nx.algorithms.dag.dag_longest_path_length(sub_graph_nx)
    if curr_len > critical_path_len:
      critical_path_len = curr_len
  
  average_active_pe= max(1 , tot_nodes//critical_path_len)
  logger.info(f"Average active pe: {average_active_pe}, critical_path_len: {critical_path_len}, tot_nodes: {tot_nodes}")

  lengths= sorted([get_w(curr_set, node_w) for curr_set in all_parts.values()], reverse= True)
  lengths= lengths[:average_active_pe]

  part_len= part_len_heuristic(lengths)

  equal_parts = {}
  for pe, part in all_parts.items():
    sub_graph_nx= graph_nx.subgraph(part)
    # TODO
    # dfs_list= useful_methods.dfs_topological_sort(sub_graph_nx)
    dfs_list= list(nx.algorithms.dag.topological_sort(sub_graph_nx))
    
    new_part= set()
    curr_len= 0
    for n in dfs_list:
      new_part.add(n)
      curr_len += get_w(n, node_w)
      if curr_len >= part_len:
        break

    equal_parts[pe]= new_part
  
  return equal_parts

def get_leaves_as_per_done_nodes(graph_nx, nodes_to_map, done_nodes):
  leaves= set([])
  for n in nodes_to_map:
    unmapped_pred= [s for s in graph_nx.predecessors(n) if s not in done_nodes]
    if len(unmapped_pred) == 0:
      leaves.add(n)

  return leaves

def equalize_parts_redistribute(graph_nx, node_w, all_parts, done_nodes, done_sets, model_fine, model_coarse, tried_partitioning, config_obj):
  
  cannot_break_pe= set()

  # depending on leaves
  for pe, part in all_parts.items():
    leaves= get_leaves_as_per_done_nodes(graph_nx, part, done_nodes)
    if len(part) != 0:
      assert len(leaves) != 0

    if len(leaves) <= 2:
      cannot_break_pe.add(pe)

    if frozenset(part) in tried_partitioning:
      cannot_break_pe.add(pe)

  TRY_AGAIN= True
  while TRY_AGAIN:
    sorted_pes= sorted(list(all_parts.keys()), key = lambda x : get_w(all_parts[x], node_w), reverse= True)

    lengths= [get_w(curr_set, node_w) for curr_set in all_parts.values()]
    part_len= config_obj.partition_len_heuristic(lengths)

    min_pe= sorted_pes[-1]
    min_part_len= get_w(all_parts[min_pe], node_w)
    min_part= all_parts[min_pe]

    TRY_AGAIN= False
    for pe in sorted_pes:
      curr_part= all_parts[pe]
      curr_part_len= get_w(all_parts[pe], node_w)

      conditition = True
      conditition &= ((curr_part_len - min_part_len) > (1 + 0.3 * curr_part_len))
      conditition &= (pe not in cannot_break_pe)

      if conditition:
        pe_tup= tuple([pe, min_pe])
        layer_parts= {p : set() for p in pe_tup}
        running_avg_obj = RunningAverage((curr_part_len + min_part_len) / 2, 2)
        nodes_to_map= curr_part | min_part
        _, _, curr_partition= partition_considering_connectivity(None, done_sets, done_nodes, pe_tup, nodes_to_map, graph_nx, node_w,
                                  model_fine, model_coarse, False, running_avg_obj, layer_parts, tried_partitioning, config_obj)

        if min([get_w(part, node_w) for part in curr_partition.values()]) > min_part_len:
          all_parts[pe] = curr_partition[pe]
          all_parts[min_pe] = curr_partition[min_pe]
          TRY_AGAIN = True
          logger.info(f"Redistribution result: previous: {curr_part_len} {min_part_len}, after: {len(curr_partition[pe])}, {len(curr_partition[min_pe])}")
          break
        else:
          cannot_break_pe.add(pe)

  lengths= [get_w(all_parts[pe], node_w) for pe in all_parts.keys()]
  print(f"lenghts after redistributing: {lengths}")
  logger.info(f"lenghts after redistributing: {lengths}")
  # after redistribution, still truncate to shave off large parts
  all_parts= equalize_parts_truncate(graph_nx, node_w, all_parts, config_obj.partition_len_heuristic)
  
  return all_parts
  
def create_and_instantiate_model_parameters(sub_graph_nx, done_CU, edges_ls, model, invert_node_mapping, mode, config_obj, node_w):
  assert mode in ['fine', 'coarse']

#  to add extra constraints
#  model.add_string(extra_str)

  solver0 = Solver.lookup("or-tools")
#  solver = Solver.lookup("gecode")
  solver1 = Solver.lookup("gurobi")
#  solver = Solver.lookup("chuffed")

  inst0 = minizinc.Instance(solver0, model)
  inst1 = minizinc.Instance(solver1, model)

  N                = len(sub_graph_nx)
  n_CU             = 2
  predecessors     = [set(sub_graph_nx.predecessors(n)) for n in sorted((sub_graph_nx.nodes()))]
  one_predecessors = [set(list(sub_graph_nx.predecessors(n))[:1]) for n in sorted((sub_graph_nx.nodes()))]

  N_edges          = len(edges_ls)
  edges            = edges_ls

  N_done           = len(done_CU)
  done_CU          = [done_CU[n] for n in sorted(done_CU.keys())]

  inst0["N"]               = N               
  inst0["n_CU"]            = n_CU            
  inst0["predecessors"]    = predecessors    
  inst0["one_predecessors"]= one_predecessors
                                             
  inst0["N_edges"]         = N_edges         
  inst0["edges"]           = edges           
                                             
  inst0["N_done"]          = N_done          
  inst0["done_CU"]         = done_CU         

  inst1["N"]               = N               
  inst1["n_CU"]            = n_CU            
  inst1["predecessors"]    = predecessors    
  inst1["one_predecessors"]= one_predecessors
                                             
  inst1["N_edges"]         = N_edges         
  inst1["edges"]           = edges           
                                             
  inst1["N_done"]          = N_done          
  inst1["done_CU"]         = done_CU         

  if mode== 'coarse':
    # if (config_obj.targe_app == config_obj.target_app_enum.SPN or 
    #     config_obj.targe_app == config_obj.target_app_enum.SPARSE_TR_SOLVE):
    #   node_w = [len(invert_node_mapping[n]) for n in sorted(sub_graph_nx.nodes())]
    # elif config_obj.targe_app == config_obj.target_app_enum.SPARSE_TR_SOLVE:
    #   node_w = [sub_graph_nx.in_degree(n) + 1 for n in sorted(sub_graph_nx.nodes())]

    curr_node_w = [node_w[n] for n in sorted(sub_graph_nx.nodes())]
    max_node_w= max(curr_node_w)
    inst0["max_node_w"]   = max_node_w 
    inst1["max_node_w"]   = max_node_w
    inst0["node_w"]        = curr_node_w          
    inst1["node_w"]        = curr_node_w          

  return solver0, solver1, inst0, inst1

def create_single_instance(sub_graph_nx, done_CU, edges_ls, model, mode):

  # Find the MiniZinc solver configuration
#  solver = Solver.lookup("or-tools")
  solver = Solver.lookup("gecode")
#  solver = Solver.lookup("gurobi")
#  solver = Solver.lookup("chuffed")

  inst = Instance(solver, model)
#
#  # instantiate variables
#  inst["N"]= len(nodes_to_map)
#  inst["n_CU"]= 2
#  inst["predecessors"]= [set(sub_graph_nx.predecessors(n)) for n in sorted((sub_graph_nx.nodes()))]
#  inst["one_predecessors"]= [set(list(sub_graph_nx.predecessors(n))[:1]) for n in sorted((sub_graph_nx.nodes()))]
#
#  inst["N_edges"] = len(edges_ls)
#  inst["edges"] = edges_ls
#
#  inst["N_done"] = len(done_CU)
#  inst["done_CU"] = [done_CU[n] for n in sorted(done_CU.keys())]
#
#  timeout= datetime.timedelta(seconds= 500)
#  result = inst.solve(timeout= timeout, processes=24, verbose=True)

#  result= asyncio.get_event_loop().run_until_complete(multiple_solvers(len(nodes_to_map), two_way_partition))


def two_way_partition_get_best_result(net, done_set_0, done_set_1, done_nodes, nodes_to_map, graph_nx, node_w, config_obj, model_fine, model_coarse):  
  # NOTE: CHANGE for disabling optimizations
  # NODES_THRESHOLD= 1e25
  NODES_THRESHOLD= 1000

  if len(nodes_to_map) < NODES_THRESHOLD: # try fine mode once
    mode= 'fine'
    curr_part_0, curr_part_1, result= two_way_partition_one_instance(net, done_set_0, done_set_1, done_nodes, nodes_to_map, graph_nx, node_w, config_obj, mode, model_fine, model_coarse)

    if result != None :
      # NOTE: CHANGE for disabling optimizations
      if not (result.status== result.status.OPTIMAL_SOLUTION):
      # if False:
      # if True:
        mode= 'coarse'
        curr_part_0_coarse, curr_part_1_coarse, result_coarse=\
            two_way_partition_one_instance(net, done_set_0, done_set_1, done_nodes, nodes_to_map, graph_nx, node_w, config_obj, mode, model_fine, model_coarse)
        
        if result_coarse != None:
          if result_coarse["obj"] > result["obj"]:
            logger.info("Using coarse results")
            curr_part_0 = curr_part_0_coarse
            curr_part_1 = curr_part_1_coarse
            result = result_coarse

  else: # only try coarse
    mode= 'coarse'
    curr_part_0, curr_part_1, result=\
        two_way_partition_one_instance(net, done_set_0, done_set_1, done_nodes, nodes_to_map, graph_nx, node_w, config_obj, mode, model_fine, model_coarse)

  return curr_part_0, curr_part_1, result

def two_way_partition_one_instance(net, done_set_0, done_set_1, done_nodes, nodes_to_map, graph_nx, node_w, config_obj, mode, model_fine, model_coarse):  
  assert mode in ['fine', 'coarse']

  # early decisions
  if len(nodes_to_map) == 0:
    return set(), set(), None

  if len(nodes_to_map) < 10 and mode == 'fine':
    inputs= set([i for n in nodes_to_map for i in graph_nx.predecessors(n)])
    set_0_inputs= inputs & done_set_0
    set_1_inputs= inputs & done_set_1

    if len(set_0_inputs) >= len(set_1_inputs):
      return nodes_to_map, set(), None
    else:
      return set(), nodes_to_map, None

  assert len(done_set_0 & done_set_1) == 0
  
  start= time.time()
  full_pred_set= set([a for n in nodes_to_map for a in graph_nx.predecessors(n)])

  done_set_0_pred= full_pred_set.intersection(done_set_0)
  done_set_1_pred= full_pred_set.intersection(done_set_1)
  done_set_pred= done_set_0_pred | done_set_1_pred
  logger.info(f"preprocess A: {time.time() - start}, {len(done_set_0_pred), len(done_set_1_pred), len(done_set_0), len(done_set_1)}")

  start= time.time()
  start_idx_done_node= 1
  id_iter= itertools.count(start_idx_done_node)
  done_node_mapping= {n:next(id_iter) for n in done_set_pred}
  done_CU= {done_node_mapping[n]: 1 for n in done_set_0_pred}
  for n in done_set_1_pred:
    done_CU[done_node_mapping[n]] = 2
  logger.info(f"preprocess B: {time.time() - start}")

  sub_graph_nx = graph_nx.subgraph(nodes_to_map)
  if mode== 'fine': 
    sub_graph_nx, node_mapping = relabel_nodes_with_contiguous_numbers(sub_graph_nx, start=1)
    invert_node_mapping= {contiguous_idx: original_idx for original_idx, contiguous_idx in node_mapping.items()}

    assert len(node_mapping) == len(invert_node_mapping)

  elif mode== 'coarse': 
    if (config_obj.targe_app == config_obj.target_app_enum.SPN or 
        config_obj.targe_app == config_obj.target_app_enum.SPARSE_TR_SOLVE):
      NODES_NORMALIZE= 7_00
      diff_threshold = max(2, useful_methods.clog2(len(nodes_to_map)//NODES_NORMALIZE))
      if len(nodes_to_map) > 1500:
        chunk_len_threshold= max(4, 2**(diff_threshold+1))
      else:
        chunk_len_threshold= max(2, 2**(diff_threshold))

      assert diff_threshold > 0
      if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
        out_degree_threshold = 4
      elif config_obj.graph_mode== config_obj.graph_mode_enum.COARSE:
        out_degree_threshold = 4* chunk_len_threshold
        out_degree_threshold = max(out_degree_threshold, sub_graph_nx.number_of_edges()/5_000)
      else:
        assert 0
      logger.info(f'chunk_len_threshold: {chunk_len_threshold}, diff_threshold: {diff_threshold}')
      sub_graph_nx, map_coarse_node_to_set, map_node_to_coarse_node = create_coarse_graph(sub_graph_nx, diff_threshold, chunk_len_threshold, out_degree_threshold, config_obj, start_idx=1)

      logger.info(f'tot_nodes coarse: {len(sub_graph_nx)}')
      # leaves= [n for n in nodes_to_map if len(set(graph_nx.predecessors(n)) & nodes_to_map) == 0]
      # print(f'tot coarse leaves: {len(set([map_node_to_coarse_node[n] for n in leaves]))}')
      leaves= [n for n in sub_graph_nx if len(list(sub_graph_nx.predecessors(n))) == 0]
      logger.info(f'tot coarse leaves: {len(leaves)}')

      node_mapping= map_node_to_coarse_node
      invert_node_mapping= map_coarse_node_to_set


      # early decision
      if len(leaves) <= 1:
        inputs= set([i for n in nodes_to_map for i in graph_nx.predecessors(n)])
        set_0_inputs= inputs & done_set_0
        set_1_inputs= inputs & done_set_1

        if len(set_0_inputs) >= len(set_1_inputs):
          return nodes_to_map, set(), None
        else:
          return set(), nodes_to_map, None

    # elif config_obj.targe_app == config_obj.target_app_enum.SPARSE_TR_SOLVE:
    #   sub_graph_nx, node_mapping = relabel_nodes_with_contiguous_numbers(sub_graph_nx, start=1)
    #   invert_node_mapping= {contiguous_idx: set([original_idx]) for original_idx, contiguous_idx in node_mapping.items()}
    else:
      assert 0
  else:
    assert 0

  edges_ls= list(nx.algorithms.boundary.edge_boundary(graph_nx, done_set_pred, nodes_to_map))
  edges_ls= [[done_node_mapping[e[0]], node_mapping[e[1]]] for e in edges_ls]

  # if config_obj.targe_app != config_obj.target_app_enum.SPARSE_TR_SOLVE or mode == 'fine':
  if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
    assert len(edges_ls) <= 2*len(nodes_to_map)
# Extra assertions in case of problem with edges_ls
#  pred_set= set([p for n in nodes_to_map for p in graph_nx.predecessors(n)])
#  pred_set = (pred_set - nodes_to_map) & done_set_pred
#  edges_ls_for_verif= [e for p in pred_set for e in graph_nx.out_edges(p) if e[1] in nodes_to_map]
#  assert len(edges_ls) == len(edges_ls_for_verif)

  if mode== 'fine': 
    node_w_to_use= {n: node_w[invert_node_mapping[n]] for n in sub_graph_nx.nodes()}
    if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
      solver_threshold= 1000
      mode_to_use= 'fine'
      model_to_use= model_fine
      # node_w_to_use= node_w
    elif config_obj.graph_mode== config_obj.graph_mode_enum.COARSE:
      solver_threshold= 200
      mode_to_use= 'coarse'
      model_to_use= model_coarse
      # node_w_to_use= {n: node_w[n] for n in sub_graph_nx.nodes()}
    else:
      assert 0
    # loc_opt_obj= local_optimization.Local_optimization_partition(set(sub_graph_nx.nodes()), sub_graph_nx, done_set_0, done_set_1, done_nodes, node_w, config_obj)
  elif mode== 'coarse': 
    node_w_to_use= {n: sum([node_w[inv_n] for inv_n in invert_node_mapping[n]]) for n in sub_graph_nx.nodes()}
    if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
      solver_threshold= 1000
      mode_to_use= 'coarse'
      model_to_use= model_coarse
      # node_w_to_use = {n:len(invert_node_mapping[n]) for n in sub_graph_nx.nodes()}
    elif config_obj.graph_mode== config_obj.graph_mode_enum.COARSE:
      solver_threshold= 200
      mode_to_use= 'coarse'
      model_to_use= model_coarse
      # node_w_to_use= {n: sum([node_w[inv_n] for inv_n in invert_node_mapping[n]]) for n in sub_graph_nx.nodes()}
    else:
      assert 0

  # NOTE: CHANGE for disabling optimizations
  # solver_threshold = 0
  logger.warning(f"Threshold too high : {solver_threshold}")
  if len(sub_graph_nx) > solver_threshold:
    start= time.time()
    solver0, solver1, inst0, inst1= create_and_instantiate_model_parameters(sub_graph_nx, done_CU, edges_ls, model_to_use, invert_node_mapping, mode_to_use, config_obj, node_w_to_use)
    logger.info(f"model time: {time.time() - start}")
    start= time.time()
    result= asyncio.get_event_loop().run_until_complete(multiple_solvers(len(nodes_to_map), solver0, solver1, inst0, inst1, config_obj))
    logger.info(f"solve time: {time.time() - start}")
  else:
    start= time.time()
    logger.info("local_optimization")
    loc_opt_obj= local_optimization.Local_optimization_partition(set(sub_graph_nx.nodes()), sub_graph_nx, set(), set(), set(), node_w_to_use, config_obj)

    result = loc_opt_obj.get_minizinc_result()
    logger.info(f"solve time: {time.time() - start}")
    curr_part_0, curr_part_1 = post_process_results(sub_graph_nx, edges_ls, invert_node_mapping, result, mode, config_obj)
    return curr_part_0, curr_part_1, None

#  result= asyncio.run(multiple_solvers(len(nodes_to_map), two_way_partition))
#  for task in asyncio.Task.all_tasks():
#    task.cancel()

  curr_part_0, curr_part_1 = post_process_results(sub_graph_nx, edges_ls, invert_node_mapping, result, mode, config_obj)
  return curr_part_0, curr_part_1, result

def post_process_results(sub_graph_nx, edges_ls, invert_node_mapping, result, mode, config_obj):
  mapped_per_CU_active= result["mapped_per_CU_active"]
  obj= result["obj"]
  logger.info(mapped_per_CU_active)
  logger.info(obj)
  logger.info(f'tot_local_edges: {result["tot_local_edges"]} \
      out of N_edges: {len(edges_ls)} and total sug_graph_edges: {sub_graph_nx.number_of_edges()}')

  curr_CU_active= result["curr_CU_active"]

  if mode == 'fine':
    curr_part_0 = set([invert_node_mapping[n+1] for n, cu in enumerate(curr_CU_active)  if cu == 1 ])
    curr_part_1 = set([invert_node_mapping[n+1] for n, cu in enumerate(curr_CU_active)  if cu == 2 ])
  elif mode== 'coarse': 
    curr_part_0 = set([n for coarse_n, cu in enumerate(curr_CU_active) for n in invert_node_mapping[coarse_n+1] if cu == 1 ])
    curr_part_1 = set([n for coarse_n, cu in enumerate(curr_CU_active) for n in invert_node_mapping[coarse_n+1] if cu == 2 ])
  else:
    assert 0

  # assertions
  # the node_w does not match to the number of nodes in that coarse node in case of SPARSE_TR_SOLVE
  # if config_obj.targe_app != config_obj.target_app_enum.SPARSE_TR_SOLVE or mode == 'fine':
  #   assert len(curr_part_0) == mapped_per_CU_active[0]
  #   assert len(curr_part_1) == mapped_per_CU_active[1]
    assert len(curr_part_0) == mapped_per_CU_active[0]
    assert len(curr_part_1) == mapped_per_CU_active[1]
  assert len(curr_part_0 & curr_part_1) == 0

  # if curr_CU_active of a node is 0, then it's successors should also be 0
  for n in sub_graph_nx.nodes():
    for s in sub_graph_nx.successors(n):
      assert (curr_CU_active[s-1] == curr_CU_active[n-1]) or (curr_CU_active[s-1] == 0)

  properly_mapped= 0
  node_active=[]
  for n in sorted(list(sub_graph_nx.nodes())):
    parent_set= set([curr_CU_active[p-1] for p in sub_graph_nx.predecessors(n)])
    parent_set.add(curr_CU_active[n-1])
    parent_set = list(parent_set)
    if len(parent_set) == 1 and parent_set[0] != 0:
      properly_mapped += 1
      node_active.append(1)
    else:
      if len(parent_set) !=1 and len(list(sub_graph_nx.predecessors(n))) < 2:
        print('NOOOO', parent_set, n, list(sub_graph_nx.predecessors(n)), curr_CU_active[n-1])
      assert curr_CU_active[n-1] == 0
      node_active.append(0)

  if mode== 'fine':
    if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
      assert properly_mapped == sum(mapped_per_CU_active)
  
  assert properly_mapped == sum(result['node_active'])

  return curr_part_0, curr_part_1

async def multiple_solvers(net_size, solver0, solver1, inst0, inst1, config_obj):
  tasks= set()

  n_processes= max(1, int(1* multiprocessing.cpu_count()/2)) # decide number of parallel threads
#  n_processes= 4
  logger.info(f"n_processes : {n_processes}")

  # Create a task for the solving of each instance
  timeout_t= max(200, int(net_size * (24/n_processes) * 0.5))
  timeout_t= min(timeout_t, 2000)
  # NOTE: CHANGE for disabling optimizations
  # timeout_t= config_obj.global_time_out

#  timeout_t= max(200, int(net_size * (12/n_processes) * 0.05))
  logger.info(f"timeout_t : {timeout_t}")
  timeout= datetime.timedelta(seconds= timeout_t)
  task = asyncio.create_task(inst0.solve_async(timeout= timeout, processes=n_processes, verbose=True))
  task.solver = solver0.name
  tasks.add(task)

  # do not use Gurobi
  # if config_obj.graph_mode== config_obj.graph_mode_enum.FINE:
  #   timeout= datetime.timedelta(seconds= timeout_t + 10)
  #   task = asyncio.create_task(inst1.solve_async(timeout= timeout, processes=n_processes, verbose=True))
  #   task.solver = solver1.name
  #   tasks.add(task)

  done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

  for t in done:
    try:
      await t
      result= t.result()
    except Exception as e:
      logger.error("exception raised: {}".format(e))
      exit(1)

    print(t.solver, result.status, result.objective)
    if result.status == result.status.UNKNOWN:
      logger.error("No solution found! aborting")
      exit(1)
    if not result.status== result.status.OPTIMAL_SOLUTION:
      for t_p in pending:
        try:
          result_p= t_p.result()
        except Exception as e:
          continue
        print("Objectives: solver: {} {} , solver: {} {}".format(t.solver, result.objective, t_p.solver, result_p.objective))
        
        if result_p.objective > result.objective:
          result= result_p
          print("Using non-optimal solution from solver: {}".format(t_p.solver))
        else:
          print("Using non-optimal solution from solver: {}".format(t.solver))
    break

  for t_p in pending:
    t_p.cancel()
    await asyncio.sleep(0.1)
    try:
       await t_p
    except asyncio.CancelledError:
      logger.info(f"task {t_p} cancelled")

  for t_p in tasks:
    t_p.cancel()
    await asyncio.sleep(0.1)
    try:
       await t_p
    except asyncio.CancelledError:
      logger.info(f"task {t_p} cancelled")
#    while not t_p.cancelled():
#      pass
#    while not t_p.done():
#      pass

  for t_p in pending:
    await asyncio.wait(pending)

  for t_p in done:
    await asyncio.wait(done)

#  for task in asyncio.Task.all_tasks():
#    task.cancel()

  return result

def two_way_partition (net, graph_nx, hw_details):
  leaf_set= set(get_leaves(graph_nx))
  internal_nodes= set(graph_nx.nodes()) - leaf_set

  sub_graph_nx = graph_nx.subgraph(internal_nodes)
  sub_graph_nx, mapping = relabel_nodes_with_contiguous_numbers(sub_graph_nx, start=1)

#  sub_graph_nx = graph_nx


  init_str=""
  init_str += "N = {};\n".format(len(sub_graph_nx))
  init_str += "n_CU = {};\n".format(2)

  predecessors_str= "one_predecessors= ["
  for n in range(1, len(sub_graph_nx)+1):
    predecessors_str += "{"
    for p in sub_graph_nx.predecessors(n):
      if p != 0:
        predecessors_str += str(p)
        break # only one predecessors

    predecessors_str += "},"
  predecessors_str = predecessors_str[:-1]
  predecessors_str += "];\n"
  init_str += str(predecessors_str)

  predecessors_str= "predecessors= ["
  for n in range(1, len(sub_graph_nx)+1):
#  for n in sub_graph_nx:
    predecessors_str += "{"
    for p in sub_graph_nx.predecessors(n):
      predecessors_str += str(p)
      predecessors_str += ','

    if len(list(sub_graph_nx.predecessors(n))) != 0:
      predecessors_str = predecessors_str[:-1]

    predecessors_str += "},"

  predecessors_str = predecessors_str[:-1]
  predecessors_str += "];\n"
  init_str += predecessors_str

  leaves_mapping={leaf: (idx+1) for idx, leaf in enumerate(list(leaf_set))}
  edges_ls= []
  for n in leaf_set:
    for s in graph_nx.successors(n):
      edges_ls.append([leaves_mapping[n] , mapping[s]])

  init_str += "N_edges= {};\n".format(len(edges_ls))
  curr= [a for b in edges_ls for a in b]
  curr= useful_methods.ls_to_str(curr)
  init_str += "edges= array2d(1..N_edges, 1..2, [{}]);\n".format(curr)

  init_str += "N_done= {};\n".format(len(leaf_set))

  done_CU= [1 if (i > len(leaf_set)//2) else 2 for i in range(len(leaf_set))]
  curr= useful_methods.ls_to_str(done_CU)
  init_str += "done_CU= [{}];\n".format(curr)

  # write to file
  prefix= "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/optimization/minizinc_code/code/async/two_way_partition/"

  init_file= prefix + "data_{}.dzn".format(net)

  with open(init_file, 'w+') as fp:
    fp.write(init_str)

  curr_CU_active= input()
  import ast
  curr_CU_active = ast.literal_eval(curr_CU_active)

  properly_mapped= 0
  node_active=[]
  for n in sorted(list(sub_graph_nx.nodes())):
    parent_set= set([curr_CU_active[p-1] for p in sub_graph_nx.predecessors(n)])
    parent_set.add(curr_CU_active[n-1])
    parent_set = list(parent_set)
    if len(parent_set) == 1 and parent_set[0] != 0:
      properly_mapped += 1
      node_active.append(1)
    else:
      if len(parent_set) !=1 and len(list(sub_graph_nx.predecessors(n))) != 2:
        print('NOOOO', parent_set, n, list(sub_graph_nx.predecessors(n)), curr_CU_active[n-1])
      node_active.append(0)

  print(properly_mapped)
  print(node_active)

def less_constraints(net, graph, graph_nx, hw_details):
  leaf_set= set(get_leaves(graph_nx))
  internal_nodes= set(graph_nx.nodes()) - leaf_set

  sub_graph_nx = graph_nx.subgraph(internal_nodes)
  sub_graph_nx, mapping = relabel_nodes_with_contiguous_numbers(sub_graph_nx, start=1)

#  sub_graph_nx = graph_nx

  init_str=""
  init_str += "N = {};\n".format(len(sub_graph_nx))
  init_str += "n_CU = {};\n".format(hw_details.N_PE)

  predecessors_str= "one_predecessors= ["
  for n in range(1, len(sub_graph_nx)+1):
    predecessors_str += "{"
    for p in sub_graph_nx.predecessors(n):
      if p != 0:
        predecessors_str += str(p)
        break # only one predecessors

    predecessors_str += "},"
  predecessors_str = predecessors_str[:-1]
  predecessors_str += "];\n"
  init_str += str(predecessors_str)

  predecessors_str= "predecessors= ["
  for n in range(1, len(sub_graph_nx)+1):
#  for n in sub_graph_nx:
    predecessors_str += "{"
    for p in sub_graph_nx.predecessors(n):
      predecessors_str += str(p)
      predecessors_str += ','

    if len(list(sub_graph_nx.predecessors(n))) != 0:
      predecessors_str = predecessors_str[:-1]

    predecessors_str += "},"

  predecessors_str = predecessors_str[:-1]
  predecessors_str += "];\n"
  init_str += predecessors_str

  prefix= "/users/micas/nshah/Downloads/no_backup/Setups/minizinc_code/code/async/less_constraints/"
  init_file= prefix + "data_{}.dzn".format(net)

  with open(init_file, 'w+') as fp:
    fp.write(init_str)

def main(net, graph, graph_nx, hw_details):
  # prepare data
  list_of_chosen_sets, status_dict= partition.first_partition(graph, graph_nx, hw_details)

  print("Lenght of first partition: ", [len(x) for x in list_of_chosen_sets])
  symmetry_break_map= [list(x)[0] for x in list_of_chosen_sets if len(x) > 0]

  graph_nx, mapping = relabel_nodes_with_contiguous_numbers(graph_nx, start=1)
  leaf_set= set(get_leaves(graph_nx))

#  for n in graph_nx.nodes():
#    print(n, list(graph_nx.predecessors(n)))

  map_cu_to_node= defaultdict(set)
  map_node_to_cu= {}
  for cu, node_set in enumerate(list_of_chosen_sets):
    for n in node_set:
      mapped_node= mapping[n]
      for p in graph_nx.predecessors(mapped_node):
        map_node_to_cu[p] = cu+1
      assert len(list(graph_nx.predecessors(mapped_node))) != 0
  
  for n, cu in map_node_to_cu.items():
    map_cu_to_node[cu].add(n)


#  print(map_cu_to_node)
  predecessors= [set(graph_nx.predecessors(n)) for n in sorted(graph_nx.nodes())]

#  n_unmapped= len(graph_nx)
#  mapped_set= "{}"
#  unmapped_set= set(graph_nx.nodes())
#  done= [False for n in graph_nx.nodes]
#
#  extra_str=""
#  final_done_CU = [5, 1, 5, 3, 3, 1, 2, 1, 2, 2, 5, 1, 2, 2, 0, 3, 5, 0, 0]
#  colors= ['red', 'green', 'blue', 'yellow', 'pink', 'white']
#  for n in list(graph_nx.nodes()):
#    graph_nx.nodes[n]['fillcolor']= colors[final_done_CU[n-1]]
#    graph_nx.nodes[n]['shape']= 'circle'
#    graph_nx.nodes[n]['style']= 'filled'
#
#  reporting_tools.plot_graph_nx_graphviz(graph_nx)
  
  mapped_set= set(map_node_to_cu.keys())
  unmapped_set= set(graph_nx.nodes()) - mapped_set
  n_unmapped= len(unmapped_set)
  done= [False for n in graph_nx.nodes]

  extra_str=""
  assert len(map_node_to_cu) != 0
  for l in leaf_set:
    if l in map_node_to_cu:
      cu= map_node_to_cu[l]
      done[l-1]= True
      extra_str += "constraint done_CU[{}] = {};\n".format(l, cu)
  
  extra_str += "% symmetry break"
  for cu, n in enumerate(symmetry_break_map):
    extra_str += "constraint done_CU[{}] = {};\n".format(mapping[n], cu)
    
  
#  print(extra_str)
  #-------------------------------------------
  #       model
  #-------------------------------------------
  prefix= "/users/micas/nshah/Downloads/no_backup/Setups/minizinc_code/code/async/"
  model_path= prefix + "intra_barrier_mapping.mzn"
  init_file= prefix + "intra_barrier_mapping_{}.dzn".format(net)
  constr_file= prefix + "init_constraints_{}.mzn".format(net)

  intra_barrier_mapping = Model(model_path)
  intra_barrier_mapping.add_string(extra_str)
  with open(constr_file, 'w+') as fp:
    fp.write(extra_str)

  # Find the MiniZinc solver configuration
#  solver = Solver.lookup("or-tools")
#  solver = Solver.lookup("gecode")
#  solver = Solver.lookup("gurobi")
  solver = Solver.lookup("chuffed")

  inst = Instance(solver, intra_barrier_mapping)

  # instantiate variables
  inst["N"]= len(graph_nx)
#  inst["n_CU"]= hw_details.N_PE
  inst["n_CU"]= 64
  inst["n_unmapped"]= n_unmapped
  inst["mapped_set"]= mapped_set
  inst["unmapped_set"]= unmapped_set
  inst["predecessors"]= predecessors
  inst["done"]= done

  init_str=""
  init_str += "N = {};\n".format(len(graph_nx))
  init_str += "n_CU = {};\n".format(hw_details.N_PE)
  init_str += "n_unmapped = {};\n".format(n_unmapped)
  init_str += "unmapped_set = {};\n".format(unmapped_set)
  init_str += "mapped_set = {};\n".format(mapped_set)

  predecessors_str= "predecessors= ["
  for n in graph_nx:
    predecessors_str += "{"
    for p in graph_nx.predecessors(n):
      predecessors_str += str(p)
      predecessors_str += ','

    if len(list(graph_nx.predecessors(n))) != 0:
      predecessors_str = predecessors_str[:-1]

    predecessors_str += "},"

  predecessors_str = predecessors_str[:-1]
  predecessors_str += "];\n"
  init_str += predecessors_str

  init_str+= "done= ["
  for d in done:
    if d: init_str+= "true,"
    else: init_str+= "false,"
  init_str= init_str[:-1]
  init_str+= "];\n"
#  init_str += "done = {};\n".format(str(done))
  
  with open(init_file, 'w+') as fp:
    fp.write(init_str)
  exit(1)

#  print(len(graph_nx), predecessors, mapped_set)
#  result = inst.solve(verbose= True, intermediate_solutions= True)
  timeout= datetime.timedelta(seconds= 600)
  
  result = inst.solve(timeout= timeout, verbose=True)
  print(result["mapped"])
  print(result["done_CU"])
  print(result["obj"])
  print(result["edge_cost"])
  print(result["workload_cost"])
  print(result["mapped_per_CU"])
