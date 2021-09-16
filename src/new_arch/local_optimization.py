#-----------------------------------------------------------------------
# Created by         : KU Leuven
# Filename           : src/new_arch/local_optimization.py
# Author             : Nimish Shah
# Created On         : 2021-01-27 14:19
# Last Modified      : 
# Update Count       : 2021-01-27 14:19
# Description        : 
#                      
#-----------------------------------------------------------------------

import networkx as nx
from .. import useful_methods 
import logging
from statistics import mean
from ..useful_methods import printlog, printcol

logger= logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # add formatter to ch
# ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

class Leaf_data():
  def __init__(self, l):
    self.l= l
    self.partition= None
    self.descendants= None
    self.descendants_topological= None

    self.descendants_in_same_part= set()
  
  def update_descendants_in_same_part(self, n_data):
    assert self.partition != None
    self.descendants_in_same_part = self.get_descendants_mappable_to_part(n_data, self.partition)

  def get_descendants_mappable_to_part(self, n_data, part, l_to_skip= set()):
    descendants_mappable_to_part= set()
    for d in self.descendants:
      d_part= n_data[d].which_partition(n_data, l_to_skip)
      if d_part == part or d_part== 'SKIPPED':
        descendants_mappable_to_part.add(d)
    return descendants_mappable_to_part

  def update_part_of_all_descendants(self, n_data):
    self.update_descendants_in_same_part(n_data)
    for d in self.descendants:
      if d not in self.descendants_in_same_part:
        n_data[d].partition = None
      else:
        n_data[d].partition = self.partition


class NonLeaf_data():
  def __init__(self, n):
    self.n = n
    self.partition= None
    self.ancestors_leaves= None
  
  def which_partition(self, n_data, l_to_skip= set()):
    l_parts= set([n_data[l].partition for l in self.ancestors_leaves if l not in l_to_skip])
    assert len(l_parts) <= 2
    if len(l_parts) == 0:
      assert len(self.ancestors_leaves - l_to_skip) == 0

    
    if len(l_parts) == 1:
      part= l_parts.pop()
      assert part != None
      return part
    elif len(l_parts) == 0:
      return "SKIPPED"
    else:
      return None

class Recent_move_check():
  def __init__(self, max_len):
    self.max_len= max_len
    self.fifo= []
    self.fifo_set= set()
  
  def is_allowed(self, m):
    return m not in self.fifo_set

  def append_move(self, m):
    assert len(self.fifo) <= self.max_len
    m_to_remove= None
    if len(self.fifo) == self.max_len:
      m_to_remove= self.fifo.pop(0)
      if m_to_remove != None:
        self.fifo_set.remove(m_to_remove)
    
    self.fifo.append(m)
    if m != None:
      self.fifo_set.add(m)

    # assert len(self.fifo) == len(self.fifo_set)
    assert len(self.fifo) <= self.max_len
    return m_to_remove

class Local_optimization_partition():
  def __init__(self, nodes_to_map, graph_nx, done_set_0, done_set_1, done_nodes, node_w, config_obj):
    """
      very inefficient implementation but is meant for smaller graph partitions
    """
    self.nodes_to_map= nodes_to_map
    self.graph_nx= graph_nx
    self.sub_graph_nx= graph_nx.subgraph(nodes_to_map)
    self.done_set_0 = done_set_0
    self.done_set_1 = done_set_1
    self.done_nodes= done_nodes
    self.config_obj= config_obj
    self.node_w= node_w
    
    # get leaves based on done nodes only
    self.leaves= set([])
    for n in nodes_to_map:
      unmapped_pred= [s for s in graph_nx.predecessors(n) if s not in done_nodes]
      if len(unmapped_pred) == 0:
        self.leaves.add(n)
    
    self.non_leaves= set(nodes_to_map) - self.leaves

    self.n_data= {l: Leaf_data(l) for l in self.leaves}
    for n in self.non_leaves:
      self.n_data[n] = NonLeaf_data(n)

    self.initialize_data_structures()
    self.part_0= set([n for n in self.nodes_to_map if self.n_data[n].partition == 0])
    self.part_1= set([n for n in self.nodes_to_map if self.n_data[n].partition == 1])
    self.part_0_l= set([n for n in self.leaves if self.n_data[n].partition == 0])
    self.part_1_l= set([n for n in self.leaves if self.n_data[n].partition == 1])


    self.best_part_0 = set(self.part_0)
    self.best_part_1 = set(self.part_1)

    # optimization
    self.ITER_LIMIT= max(10000, len(self.leaves) ** 3)
    self.MIN_OBJECTIVE_THRESHOLD= len(self.leaves) // 2
    self.BEST_OBJECTIVE= max(1, int(0.97 * len(self.nodes_to_map) // 2))
    self.ALLOWED_DEGRADATION_FACTOR= 0.9
    self.recent_move_obj= Recent_move_check(max_len= len(self.leaves) + 1)
    self.OBJ_RUN_AVERAGE_LEN= 3*len(self.leaves)

    self.optimization_main()
    
    self.print_results()

  def get_results(s):
    return s.best_part_0, s.best_part_1    

  def get_minizinc_result(s):
    result= {}
    sorted_nodes= sorted(s.nodes_to_map)
    assert sorted_nodes[0] == 1

    result["obj"]= min(s.get_obj(s.best_part_0), s.get_obj(s.best_part_1))
    result["mapped_per_CU_active"]= [s.get_obj(s.best_part_0), s.get_obj(s.best_part_1)]

    node_active= [0] * len(sorted_nodes)
    curr_CU_active= [0] * len(sorted_nodes)
    done_nodes= set()
    for n in s.best_part_0:
      assert n not in done_nodes
      node_active[n - 1] = 1
      curr_CU_active[n - 1] = 1
      done_nodes.add(n)
    for n in s.best_part_1:
      assert n not in done_nodes
      node_active[n - 1] = 1
      curr_CU_active[n - 1] = 2
      done_nodes.add(n)

    result["node_active"]= node_active
    result["curr_CU_active"]= curr_CU_active

    result["tot_local_edges"]= None

    return result

  def print_results(s):
    logger.info(f"part_0, part_1, nodes_to_map: {len(s.best_part_0)}, {len(s.best_part_1)}, {len(s.nodes_to_map)}")

  def get_obj(s, node_set):
    return sum([s.node_w[n] for n in node_set])

  def initialize_data_structures(s):
    # static data structures
    for l in s.leaves:
      descendants= set(nx.descendants(s.sub_graph_nx, l))
      assert len(descendants & s.leaves) == 0
      s.n_data[l].descendants = descendants
      sub_g= s.sub_graph_nx.subgraph(descendants)
      # assert nx.algorithms.dag.is_directed_acyclic_graph(sub_g), f"{nx.algorithms.cycles.find_cycle(sub_g)}, {list(s.sub_graph_nx.predecessors(3))}"
      s.n_data[l].descendants_topological= tuple(nx.topological_sort(sub_g))

    for n in s.non_leaves:
      s.n_data[n].ancestors_leaves= set(nx.ancestors(s.sub_graph_nx, n)) & s.leaves

    # partition leaves
    topo_ls= useful_methods.dfs_topological_sort(s.sub_graph_nx)
    topo_ls= [n for n in topo_ls if n in s.leaves]
    part_0 =topo_ls[ : len(topo_ls)//2]
    part_1 =topo_ls[len(topo_ls)//2 : ]
    for l in part_0:
      s.n_data[l].partition= 0
    for l in part_1:
      s.n_data[l].partition= 1
    
    # assign non_leaves based on original leaves partition
    for n in s.non_leaves:
      s.n_data[n].partition = s.n_data[n].which_partition(s.n_data)
    
    for l in s.leaves:
      s.n_data[l].update_descendants_in_same_part(s.n_data)
    
  def estimate_move_impact(s, l):
    src_part= s.n_data[l].partition
    dst_part= 1- src_part

    src_descendants= s.n_data[l].descendants_in_same_part
    # TODO: costly assertion, disable once code is stable
    curr_src_decscendants= s.n_data[l].get_descendants_mappable_to_part(s.n_data, src_part, l_to_skip= set())
    assert len(src_descendants) ==  len(curr_src_decscendants), f"{src_descendants}, ||||| {curr_src_decscendants}"
    dst_descendants= s.n_data[l].get_descendants_mappable_to_part(s.n_data, dst_part, l_to_skip= set([l]))

    # delta_src= -len(src_descendants) - 1
    # delta_dst= len(dst_descendants) + 1

    delta_src= -s.get_obj(src_descendants) - s.get_obj(set([l]))
    delta_dst= s.get_obj(dst_descendants) + s.get_obj(set([l]))
    
    return delta_src, delta_dst
  
  def perform_move(s, l, mode):
    assert mode in ['strict', 'random', 'non_strict' , 'allow_obj_degradation']
    if mode != 'non_strict':
      assert s.recent_move_obj.is_allowed(l)

    src_part= s.n_data[l].partition
    dst_part= 1- src_part

    src_descendants= s.n_data[l].descendants_in_same_part

    s.n_data[l].partition = dst_part
    s.n_data[l].update_part_of_all_descendants(s.n_data)

    dst_descendants= s.n_data[l].descendants_in_same_part

    if src_part == 0:
      src_part_all_nodes= s.part_0
      src_part_leaves= s.part_0_l
      dst_part_all_nodes= s.part_1
      dst_part_leaves= s.part_1_l
    else:
      src_part_all_nodes= s.part_1
      src_part_leaves= s.part_1_l
      dst_part_all_nodes= s.part_0
      dst_part_leaves= s.part_0_l

    src_part_all_nodes -= src_descendants
    src_part_all_nodes.remove(l)
    src_part_leaves.remove(l)

    dst_part_all_nodes |= dst_descendants
    dst_part_all_nodes |= set([l])
    dst_part_leaves |= set([l])

    # update state of all leaves that are ancestors
    for d in s.n_data[l].descendants:
      d_obj= s.n_data[d]
      if d_obj.partition == None:
        for a in d_obj.ancestors_leaves:
          s.n_data[a].descendants_in_same_part -= set([d])
      else:
        for a in d_obj.ancestors_leaves:
          if d_obj.partition == s.n_data[a].partition:
            s.n_data[a].descendants_in_same_part.add(d)
          else:
            s.n_data[a].descendants_in_same_part -= set([d])

    if mode != 'non_strict':
      s.recent_move_obj.append_move(l)

  def sorted_move_candidates_according_to_promise(s, src_leaves, check_if_allowed= True):
    # sorted_leaves= sorted(src_leaves, key= lambda x: len(s.n_data[x].descendants) - len(s.n_data[x].descendants_in_same_part), reverse = True)
    sorted_leaves= sorted(src_leaves, key= lambda x: s.get_obj(s.n_data[x].descendants) - s.get_obj(s.n_data[x].descendants_in_same_part), reverse = True)
    if check_if_allowed:
      sorted_leaves = [l for l in sorted_leaves if s.recent_move_obj.is_allowed(l)]

    return sorted_leaves
  
  def optimization_main(s):
    n_iter =0
    obj_run_average= [0] * s.OBJ_RUN_AVERAGE_LEN

    idle_cycles= 0
    last_l_pop= -1
    while n_iter < s.ITER_LIMIT:
      n_iter += 1

      obj= min(s.get_obj(s.part_0), s.get_obj(s.part_1))
      if obj >= s.BEST_OBJECTIVE:
        break

      if s.get_obj(s.part_0) > s.get_obj(s.part_1):
        src_part= s.part_0
        src_leaves= s.part_0_l
        dst_part= s.part_1
      else:
        src_part= s.part_1
        src_leaves= s.part_1_l
        dst_part= s.part_0

      if last_l_pop == -1:
        perform_move= s.one_iteration(src_part, dst_part, src_leaves, mode = 'strict')
      else:
        perform_move = False
      if not perform_move:
        if idle_cycles == s.recent_move_obj.max_len:
          logger.info("Need to go for non_strict mode")
          perform_move= s.one_iteration(src_part, dst_part, src_leaves, mode = 'allow_obj_degradation')
          idle_cycles += 1
          last_l_pop= -1
        else:
          idle_cycles += 1
          last_l_pop= s.recent_move_obj.append_move(None)

      else:
        last_l_pop= -1
        idle_cycles == 0
        obj= min(s.get_obj(s.part_0), s.get_obj(s.part_1))
        obj_run_average.append(obj)
        obj_run_average.pop(0)
        
        if mean(obj_run_average) >= obj:
          logger.info(f"obj_run_average is higher than obj, exiting: {obj_run_average} {obj}")
          break

      if idle_cycles > 2*s.recent_move_obj.max_len + 2:
        # print (s.recent_move_obj.fifo_set)
        # print (s.recent_move_obj.fifo)
        # # assert len(s.recent_move_obj.fifo_set) == 0
        logger.info("No move since a long time")
        break

        # if not perform_move:
        #   logger.info("No move improves the objective further")
        #   break

    best_obj= min(s.get_obj(s.best_part_0), s.get_obj(s.best_part_1))
    logger.info(f"Best objective: {best_obj}, {s.get_obj(s.best_part_0)} {s.get_obj(s.best_part_1)}")
    s.switch_according_to_done_sets()

  def switch_according_to_done_sets(s):
    inputs_0= set([i for n in s.best_part_0 for i in s.graph_nx.predecessors(n)])
    inputs_1= set([i for n in s.best_part_1 for i in s.graph_nx.predecessors(n)])

    inputs_0_done_set_0 =len(inputs_0 & s.done_set_0)
    inputs_0_done_set_1 =len(inputs_0 & s.done_set_1)
    inputs_1_done_set_0 =len(inputs_1 & s.done_set_0)
    inputs_1_done_set_1 =len(inputs_1 & s.done_set_1)
    
    max_edges= max([inputs_0_done_set_0, inputs_0_done_set_1, inputs_1_done_set_0, inputs_1_done_set_1])

    if (max_edges == inputs_0_done_set_1) or (max_edges == inputs_1_done_set_0):
      temp = set(s.best_part_1)
      s.best_part_1 = set(s.best_part_0)
      s.best_part_0= set(temp)

  def one_iteration(s, src_part, dst_part, src_leaves, mode):
    assert mode in ['strict', 'random', 'non_strict' , 'allow_obj_degradation']
    check_if_allowed = (mode != 'non_strict')
    sorted_leaves= s.sorted_move_candidates_according_to_promise(src_leaves, check_if_allowed= check_if_allowed)
    # print ([n for n in s.recent_move_obj.fifo if n != None])
    for l in sorted_leaves:
      delta_src, delta_dst= s.estimate_move_impact(l)
      old_obj= min(s.get_obj(src_part) , s.get_obj(dst_part))
      new_src_obj= s.get_obj(src_part) + delta_src
      new_dst_obj= s.get_obj(dst_part) + delta_dst
      new_obj= min(new_src_obj, new_dst_obj)

      if mode == 'allow_obj_degradation':
        perform_move= (new_obj > s.ALLOWED_DEGRADATION_FACTOR * old_obj)
        perform_move = perform_move or (new_src_obj + new_dst_obj > s.get_obj(src_part) + s.get_obj(dst_part))
      elif mode == 'strict':
        perform_move= (new_obj > old_obj)
      else:
        assert 0

      if perform_move:
        logger.info(f"Moving leaf: {l}")
        s.perform_move(l, mode)

        new_obj= min(s.get_obj(s.part_0), s.get_obj(s.part_1))
        best_obj= min(s.get_obj(s.best_part_0), s.get_obj(s.best_part_1))
        logger.info(f"old_obj: {old_obj}, new obj: {new_obj}, {new_src_obj}, {new_dst_obj}, best_obj= {best_obj}")
        
        # store if best
        if new_obj > best_obj:
          s.best_part_0 = set(s.part_0)
          s.best_part_1 = set(s.part_1)

        return True

    return False
  
  

