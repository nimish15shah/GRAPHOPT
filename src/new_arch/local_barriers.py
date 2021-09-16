
import networkx as nx
import logging as log

from ..useful_methods import printlog, printcol
from . import instr_types
from . import register_allocation

class local_status_node():
  def __init__(self, node_key):
    self.key = node_key
    self.available_in_register= False
    self.pe_id= None

    self.global_committed= False

def create_local_graph(graph_nx, partition_nodes):
  return graph_nx.subgraph(partition_nodes)

def dfs_sort(local_graph, node, topological_list, done_nodes):
  for child in local_graph.predecessors(node):
    if child not in done_nodes:
      dfs_sort(local_graph, child, topological_list, done_nodes)
  
  topological_list.append(node)
  done_nodes[node] = True

def insert_initial_loads(graph, schedule, hw_details):
  """
    assumes infinite regbank size
  """
  N_PE         = hw_details.N_PE
  REGBANK_L    = hw_details.REGBANK_L

  node_loaded= {}
  
  log.info("Before:")
  log.info([(instr_obj.instr_type, instr_obj.node) for instr_obj in schedule])

  new_schedule= []
  for instr_obj in schedule:
    node= instr_obj.node
    for child in graph[node].child_key_list:
      if child not in node_loaded:
        ld_obj= instr_types.mini_instr('ld', child)
        new_schedule.append(ld_obj)
        node_loaded[child]= True
    
    new_schedule.append(instr_obj)
    node_loaded[node]= True
  
  schedule= new_schedule
  
  log.info("")
  log.info("After:")
  log.info([(instr_obj.instr_type, instr_obj.node) for instr_obj in schedule])

def main(graph, graph_nx, list_of_partitions, status_dict, hw_details, final_output_nodes):
  N_PE         = hw_details.N_PE

  spill_cnt =0

  # NOTE: nodes_to_store is going to be modified in linear scan, 
  # new intermediate nodes are going to be added
  # Hence, create a copy of the original set
  nodes_to_store= set(final_output_nodes) # cope of n

  list_of_schedules= [[] for _ in range(N_PE)]

  # Iterate over all PEs before crossing global barriers
  for global_barrier_idx in range(len(list_of_partitions[0])): 
    for pe_id, pe_partitions in enumerate(list_of_partitions):
      partition = pe_partitions[global_barrier_idx]

      local_graph= create_local_graph(graph_nx, partition)
      

      schedule= schedule_nodes(graph, local_graph)
      schedule= create_instructions(graph, schedule)
      
      spill_cnt_partion, reg_alloc_obj_dict = register_allocation.linear_scan(graph, local_graph, schedule, nodes_to_store, hw_details)

      schedule= insert_barriers(schedule, nodes_to_store, reg_alloc_obj_dict, hw_details)
      
      list_of_schedules[pe_id].append(schedule)
      spill_cnt += spill_cnt_partion

  log.info(f"spill_cnt: {spill_cnt}")
  
  for node in nodes_to_store:
    status_dict[node].to_be_stored= True
    assert not graph[node].is_leaf()
  
  return list_of_schedules

def insert_barriers(schedule, nodes_to_store, reg_alloc_obj_dict, hw_details):
  new_schedule= []
  RESERVED_REG_OUT= hw_details.RESERVED_REG_OUT

  nodes_stored_after_prev_barrier= set()

  for idx, instr in enumerate(schedule):
    if (instr.in_0_node in nodes_stored_after_prev_barrier) or \
       (instr.in_1_node in nodes_stored_after_prev_barrier):
      barrier_instr= instr_types.full_instr()
      barrier_instr.set_op('local_barrier')      
      new_schedule.append(barrier_instr)       
      nodes_stored_after_prev_barrier= set()
      
      assert instr.reg_in_0 == hw_details.RESERVED_REG_IN_0 or instr.reg_in_1 == hw_details.RESERVED_REG_IN_1
      assert instr.in_0_node in nodes_to_store or instr.in_1_node in nodes_to_store
      assert (not reg_alloc_obj_dict[instr.in_0_node].is_reg_allocated()) or (not reg_alloc_obj_dict[instr.in_1_node].is_reg_allocated())
    
    if idx!= 0:
      prev_instr= schedule[idx-1]
      if prev_instr.reg_o == RESERVED_REG_OUT:
        nodes_stored_after_prev_barrier.add(prev_instr.node)

    new_schedule.append(instr)
  
  return new_schedule

def schedule_nodes(graph, local_graph):
  # Top nodes
  source_nodes= set([node for node in local_graph.nodes() if len(list(local_graph.successors(node)))== 0])
  source_nodes = sorted(list(source_nodes), key= lambda x:len(nx.algorithms.dag.ancestors(local_graph, x)))

  # Schedule nodes in depth-first order
  schedule= []
  done_nodes= {}
  for node in source_nodes:
    dfs_sort(local_graph, node, schedule, done_nodes)
  assert len(schedule) == len(local_graph)

  return schedule

def create_instructions(graph, schedule):
  new_schedule= []

  for node in schedule:
    obj= graph[node]
    if obj.is_sum():
      operation= 'sum'
    elif obj.is_prod():
      operation= 'prod'
    else:
      assert 0
#        instr_obj= instr_types.mini_instr(operation, node)
    instr_obj= instr_types.full_instr()
    instr_obj.node= node
    instr_obj.set_op(operation)

    assert len(obj.child_key_list) == 2
    instr_obj.in_0_node= obj.child_key_list[0]
    instr_obj.in_1_node= obj.child_key_list[1]

    new_schedule.append(instr_obj)

  schedule= new_schedule

  return schedule


