
import networkx as nx
print(nx.__file__)

#**** imports from our codebase *****
from .reporting_tools import reporting_tools

#print nx.minimum_edge_cut(G,1,7)

#print nx.edge_connectivity(G,1,7)

#part_1, part_2= nx.algorithms.community.kernighan_lin_bisection(G)

#reporting_tools.reporting_tools.plot_graph(part_1)

def partition(graph_nx, BB_graph_nx, BB_graph):
  """
    Usage: 
      self.hw_depth= 4
      self.BB_graph, self.graph, self.BB_graph_nx, self.graph_nx = src.files_parser.read_BB_graph_and_main_graph(self.global_var, self.hw_depth)
      part_1, part_2= src.partition.partition(self.graph_nx, self.BB_graph_nx) 

  """
  print('Paritioning')  
  #reporting_tools.reporting_tools.plot_graph_nx_graphviz(BB_graph_nx) 

  part_1, part_2= nx.algorithms.community.kernighan_lin_bisection(BB_graph_nx.to_undirected())
  count_crossings(BB_graph, part_1, part_2)

  return part_1, part_2

def count_crossings(BB_graph, part_1, part_2):
  #assert 0, "Erroneous counting based on BB. Must be based on nodes/variables"
  print("Warning: Erroneous counting based on BB. Must be based on nodes/variables")
  cut_set_len= 0
  cross_node= set()

  for node in part_1:
    obj= BB_graph[node]
    for child in obj.child_bb_lst_duplicates:
      if child in part_2:
        cut_set_len += 1
       
    for child in set(obj.child_bb_lst):
      if child in part_2:
        cross_node |= set(BB_graph[child].out_list).intersection(set(BB_graph[node].in_list))
  
  print(cut_set_len, len(cross_node))

  for node in part_2:
    obj= BB_graph[node]
    for child in obj.child_bb_lst_duplicates:
      if child in part_1:
        cut_set_len += 1
    
    for child in set(obj.child_bb_lst):
      if child in part_1:
        cross_node |= set(BB_graph[child].out_list).intersection(set(BB_graph[node].in_list))

  print(cut_set_len, len(cross_node))
  
  print('Cutset_len', cut_set_len)
  print('Transfer vars: ', len(cross_node))

def custom_partition(BB_graph):
  """
    Custom partitioning algo
  """

  # Start with the leaf with lowest reverse level

  curr_bb= sorted(list(BB_graph.keys()), key= lambda x: BB_graph[x].reverse_level)[0]
  
  target_len= len(BB_graph)/2
  
  curr_partition= set()
  parent_bb= set(BB_graph[curr_bb].parent_bb_lst)

  while len(curr_partition) < target_len:
    
    best_parent_cnt= len(BB_graph)
    
    for bb in parent_bb:
    #for parent in BB_graph[curr_bb].parent_bb_lst:
      children_set= bfs_children(BB_graph, parent)
      
      len(children_set)

def bfs_children(BB_graph, curr_bb):
  
  bb_list= [curr_bb]
  
  children_set= set()

  while bb_list:
    curr_bb= bb_list[0] 
    for child_bb in BB_graph[curr_bb].child_bb_lst:
      if child_bb not in children_set:
        children_set.add(child_bb)
        bb_list.append[child]
    
    del bb_list[0]
    
  
  return children_set

def greedy_partitioning(BB_graph, BB_graph_nx, num_parts):
  """
    Algo 3 from paper: "Acyclic partion of large graphs"
  """
  
  lb= 1.05 * len(BB_graph_nx)/num_parts

  # parition
  part= {k:set() for k in range(num_parts)}
  part_dict= {}

  # track if node is free 
  free= {node:True for node in BB_graph_nx.nodes()}
  
  #print BB_graph_nx.edges()
  
  for part_i in range(num_parts):

    # Construct set of eligible nodes
    set_u= set()
    for u in BB_graph_nx.nodes():
      if free[u]:
        pred_free= False
        assert len(BB_graph_nx.pred[u]) == len(BB_graph[u].child_bb_lst), [BB_graph_nx.pred[u], BB_graph[u].child_bb_lst]

        for pred in list(BB_graph_nx.pred[u].keys()):
          if free[pred]:
            pred_free = True
            break

        if not pred_free :
          set_u.add(u)

    # compute gain for set_u 
    gain= {}
    for u in set_u:
      gain[u]= CompGain(BB_graph_nx, BB_graph, u, part, part_i)

    # Add nodes to curr part
    while len(part[part_i]) < lb:
      gain_list= sorted(list(gain.keys()), key= lambda x: gain[x], reverse= True)
      if gain_list:
#        curr_u= gain_list.pop(0) 
        top_u= gain_list[0]
        same_gain_list= [u for u in gain_list if gain[u] == gain[top_u]]
        curr_u= sorted(same_gain_list, key= lambda x: BB_graph[x].sch_level)[0]
        gain_list.remove(curr_u)
      else:
        break
        assert 0

      del gain[curr_u]
      part[part_i].add(curr_u)
      part_dict[curr_u]= part_i
      free[curr_u]= False

      for v in list(BB_graph_nx.succ[curr_u].keys()):
        ready= True
        for w in list(BB_graph_nx.pred[v].keys()):
          if free[w]:
            ready= False
            break
        if ready:
          gain[v]= CompGain(BB_graph_nx, BB_graph, v, part, part_i)

      assert curr_u not in gain
  
#  print part
  
  count_crossings(BB_graph, part[0], part[1])
  kw_args= {'dot_file_name': './REPORTS/partition.dot', 'BB_graph': BB_graph, 'part_dict': part_dict, 'option': 'color_partition'}
  reporting_tools.reporting_tools.create_dot_file(**kw_args)
  
  return part

def CompGain(BB_graph_nx, BB_graph, u, part, part_i):
  """
    same as Algo 2 in "Directed acyclic partition of large graphs"
  """
  gain = 0

  for v in list(BB_graph_nx.pred[u].keys()):
    if v in part:
      curr_cost= len(set(BB_graph[v].out_list).intersection(set(BB_graph[u].in_list)))
      if part[v] == part_i:
        gain += curr_cost
      else:
        gain -= curr_cost

  for v in list(BB_graph_nx.succ[u].keys()):
    if v in part:
      curr_cost= len(set(BB_graph[u].out_list).intersection(set(BB_graph[v].in_list)))
      assert 0
      if part[v] == part_i:
        gain += curr_cost
      else:
        gain -= curr_cost
  
  return gain
