
#--
#  This file contains methods that help define the structure of ALU hardware 
#--

from collections import defaultdict
import networkx as nx

#**** imports from our codebase *****
from . import common_classes

#*****
# A function to create a DAG for HW structure described in a src file
#*****
def _read_hw_struct(hw_file_path, verbose=False):
  hw_struct= {}
  hw_depth= 0
  hw_depth_cnt={}
  hw_dict= defaultdict(list) 
  hw_misc_dict= defaultdict(defaultdict)

  hw_file= open(hw_file_path, 'r')
  
  lines= hw_file.readlines()
  for line in lines:
    line= line.rstrip()
    line= line.split(',')

    key= int(line[0])

    curr_node= common_classes.hw_node(key)
    
    curr_node.operation_type.reset_all()
    
    if 'leaf' in line:
      curr_node.operation_type.LEAF= True    
    if 'sum' in line:
      curr_node.operation_type.SUM= True    
    if 'product' in line:
      curr_node.operation_type.PRODUCT= True    
    if 'pass_1' in line:
      curr_node.operation_type.PASS_IN1= True    
    if 'pass_2' in line:
      curr_node.operation_type.PASS_IN2= True    
    if 'out' in line:
      curr_node.operation_type.OUT= True    

    if 'leaf' not in line:
      child_idx= line.index('c') + 1
      child_list= line[child_idx::] # Get all the elements from child_idx onwards
      
      max_child_depth=0
      for child in child_list:
        child= int(child)
        curr_node.child_key_list.append(child)
        hw_struct[child].parent_key_list.append(key)
        if(hw_depth_cnt[child] > max_child_depth):
          max_child_depth= hw_depth_cnt[child]
      
      hw_depth_cnt[key]= max_child_depth+1
      hw_dict[max_child_depth+1].append(key)
    else:
      hw_depth_cnt[key]=0
      hw_dict[0].append(key)

    hw_struct[key]= curr_node
  
  final_key= int(lines[-1][0])
  hw_depth= hw_depth_cnt[final_key]
  
  #** Invert key scheme of hw_dict.
  #** Currently leaf nodes are at depth 0 and final_node is at hw_depth. Invert that.
  hw_dict_invert= defaultdict(list)
  for depth in hw_dict:
    hw_dict_invert[hw_depth-depth]= hw_dict[depth]
  
  #** Misc details
  for depth in hw_dict_invert:
    sum_cnt=0
    prod_cnt=0
    leaf_cnt=0
    pass_in1_cnt=0
    pass_in2_cnt=0

    for node in hw_dict_invert[depth]:
      if hw_struct[node].operation_type.PRODUCT:
        prod_cnt= prod_cnt + 1
        
      if hw_struct[node].operation_type.SUM:
        sum_cnt= sum_cnt + 1

      if hw_struct[node].operation_type.PASS_IN1:
        pass_in1_cnt= pass_in1_cnt + 1

      if hw_struct[node].operation_type.PASS_IN2:
        pass_in2_cnt= pass_in2_cnt + 1

      if hw_struct[node].operation_type.LEAF:
        leaf_cnt= leaf_cnt + 1
    
    hw_misc_dict[depth]['sum_cnt']= sum_cnt
    hw_misc_dict[depth]['prod_cnt']= prod_cnt
    hw_misc_dict[depth]['pass_in1_cnt']= pass_in1_cnt
    hw_misc_dict[depth]['pass_in2_cnt']= pass_in2_cnt
    hw_misc_dict[depth]['leaf_cnt']= leaf_cnt
  
  return hw_struct, hw_depth, hw_dict_invert, hw_misc_dict

class hw_details_class():
  """
    This class is to keep all the hardware related details together at a single place
  """
  
  def __init__(self):
    #-- Logical description of hw
    #-- A list of hw lists. Hw list is a list of NX graph representing the hw structure
    #-- Fir loop iter corresponds to different types of Hybrid BBs
    #-- Second loop iter --> Different parts in a Hybrid BB
    self.list_of_hw_lists= None
    
    #-- A list of set of isomporph indices. Indices are in form of tuples (a,b)
    #-- Fir loop iter corresponds to different types of Hybrid BBs
    #-- Second loop iter --> List of sets for Different parts in a Hybrid BB
    #-- Set contains tuples of indices of isomporphic nx
    self.list_of_isomorph_indices= None
    
    #-- USed to name files
    self.list_of_depth_lists= None
    
    #-- Physical description of hw
    self.tree_depth= None
    self.n_tree= None

    #-- Output port map
    # Key: Tuple (tree, lvl, pe)
    #           [0,3,0]                       [1,3,0]
    #           /     \                       /     \
    #     [0,2,0]     [0,2,1]           [1,2,0]     [1,2,1]
    #     /   |        |   \             /   |        |   \
    #[0,1,0][0,1,1] [0,1,2][0,1,3]  [1,1,0][1,1,1] [1,1,2][1,1,3]
    # Val: set of banks that output of this PE can access
    self.out_port_map= {}

    self.n_banks= None
    self.reg_bank_depth= None
    self.mem_bank_depth= None
    self.mem_addr_bits= None
    self.n_bits= None
    self.n_pipe_stages= None
    
def hw_nx_graph_structs(hw_depth, max_depth, min_depth):
  """
    Creates multiple lists of nx_graphs to represent hardware.
    These nx_graphs can be used for mapping AC using subgraph isomorphism.
    
    Example:
      hw_depth = 4
      # of outputs= 3 (assumed)

      lists would look like as follows
        list_1 : tree of depth 4
        list_2 : 2 trees of depth 3

      If an extra input is allowed:
        list_3 : A tree of depth 3 + compliment of depth-3 tree
    
    Priority:
      list are arranged as elemnts of a global list.
      In the global list, list item 0 has the highest priority, item 1 is next to item 0 in priority, and so on...
  """
  
  assert min_depth > 0
  assert max_depth <= hw_depth

  list_of_hw_lists= []
  
  # -- Specifies how many smaller levels are permitted in Hybrid BBs
  # -- level_for_hybrid == 1 , means no hybdrif BB allowed
  # -- level_for_hybrid == 2 , Hybrid BBs of hw_depth-1 allowed
  # -- least depth = hw_depth - level_for_hybrid + 1
  
  level_for_hybrid= hw_depth - min_depth + 1
  #level_for_hybrid= hw_depth
  #level_for_hybrid= hw_depth - 1
  #level_for_hybrid = 2

  ##-- Auto generate same-depth hybrid BBs
  for level in range(0,level_for_hybrid):
    hw_list= []
    assert hw_depth - level > 0
    for i in range(0,2**level):
      G= nx.balanced_tree(2, hw_depth-level, nx.DiGraph) 
      G= G.reverse(copy=False)
      hw_list.append(G)
    
  #  list_of_hw_lists.append(hw_list)

  list_of_depth_lists= hw_models_for_all_combinations(hw_depth, max_depth, level_for_hybrid, list_of_hw_lists) 
  
  ##-#- A full balanced tree of hw_depth
  #hw_list= []
  #assert hw_depth > 0
  #G= nx.balanced_tree(2, hw_depth, nx.DiGraph) 
  #G= G.reverse(copy=False)
  #hw_list.append(G)
  #
  #list_of_hw_lists.append(hw_list)
  #

  ###-- Two full balanced_tree of hw_depth-1
  #hw_list= []
  #assert hw_depth - 1 > 0
  #for i in range(0,2):
  #  G= nx.balanced_tree(2, hw_depth-1, nx.DiGraph) 
  #  G= G.reverse(copy=False)
  #  hw_list.append(G)
  #
  #list_of_hw_lists.append(hw_list)

  ##-- Four full balanced_tree of hw_depth-2
  #hw_list= []
  #assert hw_depth - 2 > 0
  #for i in range(0,4):
  #  G= nx.balanced_tree(2, hw_depth-2, nx.DiGraph) 
  #  G= G.reverse(copy=False)
  #  hw_list.append(G)
  #
  #list_of_hw_lists.append(hw_list)
  
  ##-- Eight full balanced_tree of hw_depth-3
  #hw_list= []
  #assert hw_depth - 3 > 0
  #for i in range(0,8):
  #  G= nx.balanced_tree(2, hw_depth-3, nx.DiGraph) 
  #  G= G.reverse(copy=False)
  #  hw_list.append(G)
  #
  #list_of_hw_lists.append(hw_list)

  #-- A vector of single operators
  hw_list= []
  for i in range(0,8):
    G= nx.balanced_tree(2, 1, nx.DiGraph)
    G= G.reverse(copy=False)
    hw_list.append(G)
  #list_of_hw_lists.append(hw_list)
  
  list_of_isomorph_indices= create_isomorp_list(list_of_hw_lists)
  
  hw_details= hw_details_class()
  hw_details.list_of_hw_lists= list_of_hw_lists
  hw_details.list_of_isomorph_indices= list_of_isomorph_indices
  hw_details.list_of_depth_lists= list_of_depth_lists
  hw_details.tree_depth= max_depth
  hw_details.n_tree= int((2**hw_depth)//(2**max_depth))
  hw_details.out_port_map= describe_out_port_map(hw_details.tree_depth, hw_details.n_tree)
  hw_details.max_depth= max_depth
  hw_details.min_depth= min_depth

  return hw_details 

def describe_out_port_map(tree_depth, n_tree):
  """
    Assumption:
      All PEs have output ports
      No crossbar for output
  """
  # Dict of dicts
  # Key_1: Tree_id
  # Key_2: lvl
  # Key_3: pe_id
  # Val: Set of banks
  out_port_map= {}

  """
  #    two banks for PEs at all level except level 2
  #    four banks for level 2 PEs
  for tree in range(n_tree):
    base_bank= tree*(2**tree_depth)

    for rev_lvl in range(tree_depth):
      lvl= tree_depth-rev_lvl
      for pe in range(2**rev_lvl):
        if lvl != 2: # Two banks per PE
          start= base_bank + pe*(2**lvl) + 2**(lvl-1) - 1
          end= base_bank + pe*(2**lvl) + 2**(lvl-1) + 1
        else: # lvl=2, four banks per PE
          start= base_bank + pe*(2**lvl) + 2**(lvl-1) - 2
          end= base_bank + pe*(2**lvl) + 2**(lvl-1) + 2
        
        banks= set(range(start,end))
        out_port_map[(tree,lvl,pe)]= banks
  """
  
  #    2 banks for 1st level, 4 banks for 2nd level and so on
  for tree in range(n_tree):
    base_bank= tree*(2**tree_depth)

    for rev_lvl in range(tree_depth):
      lvl= tree_depth-rev_lvl
      for pe in range(2**rev_lvl):
        start= base_bank + pe*(2**lvl) 
        end= base_bank + (pe+1)*(2**lvl) 
        
        banks= set(range(start,end))
        out_port_map[(tree,lvl,pe)]= banks

  return out_port_map

def hw_models_for_all_combinations(hw_depth, max_depth, level_for_hybrid, list_of_hw_lists):
  """
    Creates all combinations of trees that can be mapped in a Hybrid BB of hw_depth
    E.G:
      For hw_depth == 4,
      All (unique) possible combinations of trees are
      4
      3,3
      3,2,2
      3,2,1,1
      3,1,1,1,1
      2,2,2,2
      2,2,2,1,1
      2,2,1,1,1,1
      2,1,1,1,1,1,1
      1,1,1,1,1,1,1,1
  """
  ## -- Autogenerate hybrid-depth hybrid BBs
  
  #-- First, list out all combinations of possible depths
  iter_list_of_depth_lists=[]
  for level in range(0, level_for_hybrid):
    depth_list= []
    for i in range(0,2**level):
      depth_list.append(hw_depth - level)
    iter_list_of_depth_lists.append(depth_list)
  
  #iter_list_of_depth_lists.remove([6])
  #iter_list_of_depth_lists.remove([5,5])
  #iter_list_of_depth_lists.remove([4,4,4,4])
  
  #iter_list_of_depth_lists.remove([5])
  #iter_list_of_depth_lists.remove([4,4])
  #iter_list_of_depth_lists.remove([3,3,3,3])
  #iter_list_of_depth_lists.remove([2,2,2,2,2,2,2,2])
  #iter_list_of_depth_lists.remove([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
  
  assert max_depth <= hw_depth

  for depth in range(max_depth + 1, hw_depth + 1):
    list_to_remove= [depth] * 2**(hw_depth - depth)
    iter_list_of_depth_lists.remove(list_to_remove)

  least_depth= hw_depth - level_for_hybrid + 1
  list_of_depth_lists=[]
  for depth_list in iter_list_of_depth_lists:
    list_of_depth_lists.append(depth_list)
    curr_depth_list= list(depth_list)
    idx_to_split= len(curr_depth_list) - 1
    while idx_to_split != 0 and curr_depth_list[idx_to_split] > least_depth:
      split_level= curr_depth_list[idx_to_split]
      del curr_depth_list[idx_to_split]
      assert split_level - 1 >= least_depth
      curr_depth_list.insert(idx_to_split, split_level - 1)
      curr_depth_list.insert(idx_to_split, split_level - 1)
      
      list_of_depth_lists.append(curr_depth_list)
      
      # Idx for next iteration. Depends if we reached lowest level or not
      if split_level-1 > least_depth:
        idx_to_split = idx_to_split + 1
      else:
        idx_to_split = idx_to_split - 1
      curr_depth_list= list(curr_depth_list)
  
  print(list_of_depth_lists)
  
  #-- Next Create nx hw_models
  for depth_list in list_of_depth_lists:
    hw_list= []
    for depth in depth_list:
      G= nx.balanced_tree(2, depth, nx.DiGraph) 
      G= G.reverse(copy=False)
      hw_list.append(G)
    
    list_of_hw_lists.append(hw_list)
  
  return list_of_depth_lists

def create_isomorp_list(list_of_hw_lists):
  """
    For every nx_graph in list_of_hw_lists, it specifies a list of other nx_graphs that are isomporphic to it
  """
  
  list_of_isomorph_indices= []

  done= []
  
  for hw_set in list_of_hw_lists:
    done.append([])
    list_of_isomorph_indices.append([])    
    for nx_graph in hw_set:
      done[-1].append(False)
      list_of_isomorph_indices[-1].append(set())
  
  for i1, hw_set_i in enumerate(list_of_hw_lists):
    for i2, nx_graph_i in enumerate(hw_set_i):
      G1= nx_graph_i
      for j1, hw_set_j in enumerate(list_of_hw_lists):
        for j2, nx_graph_j in enumerate(hw_set_j):
          G2= nx_graph_j
          if (j1,j2) != (i1,i2):
            if (j1,j2) not in list_of_isomorph_indices[i1][i2]:
              if (check_isomorphism(G1, G2)):
                list_of_isomorph_indices[i1][i2].add((j1,j2))
                list_of_isomorph_indices[j1][j2].add((i1,i2))

  return list_of_isomorph_indices

def check_isomorphism(nx_graph_1, nx_graph_2):
  """
    Efficient isomorphic check
  """

  if nx_graph_1.number_of_nodes() != nx_graph_2.number_of_nodes():
    return False

  if nx_graph_1.number_of_edges() != nx_graph_2.number_of_edges():
    return False
  
  return nx.is_isomorphic(nx_graph_1, nx_graph_2)
