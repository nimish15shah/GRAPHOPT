
import networkx as nx
import copy
import math

# Local imports
from . import files_parser
from . import common_classes
from . import useful_methods

class node_logistic_and(common_classes.node):
  def __init__(self, node_key):
    common_classes.node.__init__(self, node_key)
    self.and_gate_parameter= None

class node_logistic_leaf(common_classes.node):
  def __init__(self, node_key):
    common_classes.node.__init__(self, node_key)
    self.leaf_parameter= None

def parse_logistic_circuit(args, global_var):
  """
    Creates self.graph, self.leaf_list etc. by reading logistic circuit file
  """

  lines= files_parser.read_logistic_circuit(global_var)

  # Create nodes in graph for first upward pass that computes node probabilities
  # Corresponds to Algorithm 1 in AAAI version of Liang's paper
  raw_graph = create_raw_graph(lines)
  

  graph= copy.deepcopy(raw_graph)


  print(len(graph))
  
  map_node_to_v = graph_for_downward_pass(graph)
  
  print(len(graph))
  
  #head_node= graph_for_dot_product(graph, raw_graph, map_node_to_v)
  
  # Create trees corresponding to dot products for each class
  dot_prod_heads=[]
  for i in range(10):
    temp_head= graph_for_dot_product(graph, raw_graph, map_node_to_v)
    dot_prod_heads.append(temp_head)
  
  #### Heads of all the dotproducts are combined with a tree inorder to create a DAG
  # Find next available key
  key_count= max(list(graph.keys())) + 1
  assert key_count-1 == dot_prod_heads[-1], [key_count-1, dot_prod_heads[-1]]

  curr_key_mutable= [key_count]
  head_node= create_tree_of_nodes(graph, dot_prod_heads, curr_key_mutable, common_classes.OPERATOR.SUM)

  binarize(graph)
  print(len(graph))
  
  graph_nx= useful_methods.create_nx_graph_from_node_graph(graph)

  leaf_list= [node for node in graph_nx.nodes() if graph_nx.in_degree(node) == 0]

  ac_node_list= list(graph.keys())
  
  sanity_check(graph, graph_nx, leaf_list, head_node)
  
  #-- TEST for tree
  #graph= {}
  #for node in leaf_list:
  #  create_weight_node(graph, node, 0.0)
  #head_node= create_tree_of_nodes(graph, leaf_list, [head_node], common_classes.OPERATOR.SUM)
  #graph_nx= useful_methods.create_nx_graph_from_node_graph(graph)
  #ac_node_list= list(graph.keys())
  
  return graph, graph_nx, head_node, leaf_list, ac_node_list

def sanity_check(graph, graph_nx, leaf_list, head_node):
  head_list= [node for node in graph_nx.nodes() if graph_nx.out_degree(node) == 0]
  assert head_list[0] == head_node and len(head_list) == 1
  
  head_list= [node for node,obj in list(graph.items()) if not obj.parent_key_list]
  assert len(head_list) == 1, head_list
  assert head_list[0] == head_node

  for leaf in leaf_list:
    assert graph[leaf].operation_type == common_classes.OPERATOR.LEAF
  
  for node, obj in list(graph.items()):
   for child in obj.child_key_list:
     assert child in graph
   
   for parent in obj.parent_key_list:
     assert parent in graph

def binarize(graph):
  """
    Binarise all the OR-nodes of first upward pass
  """
  # Find next available key
  key_count= max(list(graph.keys())) + 1
  
  for node, obj in list(graph.items()):
    if len(obj.child_key_list) > 2:
      assert obj.operation_type == common_classes.OPERATOR.SUM
      
      # Reset parent list
      for child in obj.child_key_list:
        ch_obj= graph[child]
        assert ch_obj.operation_type == common_classes.OPERATOR.PRODUCT        
        ch_obj.parent_key_list.remove(node)
      
      # Create tree of nodes
      curr_key_mutable= [key_count]
      tree_top= create_tree_of_nodes(graph, obj.child_key_list, curr_key_mutable, common_classes.OPERATOR.SUM)
      key_count= curr_key_mutable[0]

      # Reset child list
      obj.child_key_list= []

      # Replace tree_top with curr_node
      for child in graph[tree_top].child_key_list:
        graph[child].parent_key_list.remove(tree_top)
        graph[child].parent_key_list.append(node)
        obj.child_key_list.append(child)
      
      del graph[tree_top]
  
  # Add an extra child for single-input OR-gates
  for node, obj in list(graph.items()):
    if len(obj.child_key_list) == 1:
      assert obj.operation_type == common_classes.OPERATOR.SUM
      
      create_weight_node(graph, key_count, 0.0)
      dummy_node= key_count
      key_count += 1
      
      obj.child_key_list.append(dummy_node)
      graph[dummy_node].parent_key_list.append(node)

  # Assert that graph is binary
  for node, obj in list(graph.items()):
    assert len(obj.child_key_list) == 2 or len(obj.child_key_list) == 0, [len(obj.child_key_list), node, obj.operation_type]
  
def graph_for_dot_product(graph, raw_graph, map_node_to_v):
  """
    Create graph for dot_product 
  """
  
  # Find next available key
  key_count= max(list(graph.keys())) + 1
  
  mul_node_list= []
  
  for node, obj in list(raw_graph.items()):
    
    if obj.operation_type == common_classes.OPERATOR.PRODUCT or obj.operation_type == common_classes.OPERATOR.LEAF:
      
      if obj.operation_type == common_classes.OPERATOR.PRODUCT:
        weight= obj.and_gate_parameter
      elif obj.operation_type == common_classes.OPERATOR.LEAF:
        weight= obj.leaf_parameter
      else:
        assert 0
      
      create_weight_node(graph, key_count, weight)
      child_0= map_node_to_v[node]
      child_1= key_count
      key_count += 1

      create_internal_node(graph, key_count, child_0, child_1, common_classes.OPERATOR.PRODUCT)
      mul_key= key_count
      key_count += 1

      mul_node_list.append(mul_key)
    
  
  curr_key_mutable= [key_count]
  head_node= create_tree_of_nodes(graph, mul_node_list, curr_key_mutable, common_classes.OPERATOR.SUM)
  
  return head_node

def create_raw_graph(lines):
  """
    A raw graph from the lines of the logistic files

    Nodes may have multiple inputs in the raw graph 

    Makes AND-nodes explicit, which are implicit in the logistic file
  """
  
  # Map ID in file to the node key used in the graph in FIRST PASS ONLY
  # Key: Id from logistic circuit's file
  # Val: Node key in graph
  map_id_to_key_in_raw_graph= {}

  graph= {}
  
  key_count= 0
  for line in lines:

    # True or False Indicator leaf nodes
    if line[0] == 'T' or line[0] == 'F':

      curr_id= line[1]
      node_key= key_count
      key_count += 1
      
      map_id_to_key_in_raw_graph[curr_id] = node_key
      
      node_obj= node_logistic_leaf(node_key)
      
      op_type = common_classes.OPERATOR.LEAF
      node_obj.set_operation(op_type)
      
      node_obj.computed = 1
      node_obj.leaf_type= node_obj.LEAF_TYPE_INDICATOR
      node_obj.leaf_BN_node_name= line[2]
      node_obj.leaf_BN_node_type= node_obj.BN_NODE_TYPE_EVIDENCE
      
      node_obj.leaf_parameter = line[3]

      assert node_key not in graph
      graph[node_key]= node_obj
    
    # OR-node
    if line[0] == 'D':
      curr_id= line[1]
      or_node_key= key_count
      key_count += 1
      
      map_id_to_key_in_raw_graph[curr_id] = or_node_key
      
      or_node_obj= common_classes.node(or_node_key)
      
      op_type = common_classes.OPERATOR.SUM
      or_node_obj.set_operation(op_type)
      
      for and_node in line[3:]:
        and_node_key= key_count
        key_count += 1

        and_node_obj= node_logistic_and(and_node_key)
        op_type = common_classes.OPERATOR.PRODUCT
        and_node_obj.set_operation(op_type)
        
        and_node_obj.add_parent(or_node_key)
        or_node_obj.add_child(and_node_key)

        child_0= map_id_to_key_in_raw_graph[and_node[0]]
        child_1= map_id_to_key_in_raw_graph[and_node[1]]
        assert graph[child_0].operation_type == common_classes.OPERATOR.SUM or graph[child_0].operation_type == common_classes.OPERATOR.LEAF 
        assert graph[child_1].operation_type == common_classes.OPERATOR.SUM or graph[child_1].operation_type == common_classes.OPERATOR.LEAF 
        
        and_node_obj.add_child(child_0)
        and_node_obj.add_child(child_1)
        graph[child_0].add_parent(and_node_key)
        graph[child_1].add_parent(and_node_key)
        
        and_node_obj.and_gate_parameter= and_node[2]
        
        assert and_node_key not in graph
        graph[and_node_key] = and_node_obj

      assert line[2] == len(or_node_obj.child_key_list)
      assert or_node_key not in graph
      graph[or_node_key]= or_node_obj
  
  # Sanity check of node count
  node_cnt= 0
  for line in lines:
    if line[0] == 'T' or line[0] == 'F':
      node_cnt += 1
    
    if line[0] == 'D':
      node_cnt += 1
      node_cnt += line[2]

  assert node_cnt == len(graph), [node_cnt, len(graph)]
  return graph

def graph_for_downward_pass(graph):
  """
    Adds operation corresponds to Algo 2 in Liang's AAAI updated paper
    Computing features from node-probabilities
  """
  
  # To find next available key
  key_count= max(list(graph.keys())) + 1
  
  graph_nx= useful_methods.create_nx_graph_from_node_graph(graph)
  
  topo_sorted_nodes= list(reversed(list(nx.topological_sort(graph_nx))))

  
  # map nodes in original graph to the node for their respective v (nomenclature from Liang's paper) 
  # Key: node key
  # Val: key of "v" node
  map_node_to_v= {}
  
  create_weight_node(graph, key_count, 1.0)
  head_node= topo_sorted_nodes[0]
  map_node_to_v[head_node]= key_count
  
  key_count += 1
  
  # list of children for v node of SUM nodes (Corresponds to line 11 Algo 2, Liang's paper)
  # Key: Key of OR node
  # Val: List of children v_node of parent v_node
  map_node_to_list_of_v_children= {}
  
  for node in topo_sorted_nodes:
    obj= graph[node]
    
    if obj.operation_type == common_classes.OPERATOR.SUM:
      assert node in map_node_to_v      
      for child in obj.child_key_list:
        assert graph[child].operation_type == common_classes.OPERATOR.PRODUCT
        assert len(graph[child].parent_key_list) == 1
        
        create_internal_node(graph, key_count, child, node, common_classes.OPERATOR.DIV)
        div_key= key_count
        key_count += 1
        
        v_n_key= map_node_to_v[node]
        create_internal_node(graph, key_count, div_key, v_n_key , common_classes.OPERATOR.PRODUCT)
        v_key= key_count
        key_count += 1
        
        # Assumes that AND gate has only one parent OR-gate
        # Assumes a leaf node cannot be children of OR-gate
        # Without this assumption, += in line 8 of Algo 2 in Liang's paper will not be valid
        assert child not in map_node_to_v
        map_node_to_v[child]= v_key
        
      del map_node_to_v[node]

    if obj.operation_type == common_classes.OPERATOR.PRODUCT:
      assert node in map_node_to_v, node
      
      for child in obj.child_key_list:
        ch_obj= graph[child]

        assert ch_obj.operation_type == common_classes.OPERATOR.SUM or ch_obj.operation_type == common_classes.OPERATOR.LEAF, ch_obj.operation_type
        
        if child not in map_node_to_list_of_v_children:
          map_node_to_list_of_v_children[child]= []
        
        v_n_key = map_node_to_v[node]

        map_node_to_list_of_v_children[child].append(v_n_key)
        
        # Create a tree of nodes when all the v_children in line 11 (Algo 2 Liang's paper) are identified
        if len(ch_obj.parent_key_list) == len(map_node_to_list_of_v_children[child]):
          node_list= map_node_to_list_of_v_children[child]
          curr_key_mutable= [key_count]
          v_key= create_tree_of_nodes(graph, node_list, curr_key_mutable, common_classes.OPERATOR.SUM)
          
          if len(ch_obj.parent_key_list) == 1:
            assert v_key == node_list[0]
            assert key_count == curr_key_mutable[0]
          else:
            assert key_count < curr_key_mutable[0]
          
          key_count = curr_key_mutable[0]
          
          assert child not in map_node_to_v
          map_node_to_v[child]= v_key
          
          del map_node_to_list_of_v_children[child]
          
  assert len(map_node_to_list_of_v_children) == 0
  
  return map_node_to_v

def create_internal_node(graph, key, child_0, child_1, op_type):
  """
    Create an internal node of given op_type
  """
  assert child_0 != child_1
  assert isinstance(key, int)

  node_obj= common_classes.node(key)

  node_obj.set_operation(op_type)

  node_obj.add_child(child_0)
  node_obj.add_child(child_1)
  graph[child_0].add_parent(key)
  graph[child_1].add_parent(key)
  
  assert key not in graph
  graph[key]= node_obj

def create_weight_node(graph, key, weight):
  """
    Creates a leaf node for a weight/parameter
  """
  assert isinstance(key, int)
  
  node_obj= common_classes.node(key)
  
  op_type = common_classes.OPERATOR.LEAF
  node_obj.set_operation(op_type)
  
  node_obj.computed = 1
  node_obj.leaf_type= node_obj.LEAF_TYPE_WEIGHT
  node_obj.curr_val= weight

  assert key not in graph, key
  graph[key]= node_obj

def graph_for_first_upward_pass(graph, lines):
  """
    # Create nodes in graph for first upward pass that computes node probabilities
    # Corresponds to Algorithm 1 in AAAI version of Liang's paper
  """
  # Map ID in file to the node key used in the graph in FIRST PASS ONLY
  # Key: Id from logistic circuit's file
  # Val: Node key in graph
  map_id_to_key_in_first_pass= {}
  
  # A dict to keep all the raw file details in one place
  # Key: Id from logistic file
  # Val: Line corresponding to it in the file
  raw_file_details= {}

  graph= {}
  key_count= 0
  for line in lines:
    
    # True or False Indicator leaf nodes
    if line[0] == 'T' or line[0] == 'F':

      curr_id= line[1]
      node_key= key_count
      key_count += 1
      
      map_id_to_key_in_first_pass[curr_id] = node_key
      raw_file_details[curr_id]= line
      
      node_obj= common_classes.node(node_key)
      
      op_type = common_classes.OPERATOR.LEAF
      node_obj.set_operation(op_type)
      
      node_obj.computed = 1
      node_obj.leaf_type= node_obj.LEAF_TYPE_INDICATOR
      node_obj.leaf_BN_node_name= line[2]
      node_obj.leaf_BN_node_type= node_obj.BN_NODE_TYPE_EVIDENCE

      graph[node_key]= node_obj
    
    # OR-node
    if line[0] == 'D':
      curr_id= line[1]

def create_tree_of_nodes(graph, node_list, curr_key_mutable, op_type):
  """
    An OR-gate can have more than 2-inputs. This creates a tree of 2-input sum nodes

    curr_key_mutable is [curr_key], i.e. curr_key as an element of list
  """
  
  len_node_list= len(node_list)
  
  assert len_node_list > 0
  
  assert isinstance(curr_key_mutable, list)

  if len_node_list == 1:
    assert node_list[0] in graph
    return node_list[0]
  
  if len_node_list > 1:
    # Slicing index is the biggest power of 2 smaller than len_node_list
    biggest_power_2= int(math.log(len_node_list,2))
    slicing_idx= 2**biggest_power_2
    if len_node_list == slicing_idx:
      slicing_idx /= 2

    child_0= create_tree_of_nodes(graph, list(node_list[ : slicing_idx]), curr_key_mutable, op_type)
    child_1= create_tree_of_nodes(graph, list(node_list[slicing_idx : ]), curr_key_mutable, op_type)
    
    key= curr_key_mutable[0]
    curr_key_mutable[0] += 1
    
    create_internal_node(graph, key, child_0, child_1, op_type)
    
    return key


  

