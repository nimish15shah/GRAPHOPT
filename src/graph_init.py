
import time
import re
import queue
import networkx as nx
import logging

#**** imports from our codebase *****
from . import common_classes
from . import useful_methods
from . import evidence_analysis
from . import logistic_circuits
from . import psdd

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def construct_graph(self, args, global_var):
  verbose= args.v
  mode= args.tmode

  #----GRAPH CONSTRUCTION
  if args.cir_type == 'ac':
    construct_graph_from_ac(self, args, global_var)

  elif args.cir_type == 'log':
    self.graph, self.graph_nx, self.head_node, self.leaf_list, self.ac_node_list = logistic_circuits.parse_logistic_circuit(args, global_var) 
  
  elif args.cir_type == 'psdd':
    self.graph, self.graph_nx, self.head_node, self.leaf_list, self.ac_node_list = psdd.main(global_var.PSDD_PATH)


  # Sanity check
  assert nx.algorithms.dag.is_directed_acyclic_graph(self.graph_nx)
  assert nx.algorithms.components.is_weakly_connected(self.graph_nx)

  #--- POST-PROCESSING---
  # Mark level, reverse_level and number of nodes under the given node
  if mode== 'hw_tree_blocks' or mode=='vectorize_inputs_of_building_blocks' or mode== 'try':
    compute_reverse_level(self.graph, self.head_node)
    compute_level(self.graph, self.head_node)
    
    # Verify reverse level
    graph=self.graph
    for node, obj in list(graph.items()):
      for child in obj.child_key_list:
        assert graph[child].reverse_level < obj.reverse_level
      
      for parent in obj.parent_key_list:
        assert graph[parent].reverse_level > obj.reverse_level
  
  # Figure out which sum_nodes are associated to which BN var elimination
  if 'float' in mode or 'fixed' in mode or 'error' in mode or mode== 'output_min_max' or  mode == 'adaptively_optimize_for_Fxpt' or mode == 'munin_single_query_verify' or mode == 'max_FixPt_err_query':
    evidence_analysis.populate_BN_list(self.graph, self.BN, self.head_node, {}) 

  logger.info(f"Critical path length: {nx.algorithms.dag.dag_longest_path_length(self.graph_nx)}")


def construct_graph_from_ac(self, args, global_var):
  verbose= args.v
  mode= args.tmode
  
  # Creating op_list from AC file  
  op_list, self.leaf_list, self.head_node= reorder_operations(self.use_ac_file, global_var)
  
  # Create graph from op_list
  self.graph = _create_graph_from_op_list(self, op_list, global_var)
  self.ac_node_list= list(self.graph.keys())
  
  if 'float' in mode or 'fixed' in mode or 'error' in mode or mode== 'output_min_max' or  mode == 'adaptively_optimize_for_Fxpt' or mode == 'munin_single_query_verify' or mode == 'max_FixPt_err_query':
    # Creating BN graph from BN file (*.net)
    net_fp= open(global_var.NET_FILE, 'r')
    net_content= net_fp.readlines()
    self.BN= read_net(net_content)
    
    # Read lmap file and update the state of object accordingly
    lmap_fp= open(global_var.LMAP_FILE, 'r')
    lmap_content= lmap_fp.readlines()
    read_lmap(self, lmap_content)
   

  
  # Create networkx (nx) graph for AC
  self.graph_nx= useful_methods.create_nx_graph_from_node_graph(self.graph)
  
def compute_level(graph, head_node):
  highest_level= 0
  graph[head_node].level= 0
  
  open_set= queue.Queue()
  open_set.put(head_node)

  closed_set= []
  while not open_set.empty():
    curr_node = open_set.get()
    obj= graph[curr_node]
    
    child_level= obj.level + 1
    for child in obj.child_key_list:
      if graph[child].level == None:
        graph[child].level= child_level
        open_set.put(child)
        if child_level > highest_level:
          highest_level= child_level
    
      elif graph[child].level < child_level: 
        # Same thing as above
        graph[child].level= child_level
        open_set.put(child)
        if child_level > highest_level:
          highest_level= child_level

    closed_set.append(curr_node)
  
  # Offset levels such that
  for node, obj in list(graph.items()):
    obj.level= highest_level- obj.level
  
  # Sanity Check
  for node, obj in list(graph.items()):
    for child in obj.child_key_list:
      assert graph[child].level < obj.level
    
    for parent in obj.parent_key_list:
      assert graph[parent].level > obj.level

def compute_reverse_level(graph, curr_node):
  obj= graph[curr_node]
  
  if obj.reverse_level is None:
    if not obj.child_key_list: # leaf node
      obj.reverse_level= 0
    else: # Not a leaf
      max_level= 0
      for child in obj.child_key_list:
        lvl= compute_reverse_level(graph, child)
        if lvl > max_level:
          max_level= lvl

      obj.reverse_level= max_level + 1
  
  return obj.reverse_level
      

def reorder_operations(use_ac_file, global_var):
  op_list= _read_operation_list(use_ac_file, global_var, 'true')
  
  if not use_ac_file: #Don't add leaf nodes while using .ac file as leaves are already there
    n_leaf, leaf_list= _find_n_leaves(use_ac_file, op_list, global_var)
    op_list= _add_leaf_nodes(global_var, op_list, n_leaf)
    op_list, map_BinaryOpListKey_To_OriginalKey= _convert_to_binary_tree(op_list, global_var)   
  else:
    op_list, map_BinaryOpListKey_To_OriginalKey= _convert_to_binary_tree(op_list, global_var)   
    n_leaf, leaf_list= _find_n_leaves(use_ac_file, op_list, global_var)

  return op_list, leaf_list, op_list[-1][0] 

def _create_graph_from_op_list(self, op_list, global_var):
  graph= {}
  for item in op_list:
    node_key= item[0]
    
    op_type= 0
    if (item[1] == global_var.OPERATION_NAME_PRODUCT):
      child_key_list= item[2:]
      op_type= common_classes.OPERATOR.PRODUCT
    elif (item[1] == global_var.OPERATION_NAME_SUM):
      child_key_list= item[2:]
      op_type= common_classes.OPERATOR.SUM
    elif (item[1] == global_var.OPERATION_NAME_LEAF):
      child_key_list= []
      op_type= common_classes.OPERATOR.LEAF
    else:
      print("Error in op_type while creating graph from op_list")
      exit(1)

    self._add_node(graph, node_key, child_key_list, op_type)
    
    if (op_type == common_classes.OPERATOR.LEAF):
      # Mark the leaf node as computed
      graph[node_key].computed= 1
    
      # Note literal corresponding to leaf node (will be useful while consuming lmap)
      literal_val = int(item[2])
      graph[node_key].leaf_literal_val = literal_val
      self.map_Literal_To_LeafKey[literal_val] = node_key

  return graph

def _read_operation_list(use_ac_file, global_var, original='false'):
  if (original):
    if not use_ac_file:
      fp= open(global_var.ARITH_OP_FILE, 'r')
      op_list= fp.readlines() 
      
      # Remove first line
      op_list= op_list[1:]

      last_operation_idx= [idx for idx,arg in enumerate(op_list) if arg=='\n']
      op_list= op_list[0:last_operation_idx[0]]
      
      # remove '\n' from the end of each line
      op_list= [i[:-1] for i in op_list]
      
      return op_list
    
    else: # For directly reading ac file
      fp= open(global_var.AC_FILE, 'r')
      ac= fp.readlines()
      new_ac= read_ac_file(ac, global_var)

      return new_ac
  else:
    with open(global_var.ARITH_OP_MERGED_FILE, 'rb') as f:
      op_list = pickle.load(f)      
      return op_list

def _find_n_leaves(use_ac_file, op_list, global_var):
  if not use_ac_file:
    # Find the last operation that use a leaf operand
    memory_indices= [idx for idx,arg in enumerate(op_list) if ','+ global_var.LEAF_NODE_NAME +',' in arg]
    
    last_op_with_mem= op_list[memory_indices[-1]]
    
    last_op_with_mem= last_op_with_mem.split(',')
    
    # Check the index associated with leaf operand
    memory_indices= [idx for idx,arg in enumerate(last_op_with_mem) if (global_var.LEAF_NODE_NAME in arg) and (global_var.INTERNAL_NODE_NAME not  in arg)]
    last_mem_idx= int(last_op_with_mem[memory_indices[-1]+1])
    
    return last_mem_idx + 1, list(range(self.n_leaf)) 
  
  else:
    # Check indices in op_list that have 'Leaf' in it
    leaves_list= useful_methods.indices(op_list, lambda x: global_var.OPERATION_NAME_LEAF in x)
    
    return len(leaves_list), leaves_list

def _convert_to_binary_tree(op_list, global_var):
  total_offset= 0
  #new_op_list= op_list[:]
  new_op_list_1= []

  # This dict maps new key in the new binarized op_list to original key of the non-binary op_list
  map_BinaryOpListKey_To_OriginalKey= {}
  
  offset_list= []

  for idx, arg in enumerate(op_list):
    # Offset all numbers in arg
    arg_w_offset= []
    for opcode_idx, opcode in enumerate(arg):
      if opcode_idx == 0:
        arg_w_offset.append(opcode + total_offset)
      if opcode_idx == 1: # operator
        arg_w_offset.append(opcode)   
      if opcode_idx > 1:
        # Add offset to rest of the numbers only if this is not a leaf node
        if not arg[1] == global_var.OPERATION_NAME_LEAF:
          assert len(offset_list) > opcode, [len(offset_list),opcode, idx, arg]
          arg_w_offset.append(opcode + offset_list[opcode])
        else:
          arg_w_offset.append(opcode)

    # check if operator has more than 2 inputs
    if (len(arg) > 4):
      new_idx= idx + total_offset
      
      # remove the big instr from the list
      #del new_op_list[new_idx]
      
      # Break up the big instruction in smaller instructions
      oper_name = arg_w_offset[1]
      n_input= len(arg_w_offset) - 2
      input_list = arg_w_offset[2:]
      curr_instr_idx= new_idx
      new_instr_list= []
      while (n_input > 1):
        n_remain_input = n_input
        output_list= [] # The list that contains all the outputs of this level, to be used as inputs for the next level
        while (n_remain_input > 1):
          output_list.append(curr_instr_idx)
          
          new_instr= []
          new_instr.append(curr_instr_idx)
          new_instr.append(oper_name) # Operation name
          new_instr.append(input_list[0])
          new_instr.append(input_list[1])

          #new_op_list.insert(curr_instr_idx, new_instr)
          new_op_list_1.append(new_instr)
          
          map_BinaryOpListKey_To_OriginalKey[curr_instr_idx] = idx

          curr_instr_idx = curr_instr_idx + 1 
          input_list = input_list[2:] # remove inputs consumed in curr instruction
          n_remain_input = n_remain_input-2
        
        n_input= n_input // 2 + (n_input & 1)
        output_list = output_list + input_list # Add the last unpaired element of input list to the output list
        input_list= output_list[:] # Copy all elements of output_list to input_list for next iteration
      
      t0= time.time()
      # Add offset to all the integers in the list
      offset= curr_instr_idx - new_idx - 1
      #new_op_list= _add_offset_to_op_list(new_op_list, curr_instr_idx, new_idx, offset)
      total_offset = total_offset + offset
      
      # Record the offset in the offset list
      offset_list.append(total_offset)
      
      t1= time.time()
      #print 2,t1-t0
    
    else: # Code will enter this else when the instruction has less than 2 inputs and do not need to be expanded in a binary tree 
      offset_list.append(total_offset) # No offset to be added for this instruction, as it is not expanded in a binary tree structure
      new_op_list_1.append(arg_w_offset)
      map_BinaryOpListKey_To_OriginalKey[arg_w_offset[0]] = arg[0]
  
  return new_op_list_1, map_BinaryOpListKey_To_OriginalKey

def _add_leaf_nodes(global_var, op_list, n_leaf):
  # Split the operation string with commas
  op_list= [i.split(',') for i in op_list]
  
  # shift operation number by the number of leaf nodes
  op_list= [ [str(int(i[0])+n_leaf)] + i[1:] for i in op_list]
  
  # Also shift temp_memory indices
  for idx, arg in enumerate(op_list):
    internal_node_indices= [idx for idx,operand in enumerate(arg) if global_var.INTERNAL_NODE_NAME in operand]
    for i,j in enumerate(internal_node_indices):
      arg[j+1]= str(int(arg[j+1]) + n_leaf)
  
  # Add leaf nodes in the head of op_list
  for i in range(n_leaf-1,-1, -1):
    op_list = [[str(i), str(global_var.OPERATION_NAME_LEAF)]] + op_list
  
  # Remove strings like 'memory' and 'temp_memory' as it is no longer required
  new_op_list=[]
  for item in op_list:
    new_op_list.append([arg for arg in item if (global_var.INTERNAL_NODE_NAME != arg) and (global_var.LEAF_NODE_NAME != arg)])
  op_list= new_op_list    
  
  # Change numbers from str to int
  new_op_list= []
  for item in op_list:
    new_op_list.append([int(arg) if idx != 1 else arg for idx,arg in enumerate(item) ])
  
  op_list= new_op_list
  
  return op_list

def _add_offset_to_op_list(op_list, start_pos, start_val, offset):
  new_op_list= []
  # Add offset to the numbers bigger than start_pos
  for idx,arg in enumerate(op_list):
    temp_list= arg
    if (idx >= start_pos):
      for item_idx,item in enumerate(arg):
        #if isinstance(item, int):
        if item_idx != 1:
          if item >= start_val:
            temp_list[item_idx] = item + offset
    new_op_list.append(temp_list)
  
  return new_op_list

def read_lmap(anal_obj, lmap_content):
  """ Reads *.net.lmap file and changes attributes of the graph accordingly
  @param anal_obj: Object of class graph_analysis_c
  @param lmap_content: Output of *.net.lmap.readlines()
  """
  for line in lmap_content:
    
    # If line does not start with "cc$" it is a comment. Skip comments
    if (line[0:3] != 'cc$'):
      continue
    # If the line indicates the compile kind, then it is of the form:
    # "cc" "K" compile kind.

    # If the line indicates the mathematical space, then it is of the form:
    # "cc" "S" space
  
    # If the line is a literal count, it is of the form:
    # "cc" "N" numLogicVars.
    
    # If the line is a variable count line, then it is of the form:
    # "cc" "v" numVars
    
    # If the line is a variable line, then it is of the form:
    # "cc" "V" srcVarName numSrcVals (srcVal)+
  
    # If the line is a potential line, then it is of the form:
    # "cc" "T" srcPotName parameterCnt.
    
    # The line must be a literal description, which looks like one of the
    # following:
    #   "cc" "I" literal weight elimOp srcVarName srcValName srcVal
    #   "cc" "P" literal weight elimOp srcPotName pos+
    #   "cc" "C" literal weight elimOp
    
    # "cc" "I" maps indicator of a variable and it's state to a literal. 
    # We have to find the leaf node in AC corresponding to that literal
    if (line[3] == 'I'):
      line_split= line.split('$')

      literal= int(line_split[2])
      BNvar_nm= line_split[5]
      BNvar_state= anal_obj.BN[BNvar_nm].states[int(line_split[6])]
      
      try: # This has to be 'tried' because there may nobe a AC leaf corresponding to the literal, if the probability of that variable state happening is zero
        AC_leaf_key= anal_obj.map_Literal_To_LeafKey[literal]
      except KeyError:
        AC_leaf_key= None

      if AC_leaf_key != None:
        # Update BN with corresponding leaf keys
        anal_obj.BN[BNvar_nm].AC_leaf_dict[BNvar_state]= AC_leaf_key

        # Update AC Leaf nodes with literal details
        #anal_obj.graph[AC_leaf_key].leaf_numeric_val= 1 # Set all indicators by default
        anal_obj.graph[AC_leaf_key].leaf_type= anal_obj.graph[AC_leaf_key].LEAF_TYPE_INDICATOR
        anal_obj.graph[AC_leaf_key].leaf_BN_node_name= BNvar_nm
        anal_obj.graph[AC_leaf_key].leaf_BN_node_state= BNvar_state
        anal_obj.graph[AC_leaf_key].min_val= 1.0 # Setting min_value of an indicator as 1
        anal_obj.graph[AC_leaf_key].max_val= 1.0 # Setting max_value of an indicator as 1
        anal_obj.graph[AC_leaf_key].curr_val= 1.0 
        

    # "cc" "C" is weight of non-indicator leaf nodes in AC
    if (line[3] == 'C'):
      line_split= line.split('$')
      
      literal= int(line_split[2])
      weight= float(line_split[3])
      
      #Get the corresponding AC_leaf_key
      if literal in anal_obj.map_Literal_To_LeafKey: # It may happen that this literal had been eliminated during AC compilation.
        AC_leaf_key= anal_obj.map_Literal_To_LeafKey[literal]

        # Update AC Leaf nodes with literal details
        #anal_obj.graph[AC_leaf_key].leaf_numeric_val= weight 
        anal_obj.graph[AC_leaf_key].leaf_type= anal_obj.graph[AC_leaf_key].LEAF_TYPE_WEIGHT
        anal_obj.graph[AC_leaf_key].min_val= weight 
        anal_obj.graph[AC_leaf_key].max_val= weight
        anal_obj.graph[AC_leaf_key].curr_val= weight
  
  # TEST: Check if all the leafs in AC are either made a weight or indicator
  for leaf in anal_obj.leaf_list:
    assert anal_obj.graph[leaf].leaf_type == anal_obj.graph[leaf].LEAF_TYPE_INDICATOR or anal_obj.graph[leaf].leaf_type == anal_obj.graph[leaf].LEAF_TYPE_WEIGHT, "One of the leaf in AC has not been assigned proper type while reading LMAP, LEAF_KEY= {}".format(leaf)
            
def read_ac_file(ac_content, global_var):
  """ Consumes reads .net.ac file and creates an op_list out of it
  @param ac_content: output of ac_file.readlines()
  @output op_list: non_binarized op_list. Each element of the list corresponds to a line in ac file, with line_number = index in op_list
  """
  ac= ac_content
  # Remove first line of type "nnf 23 26 7"
  if 'nnf' in ac[0]:
    ac = ac[1:]
  
  # Determine which type of AC file it is
  # Type 1: One with A,O and L 
  # Type 2: One with *,+ and l
  ac_type_1= 0
  ac_type_2= 1
  if ac[0].split(' ')[0] == 'L':
    ac_type= ac_type_1 
  elif ac[0].split(' ')[0] == 'l':
    ac_type= ac_type_2
  else:
    print("Unknown format of net.ac file")
    exit(1)

  if ac_type == ac_type_1: # one with A,O and L
    # remove '\n' from the end of each line and split with space
    ac= [i[:-1].split(' ') for i in ac]
    
    # Remove first element in A and two elements in O
    for arg in ac:
      if (arg[0] == 'A'):
        del arg[1]
      if (arg[0] == 'O'):
        del arg[2]
        del arg [1]
    
    new_ac=[]
    for op_idx, operation in enumerate(ac):
      new_op= []
      for idx,arg in enumerate(operation):
        # Replace 'A', 'O' and 'L' with terms in op_list
        if arg == 'L':
          new_op.append(op_idx)
          new_op.append(global_var.OPERATION_NAME_LEAF)
        elif arg == 'A':
          new_op.append(op_idx)
          new_op.append(global_var.OPERATION_NAME_PRODUCT)
        elif arg == 'O':
          new_op.append(op_idx)
          new_op.append(global_var.OPERATION_NAME_SUM)

        # Change numbers from str to int
        if idx > 0:
          new_op.append(int(arg))
    
  
      new_ac.append(new_op)
    
    return new_ac
  
  elif ac_type == ac_type_2: # one with +,* and l
    new_ac = [] 
    for op_idx, operation in enumerate(ac):
      # remove '\n' from the end of each line and split with space
      operation_spl= operation[:-1].split(' ')
      
      # Remove first element in * and + line
      if ('*' in operation_spl) or ('+' in operation_spl):
        del operation_spl[1]
      
      new_op=[]
      for idx,arg in enumerate(operation_spl):
        if arg == 'l':
          new_op.append(op_idx)
          new_op.append(global_var.OPERATION_NAME_LEAF)
        elif arg == '*':
          new_op.append(op_idx)
          new_op.append(global_var.OPERATION_NAME_PRODUCT)
        elif arg == '+':
          new_op.append(op_idx)
          new_op.append(global_var.OPERATION_NAME_SUM)
        
        # Change numbers from str to int
        if idx > 0:
          new_op.append(int(arg))

      new_ac.append(new_op)
    
    return new_ac

def read_net(net_content):
  """ Reads *.net file 
  @param net_content: output of <*.net file>.readlines()
  @return BN: BN graph where each node is of type BN_node
  """
  
  # remove '\n' from the end of each line
  net_content= [i[:-1] for i in net_content]
  
  # BN graph
  # Key: BN node name (str)
  BN={}
  
  # List of BN nodes. Contains BN node name (str)
  BNnode_list= []
  
  # Local variable: Dict to collect information
  BNnode_info= {}
  
  # List for Potential information (CPT info)
  potential_list=[]
  potential_info= {}

  # Segregate information nodewise and potential-wise
  collect_node_info= False
  collect_potent_info= False
  for line_num, line in enumerate(net_content):    
    line_split= line.split(' ')        
    # Look for the lines with 'node' in the begining
    if line_split[0] == 'node':     
      node_name= line_split[1]
      BNnode_list.append(node_name)

      BN[node_name]= common_classes.BN_node(node_name)

      BNnode_info[node_name]= []
      
      # Start collecting information from next line onwards
      collect_node_info= True

      continue
    
    # Look for the lines with 'potential' in the begining
    if line_split[0] == 'potential':
      potential_name= line.split('potential')[-1]
      #remove a space from the start
      potential_name= potential_name[1:]

      potential_list.append(potential_name)
      
      potential_info[potential_name]= []
      
      # Start collecting information from next line onwards
      collect_potent_info= True
      continue


    # Collect this line if collect_node_info flag is true
    if collect_node_info:
      last_node_name= BNnode_list[-1]
      BNnode_info[last_node_name].append(line)
    
    if collect_potent_info:
      last_potent_name= potential_list[-1]
      potential_info[last_potent_name].append(line)

    # Stop collecting if hit '}'
    if collect_node_info and line_split[0] == '}':
      collect_node_info= False
      collect_potent_info= False
  
  # Process 'state' information of each node
  quoted_re= re.compile('"[^"]*"')
  for node in BNnode_info:
    info_list= BNnode_info[node]
    
    for line in info_list:
      if 'states' in line:
        states_line= line

    # Format of states line: '  states = ("state0" "state1" );'
    list_of_states= quoted_re.findall(states_line)
    list_of_states = [state[1:-1] for state in list_of_states] # Remove " from start and end

    BN[node].states= list_of_states
  
  # Generate parent-child edges and store potential values
  word_match_re= re.compile('\S+')
  for potential in potential_info:
    # Remove ( and ) from start and end
    potential_name= potential.split('(')[-1]
    potential_name= potential_name.split(')')[0]

    # See if it has | in the line
    if '|' in potential_name:
      pot_var_list= potential_name.split('|')
    else:
      pot_var_list= [potential_name]
    
    # The node for which this potential is assigned
    curr_node= word_match_re.findall(pot_var_list[0])[0]
    assert len(curr_node) != 0, "Format of .net file unrecognizable"
    
    # Parent nodes
    if len(pot_var_list) > 1:
      parent_nodes= pot_var_list[1]
    else:
      parent_nodes= []

    # Separate multiple parents, if any. Also, check if the parent string is empty or not
    if len(parent_nodes):
      parent_nodes= word_match_re.findall(parent_nodes)
    
    
    # Update BN graph attributed accordingly
    BN[curr_node].parent_list= parent_nodes
    BN[curr_node].potential= potential_info[potential]

    for node in parent_nodes:
      assert len(node)!=0, "name of the parent node cannot be empty"
      BN[node].child_list.append(curr_node)

  return BN

def read_custom_bits_file(analysis_obj, bit_content):
  """ Updates state of graph according to *.net.bits.csv files
  This file contain number of bits for AC nodes, in the order of .net.ac
  """

  graph= analysis_obj.graph
  map_BinaryOpListKey_To_OriginalKey= analysis_obj.map_BinaryOpListKey_To_OriginalKey

  bit_list= bit_content[0].split(',')
  
  #avg_bits= sum([int(bits) for bits in bit_list])/len(bit_list)
  #print avg_bits

  for key, obj in list(graph.items()):
    bit_ls_k= map_BinaryOpListKey_To_OriginalKey[key]
    obj.bits = int(bit_list[bit_ls_k])
    

