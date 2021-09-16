#-----------------------------------------------------------------------
# Created by         : KU Leuven
# Filename           : src/new_arch/register_allocation.py
# Author             : Nimish Shah
# Created On         : 2019-11-19 17:09
# Last Modified      : 
# Update Count       : 2019-11-19 17:09
# Description        : 
#                      
#-----------------------------------------------------------------------

from . import instr_types

class reg_alloc_class():
  def __init__(self, node):
    self.node= node
    
    self.liverange_start= None
    self.liverange_end= None
    
    self.reg= None

    self.generated_in_partition= None

  def is_reg_allocated(self):
    if self.reg != None:
      return True
    else:
      return False
  
  def skip_reg_alloc(self):
    if self.liverange_end == self.liverange_start + 1:
      return True
    else:
      return False

def linear_scan(graph, local_graph, schedule, nodes_to_store, hw_details):
  ## Code inspired from the paper "Linear Scan Register Allocation" 
  
  # NOTE: local_graph should only contain nodes generated in this partition
  # and not the inputs

  reg_alloc_obj_dict = liverange_analysis(graph, local_graph, schedule)
    
  for node, obj in list(reg_alloc_obj_dict.items()):
    if obj.generated_in_partition:
      parent_outside_partition = set(graph[node].parent_key_list) - set(local_graph.nodes())
      if len(parent_outside_partition) != 0:
        nodes_to_store.add(node)
        assert not graph[node].is_leaf()
        
  spill_cnt= longer_register_allocation(graph, schedule, reg_alloc_obj_dict, hw_details, nodes_to_store)

  reserved_output_register_allocation(schedule, reg_alloc_obj_dict, hw_details)

  ld_st(graph, schedule, reg_alloc_obj_dict,nodes_to_store, hw_details)
  
  return spill_cnt, reg_alloc_obj_dict

def reserved_output_register_allocation(schedule, reg_alloc_obj_dict, hw_details):
  RESERVED_REG_OUT= hw_details.RESERVED_REG_OUT

  prev_out_node = None

  #  Allocate reserved regs to non-allocated nodes
  for instr_obj in schedule:
    in_0_node= instr_obj.in_0_node
    in_0_details= reg_alloc_obj_dict[in_0_node]
    if not in_0_details.is_reg_allocated():
      if in_0_node == prev_out_node:
        instr_obj.reg_in_0= RESERVED_REG_OUT
    
    in_1_node= instr_obj.in_1_node
    in_1_details= reg_alloc_obj_dict[in_1_node]
    if not in_1_details.is_reg_allocated():
      if in_1_node == prev_out_node:
        instr_obj.reg_in_1= RESERVED_REG_OUT

    node= instr_obj.node
    node_details= reg_alloc_obj_dict[node]
    if not node_details.is_reg_allocated():
      instr_obj.reg_o= RESERVED_REG_OUT
      prev_out_node= node
    else:
      prev_out_node= None
    
    

def ld_st(graph, schedule, reg_alloc_obj_dict, nodes_to_store, hw_details):
  RESERVED_REG_IN_0= hw_details.RESERVED_REG_IN_0
  RESERVED_REG_IN_1= hw_details.RESERVED_REG_IN_1
  # insert a nop for first compulsory loads
  first_loads= [node for node, obj in list(reg_alloc_obj_dict.items()) if obj.liverange_start == -1]
  if len(first_loads) == 2:
    instr_obj= instr_types.full_instr()
    instr_obj.set_op("nop")
    instr_obj.to_load_0= True
    instr_obj.to_load_1= True

    in_0= first_loads[0]
    instr_obj.load_0_node= in_0
    if reg_alloc_obj_dict[in_0].is_reg_allocated():
      load_0_reg= reg_alloc_obj_dict[in_0].reg
    else:
      load_0_reg= RESERVED_REG_IN_0
    instr_obj.load_0_reg= load_0_reg

    in_1= first_loads[1]
    instr_obj.load_1_node= in_1
    if reg_alloc_obj_dict[in_1].is_reg_allocated():
      load_1_reg= reg_alloc_obj_dict[in_1].reg
    else:
      load_1_reg= RESERVED_REG_IN_1
    instr_obj.load_1_reg= load_1_reg

    schedule.insert(0, instr_obj)
  else:
    assert len(schedule) == 0
  
  # insert loads/stores and RESERVED_REG_IN_0, RESERVED_REG_IN_1
  nodes_in_reg= set()
  for idx, obj in enumerate(schedule):
    if idx < len(schedule)-1:
      # load 0
      next_in_0_node= schedule[idx + 1].in_0_node
      next_in_0_details= reg_alloc_obj_dict[next_in_0_node]
      if obj.node != next_in_0_node:
        if not next_in_0_details.is_reg_allocated(): # load in reserved reg
          obj.to_load_0= True
          obj.load_0_node= next_in_0_node
          obj.load_0_reg= RESERVED_REG_IN_0
          assert (obj.load_0_node in nodes_to_store) or graph[obj.load_0_node].is_leaf(), [obj.load_0_node, graph[31].child_key_list, obj.node, schedule[idx+1].in_0_node, schedule[idx+1].in_1_node, schedule[idx+1].node]
        else: # allocated to register
          if not next_in_0_node in nodes_in_reg:
            obj.to_load_0= True
            obj.load_0_node= next_in_0_node
            obj.load_0_reg = next_in_0_details.reg
            assert obj.load_0_reg != None
            assert next_in_0_details.liverange_start == idx - 1
            nodes_in_reg.add(next_in_0_node)
            assert (obj.load_0_node in nodes_to_store) or graph[obj.load_0_node].is_leaf(), obj.load_0_node
        
      # load 1
      next_in_1_node= schedule[idx + 1].in_1_node
      next_in_1_details= reg_alloc_obj_dict[next_in_1_node]
      if obj.node != next_in_1_node:
        if not next_in_1_details.is_reg_allocated(): # load in reserved reg
          if obj.to_load_0: # load to 1 only if 0 is already used up
            obj.to_load_1= True
            obj.load_1_node= next_in_1_node
            obj.load_1_reg= RESERVED_REG_IN_1
          else:
            obj.to_load_0= True
            obj.load_0_node= next_in_1_node
            obj.load_0_reg= RESERVED_REG_IN_1
          assert (next_in_1_node in nodes_to_store) or graph[next_in_1_node].is_leaf()
        else: # allocated to register
          if not next_in_1_node in nodes_in_reg:
            if obj.to_load_0: # load to 1 only if 0 is already used up
              obj.to_load_1= True
              obj.load_1_node= next_in_1_node
              obj.load_1_reg= next_in_1_details.reg
            else:
              obj.to_load_0= True
              obj.load_0_node= next_in_1_node
              obj.load_0_reg= next_in_1_details.reg

            assert next_in_1_details.reg != None
            assert next_in_1_details.liverange_start == idx - 1, (next_in_1_details.liverange_start, idx, next_in_1_node, schedule[1].in_0_node, schedule[1].in_1_node, nodes_in_reg)
            nodes_in_reg.add(next_in_1_node)
            assert (next_in_1_node in nodes_to_store) or graph[next_in_1_node].is_leaf()
    
    # store
    if obj.node != None:
      if obj.node in nodes_to_store:
        obj.to_store = True

      if reg_alloc_obj_dict[obj.node].is_reg_allocated():
        nodes_in_reg.add(obj.node)

    # in_0
    if obj.reg_in_0 == None:
      obj.reg_in_0 = RESERVED_REG_IN_0

    if obj.reg_in_1 == None:
      obj.reg_in_1 = RESERVED_REG_IN_1

def longer_register_allocation(graph, schedule, reg_alloc_obj_dict, hw_details, nodes_to_store):
  REDUCED_REGBANK_L= hw_details.REGBANK_L - 3
  REGBANK_L = hw_details.REGBANK_L

  # Drop all liveranges of length == 1, 
  # as those would be routed through 3 registers set aside for it
  pruned_reg_alloc_obj_set= set([obj for obj in list(reg_alloc_obj_dict.values()) if not obj.skip_reg_alloc()])

  active= []
  free_regs= set(range(REGBANK_L)) 
  free_regs -= set([hw_details.RESERVED_REG_IN_0])
  free_regs -= set([hw_details.RESERVED_REG_IN_1])
  free_regs -= set([hw_details.RESERVED_REG_OUT ])
  assert len(free_regs) == REDUCED_REGBANK_L

  spill_cnt= 0
  
  ## Actual register allocation
  for new_reg_alloc_obj in sorted(list(pruned_reg_alloc_obj_set), key= lambda x: x.liverange_start):
    active= expire_old_intervals(active, new_reg_alloc_obj, free_regs)
    
    if len(active) == REDUCED_REGBANK_L:
      assert len(free_regs) == 0
      spill(graph, active, new_reg_alloc_obj, nodes_to_store)
      spill_cnt += 1
    else:
      new_reg_alloc_obj.reg= free_regs.pop()
      active.append(new_reg_alloc_obj)
      active.sort(key= lambda x: x.liverange_end)
  
  
  ## Update full_instr class object with the register allocation information
  for instr_obj in schedule:

    node= instr_obj.node
    node_details= reg_alloc_obj_dict[node]
    if node_details.is_reg_allocated():
      instr_obj.reg_o= node_details.reg
      
    in_0_node= instr_obj.in_0_node
    in_0_details= reg_alloc_obj_dict[in_0_node]
    if in_0_details.is_reg_allocated():
      instr_obj.reg_in_0= in_0_details.reg
    

    in_1_node= instr_obj.in_1_node
    in_1_details= reg_alloc_obj_dict[in_1_node]
    if in_1_details.is_reg_allocated():
      instr_obj.reg_in_1= in_1_details.reg

  return spill_cnt

# TODO: Use a better spill heuristic, like with liverange splitting 
def spill(graph, active, new_reg_alloc_obj, nodes_to_store):
  last_reg_alloc_obj= active[-1]
  
  if new_reg_alloc_obj.generated_in_partition: 
    if last_reg_alloc_obj.generated_in_partition: 
      if last_reg_alloc_obj.liverange_end > new_reg_alloc_obj.liverange_end : # Spill last_reg_alloc_obj
        TO_SPILL= 'last'
      else:
        TO_SPILL= 'new'
    else:
      TO_SPILL= 'last'
  else:
    if last_reg_alloc_obj.liverange_end > new_reg_alloc_obj.liverange_end : # Spill last_reg_alloc_obj
      TO_SPILL= 'last'
    else:
      TO_SPILL= 'new'
  
  if TO_SPILL == 'new':
    spilled_obj= new_reg_alloc_obj
  elif TO_SPILL == 'last':
    spilled_obj= last_reg_alloc_obj
  else:
    assert 0

  if spilled_obj.generated_in_partition:
    nodes_to_store.add(spilled_obj.node)
    assert not graph[spilled_obj.node].is_leaf()

  if TO_SPILL == 'last':
    new_reg_alloc_obj.reg= last_reg_alloc_obj.reg
    last_reg_alloc_obj.reg= None
    del active[-1]
    active.append(new_reg_alloc_obj)
    active.sort(key= lambda x: x.liverange_end)
  elif TO_SPILL == 'new': 
    new_reg_alloc_obj.reg= None
  else:
    assert 0

def spill_heuristic(active, new_reg_alloc_obj):
  None
  
def expire_old_intervals(active, new_reg_alloc_obj, free_regs):
  # active is already sorted by increasing end point
  
  new_active= list(active)
  for reg_alloc_obj in active:
    if reg_alloc_obj.liverange_end >= new_reg_alloc_obj.liverange_start:
      break

    assert reg_alloc_obj.reg != None
    free_regs.add(reg_alloc_obj.reg)

    new_active.remove(reg_alloc_obj)
  
  return new_active

def liverange_analysis(graph, local_graph, schedule):
  # tuple of [start, end] in the form of list
  liveranges= {}
  
  for instr_idx, instr_obj in enumerate(schedule):
    node= instr_obj.node

    for child in graph[node].child_key_list:
      if child not in liveranges:
        liveranges[child]= [instr_idx - 1, instr_idx]
      else:
        liveranges[child][1]= instr_idx
    
    liveranges[node]= [instr_idx, instr_idx + 1]
  
  reg_alloc_obj_dict= {}
  for node, liverange in list(liveranges.items()):
    reg_alloc_obj= reg_alloc_class(node)
    reg_alloc_obj.liverange_start= liverange[0]
    reg_alloc_obj.liverange_end= liverange[1]
    
    if node in local_graph:
      reg_alloc_obj.generated_in_partition= True
    else:
      reg_alloc_obj.generated_in_partition= False

    reg_alloc_obj_dict[node]= reg_alloc_obj


  return reg_alloc_obj_dict


