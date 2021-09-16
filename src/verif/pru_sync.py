
from .. import common_classes
from .. import ac_eval

import FixedPointImplementation 


def main(graph, instr_ls_obj, map_param_to_addr, n_pipe_stages):
  
  # map (addr,bank) to value
  map_addr_to_val= {}
  map_addr_to_val_normal= {}
  FRAC_BITS= 26
  INT_BITS= 4

  final_node= [node for node, obj in list(graph.items()) if len(obj.parent_key_list) == 0]
  final_node= final_node.pop()

  """
  for node, addr_bank_tup in map_param_to_addr.items():
    val= random.uniform(0.5, 1)
    graph[node].curr_val= val
    map_addr_to_val[tuple(addr_bank_tup)]= FixedPointImplementation.FloatingPntToFixedPoint(val, INT_BITS, FRAC_BITS)
  print 'Normal', ac_eval.ac_eval(graph, final_node)
  """  
  
  EXP_BITS= 8
  MANT_BITS= 23
  for node, obj in list(graph.items()):
    if obj.is_leaf():
      assert node in map_param_to_addr

  init_leaf_val(graph, mode='random')

  for node, addr_bank_tup in list(map_param_to_addr.items()):
    graph[node].curr_val= val
#    map_addr_to_val[tuple(addr_bank_tup)]= val
    map_addr_to_val[tuple(addr_bank_tup)]= FixedPointImplementation.flt_to_custom_flt(val, EXP_BITS, MANT_BITS, denorm= 0)
    map_addr_to_val_normal[tuple(addr_bank_tup)]= val

  print('Normal', ac_eval.ac_eval(graph, final_node))

  simulated_val= simulate_instr_sync(graph, instr_ls_obj,map_addr_to_val_normal, n_pipe_stages) 
  print('Simulated:', simulated_val[final_node])

  result= ac_eval.ac_eval(graph, final_node, precision='CUSTOM' ,arith_type= 'FLOAT', exp= EXP_BITS, mant= MANT_BITS)
  print('AC_eval:', result)
  
#  print [(node, obj.curr_val) for node,obj in graph.items()]

  return map_addr_to_val

class reg_bank_c():
  def __init__(self, DEPTH):
    self.free_pos= set(list(range(DEPTH)))
    self.data= [None for _ in range(DEPTH)]
  
  def read(self, pos):
    assert pos not in self.free_pos
    return self.data[pos]

  def inv(self, pos):
    assert pos not in self.free_pos
    self.data[pos]= None
    self.free_pos.add(pos)

  def write(self, data):
    pos= min(list(self.free_pos))
    self.data[pos]= data
    
    self.free_pos.remove(pos)

    return pos

def reg_rd_stage(graph, instr, reg_bank, memory, curr_val):
  this_pipestage_out_data= {}
  if instr.is_type('nop'):
    pass
  elif instr.is_type('ld') or instr.is_type('ld_sc'):
    pass
  elif instr.is_type('st'):
    mem_addr= instr.mem_addr
    for node, obj in list(instr.node_details_dict.items()):
      bank= obj.bank
      data= reg_bank[bank].read(obj.pos)
      memory[bank][mem_addr]= data
      reg_bank[bank].inv(obj.pos)

  elif instr.is_type('sh'):
    map_node_to_val= {}
    for node, src_dst_tup in list(instr.sh_dict_bank.items()):
      src_bank= src_dst_tup[0]
      src_pos= instr.sh_dict_pos[node][0]
      data= reg_bank[src_bank].read(src_pos)
      map_node_to_val[node]= data

      if node in instr.invalidate_node_set:
        reg_bank[src_bank].inv(src_pos)
        
    this_pipestage_out_data= map_node_to_val

  elif instr.is_type('bb'):
    # map pe to val
    # key: pe, val: val
    map_pe_to_val= {}

    # first-level PEs
    for pe, pe_details in list(instr.pe_details.items()):
      if pe[1]==1:
        if pe_details.input_0_reg != None:
          bank, pos= pe_details.input_0_reg
          in_0 = reg_bank[bank].read(pos)
        else:
          in_0= None
        if pe_details.input_1_reg != None:
          bank, pos= pe_details.input_1_reg
          in_1 = reg_bank[bank].read(pos)

        if pe_details.is_sum():
          map_pe_to_val[pe] = in_0 + in_1
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_prod():
          assert graph[pe_details.node].is_prod()
          map_pe_to_val[pe]= in_0 * in_1
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val, [map_pe_to_val[pe], graph[pe_details.node].curr_val, pe_details.node, graph[pe_details.node].child_key_list]
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_pass_0():
          map_pe_to_val[pe]= in_0

    # second-level PEs
    for pe, pe_details in list(instr.pe_details.items()):
      if pe[1]==2:
        in_0= (pe[0], pe[1]-1, 2*pe[2])
        in_1= (pe[0], pe[1]-1, 2*pe[2] + 1)
        if pe_details.is_sum():
          assert in_0 in map_pe_to_val, [in_0, map_pe_to_val, pe]
          assert in_1 in map_pe_to_val, [in_1, map_pe_to_val, pe]
          map_pe_to_val[pe] = map_pe_to_val[in_0] + map_pe_to_val[in_1]
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_prod():
          assert in_0 in map_pe_to_val, [in_0, map_pe_to_val, pe]
          assert in_1 in map_pe_to_val, [in_1, map_pe_to_val, pe]
          map_pe_to_val[pe]= map_pe_to_val[in_0] * map_pe_to_val[in_1]
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_pass_0():
          map_pe_to_val[pe]= map_pe_to_val[in_0]
        else:
          assert 0

    # third-level PEs
    for pe, pe_details in list(instr.pe_details.items()):
      if pe[1]==3:
        in_0= (pe[0], pe[1]-1, 2*pe[2])
        in_1= (pe[0], pe[1]-1, 2*pe[2] + 1)
        if pe_details.is_sum():
          map_pe_to_val[pe] = map_pe_to_val[in_0] + map_pe_to_val[in_1]
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_prod():
          map_pe_to_val[pe]= map_pe_to_val[in_0] * map_pe_to_val[in_1]
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_pass_0():
          map_pe_to_val[pe]= map_pe_to_val[in_0]
        else:
          assert 0
    
    # Fourth level
    for pe, pe_details in list(instr.pe_details.items()):
      if pe[1]==4:
        in_0= (pe[0], pe[1]-1, 2*pe[2])
        in_1= (pe[0], pe[1]-1, 2*pe[2] + 1)
        if pe_details.is_sum():
          map_pe_to_val[pe] = map_pe_to_val[in_0] + map_pe_to_val[in_1]
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_prod():
          map_pe_to_val[pe]= map_pe_to_val[in_0] * map_pe_to_val[in_1]
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_pass_0():
          map_pe_to_val[pe]= map_pe_to_val[in_0]
        else:
          assert 0

    # Invalidate
    for  node in instr.invalidate_node_set:
      bank= instr.in_node_details_dict[node].bank
      pos= instr.in_node_details_dict[node].pos
      reg_bank[bank].inv(pos)

    this_pipestage_out_data= map_pe_to_val

  else:
    assert False

  return this_pipestage_out_data


def reg_wr_stage(instr, reg_bank, memory, write_data):
  if instr.is_type('nop'):
    pass

  elif instr.is_type('ld') or instr.is_type('ld_sc'):
    mem_addr= instr.mem_addr
    for node, obj in list(instr.node_details_dict.items()):
      bank= obj.bank
      data= memory[bank][mem_addr]
      assert data != None, [mem_addr, bank, data, node]
      pos= reg_bank[bank].write(data)
      assert pos== obj.pos

  elif instr.is_type('st'):
    pass

  elif instr.is_type('sh'):
    for node, src_dst_tup in list(instr.sh_dict_bank.items()):
      dst_bank= src_dst_tup[1]
      data= write_data[node]
      dst_pos= reg_bank[dst_bank].write(data)
      assert dst_pos == instr.sh_dict_pos[node][1]

  elif instr.is_type('bb'):

    # Outputs
    for node, pe in list(instr.output_to_pe.items()):
      pe_details= instr.pe_details[pe]
      assert node == pe_details.node
      bank, pos= pe_details.output_reg
      
      written_pos= reg_bank[bank].write(write_data[pe])
      assert pos == written_pos
      
  else:
    assert 0

def simulate_instr_sync(graph, instr_ls_obj, map_addr_to_val, n_pipe_stages):

  instr_ls= instr_ls_obj.instr_ls
  
  MEM_DEPTH= 1024
  BANK_DEPTH= 64
  N_BANKS= 32

  reg_bank= [reg_bank_c(BANK_DEPTH) for _i in range(N_BANKS)]
  memory=  [[None for _j in range(MEM_DEPTH)] for _i in range(N_BANKS)]
  
  # to store local compute value
  # key: node, val: curr_val
  curr_val= {node: obj.curr_val for node, obj in list(graph.items()) if obj.is_leaf()}
  

  # memory init
  for addr_bank_tup, data in list(map_addr_to_val.items()):
    addr= addr_bank_tup[0]
    bank= addr_bank_tup[1]
    memory[bank][addr]= data
    assert data != None
    
  # pipestages
  instr_in_pipe= [common_classes.nop_instr()]*n_pipe_stages
  data_in_pipe= [{}] * n_pipe_stages

  for idx, instr in enumerate(instr_ls):
    write_data= reg_rd_stage(graph, instr, reg_bank, memory, curr_val)
    instr_in_pipe.append(instr)
    data_in_pipe.append(write_data)

    commit_instr= instr_in_pipe.pop(0)
    write_data= data_in_pipe.pop(0)
    reg_wr_stage(commit_instr, reg_bank, memory, write_data)

  return curr_val


