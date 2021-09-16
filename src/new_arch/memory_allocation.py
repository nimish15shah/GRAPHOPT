
import random
import logging
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class mem_alloc_detail_class():
  def __init__(self):
    self.store_in_local_mem= False
    self.store_in_global_mem= False

    self.bank= None
    self.pos= None

def main(graph, status_dict, list_of_schedules, hw_details, final_output_nodes):
  local_or_global(graph, status_dict, hw_details, final_output_nodes)
  
  memory_allocation(status_dict, graph, list_of_schedules, hw_details, final_output_nodes)

def memory_allocation(status_dict, graph, list_of_schedules, hw_details, final_output_nodes):
  """
    Updates information of memory_allocation in status_dict (and also at some places in full_instr)
  """
  LOCAL_MEM_DEPTH= hw_details.LOCAL_MEM_DEPTH
  GLOBAL_MEM_DEPTH= hw_details.GLOBAL_MEM_DEPTH
  N_PE= hw_details.N_PE

  n_global_barriers= len(list_of_schedules[0])

  # highest global barrier when it is consumed.
  map_node_to_max_global_barrier= {}

  # Iterate over all PEs before crossing global barriers
  for global_barrier_idx in range(n_global_barriers): 
    for pe_schedule in list_of_schedules:
      schedule = pe_schedule[global_barrier_idx]
      for instr in schedule:
        if instr.to_load_0 == True:
          map_node_to_max_global_barrier[instr.load_0_node]= global_barrier_idx

        if instr.to_load_1 == True:
          map_node_to_max_global_barrier[instr.load_1_node]= global_barrier_idx
  
  # keep track of allocated nodes to assert
  allocated= set()
  deallocated= set()

  global_free_pos= [set(range(GLOBAL_MEM_DEPTH)) for _ in range(N_PE)]
  local_free_pos= [set(range(LOCAL_MEM_DEPTH)) for _ in range(N_PE)]
  
  # allocate leaf nodes
  for node, obj in list(status_dict.items()):
    if obj.is_leaf():
      allocate(node, graph, status_dict, global_free_pos, local_free_pos, allocated, final_output_nodes)

  # logger.info(f'global_free_pos: {[len(x) for x in global_free_pos]}')
  # logger.info(f'local_free_pos: {[len(x) for x in local_free_pos]}')

  # allocate/deallocate a memory position to applicable nodes
  # Iterate over all PEs before crossing global barriers
  for global_barrier_idx in range(n_global_barriers): 
    # allocate
    for pe_id, pe_schedule in enumerate(list_of_schedules):
      schedule = pe_schedule[global_barrier_idx]
      for instr in schedule:
        if instr.to_store:
          mem_obj= allocate(instr.node, graph, status_dict, global_free_pos, local_free_pos, allocated, final_output_nodes)
          assert not isinstance(mem_obj.bank, list)
          instr.store_addr = mem_obj

        if instr.to_load_0 == True:
          instr.load_0_addr = create_mem_obj(instr.load_0_node, status_dict, pe_id)
        if instr.to_load_1 == True:
          instr.load_1_addr = create_mem_obj(instr.load_1_node, status_dict, pe_id)

    # deallocate
    for pe_schedule in list_of_schedules:
      schedule = pe_schedule[global_barrier_idx]
      for instr in schedule:
        if instr.to_load_0:
          node= instr.load_0_node
          if (not status_dict[node].is_leaf()) and \
             (not node in deallocated) and \
             (not node in final_output_nodes):
            if map_node_to_max_global_barrier[node] == global_barrier_idx:
              deallocate(node, status_dict, global_free_pos, local_free_pos, allocated, deallocated)

        if instr.to_load_1:
          node= instr.load_1_node
          if (not status_dict[node].is_leaf()) and \
             (not node in deallocated) and \
             (not node in final_output_nodes):
            if map_node_to_max_global_barrier[node] == global_barrier_idx:
              deallocate(node, status_dict, global_free_pos, local_free_pos, allocated, deallocated)
  
  # Sanity checks
  leaf_cnt= len([obj for obj in list(status_dict.values()) if obj.is_leaf()])
  assert len(allocated) - leaf_cnt == len(final_output_nodes), allocated
  allocated_cnt= 0
  for free_pos in global_free_pos:
    allocated_cnt += GLOBAL_MEM_DEPTH - len(free_pos)
  
  for free_pos in local_free_pos:
    allocated_cnt += LOCAL_MEM_DEPTH - len(free_pos)
  #assert allocated_cnt == leaf_cnt + 1, "{} {}".format(allocated_cnt, leaf_cnt)
  
  for status_obj in list(status_dict.values()):
    if status_obj.to_be_stored:
      assert status_obj.bank != None
      assert status_obj.pos != None


def create_mem_obj(node, status_dict, bank):
  status_obj= status_dict[node]
  
  if not status_obj.is_leaf():
    assert status_obj.to_be_stored == True
    mem_obj= mem_alloc_detail_class()
    mem_obj.bank= status_obj.bank
    mem_obj.pos= status_obj.pos

    if status_obj.store_in_global_mem:
      mem_obj.store_in_global_mem = True
    elif status_obj.store_in_local_mem:
      mem_obj.store_in_local_mem = True
    else:
      assert 0
  else: # leaf
    mem_obj= mem_alloc_detail_class()
    if status_obj.store_in_global_mem:
      mem_obj.store_in_global_mem = True
      mem_obj.bank= status_obj.bank[0]
      mem_obj.pos= status_obj.pos[0]
    elif status_obj.store_in_local_mem:
      mem_obj.store_in_local_mem = True
      mem_obj.bank= bank
      idx= status_obj.bank.index(bank)
      mem_obj.pos= status_obj.pos[idx]
    else:
      assert 0

  return mem_obj

def allocate(node, graph, status_dict, global_free_pos, local_free_pos, allocated, final_output_nodes):
  status_obj= status_dict[node]
  
  if not status_obj.is_leaf():
    assert status_obj.to_be_stored == True
  
    bank= status_obj.pe_id
    status_obj.bank= bank
    if status_obj.store_in_global_mem:
      if len(global_free_pos[bank]) == 0:
        raise OverflowError(f"global memory bank: {bank} out of memory")
      # final_node allocated to final memory
      #if len(graph[node].parent_key_list) == 0:
      if node in final_output_nodes:
        pos= max(global_free_pos[bank])
      else: # any location
        pos= min(global_free_pos[bank])

      status_obj.pos= pos
      global_free_pos[bank].remove(pos)
#        status_obj.pos= global_free_pos[bank].pop()

    elif status_obj.store_in_local_mem:
      if len(local_free_pos[bank]) == 0:
        raise OverflowError(f"local memory bank: {bank} out of memory")
      status_obj.pos= local_free_pos[bank].pop()
      if node in final_output_nodes:
        assert 0, "final node needs to go to global mem"
    else:
      assert 0
    
  else: # leaf
    bank_ls= status_obj.pe_id
    status_obj.bank= bank_ls
    pos_ls= []
    for bank in bank_ls:
      if status_obj.store_in_global_mem:
        if len(global_free_pos[bank]) == 0:
          raise OverflowError(f"global memory bank: {bank} out of memory")
        pos= global_free_pos[bank].pop()
        pos_ls.append(pos)
      elif status_obj.store_in_local_mem:
        if len(local_free_pos[bank]) == 0:
          raise OverflowError(f"local memory bank: {bank} out of memory")
        pos= local_free_pos[bank].pop()
        pos_ls.append(pos)
        if node in final_output_nodes:
          assert 0, "final node needs to go to global mem"
      else:
        assert 0
    status_obj.pos= pos_ls

    
  assert node not in allocated, node
  allocated.add(node)

  
  if not status_obj.is_leaf():
    mem_obj= create_mem_obj(node, status_dict, bank)
    return mem_obj
  else:
    return None
  

def deallocate(node, status_dict, global_free_pos, local_free_pos, allocated, deallocated):
  assert node in allocated, node
  
  status_obj= status_dict[node]

  bank= status_obj.bank
  pos= status_obj.pos

  if status_obj.store_in_global_mem:
    global_free_pos[bank].add(pos)
  elif status_obj.store_in_local_mem:
    local_free_pos[bank].add(pos)
  else:
    assert 0

  allocated.remove(node)  
  deallocated.add(node)

  
def local_or_global(graph, status_dict, hw_details, final_output_nodes):
  #assert mode in ['uniquify_leaf', 'replicate_leaf']

  N_PE= hw_details.N_PE

  for node, status_obj in list(status_dict.items()):
    if status_obj.to_be_stored:
      curr_pe_id= status_obj.pe_id
      assert curr_pe_id != None 

      status_obj.store_in_local_mem= True # overwritten below if needed

      parent_pe_id= set([status_dict[parent].pe_id for parent in graph[node].parent_key_list])
      extra_parent_pe_id= parent_pe_id - set([curr_pe_id])
#      for parent in graph[node].parent_key_list:
#        parent_pe_id = status_dict[parent].pe_id
#        assert parent_pe_id != None
#
#        if parent_pe_id != curr_pe_id:
#          status_obj.store_in_local_mem= False
#          status_obj.store_in_global_mem= True
#          break

      # either final node, or parent in other pe
      if len(parent_pe_id) == 0 or len(extra_parent_pe_id) != 0 or (node in final_output_nodes):
        status_obj.store_in_local_mem= False
        status_obj.store_in_global_mem= True

  global_mem_cnt =0
  local_mem_cnt =0

  for node, obj in list(status_dict.items()):
    if not graph[node].is_leaf():
      if obj.store_in_global_mem == True:
        global_mem_cnt += 1
      if obj.store_in_local_mem == True:
        local_mem_cnt += 1

  logger.info(f"Total stores: {len([node for node, obj in list(status_dict.items()) if obj.to_be_stored])}")
  logger.info(f"Global mem: {global_mem_cnt}")
  logger.info(f"Local mem: {local_mem_cnt}")

  
  # reserve 1/4 local memory for intermediate results
  LOCAL_MEM_DEPTH= (3*hw_details.LOCAL_MEM_DEPTH)//4
  local_free_pos= [set(range(LOCAL_MEM_DEPTH)) for _ in range(N_PE)]
  leaf_list= [node for node, obj in list(graph.items()) if obj.is_leaf()]
  leaf_list= sorted(leaf_list, key= lambda x: len(graph[x].parent_key_list), reverse= True)
  # Assign leaves to memory
  # such that leaves with more parents are assigned to local memory first
  for node in leaf_list:
    obj= graph[node]
    pe_set= set([status_dict[parent].pe_id for parent in obj.parent_key_list])
    
    local_flag= True
    for pe in pe_set:
      if len(local_free_pos[pe]) == 0:
        local_flag= False

    #if len(pe_set) == 1 and local_flag:
    if local_flag:
      status_dict[node].store_in_local_mem= True
      for pe in pe_set:
        local_free_pos[pe].pop()
    else:
#        status_dict[node].store_in_local_mem= True

      status_dict[node].store_in_global_mem= True
#        status_dict[node].store_in_local_mem= True
    
    #status_dict[node].pe_id= pe_set.pop() # pe in which this node will be consumed at least once
    if status_dict[node].store_in_local_mem:
      #status_dict[node].pe_id= pe_set.pop() 
      status_dict[node].pe_id= list(pe_set)
    elif status_dict[node].store_in_global_mem:
      status_dict[node].pe_id= [random.choice(list(range(N_PE)))]
      #log.info node, status_dict[node].pe_id, len(obj.parent_key_list), pe_set

  leaf_global_cnt= 0
  leaf_local_cnt= 0
  for node, status_obj in list(status_dict.items()):
    if status_obj.is_leaf():
      if status_obj.store_in_global_mem:
        leaf_global_cnt += 1
      elif status_obj.store_in_local_mem:
        leaf_local_cnt += 1
      else:
        assert 0
  
  logger.info(f"Leaves global: {leaf_global_cnt}")
  logger.info(f"Leaves local: {leaf_local_cnt}")
