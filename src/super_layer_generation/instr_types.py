
import itertools

class full_instr():
  id_iter= itertools.count()
  isa= ['pass', 'sum', 'prod', 'nop', 'local_barrier', 'global_barrier', 'set_ld_stream_len']

  def __init__(self):
    self.id= next(self.id_iter)
    
    self.node        = None
    self.operation   = None
    
    self.in_0_node   = None
    self.in_1_node   = None

    self.reg_in_0    = None
    self.reg_in_1    = None
    self.reg_o       = None

    self.to_load_0   = None
    self.load_0_node = None
    self.load_0_addr = None
    self.load_0_reg  = None

    self.to_load_1   = None
    self.load_1_node = None
    self.load_1_addr = None
    self.load_1_reg  = None

    self.to_store    = None
    self.store_addr  = None # object of type mem_alloc_detail_class

    # applicable to set_ld_stream_len
    self.ld_stream_len = None
  
  def set_op(self, operation):
    assert operation in full_instr.isa
    self.operation = operation
  
  def is_local_barrier(self):
    if self.operation == 'local_barrier':
      return True
    else:
      return False

  def is_sum(self):
    if self.operation == 'sum':
      return True
    else:
      return False

  def is_prod(self):
    if self.operation == 'prod':
      return True
    else:
      return False
  
  def is_set_ld_stream_len(self):
    if self.operation == 'set_ld_stream_len':
      return True
    else:
      return False

  def is_pass(self):
    if self.operation == 'pass':
      return True
    else:
      return False

  def is_nop(self):
    if self.operation == 'nop':
      return True
    else:
      return False

class mini_instr():    
  id_iter= itertools.count()
  isa= ['sum', 'prod', 'nop', 'local_barrier', 'global_barrier', 'ld', 'st']

  def __init__(self, instr_type, node):
    self.id= next(self.id_iter)
    
    assert instr_type in mini_instr.isa
    self.instr_type= instr_type
    
    self.node= node

