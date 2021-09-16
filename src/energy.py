import math

#####
from . import common_classes

def energy_est(graph, **kwargs):
  """ Estimates energy to be consumed in AC for fixed/float representation
  
  Inputs:
    graph: The AC. This is to calculate the number of adders and multipliers
    kwargs:
    'precision' = FULL or CUSTOM
    'arith_type' = FIXED or FLOAT
    
    Only valid if 'arith_type' is FIXED:
      'int' = Number of integer bits
      'frac' = Number of fraction bits
    
    Only valid if 'arith_type' is FLOAT:
      'exp' = Number of exponent bits
      'mant' = Number of mantissa bits

    Example:
      kwargs= {'precision': 'CUSTOM', 'arith_type' : 'FIXED', 'int': 8, 'frac': 23}

  Output: 
    returns Energy in pJ and the best 
  """
  
  ## Parsing and assert inputs
  precision= 'FULL'
  if kwargs is not None:
    if 'precision' in kwargs:
      assert kwargs['precision'] in ['FULL', 'CUSTOM']
      if kwargs['precision'] == 'CUSTOM':
        precision= 'CUSTOM'

    if precision == 'CUSTOM':
      assert 'arith_type' in kwargs, "arith_type has to be passed if precision is CUSTOM"
      assert kwargs['arith_type'] in ['FIXED', 'FLOAT']
      arith_type= kwargs['arith_type']

      if arith_type == 'FIXED':
        assert 'int' in kwargs, "number of int bits has to be passed if arith_type is FIXED"
        assert kwargs['int'] > 0, "number of int bits should be in range (0,50)"
        int_bits= kwargs['int']
        
        assert 'frac' in kwargs, "number of frac bits has to be passed if arith_type is FIXED"
        assert kwargs['frac'] > 0, "number of frac bits should be in range (0,50)"
        frac_bits= kwargs['frac']
        
      if arith_type == 'FLOAT':
        assert 'exp' in kwargs, "number of exp bits has to be passed if arith_type is FLOAT"
        assert kwargs['exp'] > 0, "number of exp bits should be in range (0,50)"
        exp_bits= kwargs['exp']
        
        assert 'mant' in kwargs, "number of mant bits has to be passed if arith_type is FLOAT"
        assert kwargs['mant'] > 0, "number of mant bits should be in range (0,50)"
        mant_bits= kwargs['mant']



  assert graph is not None
  assert precision is not 'FULL'
  ## 
  ## Actual functionality
  
  add_count=0
  mul_count=0

  for node, obj in list(graph.items()):
    if obj.operation_type == common_classes.OPERATOR.SUM:
      add_count += 1

    if obj.operation_type == common_classes.OPERATOR.PRODUCT:
      mul_count += 1
  
  if arith_type == 'FIXED':
    add_op_e= fix_add_energy(int_bits, frac_bits)
    mul_op_e= fix_mul_energy(int_bits, frac_bits)
  elif arith_type == 'FLOAT':
    add_op_e= flt_add_energy(exp_bits, mant_bits)
    mul_op_e= flt_mul_energy(exp_bits, mant_bits)
  
  ADD_E = add_count * add_op_e
  MUL_E = mul_count * mul_op_e

  TOT_E= ADD_E + MUL_E

  return TOT_E

def fix_add_energy(int_bits, frac_bits):
  return 7.8*(int_bits + frac_bits)

def fix_mul_energy(int_bits, frac_bits):
  N= int_bits + frac_bits
  
  E= 1.9 * N * N* math.log(N,2)
  
  return E

def flt_add_energy(exp_bits, mant_bits):
  return 44.74 * (mant_bits + 1)

def flt_mul_energy(exp_bits, mant_bits):
  M= mant_bits
  E= 2.9 * (M+1) * (M+1) * math.log((M+1),2)
  
  return E  
