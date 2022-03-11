#!/usr/bin/python3

#file namings:
import os
import logging

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_path(network):
  # Input files
  #psdd
  PSDD_PATH            = f'./workloads/psdd/circuits/{network}.psdd'

  DATASET_PATH_PREFIX =  './workloads/psdd/datasets'
  TRAIN_DATASET_PATH= DATASET_PATH_PREFIX + f'{network}/{network}.train.data'
  TEST_DATASET_PATH= DATASET_PATH_PREFIX + f'{network}/{network}.test.data'
  VALID_DATASET_PATH= DATASET_PATH_PREFIX + f'{network}/{network}.valid.data'

  # sparse linear algebra
  SPARSE_MATRIX_MATLAB_PATH= './workloads/sptrsv/mat'
  SPARSE_MATRIX_MARKET_PATH= './workloads/sptrsv/MM'
  SPARSE_MATRIX_MARKET_FACTORS_PATH= './workloads/sptrsv/MM_LU/'

  # Output files
  REPORTS_PATH= './reports/' 
  LOG_PATH= './log/' 
  logger.info(f"making directories {REPORTS_PATH} and {LOG_PATH} if not already present")
  os.system(f'mkdir -p {REPORTS_PATH}')
  os.system(f'mkdir -p {LOG_PATH}')

  # naming convention in benchmarks
  OPERATION_LIST_START_LINE= 1
  OPERATION_NAME_PRODUCT= 'Product'
  OPERATION_NAME_LEAF= 'Leaf'
  OPERATION_NAME_SUM= 'Sum'
  LEAF_NODE_NAME= 'memory'
  INTERNAL_NODE_NAME= 'temp_memory'

