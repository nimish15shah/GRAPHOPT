#!/usr/bin/python3

#file namings:
import os
import logging

network= None

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PSDD_PATH_PREFIX = './workloads/psdd/circuits/'
PSDD_PATH        = None
DATASET_PATH_PREFIX =  './workloads/psdd/datasets/'
TRAIN_DATASET_PATH= None
TEST_DATASET_PATH= None
VALID_DATASET_PATH= None

# sparse linear algebra
SPARSE_MATRIX_MATLAB_PATH= './workloads/sptrsv/mat/'
SPARSE_MATRIX_MARKET_PATH= './workloads/sptrsv/MM/'
SPARSE_MATRIX_MARKET_FACTORS_PATH= './workloads/sptrsv/MM_LU/'

# Output files
REPORTS_PATH= './reports/' 
LOG_PATH= './log/' 
NO_BACKUP_PATH= './no_backup/' 
PARTITIONS_PATH= NO_BACKUP_PATH + 'superlayers/'
OPENMP_PATH= NO_BACKUP_PATH + 'openmp/'
logger.info(f"making directories {REPORTS_PATH}, {LOG_PATH}, {NO_BACKUP_PATH} if not already present")
os.system(f'mkdir -p {REPORTS_PATH}')
os.system(f'mkdir -p {LOG_PATH}')
os.system(f'mkdir -p {NO_BACKUP_PATH}')
os.system(f'mkdir -p {PARTITIONS_PATH}')
os.system(f'mkdir -p {OPENMP_PATH}')

# naming convention in benchmarks
OPERATION_LIST_START_LINE= 1
OPERATION_NAME_PRODUCT= 'Product'
OPERATION_NAME_LEAF= 'Leaf'
OPERATION_NAME_SUM= 'Sum'
LEAF_NODE_NAME= 'memory'
INTERNAL_NODE_NAME= 'temp_memory'

def refresh(network):
  global PSDD_PATH 
  global TRAIN_DATASET_PATH
  global TEST_DATASET_PATH
  global VALID_DATASET_PATH

  PSDD_PATH            = f'./workloads/psdd/circuits/{network}.psdd'

  TRAIN_DATASET_PATH= DATASET_PATH_PREFIX + f'{network}/{network}.train.data'
  TEST_DATASET_PATH= DATASET_PATH_PREFIX + f'{network}/{network}.test.data'
  VALID_DATASET_PATH= DATASET_PATH_PREFIX + f'{network}/{network}.valid.data'

