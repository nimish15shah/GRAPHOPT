#!/usr/bin/python

network= 'None'

#file namings:

# NOTE: Add name of these files in refresh function

benchmarks_path= '/esat/puck1/users/nshah/benchmarks/'
# Input files
ARITH_OP_FILE        =benchmarks_path +network+'_operations_no_evidence.txt'
ARITH_OP_MERGED_FILE =benchmarks_path +network+'_merged_operations_no_evidence.txt'
AC_FILE              =benchmarks_path  + network +'.net.ac'
NET_FILE             =benchmarks_path  + network + '.net'
LMAP_FILE            =benchmarks_path  + network + '.net.lmap'
BITWIDTH_FILE        =benchmarks_path  + network + '.net.bits.csv'
MAP_INSTANCE_FILE    =benchmarks_path  + network + '.net.map'
MAP_30_INSTANCE_FILE =benchmarks_path  + network + '.net.map_30'
EVIDENCE_DB_FILE     =benchmarks_path  + network + '.net.dat' # The file that contains evidence simulated by SAMIAM
LOGISTIC_CIRCUIT_FILE=benchmarks_path  + network + '.txt' # File from Yitao Liang, UCLA
PSDD_PATH_PREFIX     =benchmarks_path + 'no_backup/PSDD/psdd_to_use/'
PSDD_PATH            =PSDD_PATH_PREFIX + network + '.psdd'

DATASET_PATH_PREFIX= '/esat/puck1/users/nshah/datasets/no_backup/'
TRAIN_DATASET_PATH= DATASET_PATH_PREFIX + network + '/' + network + '.train.data'
TEST_DATASET_PATH= DATASET_PATH_PREFIX + network + '/' + network + '.test.data'
VALID_DATASET_PATH= DATASET_PATH_PREFIX + network + '/' + network + '.valid.data'

# Output files
REPORTS_PATH= './no_backup/reports/' + network + '/'
LOG_PATH= './no_backup/log/' 

GRAPH_NX_FILES_PATH= benchmarks_path + 'graph_nx/' + network + '.nx'

METAFILE              ='./WORK/metafile_' + network
COMMON_CHILD_DEPTH_CSV='./no_backup/reports/' + network + '_COMMON_CHILD_DEPTH'
GRAPHVIZ_FILE         ='./no_backup/reports/' + network + '/' +network + '_graph.dot'
GRAPHVIZ_BN_VAR_FILE  =REPORTS_PATH +network + '_BN_var_under_AC_node.dot'
ASIP_CFILE            =REPORTS_PATH + 'ASIP_code.c'
ASM_FILE              =REPORTS_PATH + network + '.asm'
BB_FILE_PREFIX        =REPORTS_PATH + network + '_bb.p_'
BB_FILE               =REPORTS_PATH + network + '_bb.p'
BB_FILE_0             =REPORTS_PATH + network + '_bb0.p'
BB_FILE_1             =REPORTS_PATH + network + '_bb1.p'
BB_FILE_2             =REPORTS_PATH + network + '_bb2.p'
GRAPH_FILE_PREFIX     =REPORTS_PATH + network + '_gr.p_'
GRAPH_FILE            =REPORTS_PATH + network + '_gr.p'
GRAPH_FILE_0          =REPORTS_PATH + network + '_gr0.p'
GRAPH_FILE_1          =REPORTS_PATH + network + '_gr1.p'
GRAPH_FILE_2          =REPORTS_PATH + network + '_gr2.p'

EVIDENCE_WMC_GLD_PREFIX ='./no_backup/reports/' + network + '.net.dat_gld_csv' # AC val computed at full precision
EVIDENCE_WMC_LP_PREFIX  ='./no_backup/reports/' + network + '.net.dat_lp_csv' # AC val computed at low precision

VID_FILE                ='./no_backup/log/' + network + '.avi'
BANK_OCCUP_PROFILE_FILE ='./no_backup/reports/' + network + '.csv'

#Temp intermediate files
IMG_FILE_FOR_PYDOT      ='./no_backup/log/' + network + '.png'

# HW description file
HW_STRUCT_FILE          ='./src/hw_models/hw_structure'
HW_STRUCT_SUM_FILE      ='./src/hw_models/hw_structure_sum'
HW_STRUCT_PROD_FILE     ='./src/hw_models/hw_structure_prod'
HW_STRUCT_MAC_FILE      ='./src/hw_models/hw_structure_mac'

# naming convention in benchmarks
OPERATION_LIST_START_LINE= 1
OPERATION_NAME_PRODUCT= 'Product'
OPERATION_NAME_LEAF= 'Leaf'
OPERATION_NAME_SUM= 'Sum'
LEAF_NODE_NAME= 'memory'
INTERNAL_NODE_NAME= 'temp_memory'

# PRU systemverilog files
PRU_SV_COMMON_PKG_FILE= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/src/common_pkg.sv'
PRU_SV_INSTR_DECD_PKG_FILE= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/src/instr_decd_pkg.sv'

PRU_ASYNC_VERIF_PATH= '/esat/puck1/users/nshah/vcs_simulation_data/tb_data'

# sparse linear algebra
SPARSE_MATRIX_MATLAB_PATH= '/esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/mat/'
SPARSE_MATRIX_MARKET_PATH= '/esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM/'
SPARSE_MATRIX_MARKET_FACTORS_PATH= '/esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM_LU_factors/'
# SPARSE_MATRIX_MATLAB_PATH= '/volume1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/mat/'
# SPARSE_MATRIX_MARKET_PATH= '/volume1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM/'
# SPARSE_MATRIX_MARKET_FACTORS_PATH= '/volume1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM_LU_factors/'

def refersh():
  global network

  global REPORTS_PATH
  REPORTS_PATH= './no_backup/reports/' + network + '/'

  global LOG_PATH
  LOG_PATH= './no_backup/log/' 

  global ARITH_OP_FILE
  ARITH_OP_FILE= benchmarks_path +network+'_operations_no_evidence.txt'
  
  global ARITH_OP_MERGED_FILE
  ARITH_OP_MERGED_FILE= benchmarks_path +network+'_merged_operations_no_evidence.txt'
  
  global METAFILE
  METAFILE = './metafile_' + str(network)
  
  global COMMON_CHILD_DEPTH_CSV
  COMMON_CHILD_DEPTH_CSV= './no_backup/reports/' + network + '_COMMON_CHILD_DEPTH'
  
  global GRAPHVIZ_FILE 
  GRAPHVIZ_FILE= './no_backup/reports/' + network + '_graph.dot'
  
  global AC_FILE
  AC_FILE= benchmarks_path  + network +'.net.ac'
  
  global NET_FILE
  NET_FILE= benchmarks_path  + network +'.net'
  
  global LMAP_FILE
  LMAP_FILE= benchmarks_path  + network +'.net.lmap'
  
  global BITWIDTH_FILE
  BITWIDTH_FILE= benchmarks_path  + network + '.net.bits.csv'
  
  global MAP_INSTANCE_FILE
  MAP_INSTANCE_FILE= benchmarks_path  + network + '.net.map'
  
  global MAP_30_INSTANCE_FILE
  MAP_30_INSTANCE_FILE= benchmarks_path  + network + '.net.map_30'
  
  global GRAPHVIZ_BN_VAR_FILE
  global ASIP_CFILE
  global ASM_FILE
  global BB_FILE_PREFIX
  global BB_FILE
  global BB_FILE_0
  global BB_FILE_1
  global BB_FILE_2
  global GRAPH_FILE_PREFIX
  global GRAPH_FILE
  global GRAPH_FILE_0
  global GRAPH_FILE_1
  global GRAPH_FILE_2
  GRAPHVIZ_BN_VAR_FILE  =REPORTS_PATH +network + '_BN_var_under_AC_node.dot'
  ASIP_CFILE            =REPORTS_PATH + 'ASIP_code.c'
  ASM_FILE              =REPORTS_PATH + network + '.asm'
  BB_FILE_PREFIX        =REPORTS_PATH + network + '_bb.p_'
  BB_FILE               =REPORTS_PATH + network + '_bb.p'
  BB_FILE_0             =REPORTS_PATH + network + '_bb0.p'
  BB_FILE_1             =REPORTS_PATH + network + '_bb1.p'
  BB_FILE_2             =REPORTS_PATH + network + '_bb2.p'
  GRAPH_FILE_PREFIX     =REPORTS_PATH + network + '_gr.p_'
  GRAPH_FILE            =REPORTS_PATH + network + '_gr.p'
  GRAPH_FILE_0          =REPORTS_PATH + network + '_gr0.p'
  GRAPH_FILE_1          =REPORTS_PATH + network + '_gr1.p'
  GRAPH_FILE_2          =REPORTS_PATH + network + '_gr2.p'
  
  
  global IMG_FILE_FOR_PYDOT
  IMG_FILE_FOR_PYDOT= './no_backup/log/' + network + '.png' 

  global VID_FILE
  VID_FILE= './no_backup/log/' + network + '.avi'
  
  global EVIDENCE_WMC_GLD_PREFIX
  EVIDENCE_WMC_GLD_PREFIX= './no_backup/reports/' + network + '.net.dat_gld' # AC val computed at full precision
  
  global EVIDENCE_WMC_LP_PREFIX
  EVIDENCE_WMC_LP_PREFIX= './no_backup/reports/' + network + '.net.dat_lp' # AC val computed at low precision
  
  global EVIDENCE_DB_FILE
  EVIDENCE_DB_FILE= benchmarks_path  + network + '.net.dat' # The file that contains evidence simulated by SAMIAM
  
  global LOGISTIC_CIRCUIT_FILE
  LOGISTIC_CIRCUIT_FILE= benchmarks_path  + network + '.txt' # File from Yitao Liang, UCLA
  
  global PSDD_PATH
  PSDD_PATH            =PSDD_PATH_PREFIX + network + '.psdd'

  global TRAIN_DATASET_PATH
  global TEST_DATASET_PATH
  global VALID_DATASET_PATH
  TRAIN_DATASET_PATH= DATASET_PATH_PREFIX + network + '/' + network + '.train.data'
  TEST_DATASET_PATH= DATASET_PATH_PREFIX + network + '/' + network + '.test.data'
  VALID_DATASET_PATH= DATASET_PATH_PREFIX + network + '/' + network + '.valid.data'

  global BANK_OCCUP_PROFILE_FILE
  BANK_OCCUP_PROFILE_FILE= './no_backup/reports/' + network + '.csv'

  global GRAPH_NX_FILES_PATH
  GRAPH_NX_FILES_PATH= benchmarks_path + 'graph_nx/' + network + '.nx'

