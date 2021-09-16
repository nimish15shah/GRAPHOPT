
import os
import subprocess
import global_var
from pathlib import Path
from collections import defaultdict

import src.sparse_linear_algebra.matrix_names_list


def filter_done_matrices (name_list, required_th_set):
  path= '/esat/puck1/users/nshah/cpu_gpu_parallel/partitions/'
  search_str= '_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_COARSE'
  str_to_remove_ls= [search_str, '.p']

  path_ls= src.sparse_linear_algebra.matrix_names_list.done_matrices(path, search_str, str_to_remove_ls= str_to_remove_ls)

  map_done_mat_to_th= defaultdict(set)
  for mat, th in path_ls:
    map_done_mat_to_th[mat].add(th)

  print(f"All matrices: {len(map_done_mat_to_th)}")

  done_matrices= set([mat for mat, th_set in map_done_mat_to_th.items() if len(required_th_set - th_set)==0])

  new_name_list= []
  for n in name_list:
    if n.replace('/', '_') not in done_matrices:
      new_name_list.append(n)

  print(f"{len(name_list) - len(new_name_list)} matrices already done")

  return new_name_list, map_done_mat_to_th

name_list= src.sparse_linear_algebra.matrix_names_list.matrices_path(global_var.SPARSE_MATRIX_MARKET_FACTORS_PATH,
    global_var,
    mode= 'with_LU_factors', exclude_category_ls= ['Schenk_AFE', 'GHS_indef'])

required_th_set = set([1, 2, 4, 6, 8, 10, 12])
name_list, map_done_mat_to_th = filter_done_matrices(name_list, required_th_set)

# Too large
name_list = list(set(name_list) -  set(['GHS_indef/brainpc2', 'GHS_indef/olesnik0', 'GHS_indef_a5esindl', 'GHS_indef/c-55', 'Boeing/pwtk', 'Boeing/ct20stif', 'GHS_indef/c-59', 'GHS_indef/cont-300', 'GHS_indef/c-62ghs', 'GHS_psdef/s3dkt3m2', 'GHS_indef/bloweya', 'GHS_indef/helm3d01', 'GHS_indef/c-70', 'GHS_indef/c-63', 'GHS_psdef/oilpan', 'GHS_indef/c-71', 'Pothen/onera_dual', 'GHS_psdef/s3dkq4m2']))

# curr_name_list= name_list[-48 : ]

# name_list = ['GHS_psdef/wathen100', 'Boeing/crystm03', 'Norris/lung1', 'GHS_psdef/obstclae', 'Norris/fv1', 'Norris/fv2', 'GHS_psdef/jnlbrng1', 'HB/bcsstm26', 'FIDAP/ex35', 'Norris/lung2', 'Boeing/crystm02', 'Oberwolfach/gyro_m', 'Oberwolfach/t2dal_e', 'Boeing/crystm01', 'Oberwolfach/flowmeter5', 'GHS_psdef/vanbody', 'Bates/Chem97ZtZ']

threads= 2
# for name in curr_name_list:
# name_list= list(reversed(name_list))
print(f"Total matrices to do: {len(name_list)}")

while name_list:
  curr_name_list= name_list[:threads]
  p_ls= []
  for name in curr_name_list:
    th_to_do = required_th_set - map_done_mat_to_th[name]
    for th in th_to_do:
      print(name, th)
      cmd= f"sparse_tr_solve_full"
      p = subprocess.Popen(["make", cmd, f"NET={name}", f"THREAD={th}"])
      p_ls.append(p)

  exit_codes = [p.wait() for p in p_ls]

  name_list= name_list[threads:]
  # name_list= filter_done_matrices(name_list, required_th_set)

