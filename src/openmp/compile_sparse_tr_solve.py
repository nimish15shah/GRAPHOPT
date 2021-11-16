
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
import csv

def get_gpu_data(fname):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  # idx
  name_idx= 0
  throughput_idx= 12
  map_mat_to_throughput= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx]
    throughput= float(d[throughput_idx])
    map_mat_to_throughput[name]= throughput

  return map_mat_to_throughput

def get_intel_data(fname):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)


  # idx
  name_idx= 0
  throughput_idx= 11
  map_mat_to_throughput= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx]
    throughput= float(d[throughput_idx])
    map_mat_to_throughput[name]= throughput

  return map_mat_to_throughput

def get_data(fname, with_single_thread= True):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)
  # idx
  name_idx= 0
  th_0_idx= 1
  th_1_idx= 5
  n_compute_idx= 9
  throughput_idx= 13
  res_idx= 15
  golden_idx= 17

  # key: (name, th) tuple
  # val: [n_compute, throughput]

  compile_error_matrices= []
  improper_res_matrices= []
  data_d= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue

    name= d[name_idx]
    th= int(d[th_0_idx])
    
    # compilation error
    if len(d) < throughput_idx:
      compile_error_matrices.append((name, th))
      continue
    
    # some matrices have wrong c files
    if th != int(d[th_1_idx]):
      improper_res_matrices.append((name, th))
      print(f"Improper threads: {name}, {th}")
      continue

    # matrices with wrong results
    res= float(d[res_idx]) 
    golden= float(d[golden_idx]) 
    if abs(res-golden) > abs(0.1 * golden) and golden != 0:
      improper_res_matrices.append((name, th))
      print(f"High error: {name}, {th}, {res}, {golden}, {abs(res-golden)}, {0.1*golden}")
      continue

    n_compute = float(d[n_compute_idx])
    throughput = float(d[throughput_idx])

    data_d[(name, th)] = [n_compute, throughput]

  map_mat_to_th= defaultdict(set)
  for name, th in data_d.keys():
    map_mat_to_th[name].add(th)
  
  return map_mat_to_th, improper_res_matrices, compile_error_matrices


def done_matrices (path, search_str, str_to_remove_ls= None, exclude_str_ls= None):
  # path= '/esat/puck1/users/nshah/cpu_gpu_parallel/partitions/'
  # exclude_string= None
  # str_to_remove= '_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_COARSE'

  path_ls= list(Path(path).glob(f"*{search_str}*"))
  path_ls= [p.name for p in path_ls]

  # exclude some paths
  if exclude_str_ls != None:
    for s in exclude_str_ls:
      path_ls= [p for p in path_ls if s not in p]

  # remove the str_to_remove_ls
  if str_to_remove_ls != None:
    for s in str_to_remove_ls:
      path_ls= [p.replace(s, '') for p in path_ls]

  # the integer after last '_' indicates the thread
  new_path_ls= []
  for p in path_ls:
    sp_idx= p.rfind('_')
    matrix_name= p[:sp_idx]
    threads= int(p[sp_idx + 1: ])

    # tuple of (name, thread)
    new_path_ls.append((matrix_name, threads))
  path_ls= new_path_ls

  return path_ls

  done_matrices= set([p[0] for p in path_ls])

  new_name_list= []
  for n in name_list:
    if n.replace('/', '_') not in done_matrices:
      new_name_list.append(n)

  print(f"{len(name_list) - len(new_name_list)} matrices already done")
  return new_name_list

def nvidia_cusparse(path_ls, log_path):

  all_matrices= set([n for n,_ in path_ls])

  all_matrices -= set(['GHS_indef_brainpc2', 'GHS_indef_olesnik0', 'GHS_indef_a5esindl', 'GHS_indef_c-55', 'Boeing_pwtk', 'Boeing_ct20stif', 'GHS_indef_c-59', 'GHS_indef_cont-300', 'GHS_indef_c-62ghs', 'GHS_psdef_s3dkt3m2', 'GHS_indef_bloweya', 'GHS_indef_helm3d01', 'GHS_indef_c-70', 'GHS_indef_c-63', 'GHS_psdef_oilpan', 'GHS_indef_c-71', 'GHS_psdef_s3dkq4m2'])

  print(f"All matrices before: {len(all_matrices)}")

  map_mat_to_th = get_gpu_data(log_path)

  all_matrices -= set(map_mat_to_th.keys())
  print(f"Matrices to do : {len(all_matrices)}")


  elf_path= "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/SparseLinAlgebra_benchmarking/src/cusparse/triangular_solve"
  partition_path= "/esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM_LU_factors/"

  run_log= open(log_path, 'a+')
  print("Start", file= run_log, flush=True)

  for mat in all_matrices:
    print(mat)
    cmd= f"{elf_path} -file={partition_path}{mat}_L.mtx | grep n_iter"
    output= subprocess.check_output(cmd, shell=True)
    output = str(output)
    output = output[:-3]
    output= output[output.find('nnzA'):]
    print(output)
    print(f"{mat},{output}", file= run_log, flush= True)


def intel_mkl (path_ls, log_path):

  all_matrices= set([n for n,_ in path_ls])

  all_matrices -= set(['GHS_indef_brainpc2', 'GHS_indef_olesnik0', 'GHS_indef_a5esindl', 'GHS_indef_c-55', 'Boeing_pwtk', 'Boeing_ct20stif', 'GHS_indef_c-59', 'GHS_indef_cont-300', 'GHS_indef_c-62ghs', 'GHS_psdef_s3dkt3m2', 'GHS_indef_bloweya', 'GHS_indef_helm3d01', 'GHS_indef_c-70', 'GHS_indef_c-63', 'GHS_psdef_oilpan', 'GHS_indef_c-71', 'GHS_psdef_s3dkq4m2'])

  print(f"All matrices before: {len(all_matrices)}")
  map_mat_to_th = get_intel_data(log_path)

  all_matrices -= set(map_mat_to_th.keys())
  print(f"Matrices to do : {len(all_matrices)}")

  th= 12
  n_iter= 1000

  elf_path= "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/SparseLinAlgebra_benchmarking/src/cpu_mkl/dss/build/2_sparse_dss"
  partition_path= "/esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM_LU_factors/"

  run_log= open(log_path, 'a+')
  print("Start", file= run_log, flush=True)

  cmd= "make set_env"
  os.system(cmd)

  for mat in all_matrices:
    print(mat)
    cmd= f"{elf_path} {partition_path}{mat}_L.mtx {th} {n_iter} | grep N_threads"
    output= subprocess.check_output(cmd, shell=True)
    output = str(output)
    output = output[:-3]
    output= output[output.find('N_threads'):]
    print(output)
    print(f"{mat},{th},{output}", file= run_log, flush= True)
  
def par_sparse_tr_solve(path_ls, log_path, required_th, search_str):
  data_prefix= "\/esat\/puck1\/users\/nshah\/cpu_openmp\/"

  map_mat_to_th= defaultdict(set)
  for mat, th in path_ls:
    map_mat_to_th[mat].add(th)

  print(f"All matrices: {len(map_mat_to_th)}")

  required_th_set= set(required_th)
  filtered_mat= [mat for mat, th_set in map_mat_to_th.items() if len(required_th_set - th_set)==0]

  filtered_mat = list(set(filtered_mat) - set(['GHS_indef_brainpc2', 'GHS_indef_olesnik0', 'GHS_indef_a5esindl', 'GHS_indef_c-55', 'Boeing_pwtk', 'Boeing_ct20stif', 'GHS_indef_c-59', 'GHS_indef_cont-300', 'GHS_indef_c-62ghs', 'GHS_psdef_s3dkt3m2', 'GHS_indef_bloweya', 'GHS_indef_helm3d01', 'GHS_indef_c-70', 'GHS_indef_c-63', 'GHS_psdef_oilpan']))

  # already compiled matrices 
  map_mat_to_th, _ , _ = get_data(log_path)

  map_mat_to_th_to_do= {}
  for mat in filtered_mat:
    if mat in map_mat_to_th:
      th_to_do= required_th_set - map_mat_to_th[mat]
    else:
      th_to_do= set(required_th_set)
    
    if len(th_to_do) != 0:
      map_mat_to_th_to_do[mat] = th_to_do

  print(f"Filtered matrices: {len(filtered_mat)}")
  print(f"Mat to do: {len(map_mat_to_th_to_do)}")

  suffix= search_str
  line_number= 49
  run_log= open(log_path, 'a+')
  print("Start", file= run_log, flush=True)

  cmd= "make set_env"
  os.system(cmd)

  for mat, th_to_do in map_mat_to_th_to_do.items():
    for th in th_to_do:
      data_path= data_prefix + f"{mat}{suffix}_{th}.c"
      cmd= f"sed -i '{line_number}s/.*/#include \"{data_path}\"/' par_for_sparse_tr_solve_coarse.cpp"
      # print(cmd)
      os.system(cmd)
      print(mat, th)
      cmd= "make normal_cpp"
      err= os.system(cmd)
      if err:
        print(f"Error in compilation {mat}, {th}")
        print(f"{mat},{th},Error compilation", file= run_log, flush= True)
      else:
        print("Excuting")
        cmd= "make run"
        output= subprocess.check_output(cmd, shell=True)
        # os.system(cmd)
        output = str(output)
        output = output[:-3]
        output= output[output.find('N_layers'):]
        print(output)
        print(f"{mat},{th},{output}", file= run_log, flush= True)
    

def main():
  # path= '/esat/puck1/users/nshah/cpu_gpu_parallel/partitions/'
  path= '/esat/puck1/users/nshah/cpu_openmp/'
  # search_str= '_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_COARSE'
  search_str= '_LAYER_WISE_ALAP_CPU_COARSE'
  str_to_remove_ls= [search_str, '.c']
  path_ls= done_matrices(path, search_str, str_to_remove_ls= str_to_remove_ls)

  required_th= [6,8,10,12]
  # log_path = './run_log_sparse_tr_solve_two_way_Ofast_eridani'
  log_path = './run_log_sparse_tr_solve_layer_wise_O3_eridani'
  par_sparse_tr_solve(path_ls, log_path, required_th, search_str)

  # log_path = './run_log_sparse_tr_solve_nvidia_cusparse_gliese'
  # nvidia_cusparse(path_ls, log_path)

  # log_path = './run_log_sparse_tr_solve_intel_mkl_eridani'
  # intel_mkl(path_ls, log_path)

if __name__ == "__main__":
  main()

