
import os
import sys
import subprocess
import csv

def get_gpu_data(fname):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  # idx
  name_idx= 0
  map_mat_to_throughput= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx]
    map_mat_to_throughput[name]= None

  return map_mat_to_throughput

def run_p2p(log_path, all_matrices, n_iter, make_target):
  
  run_log= open(log_path, 'a+')
  print("Start", file= run_log, flush=True)

  factor_path= "/esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM_LU_factors"
  for mat in all_matrices:
    print(mat)
    path= f"{factor_path}/{mat}_L.mtx"
    cmd= f"make {make_target} FPATH={path} ITER={n_iter}"
    print(cmd)
    output= subprocess.check_output(cmd, shell=True)
    output = str(output)
    output = output[:-3]
    output= output[output.find('Annz'):]
    output= output.split('\\n')
    print(output)
    nnz= output[0]
    flop= output[1]
    fwd_p2p_tr_red_perm= output[-4].split(' ')
    gflops= fwd_p2p_tr_red_perm[3]
    # time= output[1]
    # time= float(time)/n_iter
    # residual= float(output[3])
    # flops= float(output[5])
    # print(output, time)
    # time= time*1.e3
    print(f"{mat},n_iter, {n_iter}, gflops, {gflops}, {flop}, {nnz}", file= run_log, flush= True)



def main():
  log_path = '../../openmp/run_log_sparse_tr_solve_nvidia_cusparse_gliese'
  map_mat_to_th = get_gpu_data(log_path)
  
  all_matrices = list(map_mat_to_th.keys())

  target= 'p2p'
  opt= 'Ofast'
  n_iter= 1000

  machine= os.uname().nodename
  machine= machine.split('.')[0]
  run_log= f'./run_log_{target}_{opt}_{machine}'
  print(f'looging at {run_log}')
  print(f'number of matrices {len(all_matrices)}')

  make_target= f'{target}_run'
  run_p2p(run_log, all_matrices, n_iter, make_target)

if __name__ == "__main__":
  main()

