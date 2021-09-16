import math
from collections import defaultdict
import csv
from statistics import mean

from matplotlib import pyplot as plt
import matplotlib
from random import uniform
import seaborn as sns

import matplotlib.patches as mpatches
import numpy as np

import get_sizes
import logging
logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def replace_slash(name_ls):
  group_names= ["Bai", 
      "Bindel",
      "HB",
      "Norris",
      "GHS_indef",
      "GHS_psdef",
      "FIDAP",
      "Boeing",
      "Oberwolfach",
      "Pothen",
      "MathWorks",
      "Nasa",
      "Pothen"
      ]
  
  new_name_ls= []
  for name in name_ls:
    for g in group_names:
      name= name.replace(g+'/', g+'_', 1)
    
    new_name_ls.append(name)

  return new_name_ls

def remove_group_name(name_ls):
  group_names= ["Bai", 
      "Bindel",
      "HB",
      "Norris",
      "GHS_indef",
      "GHS_psdef",
      "FIDAP",
      "Boeing",
      "Oberwolfach",
      "Pothen",
      "MathWorks",
      "Nasa",
      "Pothen"
  ]
  new_name_ls= []
  for name in name_ls:
    for g in group_names:
      name= name.replace(g+'_','', 1)
      name= name.replace(g+'/','', 1)
    new_name_ls.append(name)

  return new_name_ls

def replace_underscore(name_ls):
  group_names= ["Bai", 
      "Bindel",
      "HB",
      "Norris",
      "GHS_indef",
      "GHS_psdef",
      "FIDAP",
      "Boeing",
      "Oberwolfach",
      "Pothen",
      "MathWorks",
      "Nasa",
      "Pothen"
      ]
  
  new_name_ls= []
  for name in name_ls:
    for g in group_names:
      name= name.replace(g+'_', g+'/', 1)
    
    new_name_ls.append(name)

  return new_name_ls
def geo_mean(iterable):
  a = np.array(iterable)
  return a.prod()**(1.0/len(a))

def geo_mean_overflow(iterable):
  """
    avoids overflow
  """
  a = np.log(iterable)
  return np.exp(a.mean())

class Matrix_info():
  def __init__(self, name):
    self.name= name
    self.n_compute= None
    self.n_cols= None

    self.max_throughput= 0
    self.th_with_max_throughput= None
    self.acceleration_factor= None

    self.intel_mkl_throughput = None
    self.nvidia_cusparse_throughput= None
    self.max_layer_wise_throughput= None
    self.pru_throughput= None
    self.suitesparse_cxsparse_throughput= None
    self.suitesparse_umfpack_throughput= None
    self.p2p_throughput= None

    self.map_th_to_throughput= {}

def get_gpu_cuda_data(fname, target_th, target_batch):
  """
    Only looks at batch=1 data
  """
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  name_idx= 0
  n_threads_idx= 2
  batch_sz_idx= 3
  throughput_idx= 6 # GOPS

  map_psdd_to_throughput= {}
  
  for d in data:
    if d[0][0] == '#':
      continue
    th = int(d[n_threads_idx])
    batch = int(d[batch_sz_idx])

    if th== target_th and batch == target_batch:
      name= d[name_idx]
      throughput= float(d[throughput_idx])

      map_psdd_to_throughput[name]= throughput * 1e3 # MOPS
  
  return map_psdd_to_throughput
    

def get_juice_data(fname):
  map_psdd_to_compute = get_sizes.get_psdd_sizes()

  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  # idx
  name_idx= 0
  time_idx= 1

  map_psdd_to_throughput= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx]
    time= float(d[time_idx])
    
    # MOPS
    throughput = map_psdd_to_compute[name]/time * 1e-6

    map_psdd_to_throughput[name]= throughput

  return map_psdd_to_throughput

def get_p2p_data(fname):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  map_mat_to_compute = get_sizes.get_matrix_sizes()

  # idx
  name_idx= 0
  gflops_idx= 4
  map_mat_to_throughput= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx]

    try:
      gflops= float(d[gflops_idx])
    except:
      print(f"Wrong gflops for {name} matrix in p2p data")
      continue

    map_mat_to_throughput[ name ] = gflops * 1.e3 # MOPS

  return map_mat_to_throughput

def get_suitesparse_data(fname):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  map_mat_to_compute = get_sizes.get_matrix_sizes()

  # idx
  name_idx= 0
  solve_time_idx= 4
  map_mat_to_throughput= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx]
    solve_time= float(d[solve_time_idx]) #ms
    
    throughput= map_mat_to_compute[ name ] / ( 1.e3 * solve_time ) # MOPS
    map_mat_to_throughput[ name ] = throughput

  return map_mat_to_throughput



def get_gpu_data(fname):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  # idx
  name_idx= 0
  throughput_idx= 13
  col_idx= 4
  compute_idx= 6
  map_mat_to_throughput= {}
  map_mat_to_cols_compute= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx]
    throughput= float(d[throughput_idx])
    cols= int(d[col_idx])
    compute= int(d[compute_idx])
    map_mat_to_throughput[name]= throughput
    map_mat_to_cols_compute[name] = (cols, compute)

  sorted_by_acceleration = sorted(list(map_mat_to_throughput.keys()), key= lambda x:map_mat_to_throughput[x], reverse= True)
  print([(x, map_mat_to_throughput[x]) for x in sorted_by_acceleration[:20]])

  return map_mat_to_throughput, map_mat_to_cols_compute

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

    n_compute = float(d[n_compute_idx])
    throughput = float(d[throughput_idx])

    data_d[(name, th)] = [n_compute, throughput]
  
  # print(list(set([t[0] for t in compile_error_matrices])))

  # check that all threads have same n_compute
  net_with_no_1_thread= set()
  if with_single_thread:
    for name, th in data_d.keys():
      if (name,1) in data_d:
        if data_d[(name,1)][0] != data_d[(name, th)][0]:
          print("Improper n_compute in {name, th, data_d[(name, th)], data_d[(name, 1)]}")
      else:
        net_with_no_1_thread.add(name)


  all_matrices= set([name for name,_ in data_d.keys()])
  # key: name
  # value: th
  mat_detail_d= {name: Matrix_info(name) for name in all_matrices if name not in net_with_no_1_thread}

  # find max throughput tuples
  for name, th in data_d.keys():
    if name not in mat_detail_d:
      continue
    mat_obj= mat_detail_d[name]
    n_compute, throughput = data_d[(name, th)]
    
    mat_obj.n_compute = n_compute

    mat_obj.map_th_to_throughput[th] = throughput
    assert isinstance(throughput, float)

    if throughput > mat_obj.max_throughput:
      mat_obj.max_throughput= throughput
      mat_obj.th_with_max_throughput= th

  for mat, obj in mat_detail_d.items():
    assert obj.max_throughput != 0

  # acceleration
  if with_single_thread:
    for name, obj in mat_detail_d.items():
      obj.acceleration_factor= obj.max_throughput/ obj.map_th_to_throughput[1]

    # filter matrices
    # mat_detail_d= {n:obj for n, obj in mat_detail_d.items() if obj.acceleration_factor > 1}

  # sorted_be_acceleration= sorted(list(mat_detail_d.keys()), key= lambda x : mat_detail_d[x].acceleration_factor, reverse= True)
  
  print(f"Number of matrices with >1 acceleration fator {len(mat_detail_d)}, out of {len(all_matrices)}")

  # for name in sorted_be_acceleration:
  #   print(name, mat_detail_d[name].acceleration_factor, mat_detail_d[name].max_throughput, mat_detail_d[name].th_with_max_throughput)

  return mat_detail_d, compile_error_matrices, improper_res_matrices

def read_pru_sim_file(fname):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  # idx
  name_idx= 0
  cycle_idx= 2
  map_mat_to_cycles= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx].strip()
    cycles= float(d[cycle_idx])
    
    map_mat_to_cycles[name] = cycles

  return map_mat_to_cycles

def get_pru_throughput_data(fname, matrix_or_psdd= 'matrix'):
  assert matrix_or_psdd in ['matrix', 'psdd']
  if matrix_or_psdd == 'matrix':
    map_mat_to_compute= get_sizes.get_matrix_sizes()
  elif matrix_or_psdd == 'psdd':
    map_mat_to_compute= get_sizes.get_psdd_sizes()
  else:
    assert 0

  map_mat_to_cycles= read_pru_sim_file(fname)

  freq= 280 #MHz
  map_mat_to_throughput= {}
  for mat in map_mat_to_cycles.keys():
    cycles= map_mat_to_cycles[mat]
    compute= map_mat_to_compute[mat]
    map_mat_to_throughput[mat]= compute*freq/cycles

  return map_mat_to_throughput

def throughput_scaling_pru(mat_detail_d, savefig= False):
  SMALL_SIZE = 11
  MEDIUM_SIZE = 12
  BIGGER_SIZE = 14

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

  plt.rc('legend', handlelength=0.8)  # fontsize of the figure title

  x= [2,4,8,16,32,64]
  
  f= 280 # MHz
  batch_size= 4

  # psdd
  map_psdd_to_compute= get_sizes.get_psdd_sizes()
  prefix= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/async/tb/'
  target_psdd = [\
    'ad', \
    'baudio', \
    'bbc', \
    'bnetflix', \
    'book', \
    'c20ng', \
    'cr52', \
    'cwebkb', \
    'jester', \
    'kdd', \
    'mnist', \
    'msnbc', \
    'msweb', \
    'nltcs', \
    'pumsb_star', \
    'tretail', \
  ]

  target_psdd= [
    'mnist',
    'nltcs',
    'msnbc',
    'bnetflix',
    'ad',
    'bbc',
    'c20ng',
    'kdd',
    'baudio',
    'pumsb_star',
      ]

  
  y_psdd= []
  for t in x:
    fname= prefix + f'run_log_psdd_{t}_65536_65536_large_stream_mem'
    map_psdd_to_cycles = read_pru_sim_file(fname)
    throughput= [map_psdd_to_compute[name]/map_psdd_to_cycles[name] for name in target_psdd]
    throughput= [batch_size*f*1e-3*a for a in throughput]
    avg_throughput= mean(throughput)
    y_psdd.append(avg_throughput)

  for i, psdd in enumerate(target_psdd):
    print(i, psdd, map_psdd_to_compute[psdd], map_psdd_to_compute[psdd]/map_psdd_to_cycles[psdd])

  # trsv
  map_mat_to_compute= get_sizes.get_matrix_sizes()

  # select target matrices
  fname= prefix + f'run_log_sparse_tr_64_8192_8192_large_stream_mem'
  map_mat_to_cycles = read_pru_sim_file(fname)
  
  target_mat= [\
    'MathWorks_Sieber',
    'Bai_tols4000',
    'HB_west2021',
    'HB_orani678',
    'HB_gemat12',
    'HB_blckhole',
    'Bai_pde2961',
    'Bai_cryg2500',
    'HB_gemat11',
    'HB_bp_1600',
    'HB_bp_200',
    'Bai_qh1484',
    'HB_west1505',
    'HB_lshp2614',
    'HB_bp_1400',
    'Bai_dw2048',
    'Bai_dw1024',
    'HB_bp_1200',
    'HB_lshp2233',
    'Bai_qh882',
    'HB_lshp1882',
    'HB_shl_400'
  ]
  target_mat= [
    'Bai_tols4000',
    'HB_bp_200',
    'HB_west2021',
    'Bai_qh1484',
    'MathWorks_Sieber',
    'HB_gemat12',
    'Bai_dw2048',
    'HB_orani678',
    'Bai_pde2961',
    'HB_blckhole',
      ]


  # target_mat= sorted(list(map_mat_to_cycles.keys()), key= lambda a: map_mat_to_compute[a]/map_mat_to_cycles[a], reverse= True)[3:3+len(target_psdd)]
  # target_mat= sorted(list(map_mat_to_cycles.keys()), key= lambda a: map_mat_to_compute[a]/map_mat_to_cycles[a], reverse= True)[3:25]
  print(target_mat, len(target_mat))
  for mat in target_mat:
    print(mat, map_mat_to_compute[mat])

  y_mat= []
  for t in x:
    fname= prefix + f'run_log_sparse_tr_{t}_8192_8192_large_stream_mem'
    map_mat_to_cycles = read_pru_sim_file(fname)
    throughput= [map_mat_to_compute[name]/map_mat_to_cycles[name] for name in target_mat]
    throughput= [batch_size*f*1e-3*a for a in throughput]
    avg_throughput= mean(throughput)
    y_mat.append(avg_throughput)
  
  # peak
  y_peak= [batch_size*f*1e-3*t for t in x]

  # average
  y_avg= [(y_psdd[idx] + y_mat[idx])/2 for idx in range(len(y_psdd))]
  

  fig_dims = (3.4, 2.6)
  fig, ax = plt.subplots(figsize=fig_dims) 

  border_width= 1.0
  # thicker boundary
  for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(border_width)
  
  ax.plot(x, y_peak, '-o', label= "Peak")
  ax.plot(x, y_avg, '-o', label= "Average")
  ax.plot(x, y_psdd, '--', label= 'PC', marker= "o")
  ax.plot(x, y_mat, '--', label= 'SpTRSV', marker= "o")

  ax.set_ylabel("Throughput (GOPS)")
  ax.set_xlabel("Active CUs")

  # ax.annotate('Peak', xy=(35,55), ha='left', va='center', rotation=36)
  # ax.annotate('Average', xy=(35,23), ha='left', va='center', rotation=29)

  ax.set_xscale('log', basex=2)
  ax.set_yscale('log', basey=2)

  ax.set_xticks(x)
  ax.set_yticks(x)

  ax.minorticks_on()
  # ax.tick_params(which='major', length=10, width=2, direction='inout')
  # ax.tick_params(which='minor', length=5, width=2, direction='in')

  ax.grid(which='both', linestyle= 'dotted')
  # ax.tick_params(
  #   # axis='x',          # changes apply to the x-axis
  #   which='minor',      # both major and minor ticks are affected
  #   bottom=True,      # ticks along the bottom edge 
  #   top=False,         # ticks along the top edge
  #   labelbottom=True) # labels along the bottom edge


  plt.legend()
  plt.tight_layout()

  if savefig:
    path= 'throughput_scaling_pru.pdf'
    plt.savefig(path)
  else:
    plt.show()

def throughput_bar_plot_pru(mat_detail_d, all_matrices, throughput_per_cycle= False, savefig= False):
  SMALL_SIZE = 10
  MEDIUM_SIZE = 12
  BIGGER_SIZE = 14

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=8)    # legend fontsize
  plt.rc('legend', handlelength=0.5)  # fontsize of the figure title
  plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

  fig_dims = (5, 2.5)
  fig, ax = plt.subplots(2,2, figsize=fig_dims) 

  ax_sptrsv= [ax[0,1], ax[1,1]]
  ax_psdd= [ax[0,0], ax[1,0]]

  bar_spacing= 0.4
  bar_width= 0.07
  border_width= 1.0
  throughput_bar_plot_pru_sptrsv(ax_sptrsv, mat_detail_d, all_matrices, bar_spacing, bar_width, border_width, throughput_per_cycle= False)
  bar_spacing= 0.5
  bar_width= 0.09
  throughput_bar_plot_pru_psdd(ax_psdd,bar_spacing, bar_width, border_width)

  # plt.xlabel("Workloads")
  plt.tight_layout()

  if savefig:
    path= 'throughput_bar_plot_pru_2.pdf'
    plt.savefig(path)
  else:
    plt.show()


def throughput_bar_plot_pru_psdd(ax, bar_spacing, bar_width, border_width):
  target_psdd= [
    'mnist',
    'nltcs',
    'msnbc',
    'bnetflix',
    'ad',
    'bbc',
    'c20ng',
    'kdd',
    'baudio',
    'pumsb_star',
      ]
  
  # cpu_openmp
  fname= './run_log_psdd_two_way_limit_Ofast_eridani'
  psdd_detail_d, _, _= get_data(fname, with_single_thread= True)
  # manually add the missing enteries
  info=Matrix_info('pumsb_star')
  info.max_throughput= 758
  psdd_detail_d['pumsb_star']= info
  info=Matrix_info('nltcs')
  info.max_throughput= 405
  psdd_detail_d['nltcs']= info

  # cpu_juice
  fname= './run_log_psdd_juice_puck'
  map_psdd_to_throughput_juice= get_juice_data(fname)
  
  # pru
  fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/async/tb/run_log_psdd_64_65536_65536_large_stream_mem'
  map_psdd_to_throughput_pru= get_pru_throughput_data(fname, matrix_or_psdd = 'psdd')

  # gpu_cuda
  fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/cuda/gpu_compile/run_log_drive.csv'
  map_psdd_to_throughput_gpu= get_gpu_cuda_data(fname, 512, 1)

  y_cpu_openmp= [psdd_detail_d[p].max_throughput/1e3 for p in target_psdd] # GOPS
  y_cpu_juice= [map_psdd_to_throughput_juice[p]/1e3 for p in target_psdd]
  y_pru= [map_psdd_to_throughput_pru[p]/1e3 for p in target_psdd]
  y_gpu_cuda= [map_psdd_to_throughput_gpu[p]/1e3 for p in target_psdd]

  for m in target_psdd:
    print(m, map_psdd_to_throughput_pru[m], psdd_detail_d[m].max_throughput, map_psdd_to_throughput_gpu[m])
  # exit(1)

  x = np.arange(len(target_psdd))  # the label locations
  x = bar_spacing*x
  width = bar_width  # the width of the bars

  ax[0].set_title('Probabilistic Circuits (PC)') 
  
  # thicker boundary
  for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(border_width)
    ax[1].spines[axis].set_linewidth(border_width)

  # grid
  ax[0].grid(axis='y', which='both', linestyle= 'dotted')
  ax[1].grid(axis='y', which='both', linestyle= 'dotted')
  ax[0].set_axisbelow(True)
  ax[1].set_axisbelow(True)
  # ax[1].minorticks_on()
  # ax[1].tick_params(axis='x', which='minor', bottom=False)

  
  # multi-bar
  rects1 = ax[0].bar(x - 1.5* width , y_pru        , width , label='DPU'          , color= 'C0')
  rects2 = ax[0].bar(x - 0.5* width , y_cpu_openmp , width , label='CPU-OMP ' , color= 'C1')
  rects2 = ax[0].bar(x + 0.5* width , y_cpu_juice  , width , label='CPU-JUICE'  , color= 'C3')
  rects3 = ax[0].bar(x + 1.5* width , y_gpu_cuda   , width , label='GPU'          , color= 'C2')

  # ax[0].set_ylim([0,max(y_pru)+3.5])
  # ax[0].set_yticks([0,2.5,5, 7.5, 10])
  ax[0].set_yscale('log')
  ax[0].set_ylim([10**-3.5,10**1.2])
  yticks= [10**(a) for a in range(-3,2,2)]
  ax[0].set_yticks(yticks)

  y_pru_1= [t/0.23 for t in y_pru]
  y_cpu_juice_1= [t/55 for t in y_cpu_juice]
  y_cpu_openmp_1= [t/55 for t in y_cpu_openmp]
  y_gpu_cuda_1= [t/98 for t in y_gpu_cuda]

  rects4 = ax[1].bar(x - 1.5* width , y_pru_1        , width, color= 'C0')
  rects5 = ax[1].bar(x - 0.5* width , y_cpu_openmp_1  , width, color= 'C1')
  rects5 = ax[1].bar(x + 0.5* width , y_cpu_juice_1 , width, color= 'C3')
  rects6 = ax[1].bar(x + 1.5* width , y_gpu_cuda_1   , width, color= 'C2')

  ax[1].set_yscale('log')
  print(max(y_pru_1))
  ax[1].set_ylim([10**-5,150])
  # yticks= [10**(a) for a in range(int(math.log10(min(y_cpu_juice_1))), 1+ int(math.log10(max(y_pru_1))))]
  yticks= [10**(a) for a in range(-4,2,2)]
  ax[1].set_yticks(yticks)
  # ax[1].tick_params(
  #   axis='y',          # changes apply to the x-axis
  #   which='both',      # both major and minor ticks are affected
  #   right=True,      # ticks along the bottom edge are off
  #   left=True,         # ticks along the top edge are off
  #   ) 

  ax[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

  ax[1].set_xticks(x)
  xticklabels= target_psdd
  xticklabels[-1]= 'pumsb-\nstar'
  ax[1].set_xticklabels([m.lower() for m in remove_group_name(target_psdd)], rotation=90)


  # ax[0].legend(loc= 'upper center', ncol= 2)
  ax[0].legend(loc= 'upper center', ncol= 2, bbox_to_anchor= (0.5,1.4))
  
  ax[0].set_ylabel("Throughput \n(GOPS)")
  ax[1].set_ylabel("Energy eff.\n(GOPS/W)")
  
  

def throughput_bar_plot_pru_sptrsv(ax, mat_detail_d, all_matrices, bar_spacing, bar_width, border_width, throughput_per_cycle= False):
  all_matrices = list(all_matrices)
  logger.info(all_matrices)
  
  for n in all_matrices:
    assert mat_detail_d[n].pru_throughput != None, n

  y_pru= [mat_detail_d[n].pru_throughput for n in all_matrices]
  y_intel= [mat_detail_d[n].intel_mkl_throughput for n in all_matrices]
  y_nvidia= [mat_detail_d[n].nvidia_cusparse_throughput for n in all_matrices]

  for m in all_matrices:
    print(m, mat_detail_d[m].pru_throughput, mat_detail_d[m].intel_mkl_throughput, mat_detail_d[m].nvidia_cusparse_throughput)
  # exit(1)

  y_pru= [t/1000 for t in y_pru]
  y_intel= [t/1000 for t in y_intel]
  y_nvidia= [t/1000 for t in y_nvidia]

  x = np.arange(len(all_matrices))  # the label locations
  x = bar_spacing*x
  width = bar_width  # the width of the bars

  ax[0].set_title('Sparse Triangular Solves (SpTRSV)') 
  
  # thicker boundary
  for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(border_width)
    ax[1].spines[axis].set_linewidth(border_width)

  # grid
  ax[0].grid(axis='y', which='both', linestyle= 'dotted')
  ax[1].grid(axis='y', which='both', linestyle= 'dotted')
  ax[0].set_axisbelow(True)
  ax[1].set_axisbelow(True)
  # ax[1].minorticks_on()
  # ax[1].tick_params(axis='x', which='minor', bottom=False)


  
  # multi-bar
  rects1 = ax[0].bar(x - width, y_pru, width, label='DPU')
  rects2 = ax[0].bar(x, y_intel, width, label='CPU')
  rects3 = ax[0].bar(x + width, y_nvidia, width, label='GPU')

  # ax[0].set_ylim([0,max(y_pru)+3.5])
  # ax[0].set_yticks([0,2.5,5])
  ax[0].set_yscale('log')
  ax[0].set_ylim([10**-3.5,10**1.2])
  yticks= [10**(a) for a in range(-3,2,2)]
  ax[0].set_yticks(yticks)
  ax[0].tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off

  ax[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

  if throughput_per_cycle:
    y_pru_1= [t/0.28 for t in y_pru]
    y_intel_1= [t/3 for t in y_intel]
    y_nvidia_1= [t/1.35 for t in y_nvidia]
  else: # energy efficiency
    y_pru_1= [t/0.23 for t in y_pru]
    y_intel_1= [t/55 for t in y_intel]
    y_nvidia_1= [t/98 for t in y_nvidia]


  rects4 = ax[1].bar(x - width, y_pru_1, width) 
  rects5 = ax[1].bar(x, y_intel_1, width)
  rects6 = ax[1].bar(x + width, y_nvidia_1, width)

  ax[1].set_yscale('log')
  # yticks= [10**(a) for a in range(int(math.log10(min(y_nvidia_1))), 1+ int(math.log10(max(y_pru_1))))]
  ax[1].set_ylim([10**-5,150])
  yticks= [10**(a) for a in range(-4,2,2)]
  ax[1].set_yticks(yticks)
  # ax[1].tick_params(
  #   axis='y',          # changes apply to the x-axis
  #   which='both',      # both major and minor ticks are affected
  #   right=True,      # ticks along the bottom edge are off
  #   left=True,         # ticks along the top edge are off
  #   ) 

  ax[1].tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off


  ax[1].set_xticks(x)
  ax[1].set_xticklabels([m.lower() for m in remove_group_name(all_matrices)], rotation= 90)


  # ax[0].legend(loc= 'upper center', ncol= 3)
  ax[0].legend(loc= 'upper center', ncol= 3, bbox_to_anchor= (0.5,1.2))
  
  # ax[0].set_ylabel("Throughput \n(GOPS)")
  # if throughput_per_cycle:
  #   ax[1].set_ylabel("Throughput normalized by freq.\n(Operations per cycle)")
  # else:
  #   ax[1].set_ylabel("Energy efficiency \n(GOPS/W)")

def throughput_scatter_plot_pru(mat_detail_d, all_matrices, energy_eff= False, throughput_per_cycle= False):
  x= [mat_detail_d[n].n_compute for n in all_matrices]

  y_pru= [mat_detail_d[n].pru_throughput for n in all_matrices]
  y_intel= [mat_detail_d[n].intel_mkl_throughput for n in all_matrices]
  y_nvidia= [mat_detail_d[n].nvidia_cusparse_throughput for n in all_matrices]

  y_pru= [t/1000 for t in y_pru]
  y_intel= [t/1000 for t in y_intel]
  y_nvidia= [t/1000 for t in y_nvidia]

  if energy_eff: # divide by power in W
    y_pru= [t/0.2 for t in y_pru]
    y_intel= [t/50 for t in y_intel]
    y_nvidia= [t/98 for t in y_nvidia]
  elif throughput_per_cycle: # divide by freq
    y_pru= [t/0.28 for t in y_pru]
    y_intel= [t/3 for t in y_intel]
    y_nvidia= [t/1.35 for t in y_nvidia]

  pru_col= 'b'
  intel_col= 'g'
  nvidia_col= 'C7'

  c_pru= [pru_col for n in all_matrices]
  c_intel= [intel_col for n in all_matrices]
  c_nvidia= [nvidia_col for n in all_matrices]
  
  fig_dims = (5, 3.5)
  fig, ax = plt.subplots(figsize=fig_dims) 
  
  s = ax.scatter(
      x + x + x, 
      y_pru + y_intel + y_nvidia, 
      c= c_pru + c_intel + c_nvidia,
      # s= 5
      )
  ax.set_xscale('log')

  if energy_eff:
    ax.set_yscale('log')

  # legends
  classes= []
  class_colours= []

  classes.append('PRU (0.28 GHz)')
  class_colours.append(pru_col)

  classes.append('Intel MKL (3 GHz)')
  class_colours.append(intel_col)

  classes.append('Nvidia CUSPARSE (1.35 GHz)')
  class_colours.append(nvidia_col)

  recs = []
  for i in range(0,len(class_colours)):
      recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
  plt.legend(recs,classes,loc=0)
  
  if energy_eff:
    plt.ylabel("Energy efficiency (GFLOPS/W)")
  elif throughput_per_cycle:
    plt.ylabel("Operations per cycle")
  else:
    plt.ylabel("Throughput (GFLOPS)")
  plt.xlabel("Non-zero elements in matrices")
  plt.tight_layout()
  
  # plt.grid()
  # plt.show()

  if energy_eff:
    path= 'energy_eff_pru.png'
  elif throughput_per_cycle:
    path= 'operations_per_cycle_pru.png'
  else:
    path= 'throughput_scatter_plot_pru.png'
  plt.savefig(path, dpi= 300, format='png')


def throughput_scatter_plot(mat_detail_d, mat_detail_d_fast, th_wise_color= False, savefig= False):

  all_matrices= list(mat_detail_d.keys())
  x= [mat_detail_d[n].n_compute for n in all_matrices]
  x_fast= [mat_detail_d_fast[n].n_compute for n in all_matrices]
  x_baseline= [mat_detail_d[n].n_compute for n in all_matrices]
  x_intel= [mat_detail_d[n].n_compute for n in all_matrices]
  x_nvidia= [mat_detail_d[n].n_compute for n in all_matrices]
  x_layer_wise= [mat_detail_d[n].n_compute for n in all_matrices]

  y            = [mat_detail_d[n].max_throughput for n in all_matrices]
  y_fast       = [mat_detail_d_fast[n].max_throughput for n in all_matrices]
  y_baseline   = [mat_detail_d[n].map_th_to_throughput[1] for n in all_matrices]
  y_intel      = [mat_detail_d[n].intel_mkl_throughput for n in all_matrices]
  y_nvidia     = [mat_detail_d[n].nvidia_cusparse_throughput for n in all_matrices]
  y_layer_wise = [mat_detail_d[n].max_layer_wise_throughput for n in all_matrices]
  y_cxsparse   = [mat_detail_d[n].suitesparse_cxsparse_throughput for n in all_matrices]
  y_umfpack    = [mat_detail_d[n].suitesparse_umfpack_throughput for n in all_matrices]
  y_p2p        = [mat_detail_d[n].p2p_throughput for n in all_matrices]

  y            = [t/1000 for t in y]
  y_fast       = [t/1000 for t in y_fast]
  y_baseline   = [t/1000 for t in y_baseline]
  y_intel      = [t/1000 for t in y_intel]
  y_nvidia     = [t/1000 for t in y_nvidia]
  y_layer_wise = [t/1000 for t in y_layer_wise]
  y_cxsparse   = [t/1000 for t in y_cxsparse]
  y_umfpack   = [t/1000 for t in y_umfpack]
  y_p2p   = [t/1000 for t in y_p2p]

  if th_wise_color:
    color_map= {1: 'r', 2: 'r', 4: 'b', 6: 'g', 8: 'c', 10: 'm', 12: 'C1', 14: 'C8'}
  else:
    color_map= defaultdict(lambda: 'r')
    color_map_fast= defaultdict(lambda: 'b')

  intel_col= 'g'
  nvidia_col= 'C7'
  layer_wise_col= 'C5'
  baseline_col= 'k'
  cxsparse_col= 'C8'
  umfpack_col= 'C9'
  p2p_col= 'C11'
  c= [color_map[mat_detail_d[n].th_with_max_throughput] for n in all_matrices]
  c_fast= [color_map_fast[mat_detail_d[n].th_with_max_throughput] for n in all_matrices]
  c_baseline= [baseline_col for n in all_matrices]
  c_intel= [intel_col for n in all_matrices]
  c_nvidia= [nvidia_col for n in all_matrices]
  c_layer_wise= [layer_wise_col for n in all_matrices]
  c_cxsparse= [cxsparse_col for n in all_matrices]
  c_umfpack= [umfpack_col for n in all_matrices]
  c_p2p= [p2p_col for n in all_matrices]

  fig_dims = (5, 3.5)
  fig, ax = plt.subplots(figsize=fig_dims) 
  
  s = ax.scatter(
      x_baseline + x_intel + x_nvidia + x_layer_wise + x + x_fast + x + x + x, 
      y_baseline + y_intel + y_nvidia + y_layer_wise + y + y_fast + y_cxsparse + y_umfpack + y_p2p, 
      c= c_baseline + c_intel + c_nvidia + c_layer_wise + c + c_fast + c_cxsparse + c_umfpack + c_p2p,
      s= 1.5)
  ax.set_xscale('log')

  # legends
  classes= []
  class_colours= []

  classes.append('This work with parallel thread (-Ofast)')
  class_colours.append('b')
  
  if th_wise_color:
    classes += list(reversed(['2', '4', '6', '8', '10', '12']))
    # classes = list(reversed(['2', '4', '6']))
    class_colours += [color_map[int(c)] for c in classes]
  else:
    classes += ['This work with parallel threads (-O1)']
    class_colours += ['r']
  
  classes.append('This work with single thread (-O1)')
  class_colours.append(baseline_col)

  classes.append('Intel MKL')
  class_colours.append(intel_col)

  classes.append('Nvidia CUSPARSE')
  class_colours.append(nvidia_col)

  classes.append('DAG layer partitioning (-Ofast)')
  class_colours.append(layer_wise_col)

  classes.append('SuiteSparse CXSparse (-Ofast)')
  class_colours.append(cxsparse_col)

  classes.append('SuiteSparse UMFPACK (-Ofast)')
  class_colours.append(umfpack_col)

  classes.append('P2P multithreaded (-Ofast)')
  class_colours.append(p2p_col)

  recs = []

  recs = []
  for i in range(0,len(class_colours)):
      recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
  plt.legend(recs,classes,loc=0)
  
  plt.ylabel("Throughput (GFLOPS)")
  plt.xlabel("Non-zero elements in matrices")
  plt.tight_layout()
  

  if savefig:
    path= 'throughput_scatter_plot.png'
    plt.savefig(path, dpi= 300, format='png')
  else:
    plt.show()


def throughput_scaling_single_chart(mat_detail_d, legend= False):
  th_ls= [1,2,4,6,8,10,12]

  fig_dims = (5, 3.5)
  fig, ax = plt.subplots(figsize=fig_dims) 

  # use ls to preserev order
  mat_ls= list(mat_detail_d.keys())

  for mat in mat_ls:
    obj= mat_detail_d[mat]
    ref_throughput= obj.map_th_to_throughput[1]
    y= []
    skip= False
    for t in th_ls:
      y.append(obj.map_th_to_throughput[t]/1000)
#      y.append(obj.map_th_to_throughput[t]/ref_throughput)
#      if y[-1] > t+0.1:
#        skip = True

    if not skip:
      ax.plot(th_ls, y, '-o', markersize=3)

  plt.xticks(th_ls[1:])

  if legend:
    legend_ls= [f"{mat} ({int(mat_detail_d[mat].n_compute)})" for mat in mat_ls]
    plt.legend(legend_ls, loc= "lower right", ncol= 2, framealpha= 0.9)

    plt.ylabel("Throughput (GFLOPS)")
    plt.xlabel("Parallel threads (P)")


  plt.tight_layout()
#  plt.show()

  path= 'spn_throughput_scaling_single_chart.png'
  plt.savefig(path, dpi= 300, format='png')

def throughput_scaling_one_matrix(name, mat_obj_main, mat_obj_layer_wise):
  th_ls= [1,2,4,6,8,10,12]
  fig_dims = (2, 1.5)
  fig, ax = plt.subplots(figsize=fig_dims) 

  # ref_throughput= obj.map_th_to_throughput[1]
  # y= []
  # for t in th_ls:
  #   y.append(obj.map_th_to_throughput[t]/ref_throughput)

  y= [mat_obj_main.map_th_to_throughput[th]/1000 for th in th_ls]
  ax.plot(th_ls, y, '-o')

  y_l= []
  for th in th_ls:
    if th == 1:
      obj= mat_obj_main
    else:
      obj= mat_obj_layer_wise
    y_l.append(obj.map_th_to_throughput[th]/1000)
  ax.plot(th_ls, y_l, '-o')

  plt.xticks(th_ls[1:])
  # plt.rc('xtick', labelsize= 28)
  plt.tight_layout()
  # plt.show()

  path= name + '_throughput_scaling.png'
  plt.savefig(path, dpi= 300, format='png')
  # exit(1)

  
def throughput_scaling(mat_detail_d):

  sorted_by_acceleration= sorted(list(mat_detail_d.keys()), key= lambda x : mat_detail_d[x].acceleration_factor, reverse= True)
  
  dims= 5
  fig, ax = plt.subplots(dims, dims) 

  name_ls= sorted_by_acceleration[:dims * dims]

  th_ls= [1,2,4,6,8,10,12]

  for d_x in range(dims):
    for d_y in range(dims):
      if (d_x*dims + d_y) >= len(mat_detail_d):
        continue
      print(d_x, d_y)
      name= name_ls[d_x*dims + d_y]
      obj= mat_detail_d[name]

      y= [obj.map_th_to_throughput[th] for th in th_ls]
      ax[d_x, d_y].plot(th_ls, y, '-o')
      ax[d_x, d_y].set_title(name)

  plt.tight_layout()
  plt.show()

def plot_size_histogram(mat_detail_d):
  # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)o
  # hist, bins, _ = axs[0].hist(x, bins=8)
  
  x= [obj.n_compute for _, obj in mat_detail_d.items()]

  plt.subplot(211)
  hist, bins, _ = plt.hist(x, bins=10)

  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.subplot(212)
  plt.hist(x, bins=logbins)
  plt.xscale('log')
  plt.show()

def select_matrices(mat_detail_d):
  x= [obj.n_compute for _, obj in mat_detail_d.items()]
  hist, bins, _ = plt.hist(x, bins=24)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

  print(logbins)
  
  filtered_mat_ls= []
  for idx, l_thresh in enumerate(logbins):
    if idx == len(logbins) - 1:
      u_thresh = float("inf")
    else:
      u_thresh = logbins[idx+1]

    filtered_mat= [m for m, obj in mat_detail_d.items() if (obj.n_compute < u_thresh) and (obj.n_compute > l_thresh)] 
    filtered_mat= [m for m in filtered_mat if mat_detail_d[m].n_compute > mat_detail_d[m].n_cols] 
    filtered_mat = sorted(filtered_mat, key= lambda x : mat_detail_d[x].acceleration_factor, reverse= True)
    filtered_mat = filtered_mat[:1]

    print(l_thresh, u_thresh)
    for mat in filtered_mat:
      obj= mat_detail_d[mat]
      print(obj.name, obj.acceleration_factor, obj.n_compute, obj.n_cols, obj.n_compute/(obj.n_cols * obj.n_cols))

    filtered_mat_ls += filtered_mat

  print(filtered_mat_ls)

  return filtered_mat_ls

def main():
  fname= './run_log_sparse_tr_solve_two_way_O1_eridani'
  mat_detail_d, _, _= get_data(fname, with_single_thread= True)

  fname= './run_log_sparse_tr_solve_two_way_Ofast_eridani'
  mat_detail_d_fast, _, _= get_data(fname, with_single_thread= True)

  fname= './run_log_sparse_tr_solve_intel_mkl_eridani'
  map_mat_to_throughput_cpu = get_intel_data(fname)
  fname= './run_log_sparse_tr_solve_nvidia_cusparse_gliese'
  map_mat_to_throughput_gpu, map_mat_to_cols_compute = get_gpu_data(fname)

  fname= './run_log_sparse_tr_solve_layer_wise_Ofast_eridani'
  mat_detail_layer_wise_d, _, _= get_data(fname, with_single_thread= False)

  # fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/async/tb/run_log_sparse_tr_64_512_1024_large_stream_mem'
  fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/async/tb/run_log_sparse_tr_64_8192_8192_large_stream_mem'
  map_mat_to_throughput_pru= get_pru_throughput_data(fname, matrix_or_psdd = 'matrix')

  fname= '../benchmark_other_works/suitesparse/run_log_cxsparse_trsv_Ofast_eridani1'
  map_mat_to_throughput_cxsparse = get_suitesparse_data(fname)

  fname= '../benchmark_other_works/suitesparse/run_log_umfpack_Ofast_eridani1'
  map_mat_to_throughput_umfpack = get_suitesparse_data(fname)

  fname= '../benchmark_other_works/p2p/run_log_p2p_Ofast_eridani1'
  map_mat_to_throughput_p2p = get_p2p_data(fname)

  for mat, obj in mat_detail_d.items():
    obj.n_cols = map_mat_to_cols_compute[mat][0] 

  for mat, obj in mat_detail_d_fast.items():
    obj.n_cols = map_mat_to_cols_compute[mat][0]

  # sorted_by_acceleration= sorted(list(mat_detail_d.keys()), key= lambda x : mat_detail_d[x].acceleration_factor, reverse= True)
  # print(sorted_by_acceleration[:16])
  # exit(1)

  # print([n for n, obj in mat_detail_d.items() if obj.th_with_max_throughput == 10])
  # exit(1)

  print(f"Do Ofast experiments for : {set(mat_detail_d.keys()) - set(mat_detail_d_fast.keys())}")
  print(f"Do intel mkl experiments for these matrices: {set(mat_detail_d.keys()) - set(map_mat_to_throughput_cpu.keys())}")
  print(f"Do Nvidia experiments for these matrices: {set(mat_detail_d.keys()) - set(map_mat_to_throughput_gpu.keys())}")
  
  print(f"Yet to do layer wise for : {len(set(mat_detail_d.keys()) - set(mat_detail_layer_wise_d.keys()))} matrices")

  # matrices with all proper data
  filtered_mat = set(mat_detail_d.keys()) & set(map_mat_to_throughput_cpu.keys())
  filtered_mat &= set(map_mat_to_throughput_gpu.keys())
  filtered_mat &= set(mat_detail_layer_wise_d.keys())
  filtered_mat &= set(map_mat_to_throughput_cxsparse.keys())
  filtered_mat &= set(map_mat_to_throughput_umfpack.keys())
  filtered_mat &= set(map_mat_to_throughput_p2p.keys())
  # filtered_mat = set([m for m in filtered_mat if mat_detail_d[m].max_throughput <= mat_detail_d_fast[m].max_throughput])

  for name in filtered_mat:
    obj= mat_detail_d[name]
    obj.intel_mkl_throughput= map_mat_to_throughput_cpu[name]
    obj.nvidia_cusparse_throughput= map_mat_to_throughput_gpu[name]

    if name in mat_detail_layer_wise_d:
      obj.max_layer_wise_throughput= mat_detail_layer_wise_d[name].max_throughput

    if name in map_mat_to_throughput_pru:
      obj.pru_throughput= map_mat_to_throughput_pru[name]

    if name in map_mat_to_throughput_cxsparse:
      obj.suitesparse_cxsparse_throughput= map_mat_to_throughput_cxsparse[name]

    if name in map_mat_to_throughput_umfpack:
      obj.suitesparse_umfpack_throughput= map_mat_to_throughput_umfpack[name]

    if name in map_mat_to_throughput_p2p:
      obj.p2p_throughput= map_mat_to_throughput_p2p[name]

  if False:
    filtered_mat &= set(map_mat_to_throughput_pru.keys())
    # throughput_scatter_plot_pru(mat_detail_d, filtered_mat, energy_eff= False)
    # throughput_scatter_plot_pru(mat_detail_d, filtered_mat, energy_eff= False, throughput_per_cycle= True)

    # filtered_mat= sorted(filtered_mat, key= lambda x: mat_detail_d[x].pru_throughput, reverse= True)
    filtered_mat= [\
        'Bai/tols4000',
        'HB/bp_200',
        'HB/west2021',
        'Bai/qh1484',
        'MathWorks/Sieber',
        'HB/gemat12',
        'Bai/dw2048',
        'HB/orani678',
        'Bai/pde2961',
        'HB/blckhole',
      ]
    filtered_mat= replace_slash(filtered_mat)

    throughput_bar_plot_pru(mat_detail_d, filtered_mat, savefig= False)
    exit(1)

    throughput_scaling_pru(mat_detail_d, savefig= False)
    exit(1)

  if False:
    # for experiments with limited matrices
    # filtered_mat = select_matrices(mat_detail_d_fast)
    # filtered_mat = ["Bai_tols340","HB_steam1","Bindel_ted_B_unscaled","Pothen_barth5","Oberwolfach_gyro_m","Boeing_crystm02"]
    # curr_d= {}
    # for mat in filtered_mat:
    #   curr_d[mat] = mat_detail_d_fast[mat]
    # throughput_scaling(curr_d)

    filtered_mat = ["Bai_tols340","HB_steam1","Bindel_ted_B_unscaled","Pothen_barth5","Oberwolfach_gyro_m","Boeing_crystm02"]
    for mat in filtered_mat:
      print(mat)
      throughput_scaling_one_matrix(mat, mat_detail_d_fast[mat], mat_detail_layer_wise_d[mat])
    exit(1)

  print(f"Finally using {len(filtered_mat)} matrices")
  # overall performance
  fast_acceleration= [mat_detail_d_fast[name].acceleration_factor for name in filtered_mat]
  print(f"fast_acceleration: {geo_mean(fast_acceleration)}, {geo_mean_overflow(fast_acceleration)}, {mean(fast_acceleration)}")
  O1_acceleration= [mat_detail_d[name].acceleration_factor for name in filtered_mat]
  print(f"O1_acceleration: {geo_mean(O1_acceleration)}, {geo_mean_overflow(O1_acceleration)}, {mean(O1_acceleration)}")

  O1_mkl_acceleration= [mat_detail_d[name].max_throughput/mat_detail_d[name].intel_mkl_throughput for name in filtered_mat]
  print(f"O1_mkl_acceleration: {geo_mean(O1_mkl_acceleration)}, {geo_mean_overflow(O1_mkl_acceleration)}, {mean(O1_mkl_acceleration)}")
  O1_gpu_acceleration= [mat_detail_d[name].max_throughput/mat_detail_d[name].nvidia_cusparse_throughput for name in filtered_mat]
  print(f"O1_gpu_acceleration: {geo_mean(O1_gpu_acceleration)}, {geo_mean_overflow(O1_gpu_acceleration)}, {mean(O1_gpu_acceleration)}")
  O1_layer_acc= [mat_detail_d[name].max_throughput/mat_detail_d[name].max_layer_wise_throughput for name in filtered_mat]
  print(f"O1_layer_acc: {geo_mean(O1_layer_acc)}, {geo_mean_overflow(O1_layer_acc)}, {mean(O1_layer_acc)}")

  Ofast_mkl_acceleration= [mat_detail_d_fast[name].max_throughput/mat_detail_d[name].intel_mkl_throughput for name in filtered_mat]
  print(f"Ofast_mkl_acceleration: {geo_mean(Ofast_mkl_acceleration)}, {geo_mean_overflow(Ofast_mkl_acceleration)}, {mean(Ofast_mkl_acceleration)}")
  Ofast_gpu_acc= [mat_detail_d_fast[name].max_throughput/mat_detail_d[name].nvidia_cusparse_throughput for name in filtered_mat]
  print(f"Ofast_gpu_acc: {geo_mean(Ofast_gpu_acc)}, {geo_mean_overflow(Ofast_gpu_acc)}, {mean(Ofast_gpu_acc)}")
  Ofast_layer_acc= [mat_detail_d_fast[name].max_throughput/mat_detail_d[name].max_layer_wise_throughput for name in filtered_mat]
  print(f"Ofast_layer_acc: {geo_mean(Ofast_layer_acc)}, {geo_mean_overflow(Ofast_layer_acc)}, {mean(Ofast_layer_acc)}")

  # remove big outlier
  filtered_mat = [m for m in filtered_mat if mat_detail_d_fast[m].max_throughput < 12*1.e3]

  print(f"Plotting for {len(filtered_mat)} matrices")
  mat_detail_filtered_d={}
  for name in filtered_mat:
    mat_detail_filtered_d[name] = mat_detail_d[name]

  # throughput_scaling(mat_detail_d)

  # throughput_scaling_single_chart(mat_detail_filtered_d)
  # plot_size_histogram(mat_detail_filtered_d)
  # exit(1)

  throughput_scatter_plot(mat_detail_filtered_d, mat_detail_d_fast, th_wise_color = False, savefig= False)
  # throughput_scatter_plot(mat_detail_filtered_d, mat_detail_d_fast, th_wise_color = True)

def plot_spn():
  fname= './run_log_psdd_two_way_limit_Ofast_eridani'
  mat_detail_d, _, _= get_data(fname, with_single_thread= True)
  del mat_detail_d['mnist']
  del mat_detail_d['bnetflix']

  spn_acc= [mat_detail_d[name].acceleration_factor for name in mat_detail_d.keys()]
  print(f"spn_acc: {geo_mean(spn_acc)}, {geo_mean_overflow(spn_acc)}, {mean(spn_acc)}")
  
  throughput_scaling_single_chart(mat_detail_d, legend= True)

if __name__ == "__main__":
  # plot_spn()
  main()
