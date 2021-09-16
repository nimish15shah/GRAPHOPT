
import os
import subprocess

def par_for_v1():
  data_prefix= "\/esat\/puck1\/users\/nshah\/cpu_openmp\/"
  out_prefix= "/esat/puck1/users/nshah/cpu_openmp/compiled_bin/"

  # batch_sz_ls= [1,2,4,8,16,32,64,128,256,512]
  n_threads_ls= [1,2,4,6,8,10,12]
  partition_datasets= [\
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

  for n_threads in n_threads_ls:
    for batch_sz in batch_sz_ls:
      for net in partition_datasets:
        data_path= data_prefix + '{}_{}threads_{}batch.c'.format(net, n_threads, batch_sz)
        cmd= "sed -i '8s/.*/#include \"" + data_path + "\"/' par_for.c"
        print(cmd)
        os.system(cmd)
        outpath= out_prefix + "{}_{}threads_{}batch.out".format(net, n_threads, batch_sz)
        cmd= "make openmp_conda OUT_FILE=" + outpath
        print(cmd)
        os.system(cmd)


def par_for_v2(name_ls, thread_ls, log_path, suffix):
  data_prefix= "\/esat\/puck1\/users\/nshah\/cpu_openmp\/"
  line_number= 8
  run_log= open(log_path, 'a+')
  print("Start", file= run_log, flush=True)

  cmd= "make set_env"
  os.system(cmd)
  for net in name_ls:
    for th in thread_ls:
      data_path= f"{data_prefix}{net}_{suffix}_{th}.c" 
      cmd= "sed -i '8s/.*/#include \"" + data_path + "\"/' par_for_v2.cpp"
      print(cmd)
      os.system(cmd)
      cmd= "make normal_cpp_psdd"
      print(cmd)
      err= os.system(cmd)
      if err:
        print(f"Error in compilation {net}, {th}")
        print(f"{net},{th},Error compilation", file= run_log, flush= True)
      else:
        print("Excuting")
        cmd= "make run_psdd"
        output= subprocess.check_output(cmd, shell=True)
        # os.system(cmd)
        output = str(output)
        print(output)
        output = output[:-3]
        output= output[output.find('N_layers'):]
        print(output)
        print(f"{net},{th},{output}", file= run_log, flush= True)
    
def main():
  name_ls = [\
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
  name_ls= ['nltcs', 'pumsb_star']

  log_path = './run_log_psdd_two_way_limit_Ofast_eridani'
  # log_path = './run_log_psdd_layer_wise_O1_eridani'
  thread_ls= [2,4,8,16]
  suffix= 'TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_FINE'
  # suffix= 'LAYER_WISE_ALAP_CPU_FINE'

  par_for_v2(name_ls, thread_ls, log_path, suffix)


if __name__ == "__main__":
  main()

