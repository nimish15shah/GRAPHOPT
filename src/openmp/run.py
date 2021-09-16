
import os
import subprocess

out_prefix= "/esat/puck1/users/nshah/cpu_openmp/compiled_bin/"

batch_sz_ls= reversed([1,2,4,8,16,32,64,128,256,512])
n_threads_ls= reversed([1,2,4,8,16,32,64,128,256,512])
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
      outpath= out_prefix + "{}_{}threads_{}batch.out".format(net, n_threads, batch_sz)
      cmd= outpath + " " + str(10000000)
      print(cmd)
      os.system(cmd)
      exit(1)

