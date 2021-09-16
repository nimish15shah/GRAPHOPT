
import os
import subprocess

#datasets= ['ad_psdd' , \
#    'kdd_psdd', \
#    'adult_psdd', \
#    'audio_psdd', \
#    'bbc_psdd', \
#    'book_psdd', \
#    'cpu_psdd', \
#    'dna_psdd', \
#    'jester_psdd', \
#    'msnbc_psdd', \
#    'nltcs_psdd', \
#    'tmovie_psdd', \
#    'wilt_psdd', \
#   ]
#
all_datasets= [\
    'ad', \
    'baudio', \
    'bbc', \
    'bnetflix', \
    'book', \
    'c20ng', \
    'cr52', \
    'cwebkb', \
    'elevators', \
    'insurance', \
    'jester', \
    'kdd', \
    'mnist', \
    'msnbc', \
    'msweb', \
    'nltcs', \
    'plants', \
    'pumsb_star', \
    'simple2.1', \
    'simple2.2', \
    'simple2.3', \
    'simple2.4', \
    'simple2.5', \
    'simple2.6', \
    'tmovie', \
    'tretail', \
    'wilt', \
  ]

#fit_datasets= [\
#  'bnetflix', \
#  'dna', \
#  'elevators', \
#  'insurance', \
#  'jester', \
#  'mnist', \
#  'msnbc', \
#  'msweb', \
#  'nltcs', \
#  'simple2.1', \
#  'simple2.6', \
#  'tretail', \
#]
fit_datasets= [\
  'bnetflix', \
  'jester', \
  'mnist', \
  'kdd', \
  'msnbc', \
  'msweb', \
  'nltcs', \
  'tretail', \
]
partition_datasets= [\
#    'nltcs', \
#    'mnist', \
#    'msnbc', \
#    'msweb', \
#    'tretail', \
#    'bnetflix', \
#    'ad', \
    'jester', \
    'bbc', \
    'book', \
    'cr52', \
    'cwebkb', \
    'c20ng', \
    'kdd', \
#    'baudio', \
#    'pumsb_star', \
  ]

temp_list= [\
#    'ad', \
    'jester', \
    'mnist', \
    'msnbc', \
    'msweb', \
    'nltcs', \
    'tretail' \
  ]

for name in partition_datasets:
#for name in temp_list:
#  cmd = "make verif_psdd NET=" + name
  print(name)
#  cmd = "make null_psdd NET=" + name
#  cmd += " | grep arith"
#  cmd = "make compile_async_psdd NET=" + name
#  cmd = "make async_partition_psdd NET=" + name
#  cmd = "make batched_cuda_psdd NET=" + name
#  cmd = "make openmp_psdd NET=" + name
#  cmd = "make null_psdd NET=" + name
  cmd= "openmp_psdd"
  cmd = f"make async_partition_psdd NET={name}"
  cmd= f"python main.py {name} -t --tmode async_partition --cir_type psdd --targs heuristic 1"
  cmd= f"python main.py {name} -t --tmode async_partition --cir_type psdd --targs two_way_partition 1"
  cmd= f"python main.py {name} -t --tmode compile_for_async_arch  --cir_type psdd --targs 64 layer_wise full"
  cmd= "compile_async_psdd"
  print(cmd)
  subprocess.Popen(["make", cmd, "NET=" + name])
#  os.system(cmd)

