from pathlib import Path
import scipy.io
import csv

from . import file_io

matrix_names= [\
  'HB/arc130',
  'Nasa/nasa2910',
  'HB/bcsstk21',
  'HB/bcsstk01',
  'Boeing/msc00726',
  'HB/bcsstk19',
  'Boeing/msc04515',
  'HB/plat1919',
  'Norris/fv1',
  'Okunbor/aft01',
  'NASA/nasa1824',
  'HB/bcsstk09',
  'HB/bcsstk01',
  'HB/bcsstk23',
  'HB/bcsstk07',
  'HB/bcsstk17',
  'HB/bcsstk23',
  'HB/bcsstk13',
  'HB/bcsstk01',
  'HB/bcsstk01',
  'HB/bcsstk17',
  'HB/ash85',
  'HB/ash292',
  'HB/494_bus',
  'HB/1138_bus',
  'HB/bcspwr02',
  'HB/bcspwr03',
  'HB/bcspwr04',
  'HB/bcspwr05',
  'Schenk_AFE/af_shell3',
]

def matrices_path(mat_path, global_var, mode='full', exclude_category_ls= [], include_category_ls= None, ):
  """
   returns a list of posix path objects or a list of path strings.
   posix path can directly be used with file_io.read_mat function
   str(posix_path) can be used to get the path as a string
  """
  assert mode in ['full', 'only_category', 'with_LU_factors']
  all_caregories= list(Path(global_var.SPARSE_MATRIX_MATLAB_PATH).glob("*/"))
  all_caregories= [p.name for p in all_caregories]

  filtered_categories= [c for c in all_caregories if c not in exclude_category_ls]
  if include_category_ls != None:
    assert isinstance(include_category_ls, list)
    filtered_categories= [c for c in filtered_categories if c in include_category_ls]

  if mode == 'full' or mode == 'only_category':
    result= []
    for c in filtered_categories:
      path_ls= list(Path(mat_path).rglob(c + "*.mat")) 
      if mode == 'full':
        pass
      elif mode == 'only_category':
        path_ls= [c + '/' + p.stem for p in path_ls]
      else:
        assert 0
      result += path_ls

  elif mode == 'with_LU_factors':
    result= []
    for c in filtered_categories:
      path_ls= list(Path(mat_path).glob(c + "*.mtx")) 
      # only keep the prefix before the '.'
      path_ls= [p.name[:p.name.index('.')] for p in path_ls]

      # remove category to get the matrix name
      path_ls= [p.replace(c + '_', '') for p in path_ls]

      # combine category with matrix name,
      # and remove U and L from the end 
      # remove duplicates
      path_ls = [c + '/' + p[:-2] for p in path_ls]
      path_ls = list(set(path_ls))

      result += path_ls

  assert len(result) != 0

  return result

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

def matrices_based_on_size(log_path, lower_limit= 0, upper_limit= float("inf")):
  with open(log_path, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

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

  # idx
  name_idx= 0
  n_compute_idx= 9

  name_set= set()
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue

    if len(d) < n_compute_idx:
      continue

    name= d[name_idx]
    for g in group_names:
      name= name.replace(g+'_', g+'/', 1)

    n_compute= int(float(d[n_compute_idx]))

    if n_compute >= lower_limit and n_compute <= upper_limit:
      name_set.add(name)
  
  print(len(name_set), name_set)

  return name_set
