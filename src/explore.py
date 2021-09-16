
from collections import defaultdict
from .useful_methods import printcol
from .useful_methods import printlog

def reuse_factor(graph, leaf_list):
  """
    Prints reuse factor of every node
  """
  out_degree_count= defaultdict(int)
  
  for node, obj in list(graph.items()):
    if obj.is_leaf():
#    if not obj.is_leaf():
      length= len(obj.parent_key_list)
    
      out_degree_count[length] += 1
  
  avg= 0
  for key in sorted(out_degree_count.keys()):
    print(key, out_degree_count[key])
    
    avg += (key - 1)* out_degree_count[key]   
  
  msg= 'Total vector shuffle: ' + str(avg) + ' ,' +str(avg/32)
  printlog(msg, 'red')

def nice_bb(graph,BB_graph):
  count=0

  for bb,obj in list(BB_graph.items()):
    nz_set= set()

    nz_set |= set([node for node in obj.in_list_unique if len(graph[node].parent_key_list)>1])
#    nz_set |= set([node for node in obj.out_list if len(graph[node].parent_key_list)>1])

    if len(nz_set) < 3:
      count += 1

  printlog(str(len(BB_graph)) + ' ' + str(count), 'red')
