import queue

def _node_reuse_profile(graph, final_key):
  # Breadth first search
  open_set= queue.Queue()
  closed_set= set()

  open_set.put(final_key)
  
  keys_visited= 0
  total_reuse= 0
  most_parents= 0

  while not open_set.empty():
    subtree_root= open_set.get()
    if subtree_root in closed_set:
      continue
    
    num_parents= len(graph[subtree_root].parent_key_list)
    total_reuse= total_reuse + num_parents
              
    if (num_parents > most_parents):
      most_parents= num_parents
      most_parents_key= subtree_root

    keys_visited= keys_visited + 1

    for child in graph[subtree_root].child_key_list:
      open_set.put(child)
    
    closed_set.add(subtree_root)
  
  avg_reuse= float(total_reuse)/float(keys_visited)
  
  print('total keys_visited:', keys_visited)
  print('most_num of paretnsi, corresponding key:', most_parents, most_parents_key)
  return avg_reuse 
  

