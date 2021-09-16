import queue

def _depth_profile_for_common_child(graph, node, depth_list, depth_to_be_checked, visited= []):
  visited.append(node)
  if (len(visited) % 1000==0):
    print(len(visited))
    if (len(visited)>100000):
      exit(0)

  if (len(graph[node].child_key_list) != 0): # Only find depth for non-leaf nodes
    depth= _breadth_first_search_for_common_child(graph, node, depth_to_be_checked)
    depth_list.append(depth)
    graph[node].depth_of_first_common_child= depth
  
  for child in graph[node].child_key_list:
    if child not in visited:
      _depth_profile_for_common_child(graph, child, depth_list, depth_to_be_checked, visited)

def _breadth_first_search_for_common_child(graph, node, depth_to_be_checked):
  open_set= queue.Queue()
  closed_set= set()
  
  open_set.put(node)
  open_set.put(None)

  first_common_child= None
  depth= 0
  while not open_set.empty():
    subtree_root = open_set.get()
    
    if depth > depth_to_be_checked :
      break
    
    if (subtree_root == None):
      depth = depth + 1
      if (open_set.qsize() == 0):
        break
      else:
        open_set.put(None)
        continue

    if subtree_root in closed_set:
      first_common_child= subtree_root
      break
    
    for child in graph[subtree_root].child_key_list:
      open_set.put(child)

    closed_set.add(subtree_root)
  
  if (first_common_child== None):
    return 0 # Did not find common child in given depth
  else:
    return depth 
  

