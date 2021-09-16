
import networkx as nx
from .. import useful_methods 
import logging

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def par_for_sparse_tr_solve_full_coarse(graph, graph_nx, b, list_of_partitions_combined, matrix, config_obj, head_node, golden_val):
  logger.info("Generating openmp data structures")
  N_THREADS= len(list_of_partitions_combined)
  assert N_THREADS != 0

  for parts in list_of_partitions_combined:
    assert len(parts) == len(list_of_partitions_combined[0])
  N_layers= len(list_of_partitions_combined[0])
  
  n_total_nodes= len([n for pe in range(N_THREADS) for l in range(len(list_of_partitions_combined[0])) for n in list_of_partitions_combined[pe][l]])

  assert len(graph_nx) == n_total_nodes, f"{len(graph_nx)} {n_total_nodes}"

  nnz= [[] for _ in range(N_THREADS)]
  nnz_offset= [None for _ in range(N_THREADS)]

  ptr_in= [[] for _ in range(N_THREADS)]
  ptr_offset= [None for _ in range(N_THREADS)]

  b_val= [[] for _ in range(N_THREADS)]
  op_len   = [[] for _ in range(N_THREADS)]
  per_t_nodes_ls   = [[] for _ in range(N_THREADS)] # needed for memory mapping
  thread_offset= [None for _ in range(N_THREADS)]

  # layer_len
  layer_len= [[len(curr_partition) for curr_partition in parts] for parts in list_of_partitions_combined]
  cum_layer_len= [[0] for _ in range(N_THREADS)]
  if N_layers > 0:
    for t in range(N_THREADS):
      for idx, l in enumerate(layer_len[t]):
        cum_layer_l = cum_layer_len[t][idx] + l
        cum_layer_len[t].append(cum_layer_l)

  for l in range(N_layers):
    logger.info(f"working on layer: {l}")
    for t in range(N_THREADS):
      curr_partition= list_of_partitions_combined[t][l]
      # logger.info(f"working on layer: {l} with size: {len(curr_partition)}")
      subg= graph_nx.subgraph(curr_partition)
      topological_list= list(nx.algorithms.dag.topological_sort(subg))
      assert len(topological_list) == len(subg)
      # topological_list= useful_methods.dfs_topological_sort(subg)
      per_t_nodes_ls[t] += topological_list
      for coarse_n in topological_list:
        row_matrix= matrix.getrow(coarse_n)
        row_indices= list(row_matrix.indices)
        row_data=  list(row_matrix.data)

        row_data = [-1.0 * d for d in row_data]
        row_data[-1] = 1.0 / (-1.0 * row_data[-1])

        nnz[t] += row_data
        ptr_in[t] += row_indices

        op_len[t].append(len(row_indices))

        b_val[t].append(coarse_n)

        assert row_indices[-1] == coarse_n

  # assertions
  assert len([a for b in nnz for a in b]) == graph_nx.number_of_edges() + graph_nx.number_of_nodes()

  # offset for every thread
  thread_offset[0] = 0
  nnz_offset[0] = 0
  ptr_offset[0] = 0
  for t in range(1, N_THREADS):
    thread_offset[t] = thread_offset[t-1] + len(op_len[t-1])
    nnz_offset[t] = nnz_offset[t-1] + len(nnz[t-1])
    ptr_offset[t] = ptr_offset[t-1] + len(ptr_in[t-1])

  # map node to memory location
  map_coarse_n_to_mem_location= {}
  # map_nnz_to_mem_location= {}
  # map_b_to_mem_location= {}
  for t in range(N_THREADS):
    curr_offset= thread_offset[t]
    for idx, n in enumerate(per_t_nodes_ls[t]):
      assert n not in map_coarse_n_to_mem_location
      map_coarse_n_to_mem_location[n] = curr_offset + idx
    
  assert len(map_coarse_n_to_mem_location) == len(graph_nx)    

  # popultate pointer and data structures with appropriate memory locations and values
  ptr_in= [[map_coarse_n_to_mem_location[n] for n in t_part] for t_part in ptr_in]
  b_val= [[b[map_coarse_n_to_mem_location[n]] for n in t_part] for t_part in b_val]

  # misc variables
  n_tot= len(graph_nx)
  n_compute= 2*graph_nx.number_of_edges() + graph_nx.number_of_nodes()

  out_str=""
  out_str += "int N_for_threads= {};\n".format(N_THREADS)
  out_str += "int N_layers= {};\n".format(N_layers)
  out_str += "int n_compute= {};\n".format(n_compute)
  out_str += "int head_node_idx= {};\n".format(map_coarse_n_to_mem_location[head_node])
  out_str += "float golden_val= {};\n".format(golden_val)

  curr= [a for b in layer_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int layer_len[{}] = {}{}{};\n".format(N_THREADS * N_layers, '{' ,curr, '}')

  curr= [a for b in cum_layer_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int cum_layer_len[{}] = {}{}{};\n".format(N_THREADS * (N_layers+1), '{' ,curr, '}')

  curr= [a for b in ptr_in for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"int ptr_in[{len(curr)}] = {{ {curr} }};\n"

  curr= useful_methods.ls_to_str(ptr_offset)
  out_str += "int ptr_offset[{}] = {}{}{};\n".format(N_THREADS, '{' ,curr, '}')

  curr= [a for b in op_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"int op_len[{len(graph_nx)}] = {{ {curr} }};\n"

  out_str += f"float res[{len(graph_nx)}] __attribute__ ((aligned (1024))) = {{}};\n"

  curr= [a for b in nnz for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"float nnz[{len(curr)}] __attribute__ ((aligned (1024))) = {{ {curr} }};\n"

  curr= useful_methods.ls_to_str(nnz_offset)
  out_str += f"int nnz_offset[{N_THREADS}] = {{ {curr} }};\n"

  curr= [a for b in b_val for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"float b_val[{len(curr)}] __attribute__ ((aligned (1024))) = {{ {curr} }};\n"

  curr= useful_methods.ls_to_str(thread_offset)
  out_str += "int thread_offset[{}] = {}{}{};\n".format(N_THREADS, '{' ,curr, '}')

#  print(out_str)
  with open(config_obj.get_openmp_file_name(), 'w+') as fp:
    logger.info(f"Writing openmp structures at : {config_obj.get_openmp_file_name()}")
    fp.write(out_str)


def par_for_sparse_tr_solve_semi_coarse(graph, graph_nx, list_of_partitions_coarse, semi_coarse_g, \
    map_semi_coarse_to_tup, map_n_to_r, map_n_to_semi_coarse, map_r_to_nodes_info, 
    config_obj, head_node, golden_val):
  """
    This code does not work for spn because here nnz leaf is only consumed once. and this fact is used
  """
  logger.info("Generating openmp data structures")
  N_THREADS= len(list_of_partitions_coarse)
  assert N_THREADS != 0

  for parts in list_of_partitions_coarse:
    assert len(parts) == len(list_of_partitions_coarse[0])
  N_layers= len(list_of_partitions_coarse[0])

  n_total_nodes= len([n for pe in range(N_THREADS) for l in range(len(list_of_partitions_coarse[0])) for n in list_of_partitions_coarse[pe][l]])

  print("head node details: ", head_node, map_semi_coarse_to_tup[map_n_to_semi_coarse[head_node]])
  assert map_semi_coarse_to_tup[map_n_to_semi_coarse[head_node]][0] == 'final'
  assert head_node in map_semi_coarse_to_tup[map_n_to_semi_coarse[head_node]][1]

  leaves = useful_methods.get_leaves(graph_nx)

  nnz= [[] for _ in range(N_THREADS)]
  nnz_offset= [None for _ in range(N_THREADS)]

  b_key_ls= [[] for _ in range(N_THREADS)]
  b_key_offset= [None for _ in range(N_THREADS)]

  ptr_in= [[] for _ in range(N_THREADS)]
  ptr_offset= [None for _ in range(N_THREADS)]

  op   = [[] for _ in range(N_THREADS)]
  op_len   = [[] for _ in range(N_THREADS)]
  per_t_nodes_ls   = [[] for _ in range(N_THREADS)] # needed for memory mapping
  thread_offset= [None for _ in range(N_THREADS)]

  # layer_len
  layer_len= [[len(curr_partition) for curr_partition in parts] for parts in list_of_partitions_coarse]
  cum_layer_len= [[0] for _ in range(N_THREADS)]
  if N_layers > 0:
    for t in range(N_THREADS):
      for idx, l in enumerate(layer_len[t]):
        cum_layer_l = cum_layer_len[t][idx] + l
        cum_layer_len[t].append(cum_layer_l)

  done_coarse= set()
  tot_op_assert =0
  for l in range(N_layers):
    logger.info(f"working on layer: {l}")
    for t in range(N_THREADS):
      curr_partition= list_of_partitions_coarse[t][l]
      # logger.info(f"working on layer: {l} with size: {len(curr_partition)}")
      subg= semi_coarse_g.subgraph(curr_partition)
      topological_list= list(nx.algorithms.dag.topological_sort(subg))
      assert len(topological_list) == len(subg)
      # topological_list= useful_methods.dfs_topological_sort(subg)
      per_t_nodes_ls[t] += topological_list
      for semi_coarse_n in topological_list:
        operation, nodes = map_semi_coarse_to_tup[semi_coarse_n] 

        coarse_pred = list(semi_coarse_g.predecessors(semi_coarse_n))

        for p in coarse_pred:
          assert p in done_coarse

        op[t].append(operation)
        op_len[t].append(len(coarse_pred))

        if operation == 'final':
          final_prod_node = list(nodes)[0]
          r= map_n_to_r[final_prod_node]
          b_key= map_r_to_nodes_info[r].b_key
          assert final_prod_node == map_r_to_nodes_info[r].final_prod_node

          pred= graph_nx.predecessors(final_prod_node)
          pred= [p for p in pred if (p in leaves) and (p != b_key)]
          assert len(pred) == 1

          nnz[t].append(pred[0])
          b_key_ls[t].append(b_key)

          ptr_in[t] += list(coarse_pred)

          assert len(coarse_pred) <=1

          tot_op_assert += len(coarse_pred) + 1

        elif operation == "mac":
          assert len(coarse_pred) != 0
          mul_nodes = [n for n in nodes if graph[n].is_prod()]
          assert len(mul_nodes) != 0
          
          pred= [p for n in mul_nodes for p in graph_nx.predecessors(n)]

          nnz_input= [p for p in pred if p in leaves]
          other_input= [p for p in pred if p not in leaves]
          other_input_coarse= [map_n_to_semi_coarse[p] for p in other_input]
          assert set(other_input_coarse) == set(coarse_pred)

          assert len(nnz_input) == len(other_input)
          assert len(nnz_input) == len(mul_nodes)

          nnz[t] += nnz_input
          ptr_in[t] += other_input_coarse

          tot_op_assert += 2 * len(coarse_pred) - 1

        elif operation == "add":
          # TODO: there might be sum operations that are useless with no inputs or only 1 input
          assert len(coarse_pred) > 1, f"{coarse_pred}, {semi_coarse_n}"
          ptr_in[t] += list(coarse_pred)

          tot_op_assert += len(coarse_pred)

        done_coarse.add(semi_coarse_n)
  
  # assertions
  assert len([a for b in op for a in b]) == n_total_nodes
  assert len([a for b in op_len for a in b]) == n_total_nodes
  print("tot_op_assert : ", tot_op_assert)

  tot_nnz= 0
  tot_b_key= 0
  assert_nnz= set()
  assert_b_key= set()

  for t in range(0, N_THREADS):
    assert len(op[t]) == len(per_t_nodes_ls[t])
    assert len(op_len[t]) == len(per_t_nodes_ls[t])

    tot_nnz += len(nnz[t])
    tot_b_key += len(b_key_ls[t])
    assert_nnz |= set(nnz[t])
    assert_b_key |= set(b_key_ls[t])
    
    # if the nnz of 'final' operation is subtracted, ptr_in len should be longer than nnz because of "add" operation
    assert len(nnz[t]) - len(b_key_ls[t]) <= len(ptr_in[t])

    for n in nnz[t]:
      assert graph[n].is_leaf()

  assert tot_nnz + tot_b_key == len(leaves)
  assert assert_nnz | assert_b_key == leaves

  # offset for every thread
  thread_offset[0] = 0
  b_key_offset[0] = 0
  nnz_offset[0] = 0
  ptr_offset[0] = 0
  for t in range(1, N_THREADS):
    thread_offset[t] = thread_offset[t-1] + len(op[t-1])
    b_key_offset[t] = b_key_offset[t-1] + len(b_key_ls[t-1])
    nnz_offset[t] = nnz_offset[t-1] + len(nnz[t-1])
    ptr_offset[t] = ptr_offset[t-1] + len(ptr_in[t-1])

  # map node to memory location
  map_coarse_n_to_mem_location= {}
  # map_nnz_to_mem_location= {}
  # map_b_to_mem_location= {}
  for t in range(N_THREADS):
    curr_offset= thread_offset[t]
    for idx, n in enumerate(per_t_nodes_ls[t]):
      assert n not in map_coarse_n_to_mem_location
      map_coarse_n_to_mem_location[n] = curr_offset + idx
    
    # curr_offset= b_key_offset[t]
    # for idx, n in enumerate(b_key_ls[t]):
    #   assert n not in map_b_to_mem_location
    #   map_b_to_mem_location[n] = curr_offset + idx
    
    # curr_offset= nnz_offset[t]
    # for idx, n in enumerate(nnz[t]):
    #   assert n not in map_nnz_to_mem_location
    #   map_nnz_to_mem_location[n] = curr_offset + idx
    
  assert len(map_coarse_n_to_mem_location) == len(semi_coarse_g)    
  assert max(list(map_coarse_n_to_mem_location.values())) == thread_offset[-1] + len(per_t_nodes_ls[-1]) - 1, f"{max(list(map_coarse_n_to_mem_location.values()))}, {thread_offset[-1]}, {len(per_t_nodes_ls[-1])}"

  # popultate pointer and data structures with appropriate memory locations and values
  ptr_in= [[map_coarse_n_to_mem_location[n] for n in t_part] for t_part in ptr_in]
  nnz= [[graph[n].curr_val for n in t_part] for t_part in nnz]
  b_key_ls= [[graph[n].curr_val for n in t_part] for t_part in b_key_ls]

  op_dict= {'add' : 0, 'mac' : 1, 'final' : 2}
  op= [[op_dict[o] for o in t_part] for t_part in op]

  # misc variables
  n_tot= len(semi_coarse_g)
  n_compute= len(graph_nx) - len(leaves)

  out_str=""
  out_str += "int N_for_threads= {};\n".format(N_THREADS)
  out_str += "int N_layers= {};\n".format(N_layers)
  out_str += "int n_compute= {};\n".format(n_compute)
  out_str += "int head_node_idx= {};\n".format(map_coarse_n_to_mem_location[map_n_to_semi_coarse[head_node]])
  out_str += "float golden_val= {};\n".format(golden_val)

  curr= [a for b in layer_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int layer_len[{}] = {}{}{};\n".format(N_THREADS * N_layers, '{' ,curr, '}')

  curr= [a for b in cum_layer_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int cum_layer_len[{}] = {}{}{};\n".format(N_THREADS * (N_layers+1), '{' ,curr, '}')

  curr= [a for b in ptr_in for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"int ptr_in[{len(curr)}] = {{ {curr} }};"

  curr= useful_methods.ls_to_str(ptr_offset)
  out_str += "int ptr_offset[{}] = {}{}{};\n".format(N_THREADS, '{' ,curr, '}')

  curr= [a for b in op for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"int op[{len(semi_coarse_g)}] = {{ {curr} }};\n"

  curr= [a for b in op_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"int op_len[{len(semi_coarse_g)}] = {{ {curr} }};\n"

  out_str += f"float res[{len(semi_coarse_g)}] __attribute__ ((aligned (1024))) = {{}};\n"

  curr= [a for b in nnz for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"float nnz[{len(curr)}] __attribute__ ((aligned (1024))) = {{ {curr} }};\n"

  curr= useful_methods.ls_to_str(nnz_offset)
  out_str += f"int nnz_offset[{N_THREADS}] = {{ {curr} }};\n"

  curr= [a for b in b_key_ls for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += f"float b_key_ls[{len(curr)}] __attribute__ ((aligned (1024))) = {{ {curr} }};\n"

  curr= useful_methods.ls_to_str(b_key_offset)
  out_str += f"int b_key_offset[{N_THREADS}] = {{ {curr} }};\n"

  curr= useful_methods.ls_to_str(thread_offset)
  out_str += "int thread_offset[{}] = {}{}{};\n".format(N_THREADS, '{' ,curr, '}')

#  print(out_str)
  with open(config_obj.get_openmp_file_name(), 'w+') as fp:
    logger.info(f"Writing openmp structures at : {config_obj.get_openmp_file_name()}")
    fp.write(out_str)


def par_for_sparse_tr_solve(graph, graph_nx, list_of_partitions, config_obj, head_node, golden_val):
  logger.info("Generating openmp data structures")
  N_THREADS= len(list_of_partitions)
  assert N_THREADS != 0

  for parts in list_of_partitions:
    assert len(parts) == len(list_of_partitions[0])
  N_layers= len(list_of_partitions[0])
  
  # NOTE: ONLY value of 1 is supported, ptr_0 etc. is not adjusted according to offset for CACHE_LINE_SIZE > 1
  # this optimization is not actually very useful so using actual value does not bring any benefit
  # CACHE_LINE_SIZE= 1 # words

  # NOTE: this is ideal but not implemented right now
  # memory datastructure:
  # 0 to nnz: nz leaves
  # nnz to b+nnz: b vecotr
  # b+nnz to end: internal operations
  # organize leaves (nz, b) in to contiguous block according to csr format

  # NOTE: implemented a simpler version, 
  # where leaves are also organized according to partitions

  # ptrs
  ptr_0= [[] for _ in range(N_THREADS)]
  ptr_1= [[] for _ in range(N_THREADS)]
  ptr_out= [[] for _ in range(N_THREADS)] # in this scheme, output is contiguous
  op   = [[] for _ in range(N_THREADS)]
  per_t_leaves_ls   = [[] for _ in range(N_THREADS)]
  per_t_nodes_ls   = [[] for _ in range(N_THREADS)]
  thread_offset= [None for _ in range(N_THREADS)]
  ptr_offset= [None for _ in range(N_THREADS)]
  output_offset= [None for _ in range(N_THREADS)]

  leaves = useful_methods.get_leaves(graph_nx)

  done_leaves=set()

  for t, parts in enumerate(list_of_partitions):
    logger.info(f"working on thread: {t}")
    for l, curr_partition in enumerate(parts):
      # logger.info(f"working on layer: {l} with size: {len(curr_partition)}")
      subg= graph_nx.subgraph(curr_partition)
      topological_list= list(nx.algorithms.dag.topological_sort(subg))
      assert len(topological_list) == len(subg)
      # topological_list= useful_methods.dfs_topological_sort(subg)
      per_t_nodes_ls[t] += topological_list
      for n in topological_list:
        pred= list(graph_nx.predecessors(n))
        assert n not in leaves
        if len(pred) > 0:
          assert len(pred) == 2
          in_0= pred[0]
          in_1= pred[1]
        else:
          assert 0, n

        if graph[in_0].is_leaf() and (in_0 not in done_leaves):
          per_t_leaves_ls[t].append(in_0)
          done_leaves.add(in_0)

        if graph[in_1].is_leaf() and (in_1 not in done_leaves):
          per_t_leaves_ls[t].append(in_1)
          done_leaves.add(in_1)

        ptr_0[t].append(in_0)
        ptr_1[t].append(in_1)
        ptr_out[t].append(n)
        
        if graph[n].is_sum():
          op[t].append(0)
        else:
          op[t].append(1)
  
  # offset for every thread
  thread_offset[0] = 0
  for t in range(1, N_THREADS):
    curr_offset= thread_offset[t-1]
    # assert curr_offset % CACHE_LINE_SIZE == 0

    curr_offset += len(per_t_leaves_ls[t-1])
    curr_offset += len(per_t_nodes_ls[t-1])

    # curr_offset = (curr_offset + CACHE_LINE_SIZE - 1) // CACHE_LINE_SIZE
    
    thread_offset[t] = curr_offset

  for t in range(N_THREADS):
    output_offset[t] = thread_offset[t] + len(per_t_leaves_ls[t])

  ptr_offset[0] = 0
  for t in range(1, N_THREADS):
    ptr_offset[t] = ptr_offset[t-1] + len(per_t_nodes_ls[t-1])

  # layer_len
  layer_len= [[len(curr_partition) for curr_partition in parts] for parts in list_of_partitions]
  cum_layer_len= [[0] for _ in range(N_THREADS)]
  if N_layers > 0:
    for t in range(N_THREADS):
      for idx, l in enumerate(layer_len[t]):
        cum_layer_l = cum_layer_len[t][idx] + l
        cum_layer_len[t].append(cum_layer_l)

  # map node to memory location
  map_n_to_mem_location= {}
  for t in range(N_THREADS):
    curr_offset= thread_offset[t]
    for idx, n in enumerate(per_t_leaves_ls[t]):
      assert n not in map_n_to_mem_location
      map_n_to_mem_location[n] = curr_offset + idx
    
    curr_offset += len(per_t_leaves_ls[t])
    for idx, n in enumerate(per_t_nodes_ls[t]):
      assert n not in map_n_to_mem_location
      map_n_to_mem_location[n] = curr_offset + idx
    
  assert len(map_n_to_mem_location) == len(graph_nx)    
  assert max(list(map_n_to_mem_location.values())) == thread_offset[-1] + len(per_t_leaves_ls[-1]) + len(per_t_nodes_ls[-1]) - 1

  ptr_0= [[map_n_to_mem_location[n] for n in t_part] for t_part in ptr_0]
  ptr_1= [[map_n_to_mem_location[n] for n in t_part] for t_part in ptr_1]
  ptr_out= [[map_n_to_mem_location[n] for n in t_part] for t_part in ptr_out]
  assert len(ptr_0) == N_THREADS

  # assertion to make sure that ptr_out mem_locations are contiguous
  for t_part in ptr_out:
    if len(t_part) != 0:
      assert len(t_part) == t_part[-1] - t_part[0] + 1
      assert max(t_part) == t_part[-1]
      assert min(t_part) == t_part[0]
      assert len(set(t_part)) == len(t_part)
  
  # leaves value initalization
  l_mem_idx_ls=[]
  l_val_ls= []
  for l in leaves:
    l_mem_idx_ls.append(map_n_to_mem_location[l])
    l_val_ls.append(graph[l].curr_val)
    

  # misc variables
  n_tot= len(graph)
  n_leaves= len(leaves)
  n_compute= n_tot - n_leaves

  out_str=""
  out_str += "int N_for_threads= {};\n".format(N_THREADS)
  out_str += "int N_layers= {};\n".format(N_layers)
  out_str += "int n_tot= {};\n".format(n_tot)
  out_str += "int n_leaves= {};\n".format(n_leaves)
  out_str += "int n_compute= {};\n".format(n_compute)
  out_str += "int head_node_idx= {};\n".format(map_n_to_mem_location[head_node])
  out_str += "float golden_val= {};\n".format(golden_val)
  

  curr= [a for b in layer_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int layer_len[{}] = {}{}{};\n".format(N_THREADS * N_layers, '{' ,curr, '}')

  curr= [a for b in cum_layer_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int cum_layer_len[{}] = {}{}{};\n".format(N_THREADS * (N_layers+1), '{' ,curr, '}')

  curr= [a for b in ptr_0 for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int ptr_0[{}] = {}{}{};\n".format(n_compute, '{' ,curr, '}')

  curr= [a for b in ptr_1 for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int ptr_1[{}] = {}{}{};\n".format(n_compute, '{' ,curr, '}')

  # curr= [a for b in ptr_out for a in b]
  # curr= useful_methods.ls_to_str(curr)
  # out_str += "int ptr_out[{}] = {}{}{};\n".format(N_THREADS * tot_layer_len, '{' ,curr, '}')

  curr= [a for b in op for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "bool op[{}] = {}{}{};\n".format(n_compute, '{' ,curr, '}')

  #curr= [a for b in res for a in b]
  #curr= ["{:.7f}".format(x) for x in curr]
  #curr= ','.join(curr)
  ##curr= useful_methods.ls_to_str(curr)
  #out_str += "float res[{}] __attribute__ ((aligned (1024))) = {}{}{};\n".format(n_tot* batch_sz, '{' ,curr, '}')
  out_str += f"float res[{n_tot}] __attribute__ ((aligned (1024))) = {{}};\n"

  curr= useful_methods.ls_to_str(l_mem_idx_ls)
  out_str += "int l_mem_idx_ls[{}] = {}{}{};\n".format(len(leaves), '{' ,curr, '}')
  
  curr= useful_methods.ls_to_str(l_val_ls)
  out_str += "float l_val_ls[{}] = {}{}{};\n".format(len(leaves), '{' ,curr, '}')

  curr= useful_methods.ls_to_str(thread_offset)
  out_str += "int thread_offset[{}] = {}{}{};\n".format(N_THREADS, '{' ,curr, '}')

  curr= useful_methods.ls_to_str(output_offset)
  out_str += "int output_offset[{}] = {}{}{};\n".format(N_THREADS, '{' ,curr, '}')

  curr= useful_methods.ls_to_str(ptr_offset)
  out_str += "int ptr_offset[{}] = {}{}{};\n".format(N_THREADS, '{' ,curr, '}')

#  print(out_str)
  with open(config_obj.get_openmp_file_name(), 'w+') as fp:
    logger.info(f"Writing openmp structures at : {config_obj.get_openmp_file_name()}")
    fp.write(out_str)

def par_for(outpath, graph, graph_nx, list_of_partitions, golden_val, batch_sz):
  N_THREADS= len(list_of_partitions)
  assert N_THREADS != 0

  for parts in list_of_partitions:
    assert len(parts) == len(list_of_partitions[0])
  N_layers= len(list_of_partitions[0])
  
  new_graph_nx, map_old_to_new = useful_methods.relabel_nodes_with_contiguous_numbers_leaves(graph_nx, start= 0)
  new_list_of_partitions= [[set([map_old_to_new[n] for n in curr_partition])for curr_partition in parts] for parts in list_of_partitions]
  
  map_new_to_old= {y:x for x,y in map_old_to_new.items()}

  # init leaves
#  res= [[0.0 for _ in range(batch_sz)] for _ in range(len(graph_nx))]
  leaves = useful_methods.get_leaves(new_graph_nx)
  len_leaves= len(leaves)
  res= [[0.0 for _ in range(batch_sz)] for _ in range(len_leaves)]
  for b in range(batch_sz):
    for n in leaves:
      old_n= map_new_to_old[n]
      assert graph[old_n].is_leaf()
      val= graph[old_n].curr_val
      res[n][b] = val
  
  # layer_len
  layer_len= [[len(curr_partition) for curr_partition in parts] for parts in new_list_of_partitions]
  cum_layer_len= [[0] for _ in range(N_THREADS)]
  if N_layers > 0:
    for t in range(N_THREADS):
      for idx, l in enumerate(layer_len[t]):
        cum_layer_l = cum_layer_len[t][idx] + l
        cum_layer_len[t].append(cum_layer_l)

  # ptrs
  ptr_0= [[] for _ in range(N_THREADS)]
  ptr_1= [[] for _ in range(N_THREADS)]
  ptr_out= [[] for _ in range(N_THREADS)]
  op   = [[] for _ in range(N_THREADS)]

  done= list(leaves)
  for t, parts in enumerate(new_list_of_partitions):
    for l, curr_partition in enumerate(parts):
      subg= new_graph_nx.subgraph(curr_partition)
      topological_list= list(nx.algorithms.dag.topological_sort(subg))
      for n in topological_list:
        pred= list(new_graph_nx.predecessors(n))
        assert n not in leaves
        if len(pred) > 0:
          assert len(pred) == 2
          in_0= pred[0]
          in_1= pred[1]
        else:
          assert 0, n
        done.append(n)
        ptr_0[t].append(in_0)
        ptr_1[t].append(in_1)
        ptr_out[t].append(n)
        
        old_n = map_new_to_old[n]
        if graph[old_n].is_sum():
          op[t].append(0)
        else:
          op[t].append(1)
  
  # pad 0s at the end of ptrs and op to make them of same len
  tot_layer_len = max(len(part) for part in ptr_0)
  for t in range(N_THREADS):
    ptr_0[t] += [0]*(tot_layer_len - len(ptr_0[t]))
    ptr_1[t] += [0]*(tot_layer_len - len(ptr_1[t]))
    ptr_out[t] += [0]*(tot_layer_len - len(ptr_out[t]))
    op[t] += [0]*(tot_layer_len - len(op[t]))
  

  # misc variables
  n_tot= len(graph)
  n_leaves= len_leaves
  n_compute= n_tot - n_leaves

  out_str=""
  out_str += "int N_for_threads= {};\n".format(N_THREADS)
  out_str += "int N_layers= {};\n".format(N_layers)
  out_str += "int batch_sz= {};\n".format(batch_sz)
  out_str += "int n_tot= {};\n".format(n_tot)
  out_str += "int n_leaves= {};\n".format(n_leaves)
  out_str += "int n_compute= {};\n".format(n_compute)
  out_str += "int tot_layer_len= {};\n".format(tot_layer_len)
  out_str += "float golden_val= {};\n".format(golden_val)
  

  curr= [a for b in layer_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int layer_len[{}] = {}{}{};\n".format(N_THREADS * N_layers, '{' ,curr, '}')

  curr= [a for b in cum_layer_len for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int cum_layer_len[{}] = {}{}{};\n".format(N_THREADS * (N_layers+1), '{' ,curr, '}')

  curr= [a for b in ptr_0 for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int ptr_0[{}] = {}{}{};\n".format(N_THREADS * tot_layer_len, '{' ,curr, '}')

  curr= [a for b in ptr_1 for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int ptr_1[{}] = {}{}{};\n".format(N_THREADS * tot_layer_len, '{' ,curr, '}')

  curr= [a for b in ptr_out for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "int ptr_out[{}] = {}{}{};\n".format(N_THREADS * tot_layer_len, '{' ,curr, '}')

  curr= [a for b in op for a in b]
  curr= useful_methods.ls_to_str(curr)
  out_str += "bool op[{}] = {}{}{};\n".format(N_THREADS * tot_layer_len, '{' ,curr, '}')

  curr= [a for b in res for a in b]
  curr= ["{:.7f}".format(x) for x in curr]
  curr= ','.join(curr)
  #curr= useful_methods.ls_to_str(curr)
  out_str += "float res[{}] __attribute__ ((aligned (1024))) = {}{}{};\n".format(n_tot* batch_sz, '{' ,curr, '}')

#  print(out_str)
  with open(outpath, 'w+') as fp:
    logger.info(f"writing openmp code to file: {outpath}")
    fp.write(out_str)

