import networkx as nx
from collections import defaultdict
from .. import useful_methods

def statistics_of_IO_graph(graph_nx, IO_graph):
  
  internal_nodes= useful_methods.get_non_leaves(graph_nx)
  IO_subgraph= IO_graph.subgraph(internal_nodes)
  connected_components= nx.algorithms.components.connected_components(IO_subgraph)

  useful_methods.plot_graph(IO_graph)

  print([len(c) for c in connected_components])

def statistics_of_BB_graph(BB_graph_nx):
  useful_methods.plot_graph(BB_graph_nx)

def graph_color_clique(net, graph_nx, BB_graph, IO_graph):
  leaves= set(useful_methods.get_leaves(graph_nx))
  internal_nodes= useful_methods.get_non_leaves(graph_nx)
  
  IO_graph_internal= IO_graph.subgraph(internal_nodes) 

  new_IO_graph, mapping= useful_methods.relabel_nodes_with_contiguous_numbers(IO_graph_internal, start= 1)

#  from networkx.algorithms import approximation
#  clique= approximation.clique.max_clique(IO_graph_internal)

  output_constr_str = output_constr(BB_graph, mapping)

  cliques= []
  for BB, obj in list(BB_graph.items()):
    # inputs
    clique= set(obj.in_list) - leaves
    clique= [mapping[n] for n in clique]
    cliques.append(clique)
    
    # outputs
    clique= set(obj.out_list) - leaves
    clique= [mapping[n] for n in clique]
    cliques.append(clique)
  
  # constraints
  constr_str= ''
  for clique in cliques:
    if len(clique) > 1:
      curr_str= 'constraint alldifferent(['
#      curr_str= 'constraint alldifferent_except_0(['
      for n in clique:
        curr_str += f'col[{n}],'

      curr_str = curr_str[:-1]
      curr_str += ']);\n'

      constr_str += curr_str

  # data      
  data_str= f'N= {len(IO_graph_internal)};'

  path= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/optimization/minizinc_code/no_backup/'
  
  file_path = path + f'graph_color_cliques_{net}.mzn'
  with open(file_path, 'w+') as fp:
    fp.write(constr_str)
    fp.write(output_constr_str)

  file_path = path + f'graph_color_cliques_{net}.dzn'
  with open(file_path, 'w+') as fp:
    fp.write(data_str)

def sub_graph_nx_constr(struct_graph_nx, curr_out_nodes, map_v_to_lvl, node_mapping):
  curr_str= ""

  # key:  tup : max_node, min_node
  # val: div_factor
  inequality_dict= defaultdict(lambda: 0)
  equality_dict= defaultdict(lambda: 1000) # very high number
  
  for n in curr_out_nodes:
    # NOTE: struct_graph_nx also has leaves
    # NOTE: div_factor is minimum 2, when map_v_to_lvl is 1
    div_factor= 2**(map_v_to_lvl[n])

    ancestors = nx.algorithms.dag.ancestors(struct_graph_nx, n)
    ancestors &= curr_out_nodes
    for a in ancestors: 
      tup= (max(n,a), min(n,a))
      equality_dict[tup]= min(equality_dict[tup], div_factor)

    neither_ancestors_nor_descendents = useful_methods.neither_ancestors_nor_descendents(struct_graph_nx, n)
    neither_ancestors_nor_descendents &= curr_out_nodes
    for non_n in neither_ancestors_nor_descendents:
      tup= (max(n,non_n), min(n,non_n))
      inequality_dict[tup]= max(inequality_dict[tup], div_factor)

  for tup, div_factor in inequality_dict.items():
    a= node_mapping[tup[0]]
    b= node_mapping[tup[1]]
    curr_str += inequality_constr(a, b, div_factor)

  for tup, div_factor in equality_dict.items():
    a= node_mapping[tup[0]]
    b= node_mapping[tup[1]]
    curr_str += equality_constr(a, b, div_factor)

  return curr_str

def equality_constr(n1, n2, div_factor):
  assert n1 != n2
  return f'constraint div_{div_factor}[{n1}] == div_{div_factor}[{n2}];\n'

def inequality_constr(n1, n2, div_factor):
  assert n1 != n2
  return f'constraint div_{div_factor}[{n1}] != div_{div_factor}[{n2}];\n'
  
def output_constr(BB_graph, node_mapping):    

  output_constr_str = ""
  for BB, obj in list(BB_graph.items()):
    done_outputs= []
    bb_map_v_to_lvl= {}
    out_set= set(obj.out_list)

    # constraints of single subgraph
    for unit in obj.set_of_decompose_units:
      struct_graph_nx = unit.struct_graph_nx

      map_v_to_lvl= useful_methods.compute_lvl(struct_graph_nx)
      nodes= set(struct_graph_nx.nodes())
      curr_out_nodes= (nodes & out_set) - set(done_outputs)

      for n in curr_out_nodes:
        bb_map_v_to_lvl[n] = map_v_to_lvl[n]

      output_constr_str += sub_graph_nx_constr(struct_graph_nx, curr_out_nodes, map_v_to_lvl, node_mapping)

      done_outputs += list(curr_out_nodes)
      
    assert set(done_outputs) == out_set
    assert len(bb_map_v_to_lvl) == len(out_set)

    # constraints across subgraphs
    top_nodes= set([unit.parent for unit in obj.set_of_decompose_units])
    inequality_dict= defaultdict(lambda: 0)
    for n in top_nodes:
      other_nodes= top_nodes - set([n])
      # NOTE: struct_graph_nx also has leaves
      # NOTE: div_factor is minimum 2, when map_v_to_lvl is 1
      div_factor= 2**(bb_map_v_to_lvl[n])
      lvl = bb_map_v_to_lvl[n]
      for non_n in other_nodes:
        if bb_map_v_to_lvl[non_n] != lvl:
          tup= (max(n,non_n), min(n,non_n))
          inequality_dict[tup]= max(inequality_dict[tup], div_factor)
    
    for tup, div_factor in inequality_dict.items():
      a= node_mapping[tup[0]]
      b= node_mapping[tup[1]]
      output_constr_str += inequality_constr(a, b, div_factor)

    # cliques for top nodes
    for lvl in set(bb_map_v_to_lvl.values()):
      nodes_at_this_lvl= [n for n, n_lvl in bb_map_v_to_lvl.items() if n_lvl == lvl]
      if len(nodes_at_this_lvl) > 1:
        curr_str= 'constraint alldifferent(['
        div_factor= 2**(lvl)
        for n in nodes_at_this_lvl:
          curr_str += f'div_{div_factor}[{node_mapping[n]}],'

        curr_str = curr_str[:-1]
        curr_str += ']);\n'

        output_constr_str += curr_str


    
  return output_constr_str
