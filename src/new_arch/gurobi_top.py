import gurobipy as gp
from gurobipy import GRB

from . import partition
from . import output
from ..useful_methods import get_leaves
from collections import defaultdict
from ..optimization.write_to_file import relabel_nodes_with_contiguous_numbers
import datetime
import random
import os

from ..reporting_tools import reporting_tools

def main(graph, graph_nx, hw_details):
  leaf_set= set(get_leaves(graph_nx))
  list_of_chosen_sets, status_dict= partition.first_partition(graph, graph_nx, hw_details)

#  graph_nx, mapping = relabel_nodes_with_contiguous_numbers(graph_nx, start=0)

  map_cu_to_node= defaultdict(set)
  map_node_to_cu= {}
  for cu, node_set in enumerate(list_of_chosen_sets):
    for n in node_set:
      for p in graph_nx.predecessors(n):
        map_node_to_cu[p] = cu + 1
      assert len(list(graph_nx.predecessors(n))) != 0
  
  for n, cu in map_node_to_cu.items():
    map_cu_to_node[cu].add(n)
  

  all_nodes= set(graph_nx.nodes())
  mapped_set= set(map_node_to_cu.keys())
  unmapped_set= all_nodes - mapped_set
  n_unmapped= len(unmapped_set)
  cu_set= list(sorted(list(range(1, hw_details.N_PE + 1))))
  n_CU= hw_details.N_PE

  active_edges= [e for n in unmapped_set for e in graph_nx.in_edges(n)]
  
  print(map_node_to_cu)
  print(cu_set)
  print('Constructing model ..')
  m= gp.Model('mapping')
  fname= os.path.basename(__file__)
  log_file= './gurobi_' + fname +'.log'
  m.Params.LogFile= log_file
#  m.Params.Method= 3
#  m.Params.Presolve= -1
##  m.Params.MIPFocus= 1
  m.Params.Threads= 12
##  m.Params.ConcurrentMIP=4
#  # Setting MIPGap is risky for the output mapping in muxes
##  m.Params.MIPGap= 0.1
#  m.Params.TimeLimit= 1000
  m.Params.Symmetry= 2
#  obj= m.getAttr("ObjBound")
#  print(dir(m))
#  print(dir(GRB))

  # Variables
  done= m.addVars(all_nodes, vtype= GRB.BINARY)
  done_CU= m.addVars(all_nodes, vtype= GRB.INTEGER, lb=0, ub= max(cu_set)) # cu= 0 means it is not assigned
  mapped= m.addVars(unmapped_set, vtype= GRB.BINARY)

#  mapped_per_CU= m.addVars(cu_set, vtype= GRB.INTEGER, lb=0, ub= n_unmapped)
  mapped_per_CU= m.addVars(cu_set, vtype= GRB.INTEGER, lb=55, ub= 65)
  cu_indicator= m.addVars(cu_set, unmapped_set, vtype= GRB.BINARY)
  max_mapped= m.addVar(vtype= GRB.INTEGER, lb=0, ub= n_unmapped)
  min_mapped= m.addVar(vtype= GRB.INTEGER, lb=0, ub= n_unmapped)
  gap_per_CU= m.addVars(cu_set, vtype= GRB.INTEGER, lb=0, ub= n_unmapped)
  
  edge_cost= m.addVars(active_edges, vtype= GRB.BINARY)
  same_cu= m.addVars(active_edges, vtype= GRB.BINARY)

  # Constraints
  print('Adding constraints ..')
  
  # init constraints
  m.addConstrs(done_CU[n] == map_node_to_cu[n] for n in mapped_set)
  m.addConstrs(done_CU[n] >= 1 for n in mapped_set)
  m.addConstrs(done[n] == 1 for n in mapped_set)
  m.addConstrs(done[n] == 0 for n in unmapped_set)
  
  m.addConstrs(((same_cu[e] == 1) >> (done_CU[e[0]] == done_CU[e[1]]) for e in active_edges))
#  m.addConstrs(((done_CU[e[0]] == done_CU[e[1]]) >> (same_cu[e] == 1) for e in active_edges))
  m.addConstrs(((done_CU[e[0]] - done_CU[e[1]]) <=  n_CU * (1 - same_cu[e]) for e in active_edges))
  m.addConstrs(((done_CU[e[1]] - done_CU[e[0]]) <=  n_CU * (1 - same_cu[e]) for e in active_edges))
  m.addConstrs(((done[e[0]] + same_cu[e] >= mapped[e[1]]) for e in active_edges))
  m.addConstrs((mapped[n] == 1) >> (done_CU[n] >= 1) for n in unmapped_set)
  m.addConstrs(done_CU[n] >= mapped[n] for n in unmapped_set)
  m.addConstrs(done_CU[n] <= n_CU*mapped[n] for n in unmapped_set)


  # edge cost
  m.addConstrs(edge_cost[e] <= mapped[e[1]] for e in active_edges)
  m.addConstrs(edge_cost[e] >= (mapped[e[1]] - same_cu[e]) for e in active_edges)

  # workload balance cost
#  m.addConstrs(mapped_per_CU[c] == sum([1 for n in unmapped_set if (mapped[n] and (done_CU[n]==cu))]) for c in cu_set)
#  m.addConstrs((((done_CU[n] == cu) >> (cu_indicator[cu, n] == 1)) for cu in cu_set for n in unmapped_set))
  m.addConstrs((done_CU[n] == sum([cu_indicator[cu, n] * cu for cu in cu_set]) for n in unmapped_set))
  m.addConstrs(cu_indicator.sum('*', n) <= 1 for n in unmapped_set)
  m.addConstrs((mapped_per_CU[cu] == cu_indicator.sum(cu, '*') for cu in cu_set))

  m.addConstrs(mapped_per_CU[cu] <= max_mapped for cu in cu_set)
  m.addConstrs(mapped_per_CU[cu] >= min_mapped for cu in cu_set)
  m.addConstrs(gap_per_CU[cu] == max_mapped - mapped_per_CU[cu] for cu in cu_set)

  total_mapped_nodes = mapped.sum('*')
  w_tot= 10
  w_balance= 5
  w_edges= 1
  
  obj= w_tot* total_mapped_nodes - w_edges*(edge_cost.sum('*')) - w_balance*(gap_per_CU.sum('*'))
#  obj= w_tot* total_mapped_nodes - w_edges*(edge_cost.sum('*')) - w_balance*(max_mapped - min_mapped)
  m.setObjective(obj, GRB.MAXIMIZE)
  m.optimize()

  print('done_CU', [(n, done_CU[n].X) for n in all_nodes])
  print('cu_indicator', [cu_indicator[round(done_CU[n].X), n].X for n in unmapped_set if mapped[n].X > 0.5])
  print('mapped', [(n, mapped[n].X) for n in unmapped_set])
  print('max_mapped',round( max_mapped.X))
  print('min_mapped', round(min_mapped.X))
  print('total',sum([round(mapped[n].X) for n in unmapped_set]))

  final_done_CU= {n:round(done_CU[n].X)%4 for n in all_nodes}
  
  output.plot_figures(graph_nx, final_done_CU)
