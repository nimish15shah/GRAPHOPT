
from __future__ import print_function
from ortools.sat.python import cp_model

import networkx as nx

def main(G, PIPE_STAGES= 3):
  # Creates the model.
  m= cp_model.CpModel()

  V= set(G.nodes())
  E= list(G.edges())
  nodes_wo_successors= set([v for v in V if len(list(G.successors(v))) == 0])
  assert len(nodes_wo_successors) != 0

  longest_path= nx.algorithms.dag.dag_longest_path(G)
  print('len: ', len(G), len(longest_path))
#  T_max= len(V) + (PIPE_STAGES + 1) * len(longest_path) + 1
  T_max= len(V) + (PIPE_STAGES + 1) * 3

  T_i= {}
  for v in V:
    T_i[v]= m.NewIntVar(0, T_max, 'T_' + str(v))

  m.AddAllDifferent(list(T_i.values()))
  
  for src, dst in E:
    m.Add(T_i[dst] >= T_i[src] + PIPE_STAGES + 1)


  LAT= m.NewIntVar(len(V) + PIPE_STAGES + 1, T_max, 'LAT')
  for v in nodes_wo_successors:
    m.Add( LAT >= T_i[v])
#    m.Add(T_i[v] == (T_max - 1))
  
  m.Minimize(LAT)

  solver = cp_model.CpSolver()
  solver.Solve(m)
  print(solver.StatusName())
  # Assert
  for src, dst in E:
#    print(solver.Value(T_i[dst]), solver.Value(T_i[src]) + PIPE_STAGES + 1)
    assert solver.Value(T_i[dst]) >= solver.Value(T_i[src]) + PIPE_STAGES + 1

  for v in nodes_wo_successors:
    print(solver.Value(T_i[v]))

  print(solver.Value(LAT))
