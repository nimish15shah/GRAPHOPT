import gurobipy as gp
from gurobipy import GRB

import networkx as nx
import pickle
import sys
import os
import global_vars

from collections import defaultdict

def main(G, PIPE_STAGES= 3):
  assert nx.is_directed(G)

  print('Constructing model ..')
  m= gp.Model('pipelining')
  fname= os.path.basename(__file__)
  log_file= './gurobi_' + fname +'.log'
  m.Params.LogFile= log_file
#  m.Params.Method= 3
  m.Params.Presolve= -1
#  m.Params.MIPFocus= 1
#  m.Params.Threads= 48
#  m.Params.ConcurrentMIP=4
  # Setting MIPGap is risky for the output mapping in muxes
#  m.Params.MIPGap= 0.1
#= m.Params.TimeLimit= 200
  
  V= set(G.nodes())
  E= gp.tuplelist(list(G.edges()))

  nodes_wo_successors= set([v for v in V if len(list(G.successors(v))) == 0])
  
  longest_path= nx.algorithms.dag.dag_longest_path(G)
  print('len: ', len(G), len(longest_path))
  T_max= len(V) + (PIPE_STAGES + 1) * len(longest_path) + 1
#  T_max= len(V) * PIPE_STAGES + 2
#  T_max= 118
  T= list(range(T_max))
  # Variables
  M_i= m.addVars(V | set(['NOP']), T, vtype=GRB.BINARY)
  T_i= m.addVars(V, vtype=GRB.INTEGER, lb=0, ub= T_max, name= 'T_i')
  LAT= m.addVar(vtype= GRB.INTEGER)
  LIVE_RANGE= m.addVars(E, vtype= GRB.INTEGER)

  # Starting values

#  instr_schedule= list(nx.algorithms.dag.topological_sort(G))
  instr_schedule= init_breadth_wise_topological_sort(G, PIPE_STAGES)
#  for t, v in enumerate(instr_schedule):
#    M_i[v, t].Start= 1
#    if v != 'NOP':
#      T_i[v].Start = t

  # Constraints
  # Getting execution time
#  for v in V:
#    m.addConstr(T_i[v] == sum(t*M_i[v,t] for t in T))
#    m.addConstr(M_i.sum(v, '*') == 1)
#
#  # One instruction in every cycle
#  m.addConstrs(M_i.sum('*', t) == 1 for t in T)
#    
  # pipeline dependency
  for v in V:
    for s in G.successors(v):
      m.addConstr(T_i[s] >= T_i[v] + PIPE_STAGES + 1)
  
  # LAT constraint
  assert len(nodes_wo_successors) != 0
  m.addConstrs(( LAT >= T_i[v] for v in nodes_wo_successors))
#  m.addConstr(LAT >= len(V))
  
  # LIVE_RANGE constraint
  for s, d in E:
    m.addConstr(LIVE_RANGE[(s,d)] >= T_i[d] - T_i[s])


  # Symmetry breaking constraint
#  top= list(nx.algorithms.dag.topological_sort(G))
#  print (top)
#  for idx, v in enumerate(top):
#    if idx != 0:
#      print(v, top[idx - 1])
#      m.addConstr(T_i[v] >= T_i[top[idx -1]] + 1)
#    else:
#      m.addConstr(T_i[v]== 0)

  # Objective
  w_live_range= 0.1/len(T)
  tot_LIVE_RANGE= sum(LIVE_RANGE[e] for e in E)
#  obj= LAT + w_live_range*tot_LIVE_RANGE
  obj= sum(T_i[v] for v in V)
#  obj= LAT 
#  obj= tot_LIVE_RANGE
  m.setObjective(obj, GRB.MINIMIZE)
  m.optimize()

  map_v_to_lb= {v : T_i[v].X for v in V}

  m.setObjective(obj, GRB.MAXIMIZE)
  m.optimize()

  map_v_to_ub= {v : T_i[v].X for v in V}
  
  tot_var= 0
  for v in V:
    tot_var += map_v_to_ub[v] - map_v_to_lb[v]
  
  print('tot_var :', tot_var)
  exit(1)
  print ('LAT: ', LAT.X)
  print ('tot_LIVE_RANGE: ', tot_LIVE_RANGE.getValue())
  print ('weighted tot_LIVE_RANGE: ', w_live_range*tot_LIVE_RANGE.getValue())


  obj= LAT + w_live_range*tot_LIVE_RANGE
#  obj= tot_LIVE_RANGE
  m.setObjective(obj, GRB.MINIMIZE)
  m.optimize()

  print ('LAT: ', LAT.X)
  print ('tot_LIVE_RANGE: ', tot_LIVE_RANGE.getValue())
  print ('weighted tot_LIVE_RANGE: ', w_live_range*tot_LIVE_RANGE.getValue())

  
def init_breadth_wise_topological_sort(G, PIPE_STAGES):
  map_v_to_level= {}

  top= list(nx.algorithms.dag.topological_sort(G))
  
  # treversing from root to leaves
  for v in reversed(top):
    max_level= 0
    for s in G.successors(v):
      if map_v_to_level[s] >= max_level:
        max_level= map_v_to_level[s] + 1
    
    map_v_to_level[v]= max_level
  
  map_level_to_v= defaultdict(list)
  for v, level in map_v_to_level.items():
    map_level_to_v[level].append(v)

  instr_schedule= []
  for l in sorted(list(map_level_to_v.keys()), reverse= True):
    instr_schedule += list(map_level_to_v[l])
    instr_schedule += ['NOP' for i in range(PIPE_STAGES + 1)]
  
#  print(instr_schedule)
  return instr_schedule

if __name__ == "__main__":
  print('Cannot be executed stand-alone')
