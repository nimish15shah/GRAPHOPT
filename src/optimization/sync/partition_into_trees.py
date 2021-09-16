#-----------------------------------------------------------------------
# Created by         : KU Leuven
# Filename           : partition_into_trees.py
# Author             : Nimish Shah
# Created On         : 2020-10-29 15:34
# Last Modified      : 
# Update Count       : 2020-10-29 15:34
# Description        : 
#                      
#-----------------------------------------------------------------------

from minizinc import Instance, Model, Solver
import minizinc
import multiprocessing
import pickle

from .. import useful_methods 
from collections import defaultdict

import networkx as nx
from typing import Mapping, MutableMapping, MutableSequence, Sequence, Iterable, List, Set, Dict
import logging

class Partition_into_trees():
  def __init__(self, net, graph, graph_nx, hw_details, model_path_root):
    self.net= net
    self.graph= graph
    self.graph_nx = graph_nx
    self.hw_details= hw_details
    self.model_path_root = model_path_root

  def main(self):
    tree_depth= self.hw_details.tree_depth


  def model_instance(self, solver, var_dict):
    model_path= self.model_path_root + 'layers_of_trees_minimal_2.mzn'
    model= Model(model_path)

    inst= minizinc.Instance(solver, model)

    inst['TMAX']= var_dict['TMAX']
    inst['V']= var_dict['V']
    inst['D']= var_dict['D']
    inst['successors']= var_dict['successors']
    inst['SLACK_DICT']= var_dict['SLACK_DICT']
    inst['predecessors']= var_dict['predecessors']
    inst['edges']= var_dict['edges']
    inst['out_going_edges']= var_dict['out_going_edges']
    inst['incoming_edges']= var_dict['incoming_edges']
