
import networkx as nx
import pickle

def write_graph(path, graph, graph_nx):
#  print(graph_nx.nodes())
#  print(graph_nx.edges())
#  print([node for node in graph_nx.nodes() if len(graph_nx.in_edges(node)) == 0])
   with open(path, 'wb+') as fp:
     pickle.dump(graph_nx, fp)
