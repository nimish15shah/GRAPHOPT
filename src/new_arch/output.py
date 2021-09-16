
from ..reporting_tools import reporting_tools

def plot_figures(graph_nx, done_CU):
  print(done_CU)
  colors= ['red', 'green', 'blue', 'yellow', 'pink', 'white']
  for n in list(graph_nx.nodes()):
    graph_nx.nodes[n]['shape']= 'circle'
#    graph_nx.nodes[n]['fillcolor']= colors[done_CU[n]]
    graph_nx.nodes[n]['fillcolor']= colors[done_CU[n]]
    graph_nx.nodes[n]['style']= 'filled'

  reporting_tools.plot_graph_nx_graphviz(graph_nx)


