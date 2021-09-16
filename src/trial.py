from igraph import *

g= Graph()

g.add_vertices(9)

g.add_edges([
  (6,7),
  (6,4),
  (7,5),
  (7,3),
  (5,2),
  (4,3),
  (4,8),
  (3,2),
  (3,0),
  (8,0),
  (0,1),
  (2,1),
  ])

print(g)

permutation= g.canonical_permutation()
result= g.permute_vertices(permutation)
print(g)
print(result)

print(result.get_adjacency())
