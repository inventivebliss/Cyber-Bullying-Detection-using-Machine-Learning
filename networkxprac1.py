import networkx as nx
import matplotlib.pyplot as plt
G=nx.DiGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(3,1,weight=4)
G.add_edge(1,2,weight=5)
#G.add_edge(2,3,weight=6)
#G.add_edge(2,3)
#G.add_edge(1,2)
nx.draw(G)
plt.show()
a=nx.authority_matrix(G)
print a
