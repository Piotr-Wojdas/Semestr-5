import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network

# Krok 1: Wczytanie danych z pliku i przygotowanie listy krawędzi
edges = []
nodes = set()
with open('Sieci złożone/lista 2/edges.edges', 'r') as f:
    for line in f:
        u, v = map(int, line.strip().split())
        edges.append((u, v))
        nodes.add(u)
        nodes.add(v)

# Mapowanie numerów węzłów na indeksy 0...n-1
node_list = sorted(list(nodes))
node_map = {node: i for i, node in enumerate(node_list)}
num_nodes = len(node_list)
num_edges = len(edges)


# Macierz sąsiedztwa
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
for u, v in edges:
    u_idx, v_idx = node_map[u], node_map[v]
    adj_matrix[u_idx, v_idx] = 1

# Macierz incydencji
inc_matrix = np.zeros((num_nodes, num_edges), dtype=int)
for i, (u, v) in enumerate(edges):
    u_idx, v_idx = node_map[u], node_map[v]
    inc_matrix[u_idx, i] = -1
    inc_matrix[v_idx, i] = 1

print("Macierz sąsiedztwa (fragment):")
print(adj_matrix[:5, :5])
print("\nMacierz incydencji (fragment):")
print(inc_matrix[:5, :5])

# Zapis macierzy do plików tekstowych
np.savetxt("adjacency_matrix.txt", adj_matrix, fmt='%d')
np.savetxt("incidence_matrix.txt", inc_matrix, fmt='%d')
print("\nMacierze zostały zapisane do plików 'adjacency_matrix.txt' i 'incidence_matrix.txt'")

# Wczytanie grafu do networkx
G = nx.DiGraph()
G.add_edges_from(edges)

# Wizualizacja grafu przy pomocy networkx i matplotlib
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, arrows=True)
plt.title("Wizualizacja grafu (networkx)")
plt.savefig("graph_networkx.png")
print("\nGraf został zwizualizowany i zapisany jako 'graph_networkx.png'")

# Krok 5: Wizualizacja grafu przy pomocy pyvis
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
net.from_nx(G)
net.show_buttons(filter_=['physics'])
net.show("graph_pyvis.html", notebook=False)
print("Graf został zwizualizowany interaktywnie i zapisany jako 'graph_pyvis.html'")

































