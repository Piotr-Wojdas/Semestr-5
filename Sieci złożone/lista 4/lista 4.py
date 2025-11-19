
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite

# Create a new graph
B = nx.Graph()

with open("Sieci złożone/lista 4/davis_edgelist.txt", "r") as f:
    for line in f:
        actor, event = line.strip().split()
        B.add_node(actor, bipartite=0)
        B.add_node(event, bipartite=1)
        B.add_edge(actor, event)


women_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
event_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 1}

# --- 1. Draw the original bipartite graph ---
pos = dict()
pos.update( (n, (i, 1)) for i, n in enumerate(women_nodes) )
pos.update( (n, (i, 0)) for i, n in enumerate(event_nodes) )

plt.figure(figsize=(12, 8))
nx.draw(B, pos=pos, with_labels=True, node_size=500, font_size=8)
plt.title("Davis Southern Women Network (Bipartite Graph)")


# --- 2. Create and draw the projection on women nodes ---
women_graph = bipartite.projected_graph(B, women_nodes)
plt.figure(figsize=(10, 10))
nx.draw(women_graph, with_labels=True, node_color='lightblue', node_size=500, font_size=8)
plt.title("Women's Network (Projection)")


# --- 3. Create and draw the projection on event nodes ---
event_graph = bipartite.projected_graph(B, event_nodes)
plt.figure(figsize=(10, 10))
nx.draw(event_graph, with_labels=True, node_color='lightgreen', node_size=500, font_size=8)
plt.title("Events Network (Projection)")

# --- 4. Display the incidence matrix  ---
women_list = sorted(list(women_nodes))
event_list = sorted(list(event_nodes))
incidence_matrix = bipartite.biadjacency_matrix(B, row_order=women_list, column_order=event_list)

print("Incidence Matrix (Women x Events):")
print(incidence_matrix.toarray())

# --- 5. Display Adjacency Matrices ---
# Adjacency matrix for the women's projection
adj_matrix_women = nx.adjacency_matrix(women_graph, nodelist=women_list)
print("\nAdjacency Matrix (Women's Network):")
print(adj_matrix_women.toarray())

# Adjacency matrix for the events' projection
adj_matrix_events = nx.adjacency_matrix(event_graph, nodelist=event_list)
print("\nAdjacency Matrix (Events' Network):")
print(adj_matrix_events.toarray())


# Show all plots
plt.show()