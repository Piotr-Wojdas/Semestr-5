import networkx as nx
import requests
import gzip
import io
import random
import matplotlib.pyplot as plt
from collections import deque

URL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"      # Plik to lista krawędzi "u v" w skierowanej sieci



# Dodatkowe funkcje ======================================================================================================================
def load_graph(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), "rt") as f:
        G = nx.read_edgelist(f, create_using=nx.Graph(), comments="#", nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

Network = load_graph(URL)

def shortest_path_algo(G, start, end):
    if start == end:
        return [start]

    visited = {start}
    parent = {start: None}
    q = deque([start])

    while q:
        u = q.popleft()
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                if v == end:
                    
                    path = [end]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(v)
    return None

def subgraph(G, n_nodes, seed=None):
    all_nodes = list(G.nodes())
    k = min(n_nodes, len(all_nodes))
    rng = random.Random(seed) if seed is not None else random
    sampled_nodes = rng.sample(all_nodes, k)
    return G.subgraph(sampled_nodes).copy()

def max_flow(G, source, sink):    
    if source == sink:
        return 0, {}
    
    
    H = nx.DiGraph()    # zamieniamy na graf skierowany w obie strony o wadze 1
    for u, v in G.edges():
        H.add_edge(u, v, capacity=1)
        H.add_edge(v, u, capacity=1)
    
    flow_value, flow_dict = nx.maximum_flow(H, source, sink)    
    return flow_value, flow_dict


# Zad 1 ==================================================================================================================================

def Zad1(G, start, end):
     # Znajdź najkrótszą ścieżkę
    path = shortest_path_algo(G, start, end)
    print(f"Najkrótsza ścieżka od {start} do {end}: {path}")
    print(f"Długość ścieżki: {len(path) - 1}")
    
    # Zbierz węzły do wizualizacji: ścieżka + sąsiedzi
    nodes_to_show = set(path)
    for node in path:
        neighbors = set(G.neighbors(node))
        nodes_to_show.update(neighbors)

    subgraph = G.subgraph(nodes_to_show)
        
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, k=0.8, iterations=50)
    path_nodes = set(path)
    neighbor_nodes = nodes_to_show - path_nodes        

    nx.draw_networkx_nodes(subgraph, pos, nodelist=neighbor_nodes, node_size=200, node_color='lightgray', alpha=0.6)        
    nx.draw_networkx_nodes(subgraph, pos, nodelist=list(path_nodes)[1:-1], node_size=300, node_color='lightblue', alpha=0.9)        
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[start], node_size=400, node_color='lightgreen', alpha=1)       
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[end],  node_size=400, node_color='salmon', alpha=1)        
        
        # Draw edges only connecting to the main path
    connecting_edges = [(u, v) for u in path for v in G.neighbors(u) if v not in path]
    nx.draw_networkx_edges(subgraph, pos, edgelist=connecting_edges, edge_color='gray', alpha=0.3, width=1)

    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, edge_color='red', alpha=0.8, width=3, arrows=True, arrowsize=20)      
        
    nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight='bold')

    plt.title(f"Najkrótsza ścieżka: {start} → {end} (długość: {len(path)-1})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
        

#Zad1(Network, start=0, end=30)

# Zad 2 ==================================================================================================================================
def zad2(G,n_nodes):
    sub = subgraph(G, n_nodes)  
    largest = max(nx.connected_components(sub), key=len, default=set())
    sub = sub.subgraph(largest).copy()
    if nx.is_eulerian(sub):
        print(f'Podany podgraf posiada cykl Eulera: {list(nx.eulerian_circuit(sub))}')
    elif nx.has_eulerian_path(sub):
        print(f'Podany podgraf posiada ścieżkę Eulera: {list(nx.eulerian_path(sub))}')
    else:
        odd = [v for v, d in sub.degree() if d % 2 == 1]
        print('Podany podgraf nie posiada cyklu ani ścieżki Eulera.')
        print(f'Sprawdzono składową o |V|={sub.number_of_nodes()}, |E|={sub.number_of_edges()}, liczba wierzch. nieparzystych={len(odd)}')

# zad2(Network, 1000)

# Zad 3 ==================================================================================================================================
def zad3(G, source, sink):

    flow_value, flow_dict = max_flow(G, source, sink)    
    print(f"\nMaksymalny przepływ od {source} do {sink}: {flow_value}")
    
    flow_edges = []
    for u in flow_dict:
        for v, f in flow_dict[u].items():
            if f > 0:
                edge = tuple(sorted([u, v]))
                if edge not in [tuple(sorted([a, b])) for a, b in flow_edges]:
                    flow_edges.append((u, v))
    
    
    flow_nodes = set([source, sink])
    for u, v in flow_edges:
        flow_nodes.add(u)
        flow_nodes.add(v)
    
    
    
    viz_graph = G.subgraph(flow_nodes).copy()
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(viz_graph, k=1.2, iterations=50, seed=42)
    
    other_nodes = set(viz_graph.nodes()) - flow_nodes
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=list(other_nodes), 
                          node_size=200, node_color='lightgray', alpha=0.4)
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=list(flow_nodes - {source, sink}), 
                          node_size=300, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=[source], 
                          node_size=500, node_color='lightgreen', alpha=1, label='Źródło')
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=[sink], 
                          node_size=500, node_color='salmon', alpha=1, label='Ujście')
    
    other_edges = [(u, v) for u, v in viz_graph.edges() 
                   if (u, v) not in flow_edges and (v, u) not in flow_edges]
    nx.draw_networkx_edges(viz_graph, pos, edgelist=other_edges, 
                          edge_color='lightgray', alpha=0.3, width=1)
    nx.draw_networkx_edges(viz_graph, pos, edgelist=flow_edges, 
                          edge_color='red', alpha=0.8, width=3, arrows=False)
    
    nx.draw_networkx_labels(viz_graph, pos, font_size=9, font_weight='bold')
    
    plt.title(f"Maksymalny przepływ: {source} → {sink} = {flow_value}\n"
              f"Czerwone krawędzie: ścieżki przepływu")
    plt.legend(scatterpoints=1, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return flow_value, flow_dict

flow_val, flow_data = zad3(Network, source=0, sink=30)












