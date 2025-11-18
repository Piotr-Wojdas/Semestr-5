import networkx as nx
import requests
import gzip
import io
import random
import matplotlib.pyplot as plt
from collections import deque

URL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"      # Plik to lista krawędzi "u v" w skierowanej sieci

def load_graph(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), "rt") as f:
        G = nx.read_edgelist(f, create_using=nx.Graph(), comments="#", nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

Network = load_graph(URL)
if nx.is_connected(Network):
        print("Graf jest spójny.")
        component_to_analyze = Network
else:
    print("Graf nie jest spójny. Analiza dla największej spójnej składowej.")
    largest_cc = max(nx.connected_components(Network), key=len)
    Network = Network.subgraph(largest_cc)


def analyze_graph(G):
    print("Analiza grafu:")

    # Najważniejszy wierzchołek stopniem
    degree_centrality = nx.degree_centrality(G)
    most_important_degree = max(degree_centrality, key=degree_centrality.get)
    print(f"Najważniejszy wierzchołek (stopień): {most_important_degree} (centralność: {degree_centrality[most_important_degree]:.4f})")

    # Najważniejszy wierzchołek bliskością
    closeness_centrality = nx.closeness_centrality(G)
    most_important_closeness = max(closeness_centrality, key=closeness_centrality.get)
    print(f"Najważniejszy wierzchołek (bliskość): {most_important_closeness} (centralność: {closeness_centrality[most_important_closeness]:.4f})")

    # Najważniejszy wierzchołek pośredniością
    betweenness_centrality = nx.betweenness_centrality(G)
    most_important_betweenness = max(betweenness_centrality, key=betweenness_centrality.get)
    print(f"Najważniejszy wierzchołek (pośrednictwo): {most_important_betweenness} (centralność: {betweenness_centrality[most_important_betweenness]:.4f})")

    # Gęstość
    density = nx.density(G)
    print(f"Gęstość grafu: {density:.4f}")

    # Średnica i średnia długość ścieżki (dla największej spójnej składowej, jeśli graf nie jest spójny)
    try:
        diameter = nx.diameter(G)
        print(f"Średnica grafu (lub największej składowej): {diameter}")
    except nx.NetworkXError as e:
        print(f"Nie można obliczyć średnicy: {e}")

    try:
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"Średnia długość najkrótszej ścieżki (lub największej składowej): {avg_path_length:.4f}")
    except nx.NetworkXError as e:
        print(f"Nie można obliczyć średniej długości ścieżki: {e}")




density = nx.density(Network)
if density == 1:
    print("Graf jest pełny.")
else: 
    print("szukamy klik")



def clique_finder(G, n=5):
    print("\n--- Wyszukiwanie i wizualizacja klik ---")
    
    # --- 1. Znajdowanie kliki maksymalnej ---
    print("Szukanie kliki maksymalnej (może to potrwać)...")
  
    max_clique = max(nx.find_cliques(G), key=len)
    print(f"Znaleziono klikę maksymalną o rozmiarze: {len(max_clique)}")
    
    # Wizualizacja kliki maksymalnej
    plt.figure(figsize=(12, 12))
    plt.title(f"Klika maksymalna (rozmiar: {len(max_clique)})")
    
    # Tworzenie subgrafu tylko z kliką
    clique_subgraph = G.subgraph(max_clique)
    
    # Ustalenie pozycji wierzchołków
    pos = nx.spring_layout(G, seed=42)
    
    # Rysowanie całego grafu
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='gray', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4)
    
    # Wyróżnienie kliki
    nx.draw_networkx_nodes(clique_subgraph, pos, node_size=100, node_color='red')
    nx.draw_networkx_edges(clique_subgraph, pos, edge_color='red', width=1.5)
    nx.draw_networkx_labels(clique_subgraph, pos, font_size=8, font_color='white')
    
    plt.show()


    # --- 2. Znajdowanie przykładowej n-kliki ---
    print(f"\nSzukanie przykładowej kliki o rozmiarze co najmniej {n}...")
    try:
        # Znajdź pierwszą napotkaną klikę o rozmiarze >= n
        n_clique = next(c for c in nx.find_cliques(G) if len(c) == n)
        print(f"Znaleziono klikę o rozmiarze {len(n_clique)}: {n_clique}")

        # Wizualizacja n-kliki
        plt.figure(figsize=(12, 12))
        plt.title(f"Przykładowa klika o rozmiarze >= {n} (znaleziono: {len(n_clique)})")
        
        n_clique_subgraph = G.subgraph(n_clique)
        pos = nx.spring_layout(G, seed=42) # Użyj tego samego layoutu dla spójności
        
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='gray', alpha=0.6)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4)
        
        nx.draw_networkx_nodes(n_clique_subgraph, pos, node_size=100, node_color='blue')
        nx.draw_networkx_edges(n_clique_subgraph, pos, edge_color='blue', width=1.5)
        nx.draw_networkx_labels(n_clique_subgraph, pos, font_size=8, font_color='white')
        
        plt.show()

    except StopIteration:
        print(f"Nie znaleziono żadnej kliki o rozmiarze {n}.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")


def find_connectivity_properties(G):
    # --- Przeguby  ---
   
    articulation_points = list(nx.articulation_points(G))
    if articulation_points:
        print(f"Znaleziono {len(articulation_points)} przegubów. Oto kilka z nich: {articulation_points[:10]}")
    else:
        print("W grafie nie ma żadnych przegubów.")

    # --- Mosty (Bridges) ---

    bridges = list(nx.bridges(G))
    if bridges:
        print(f"Znaleziono {len(bridges)} mostów")
    else:
        print("W grafie nie ma żadnych mostów.")

    # --- k-spójność (k-connectivity) ---
    
    # k-spójność wierzchołkowa
    node_conn = nx.node_connectivity(G)
    print(f"k-spójność wierzchołkowa grafu wynosi: {node_conn}")
   
    # k-spójność krawędziowa
    edge_conn = nx.edge_connectivity(G)
    print(f"k-spójność krawędziowa grafu wynosi: {edge_conn}")
       
    
find_connectivity_properties(Network)

analyze_graph(Network)

clique_finder(Network, n=5)















