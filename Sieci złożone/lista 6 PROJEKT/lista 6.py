import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# Wczytanie danych
df = pd.read_csv('C:\\Users\\RODO\\Desktop\\Semestr 5\\Sieci złożone\\lista 6 PROJEKT\\trending_hashtags.csv')

# Konwersja kolumny peak_date na datetime
df['peak_date'] = pd.to_datetime(df['peak_date'])

print(f"Liczba wpisów w CSV: {len(df)}")
print(f"Zakres dat: {df['peak_date'].min()} - {df['peak_date'].max()}")

# Tworzenie grafu
# WAŻNE: Każdy wpis (tag + data) jest osobnym wierzchołkiem!
# Identyfikator wierzchołka: "tag_YYYY-MM-DD"
G = nx.Graph()

# Tworzenie unikalnych identyfikatorów dla każdego wpisu
df['node_id'] = df['tag'] + '_' + df['peak_date'].dt.strftime('%Y-%m-%d')

print(f"Unikalnych wierzchołków: {df['node_id'].nunique()}")

# Dodanie wszystkich hashtagów jako wierzchołków
for _, row in df.iterrows():
    G.add_node(row['node_id'], 
               tag=row['tag'],  # oryginalny tag
               year=row['year'], 
               peak_date=str(row['peak_date'].date()),
               tweets=row['tweets'], 
               rank=row['rank'])

# Dodanie krawędzi - łączymy hashtagi, których szczyty były w odległości <= 14 dni
max_days = 14  # 2 tygodnie

print("\nTworzenie krawędzi grafu (wektoryzacja numpy)...")
start_time = time.time()

# Przygotowanie danych do wektoryzacji
node_ids = df['node_id'].values
# Konwersja dat na liczby dni od epoch dla szybkich obliczeń
dates_numeric = (df['peak_date'] - pd.Timestamp('1970-01-01')).dt.days.values

n = len(node_ids)
print(f"Liczba par do sprawdzenia: {n * (n-1) // 2:,}")

# Wektoryzowane obliczenie wszystkich krawędzi
chunk_size = 2000
edges = []

num_chunks = (n + chunk_size - 1) // chunk_size
total_chunks = num_chunks * (num_chunks + 1) // 2

print(f"Przetwarzanie w {total_chunks} chunkach...")

chunk_num = 0
for i_start in range(0, n, chunk_size):
    i_end = min(i_start + chunk_size, n)
    dates_i = dates_numeric[i_start:i_end]
    
    for j_start in range(i_start, n, chunk_size):
        j_end = min(j_start + chunk_size, n)
        dates_j = dates_numeric[j_start:j_end]
        
        # Macierz różnic dat (broadcasting)
        diff_matrix = np.abs(dates_i[:, np.newaxis] - dates_j[np.newaxis, :])
        
        # Znajdź pary spełniające warunek
        mask = diff_matrix <= max_days
        
        # Dla chunków na przekątnej - tylko górny trójkąt
        if i_start == j_start:
            mask = np.triu(mask, k=1)
        
        # Indeksy par spełniających warunek
        local_i, local_j = np.where(mask)
        
        # Konwersja na globalne indeksy i dodanie krawędzi
        for li, lj in zip(local_i, local_j):
            edges.append((node_ids[i_start + li], node_ids[j_start + lj]))
        
        chunk_num += 1
        if chunk_num % 100 == 0:
            print(f"  Postęp: {chunk_num}/{total_chunks} chunków ({100*chunk_num/total_chunks:.1f}%)")

# Dodanie wszystkich krawędzi do grafu
G.add_edges_from(edges)

elapsed = time.time() - start_time
print(f"Czas tworzenia krawędzi: {elapsed:.2f}s")

print(f"\nStatystyki grafu:")
print(f"Liczba wierzchołków: {G.number_of_nodes()}")
print(f"Liczba krawędzi: {G.number_of_edges()}")
print(f"Gęstość grafu: {nx.density(G):.4f}")

# Podstawowe metryki
if G.number_of_edges() > 0:
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    max_degree_node = max(degrees, key=degrees.get)
    
    print(f"Średni stopień wierzchołka: {avg_degree:.2f}")
    print(f"Wierzchołek o najwyższym stopniu: {max_degree_node} (stopień: {degrees[max_degree_node]})")
    
    # Liczba spójnych składowych
    num_components = nx.number_connected_components(G)
    print(f"Liczba spójnych składowych: {num_components}")
    
    # Największa spójna składowa
    largest_cc = max(nx.connected_components(G), key=len)
    print(f"Rozmiar największej spójnej składowej: {len(largest_cc)}")

# Zapisanie grafu do pliku
nx.write_gexf(G, 'hashtags_graph.gexf')
print("\nGraf zapisano do pliku 'hashtags_graph.gexf'")

# =============================================================================
# ZAAWANSOWANA ANALIZA GRAFU
# =============================================================================
print("\n" + "="*70)
print("ZAAWANSOWANA ANALIZA GRAFU HASHTAGÓW")
print("="*70)

# Pracujemy na największej spójnej składowej dla metryk wymagających spójności
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
print(f"\nAnaliza na największej spójnej składowej: {G_lcc.number_of_nodes()} wierzchołków")

# -----------------------------------------------------------------------------
# 1. ROZKŁAD STOPNI
# -----------------------------------------------------------------------------
print("\n" + "-"*50)
print("1. ROZKŁAD STOPNI WIERZCHOŁKÓW")
print("-"*50)

degrees = dict(G.degree())
degree_values = list(degrees.values())

print(f"Minimalny stopień: {min(degree_values)}")
print(f"Maksymalny stopień: {max(degree_values)}")
print(f"Średni stopień: {np.mean(degree_values):.2f}")
print(f"Mediana stopnia: {np.median(degree_values):.2f}")
print(f"Odchylenie standardowe: {np.std(degree_values):.2f}")

# Top 10 wierzchołków wg stopnia
sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 hashtagów wg stopnia (liczby połączeń):")
for i, (tag, deg) in enumerate(sorted_degrees, 1):
    print(f"  {i}. {tag}: {deg}")

# Wizualizacja rozkładu stopni
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(degree_values, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Stopień wierzchołka')
axes[0].set_ylabel('Liczba wierzchołków')
axes[0].set_title('Rozkład stopni wierzchołków')
axes[0].axvline(np.mean(degree_values), color='red', linestyle='--', label=f'Średnia: {np.mean(degree_values):.1f}')
axes[0].legend()

# Log-log plot (sprawdzenie prawa potęgowego)
degree_count = {}
for d in degree_values:
    degree_count[d] = degree_count.get(d, 0) + 1
degrees_unique = sorted(degree_count.keys())
counts = [degree_count[d] for d in degrees_unique]

axes[1].loglog(degrees_unique, counts, 'bo', markersize=3, alpha=0.6)
axes[1].set_xlabel('Stopień (log)')
axes[1].set_ylabel('Liczba wierzchołków (log)')
axes[1].set_title('Rozkład stopni (skala log-log)')

plt.tight_layout()
plt.savefig('rozklad_stopni.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 2. MIARY CENTRALNOŚCI
# -----------------------------------------------------------------------------
print("\n" + "-"*50)
print("2. MIARY CENTRALNOŚCI")
print("-"*50)

# 2a. Degree Centrality
print("\n2a. Degree Centrality (centralność stopnia)")
degree_centrality = nx.degree_centrality(G)
top_degree_cent = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10:")
for i, (tag, cent) in enumerate(top_degree_cent, 1):
    print(f"  {i}. {tag}: {cent:.4f}")

# 2b. Betweenness Centrality (pełne obliczenie - wielowątkowe)
print("\n2b. Betweenness Centrality (centralność pośrednictwa) - PEŁNE OBLICZENIE")
print("    To może potrwać kilka minut...")
start_bc = time.time()
# Używamy wszystkich rdzeni CPU
betweenness = nx.betweenness_centrality(G_lcc, k=None, normalized=True)
print(f"    Czas: {time.time() - start_bc:.1f}s")
top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10:")
for i, (tag, cent) in enumerate(top_betweenness, 1):
    print(f"  {i}. {tag}: {cent:.6f}")

# 2c. Closeness Centrality
print("\n2c. Closeness Centrality (centralność bliskości)")
closeness = nx.closeness_centrality(G_lcc)
top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10:")
for i, (tag, cent) in enumerate(top_closeness, 1):
    print(f"  {i}. {tag}: {cent:.4f}")

# 2d. Eigenvector Centrality
print("\n2d. Eigenvector Centrality (centralność wektora własnego)")
try:
    eigenvector = nx.eigenvector_centrality(G_lcc, max_iter=1000)
    top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10:")
    for i, (tag, cent) in enumerate(top_eigenvector, 1):
        print(f"  {i}. {tag}: {cent:.4f}")
except:
    print("  Nie udało się obliczyć (graf może być zbyt rzadki)")

# 2e. PageRank
print("\n2e. PageRank")
pagerank = nx.pagerank(G_lcc, alpha=0.85)
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10:")
for i, (tag, cent) in enumerate(top_pagerank, 1):
    print(f"  {i}. {tag}: {cent:.6f}")

# Wizualizacja porównania centralności
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top 15 dla każdej miary
n_top = 15

# Degree
top_deg = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:n_top]
axes[0, 0].barh([x[0] for x in top_deg][::-1], [x[1] for x in top_deg][::-1])
axes[0, 0].set_title('Degree Centrality (Top 15)')
axes[0, 0].set_xlabel('Centralność')

# Betweenness
top_bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:n_top]
axes[0, 1].barh([x[0] for x in top_bet][::-1], [x[1] for x in top_bet][::-1])
axes[0, 1].set_title('Betweenness Centrality (Top 15)')
axes[0, 1].set_xlabel('Centralność')

# Closeness
top_clo = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:n_top]
axes[1, 0].barh([x[0] for x in top_clo][::-1], [x[1] for x in top_clo][::-1])
axes[1, 0].set_title('Closeness Centrality (Top 15)')
axes[1, 0].set_xlabel('Centralność')

# PageRank
top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:n_top]
axes[1, 1].barh([x[0] for x in top_pr][::-1], [x[1] for x in top_pr][::-1])
axes[1, 1].set_title('PageRank (Top 15)')
axes[1, 1].set_xlabel('Centralność')

plt.tight_layout()
plt.savefig('centralnosci.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 3. KLASTERYZACJA I TRÓJKĄTY
# -----------------------------------------------------------------------------
print("\n" + "-"*50)
print("3. KLASTERYZACJA I TRÓJKĄTY")
print("-"*50)

# Współczynnik klasteryzacji
avg_clustering = nx.average_clustering(G)
print(f"Średni współczynnik klasteryzacji: {avg_clustering:.4f}")

# Globalny współczynnik klasteryzacji (tranzytywność)
transitivity = nx.transitivity(G)
print(f"Tranzytywność (globalny wsp. klasteryzacji): {transitivity:.4f}")

# Liczba trójkątów
triangles = nx.triangles(G)
total_triangles = sum(triangles.values()) // 3
print(f"Liczba trójkątów w grafie: {total_triangles:,}")

# Top 10 wierzchołków z największą liczbą trójkątów
top_triangles = sorted(triangles.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 hashtagów wg liczby trójkątów:")
for i, (tag, tri) in enumerate(top_triangles, 1):
    print(f"  {i}. {tag}: {tri}")

# Rozkład współczynnika klasteryzacji
clustering_coeffs = nx.clustering(G)
clustering_values = list(clustering_coeffs.values())

plt.figure(figsize=(10, 5))
plt.hist(clustering_values, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Współczynnik klasteryzacji')
plt.ylabel('Liczba wierzchołków')
plt.title('Rozkład współczynnika klasteryzacji')
plt.axvline(avg_clustering, color='red', linestyle='--', label=f'Średnia: {avg_clustering:.3f}')
plt.legend()
plt.tight_layout()
plt.savefig('klasteryzacja.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 4. KLIKI
# -----------------------------------------------------------------------------
print("\n" + "-"*50)
print("4. KLIKI (PODGRAFY PEŁNE)")
print("-"*50)

# Znajdowanie klik maksymalnych (może być czasochłonne)
print("Szukanie klik maksymalnych...")
start_time = time.time()

# Dla wydajności szukamy tylko większych klik
all_cliques = list(nx.find_cliques(G))
clique_sizes = [len(c) for c in all_cliques]

print(f"Czas: {time.time() - start_time:.2f}s")
print(f"Liczba klik maksymalnych: {len(all_cliques):,}")
print(f"Rozmiar najmniejszej kliki: {min(clique_sizes)}")
print(f"Rozmiar największej kliki: {max(clique_sizes)}")
print(f"Średni rozmiar kliki: {np.mean(clique_sizes):.2f}")

# Rozkład rozmiarów klik
size_distribution = {}
for s in clique_sizes:
    size_distribution[s] = size_distribution.get(s, 0) + 1

print("\nRozkład rozmiarów klik:")
for size in sorted(size_distribution.keys(), reverse=True)[:15]:
    print(f"  Rozmiar {size}: {size_distribution[size]} klik")

# Największe kliki
largest_cliques = sorted(all_cliques, key=len, reverse=True)[:5]
print("\nNajwiększe kliki (top 5):")
for i, clique in enumerate(largest_cliques, 1):
    print(f"  {i}. Rozmiar {len(clique)}: {clique[:10]}{'...' if len(clique) > 10 else ''}")

# Wizualizacja rozkładu klik
plt.figure(figsize=(10, 5))
sizes = sorted(size_distribution.keys())
counts = [size_distribution[s] for s in sizes]
plt.bar(sizes, counts, edgecolor='black', alpha=0.7)
plt.xlabel('Rozmiar kliki')
plt.ylabel('Liczba klik')
plt.title('Rozkład rozmiarów klik maksymalnych')
plt.yscale('log')
plt.tight_layout()
plt.savefig('kliki.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 5. ŚCIEŻKI I ODLEGŁOŚCI - PEŁNE OBLICZENIE
# -----------------------------------------------------------------------------
print("\n" + "-"*50)
print("5. ŚCIEŻKI I ODLEGŁOŚCI (PEŁNE OBLICZENIE)")
print("-"*50)

from concurrent.futures import ThreadPoolExecutor
import os

num_cores = os.cpu_count()
print(f"Dostępne rdzenie CPU: {num_cores}")

# Funkcja do obliczenia ścieżek z jednego wierzchołka
def compute_paths_from_node(node):
    lengths = nx.single_source_shortest_path_length(G_lcc, node)
    return list(lengths.values())

# PEŁNA średnia długość najkrótszej ścieżki
print("\nObliczanie DOKŁADNEJ średniej długości ścieżki...")
print("(Obliczanie dla wszystkich par wierzchołków - to może potrwać)")
start_path = time.time()

all_path_lengths = []
nodes_list = list(G_lcc.nodes())
total_nodes = len(nodes_list)

# Wielowątkowe obliczanie
with ThreadPoolExecutor(max_workers=num_cores) as executor:
    futures = {executor.submit(compute_paths_from_node, node): i for i, node in enumerate(nodes_list)}
    completed = 0
    for future in futures:
        result = future.result()
        all_path_lengths.extend(result)
        completed += 1
        if completed % 500 == 0 or completed == total_nodes:
            elapsed = time.time() - start_path
            eta = (elapsed / completed) * (total_nodes - completed) if completed > 0 else 0
            print(f"  Postęp: {completed}/{total_nodes} ({100*completed/total_nodes:.1f}%) - ETA: {eta:.0f}s")

# Usunięcie zer (odległość do samego siebie)
all_path_lengths = [x for x in all_path_lengths if x > 0]

avg_path_length = np.mean(all_path_lengths)
print(f"\nDOKŁADNA średnia długość najkrótszej ścieżki: {avg_path_length:.6f}")
print(f"Czas obliczenia: {time.time() - start_path:.1f}s")

# DOKŁADNA średnica grafu
print("\nObliczanie DOKŁADNEJ średnicy grafu...")
diameter = max(all_path_lengths)
print(f"DOKŁADNA średnica grafu: {diameter}")

# Statystyki rozkładu
print(f"\nStatystyki długości ścieżek:")
print(f"  Min: {min(all_path_lengths)}")
print(f"  Max (średnica): {diameter}")
print(f"  Średnia: {avg_path_length:.4f}")
print(f"  Mediana: {np.median(all_path_lengths):.4f}")
print(f"  Odchylenie std: {np.std(all_path_lengths):.4f}")

# Rozkład długości ścieżek
path_distribution = {}
for p in all_path_lengths:
    path_distribution[p] = path_distribution.get(p, 0) + 1

print("\nRozkład długości ścieżek:")
for length in sorted(path_distribution.keys()):
    count = path_distribution[length]
    percentage = 100 * count / len(all_path_lengths)
    print(f"  Długość {length}: {count:,} par ({percentage:.2f}%)")

# Wizualizacja
plt.figure(figsize=(10, 5))
plt.hist(all_path_lengths, bins=range(int(max(all_path_lengths))+2), edgecolor='black', alpha=0.7)
plt.xlabel('Długość najkrótszej ścieżki')
plt.ylabel('Liczba par wierzchołków')
plt.title(f'Rozkład długości najkrótszych ścieżek (średnia: {avg_path_length:.3f}, średnica: {diameter})')
plt.axvline(avg_path_length, color='red', linestyle='--', label=f'Średnia: {avg_path_length:.3f}')
plt.legend()
plt.tight_layout()
plt.savefig('sciezki.png', dpi=150)
plt.show()

# Dodatkowa metryka: promień grafu
radius = min(max(nx.single_source_shortest_path_length(G_lcc, node).values()) for node in G_lcc.nodes())
print(f"\nPromień grafu: {radius}")

# -----------------------------------------------------------------------------
# 6. ASSORTATIVITY (KORELACJA STOPNI)
# -----------------------------------------------------------------------------
print("\n" + "-"*50)
print("6. ASSORTATIVITY")
print("-"*50)

assortativity = nx.degree_assortativity_coefficient(G)
print(f"Współczynnik assortatywności stopni: {assortativity:.4f}")
if assortativity > 0:
    print("  -> Graf jest assortatywny (węzły o wysokim stopniu łączą się z węzłami o wysokim stopniu)")
else:
    print("  -> Graf jest disassortatywny (węzły o wysokim stopniu łączą się z węzłami o niskim stopniu)")

# -----------------------------------------------------------------------------
# 7. WYKRYWANIE SPOŁECZNOŚCI
# -----------------------------------------------------------------------------
print("\n" + "-"*50)
print("7. WYKRYWANIE SPOŁECZNOŚCI (Louvain)")
print("-"*50)

try:
    from community import community_louvain
    
    partition = community_louvain.best_partition(G)
    num_communities = max(partition.values()) + 1
    print(f"Liczba wykrytych społeczności: {num_communities}")
    
    # Rozmiary społeczności
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 największych społeczności:")
    for i, (comm_id, size) in enumerate(sorted_communities[:10], 1):
        # Znajdź przykładowe hashtagi z tej społeczności
        members = [n for n, c in partition.items() if c == comm_id][:5]
        print(f"  {i}. Społeczność {comm_id}: {size} członków")
        print(f"     Przykłady: {members}")
    
    # Modularność
    modularity = community_louvain.modularity(partition, G)
    print(f"\nModularność: {modularity:.4f}")
    
    # Wizualizacja rozmiarów społeczności
    plt.figure(figsize=(12, 5))
    sizes_sorted = [x[1] for x in sorted_communities]
    plt.bar(range(len(sizes_sorted)), sizes_sorted, edgecolor='black', alpha=0.7)
    plt.xlabel('ID społeczności (posortowane wg rozmiaru)')
    plt.ylabel('Liczba członków')
    plt.title(f'Rozmiary społeczności (Louvain, modularność: {modularity:.3f})')
    plt.tight_layout()
    plt.savefig('spolecznosci.png', dpi=150)
    plt.show()
    
except ImportError:
    print("Biblioteka python-louvain nie jest zainstalowana.")
    print("Zainstaluj: pip install python-louvain")

# -----------------------------------------------------------------------------
# 8. PODSUMOWANIE
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("PODSUMOWANIE ANALIZY")
print("="*70)
print(f"""
PODSTAWOWE STATYSTYKI:
  - Liczba wierzchołków: {G.number_of_nodes():,}
  - Liczba krawędzi: {G.number_of_edges():,}
  - Gęstość: {nx.density(G):.4f}
  - Liczba spójnych składowych: {nx.number_connected_components(G)}

STOPNIE:
  - Średni stopień: {np.mean(degree_values):.2f}
  - Max stopień: {max(degree_values)} ({sorted_degrees[0][0]})

KLASTERYZACJA:
  - Średni wsp. klasteryzacji: {avg_clustering:.4f}
  - Tranzytywność: {transitivity:.4f}
  - Liczba trójkątów: {total_triangles:,}

KLIKI:
  - Liczba klik maksymalnych: {len(all_cliques):,}
  - Największa klika: {max(clique_sizes)} wierzchołków

ŚCIEŻKI (DOKŁADNE WARTOŚCI):
  - Średnia długość ścieżki: {avg_path_length:.6f}
  - Średnica: {diameter}
  - Promień: {radius}

ASSORTATIVITY:
  - Współczynnik: {assortativity:.4f}
""")

print("\nWszystkie wykresy zapisano do plików PNG.")

# Wizualizacja top 50 hashtagów
print("\n" + "-"*50)
print("WIZUALIZACJA TOP 50 HASHTAGÓW")
print("-"*50)

top_node_ids = df.nsmallest(50, 'rank')['node_id'].tolist()
G_sub = G.subgraph(top_node_ids)

# Etykiety - tylko nazwa tagu (bez daty)
labels = {node: G.nodes[node]['tag'] for node in G_sub.nodes()}

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G_sub, k=2, iterations=50)
nx.draw(G_sub, pos, 
        node_color='lightblue', 
        node_size=500,
        labels=labels,
        font_size=8,
        edge_color='gray',
        alpha=0.7)
plt.title('Graf hashtagów (top 50) - krawędź = szczyt w odległości ≤ 2 tygodni')
plt.tight_layout()
plt.savefig('hashtags_graph_top50.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAnaliza zakończona!")
