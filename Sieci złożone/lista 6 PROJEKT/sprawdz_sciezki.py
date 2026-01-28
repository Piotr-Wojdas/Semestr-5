import pandas as pd
import networkx as nx
import numpy as np
import random

# Wczytanie danych i budowa grafu
df = pd.read_csv('trending_hashtags.csv')
df['peak_date'] = pd.to_datetime(df['peak_date'])

G = nx.Graph()
for _, row in df.iterrows():
    G.add_node(row['tag'], year=row['year'], peak_date=row['peak_date'])

tags = df['tag'].values
dates_numeric = (df['peak_date'] - pd.Timestamp('1970-01-01')).dt.days.values
n = len(tags)

# Budowa krawędzi
for i in range(n):
    for j in range(i+1, n):
        if abs(dates_numeric[i] - dates_numeric[j]) <= 14:
            G.add_edge(tags[i], tags[j])

# Znajdź hashtagi z 2020 i 2025
tags_2020 = [n for n, d in G.nodes(data=True) if d.get('year') == 2020]
tags_2025 = [n for n, d in G.nodes(data=True) if d.get('year') == 2025]

print(f'Hashtagów z 2020: {len(tags_2020)}')
print(f'Hashtagów z 2025: {len(tags_2025)}')

# Znajdź przykładową najkrótszą ścieżkę 2020 -> 2025
random.seed(42)

# Znajdź pary o różnych odległościach
found_paths = {}
for _ in range(200):
    t2020 = random.choice(tags_2020)
    t2025 = random.choice(tags_2025)
    try:
        path = nx.shortest_path(G, t2020, t2025)
        length = len(path) - 1
        if length not in found_paths:
            found_paths[length] = (t2020, t2025, path)
    except:
        pass

print('\n' + '='*60)
print('PRZYKŁADOWE ŚCIEŻKI 2020 -> 2025:')
print('='*60)

for length in sorted(found_paths.keys()):
    t2020, t2025, path = found_paths[length]
    d2020 = G.nodes[t2020]['peak_date'].date()
    d2025 = G.nodes[t2025]['peak_date'].date()
    print(f'\nŚcieżka długości {length}:')
    print(f'  Start: {t2020} ({d2020})')
    print(f'  Koniec: {t2025} ({d2025})')
    print(f'  Pełna ścieżka:')
    for i, node in enumerate(path):
        date = G.nodes[node]['peak_date'].date()
        year = G.nodes[node]['year']
        print(f'    {i}. {node} ({date}, rok {year})')

# Sprawdź minimalną i maksymalną odległość między 2020 a 2025
print('\n' + '='*60)
print('STATYSTYKI ODLEGŁOŚCI 2020 <-> 2025:')
print('='*60)

distances = []
sample_2020 = random.sample(tags_2020, min(100, len(tags_2020)))
sample_2025 = random.sample(tags_2025, min(100, len(tags_2025)))

for t20 in sample_2020:
    for t25 in sample_2025:
        try:
            d = nx.shortest_path_length(G, t20, t25)
            distances.append(d)
        except:
            pass

print(f'Próbka: {len(distances)} par')
print(f'Min odległość: {min(distances)}')
print(f'Max odległość: {max(distances)}')
print(f'Średnia odległość: {np.mean(distances):.2f}')

# Rozkład
from collections import Counter
dist_count = Counter(distances)
print('\nRozkład odległości:')
for d in sorted(dist_count.keys()):
    print(f'  Odległość {d}: {dist_count[d]} par ({100*dist_count[d]/len(distances):.1f}%)')

# Znajdź ścieżkę o długości 4 (jeśli istnieje między 2020 a 2025)
print('\n' + '='*60)
print('SZUKANIE NAJDŁUŻSZEJ ŚCIEŻKI (długość 4):')
print('='*60)

for t20 in tags_2020:
    for t25 in tags_2025:
        try:
            if nx.shortest_path_length(G, t20, t25) == 4:
                path = nx.shortest_path(G, t20, t25)
                print(f'\nZnaleziono ścieżkę długości 4!')
                for i, node in enumerate(path):
                    date = G.nodes[node]['peak_date'].date()
                    year = G.nodes[node]['year']
                    print(f'  {i}. {node} ({date}, rok {year})')
                break
        except:
            pass
    else:
        continue
    break
