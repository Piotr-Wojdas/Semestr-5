import pandas as pd
import networkx as nx

df = pd.read_csv('trending_hashtags.csv')
df['peak_date'] = pd.to_datetime(df['peak_date'])

G = nx.Graph()
for _, row in df.iterrows():
    G.add_node(row['tag'], year=row['year'], peak_date=row['peak_date'])

# Ile mamy wierzchołków?
print('Wierzchołków w grafie:', G.number_of_nodes())
print('Wierszy w CSV:', len(df))
print()

# Co jest w Father's Day?
tag = "Father's Day"
fd = G.nodes.get(tag)
print(f"{tag} w grafie: {fd}")

# Ale w CSV mamy wiele rekordów!
print()
print("Father's Day w CSV:")
print(df[df['tag'] == tag][['tag','year','peak_date']])

# PROBLEM: Krawędzie są tworzone na podstawie WSZYSTKICH wierszy CSV,
# ale wierzchołki są nadpisywane!
print()
print('='*60)
print('PROBLEM: Jak tworzone są krawędzie?')
print('='*60)

# Sprawdźmy jakie krawędzie ma Father's Day
tags = df['tag'].values
dates_numeric = (df['peak_date'] - pd.Timestamp('1970-01-01')).dt.days.values

# Znajdź wszystkie indeksy Father's Day
fd_indices = [i for i, t in enumerate(tags) if t == tag]
print(f"\nIndeksy Father's Day w CSV: {fd_indices}")
print("Daty:")
for i in fd_indices:
    print(f"  Index {i}: {df.iloc[i]['peak_date'].date()} (rok {df.iloc[i]['year']})")

# Jakie krawędzie tworzy każdy z tych indeksów?
print("\nKrawędzie tworzone przez każdy wpis Father's Day:")
for idx in fd_indices:
    date_fd = dates_numeric[idx]
    neighbors = []
    for j in range(len(tags)):
        if idx != j and abs(date_fd - dates_numeric[j]) <= 14:
            neighbors.append((tags[j], df.iloc[j]['year'], df.iloc[j]['peak_date'].date()))
    print(f"\n  Index {idx} ({df.iloc[idx]['peak_date'].date()}, rok {df.iloc[idx]['year']}):")
    print(f"    Liczba sąsiadów: {len(neighbors)}")
    # Pokaż kilku sąsiadów z różnych lat
    years_sample = {}
    for n, y, d in neighbors:
        if y not in years_sample:
            years_sample[y] = (n, d)
    for y in sorted(years_sample.keys()):
        n, d = years_sample[y]
        print(f"    Rok {y}: np. {n} ({d})")
