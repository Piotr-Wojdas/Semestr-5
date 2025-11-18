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

"""
własności do implementacji:
klika/klika maksymalna/n-podklika
spójne składowe
macierz sąsiedztwa/incydencji





"""















