"""
Character relationship analysis in "The Witcher - The Tower of the Swallow"
Uses NetworkX for relationship graph and NLTK for text analysis
"""

import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import re
import os

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Analysis parameters
WINDOW_SIZE = 150  # words in sliding window
OVERLAP = 50       # word overlap between windows
MIN_OCCURRENCES = 3  # minimum co-occurrences to draw an edge

# Character definitions and name variants
CHARACTERS = {
    'Ciri': ['ciri', 'cirilla', 'zireael', 'falka', 'lwie źrebię', 'źrebię'],
    'Geralt': ['geralt', 'biały wilk', 'gwynbleidd', 'wiedźmin', 'białym wilkiem', 'białego wilka'],
    'Yennefer': ['yennefer', 'yenn', 'jennefer', 'yen'],
    'Jaskier': ['jaskier', 'dandelion'],
    'Regis': ['regis', 'emiel'],
    'Cahir': ['cahir'],
    'Bonhart': ['bonhart', 'leo bonhart'],
    'Skellen': ['skellen', 'stefan skellen', 'puszczyk'],
    'Vilgefortz': ['vilgefortz'],
    'Crach': ['crach', 'an craite'],
    'Avallach': ['avallach', "avallac'h", 'eredin'],
    'Dijkstra': ['dijkstra', 'sigi'],
    'Philippa': ['philippa', 'filippa', 'eilhart'],
    'Triss': ['triss', 'merigold'],
    "milva": ["milva", "milwą", "milvę", "milvy"]
}


def load_book(filepath):
    print("Loading book...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Remove headers, page numbers and unnecessary elements
    text = re.sub(r'Sapkowski Andrzej.*?Wieża Jaskółki', '', text)
    text = re.sub(r'/\* Lines.*?\*/', '', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    print(f"Book loaded: {len(text)} characters")
    return text


def split_into_windows(text):
    """Splits text into sliding windows of words"""
    words = word_tokenize(text.lower())
    windows = []
    
    i = 0
    while i < len(words):
        window = words[i:i + WINDOW_SIZE]
        if len(window) >= 20:  # Skip too short windows
            windows.append(' '.join(window))
        i += WINDOW_SIZE - OVERLAP
    
    print(f"Split into {len(windows)} windows (size: {WINDOW_SIZE} words, overlap: {OVERLAP})")
    return windows


def detect_character_in_text(text, character_variants):
    """Checks which characters appear in text"""
    found = set()
    text_lower = text.lower()
    
    for character, variants in character_variants.items():
        for variant in variants:
            # Search for variant as whole word (with boundaries)
            if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                found.add(character)
                break
    
    return found


def build_relationship_graph(windows):
    """Builds relationship graph based on co-occurrences in windows"""
    print("\nAnalyzing character co-occurrences...")
    
    # Counters for relationships and individual character occurrences
    relationships = defaultdict(int)
    occurrences = Counter()
    character_context = defaultdict(list)  # Words around each character
    
    for window in windows:
        characters_in_window = detect_character_in_text(window, CHARACTERS)
        
        # Count individual occurrences
        for character in characters_in_window:
            occurrences[character] += 1
            character_context[character].append(window)
        
        # Count pairs (relationships)
        character_list = list(characters_in_window)
        for i in range(len(character_list)):
            for j in range(i + 1, len(character_list)):
                pair = tuple(sorted([character_list[i], character_list[j]]))
                relationships[pair] += 1
    
    # Build graph
    G = nx.Graph()
    
    # Add nodes with weights (occurrence count)
    for character, count in occurrences.items():
        G.add_node(character, occurrences=count)
    
    # Add edges (relationships) with weights
    for (char1, char2), count in relationships.items():
        if count >= MIN_OCCURRENCES:  # Filter weak relationships
            G.add_edge(char1, char2, weight=count)
    
    print(f"Graph: {G.number_of_nodes()} characters, {G.number_of_edges()} relationships")
    return G, occurrences, character_context


def analyze_graph(G, occurrences):
    """Analyzes relationship graph structure"""
    print("\n" + "="*70)
    print("CHARACTER RELATIONSHIP GRAPH ANALYSIS")
    print("="*70)
    
    print(f"\nNumber of characters: {G.number_of_nodes()}")
    print(f"Number of relationships: {G.number_of_edges()}")
    
    if G.number_of_nodes() == 0:
        print("No data to analyze!")
        return
    
    # Most frequent characters
    print("\nMost frequent characters:")
    for i, (character, count) in enumerate(occurrences.most_common(7), 1):
        print(f"  {i}. {character}: {count} occurrences")
    
    # Strongest relationships
    if G.number_of_edges() > 0:
        print("\nStrongest relationships (frequent co-occurrences):")
        relationships_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        for i, (p1, p2, data) in enumerate(relationships_sorted[:10], 1):
            print(f"  {i}. {p1} ↔ {p2}: {data['weight']} times")
    
    # Centrality (who connects others)
    if G.number_of_nodes() > 1:
        degree_centrality = nx.degree_centrality(G)
        print("\nMost important characters (degree centrality):")
        for i, (character, cent) in enumerate(sorted(degree_centrality.items(), 
                                                   key=lambda x: x[1], reverse=True)[:5], 1):
            print(f"  {i}. {character}: {cent:.3f} (connections: {G.degree(character)})")
    
    # Communities
    if G.number_of_edges() > 0 and G.number_of_nodes() > 2:
        try:
            communities = list(nx.community.louvain_communities(G, seed=42))
            print(f"\nDetected {len(communities)} character groups:")
            for i, comm in enumerate(communities, 1):
                print(f"  Group {i}: {', '.join(sorted(comm))}")
        except:
            print("\nCannot detect communities (insufficient data)")


def analyze_context_nltk(character_context, top_n=5):    
    stop_words = {
        'i', 'w', 'na', 'z', 'to', 'się', 'że', 'nie', 'a', 'o', 'do', 'po', 'za', 'przez',
        'przy', 'od', 'dla', 'jak', 'jego', 'jej', 'go', 'mu', 'ale', 'czy', 'być', 'oraz',
        'ja', 'ty', 'on', 'ona', 'ono', 'my', 'wy', 'oni', 'one', 'mnie', 'tobie', 'sobie',
        'ciebie', 'cię', 'mi', 'ci', 'nas', 'wam', 'was', 'im', 'nim', 'jej', 'nich',
        'tak', 'nie', 'tam', 'tu', 'tutaj', 'teraz', 'wtedy', 'potem', 'więc', 'bardzo',
        'jeszcze', 'już', 'tylko', 'nawet', 'może', 'jednak', 'zawsze', 'nigdy', 'czasem',
        'też', 'także', 'również', 'więcej', 'mniej', 'dużo', 'trochę', 'całkiem',
        'ten', 'ta', 'to', 'tego', 'tej', 'tym', 'tych', 'tamten', 'tamta', 'tamto',
        'co', 'kto', 'gdzie', 'kiedy', 'dlaczego', 'jak', 'który', 'która', 'które',
        'był', 'była', 'było', 'były', 'jest', 'są', 'będzie', 'będą', 'został', 'była',
        'zostać', 'stać', 'mieć', 'może', 'móc', 'chcieć', 'wiedzieć', 'wiem', 'można',
        'powiedział', 'powiedziała', 'rzekł', 'rzekła', 'spytał', 'spytała', 'odrzekł',
        'mówił', 'mówiła', 'odpowiedział', 'odpowiedziała', 'zapytał', 'zapytała',
        'gdy', 'gdyż', 'jeśli', 'jeżeli', 'żeby', 'aby', 'choć', 'chociaż', 'bo', 'lecz',
        'ani', 'albo', 'lub', 'bądź', 'czy', 'jakby', 'niż', 'niby',
        'coś', 'ktoś', 'nic', 'nikt', 'każdy', 'wszystko', 'wszyscy', 'żaden', 'inny',
        'sam', 'sama', 'samo', 'swój', 'swoja', 'swoje', 'własny', 'taki', 'taka', 'takie'
    }
    
    for character in sorted(character_context.keys())[:top_n]:
        # Combine all windows with given character
        full_context = ' '.join(character_context[character])
        
        # Tokenization
        tokens = word_tokenize(full_context.lower())
        words = [w for w in tokens if w.isalnum() and len(w) > 2 and w not in stop_words]
        
        # Remove other character names from context
        for other_chars in CHARACTERS.values():
            words = [w for w in words if w not in [v.lower() for v in other_chars]]
        
        # Most frequent words
        freq = Counter(words)
        
        print(f"\n{character} - most frequent words in context:")
        for word, count in freq.most_common(8):
            print(f"  '{word}': {count}x")


def visualize_graph(G, occurrences):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Relationship graph
    ax1.set_title('Character relationship graph\n(thickness = co-occurrence frequency)', 
                  fontsize=13, fontweight='bold')
    
    # Layout
    if G.number_of_nodes() > 1:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = {list(G.nodes())[0]: (0, 0)}
    
    # Node sizes proportional to occurrence count
    node_sizes = [occurrences[node] * 10 for node in G.nodes()]
    
    # Draw edges with thickness proportional to weight
    if G.number_of_edges() > 0:
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        widths = [3 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, ax=ax1)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                          edgecolors='black', linewidths=2, ax=ax1)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax1)
    
    ax1.axis('off')
    
    # Plot 2: Bar chart of occurrences
    ax2.set_title('Character occurrence frequency', fontsize=13, fontweight='bold')
    
    characters_sorted = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)[:10]
    if characters_sorted:
        names = [p[0] for p in characters_sorted]
        counts = [p[1] for p in characters_sorted]
        
        bars = ax2.barh(range(len(names)), counts, color='steelblue', alpha=0.8)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Number of occurrences', fontsize=11)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax2.text(count + 5, i, str(count), va='center', fontweight='bold')
    
    plt.tight_layout()
    
    output_file = 'Sieci złożone/lista 5/witcher_relacje.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_file}")
    plt.close()


def conclusions(G, occurrences):        
    # Main character
    main_character = occurrences.most_common(1)[0][0] if occurrences else "None"
    
    # Most connected pair
    if G.number_of_edges() > 0:
        strongest = max(G.edges(data=True), key=lambda x: x[2]['weight'])
        pair = f"{strongest[0]} ↔ {strongest[1]} ({strongest[2]['weight']} times)"
    else:
        pair = "No significant relationships"
    
    print(f"""
1. MAIN CHARACTER: {main_character}
   - Most frequently appears in narrative

2. STRONGEST RELATIONSHIP: {pair}
   - These characters most often appear together

3. NARRATIVE STRUCTURE:
   - Detected {G.number_of_nodes()} active characters from list
   - Found {G.number_of_edges()} significant relationships (min. {MIN_OCCURRENCES} co-occurrences)
    """)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "38904454-wiedzmin-04-wieza-jaskolki.txt")
    text = load_book(filepath)
    
    # Split into windows
    windows = split_into_windows(text)
    
    # Build relationship graph
    G, occurrences, character_context = build_relationship_graph(windows)
    
    # Graph analysis
    analyze_graph(G, occurrences)
    
    # NLTK context analysis
    analyze_context_nltk(character_context, top_n=7)
    
    # Visualization
    visualize_graph(G, occurrences)
    
    # Conclusions
    conclusions(G, occurrences)
    


if __name__ == "__main__":
    main()
