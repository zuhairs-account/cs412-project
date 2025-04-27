import matplotlib.pyplot as plt
import networkx as nx
import math

# Adjacency matrix
inf = math.inf
graph = [ 
    [inf,   2,   5,   4,   inf, inf, inf, inf],   # Node 0
    [inf, inf,   2,   inf,   7,   inf, inf,  inf],  # Node 1
    [inf, inf,inf,   inf,   inf, inf, inf, inf],   # Node 2
    [inf, inf, inf, inf,   inf,   4,   3, inf],  # Node 3
    [inf, inf, inf, inf, inf, inf, inf,   5],    # Node 4
    [inf, inf, inf, inf, inf, inf, inf,   7],    # Node 5
    [inf, inf, inf, inf, inf, inf, inf,   3],    # Node 6
    [inf, inf, inf, inf, inf, inf, inf, inf]     # Node 7
]

# Create the directed graph
G = nx.DiGraph()

# Add edges with weights
for i in range(len(graph)):
    for j in range(len(graph[i])):
        if graph[i][j] != inf:
            G.add_edge(i, j, weight=graph[i][j])

# Custom layout: Move 3 to middle, 4 to bottom left
pos = {
    0: (0, 3),
    1: (1, 3),
    2: (2, 3),
    3: (1.5, 2),  # Moved to middle
    4: (0, 0),    # Moved to bottom left
    5: (2.5, 1),
    6: (3.5, 1),
    7: (2.5, 0)
}

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="#a7d8de", edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

# Apply variable curvature to edges to avoid overlap
edge_styles = {
    (0, 2): 'arc3,rad=0.3',
    (0, 3): 'arc3,rad=0.15',
    (1, 2): 'arc3,rad=0.1',
    (1, 3): 'arc3,rad=0.2',
    (3, 5): 'arc3,rad=0.15',
    (3, 6): 'arc3,rad=-0.15',
    (5, 7): 'arc3,rad=0.2',
    (6, 7): 'arc3,rad=-0.2'
}
default_style = 'arc3,rad=0.2'

for u, v in G.edges():
    style = edge_styles.get((u, v), default_style)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                           connectionstyle=style, edge_color='gray',
                           arrowsize=20, width=2)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

# Final layout touches
plt.title("Directed Graph", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
