import sys

def dijkstra_with_updates(vertices, edges, updates, source=0):
    # Apply updates to the edges and vertices
    for u, v, new_weight in updates:
        if new_weight == 0:
            vertices[u][v] = 0
            edges[u][v] = 0
        else:
            vertices[u][v] = 1
            edges[u][v] = new_weight

    num_of_vertices = len(vertices[0])
    visited_and_distance = [[0, 0]]  # source distance = 0

    for i in range(num_of_vertices - 1):
        visited_and_distance.append([0, sys.maxsize])

    def to_be_visited():
        v = -1
        for index in range(num_of_vertices):
            if visited_and_distance[index][0] == 0 and \
               (v < 0 or visited_and_distance[index][1] <= visited_and_distance[v][1]):
                v = index
        return v

    for vertex in range(num_of_vertices):
        to_visit = to_be_visited()
        for neighbor_index in range(num_of_vertices):
            if vertices[to_visit][neighbor_index] == 1 and \
               visited_and_distance[neighbor_index][0] == 0:
                new_distance = visited_and_distance[to_visit][1] + edges[to_visit][neighbor_index]
                if visited_and_distance[neighbor_index][1] > new_distance:
                    visited_and_distance[neighbor_index][1] = new_distance

        visited_and_distance[to_visit][0] = 1

    # Return distances in a list
    return [dist for visited, dist in visited_and_distance]


import numpy as np

def generate_adjacency_matrix(n_vertices, n_edges, directed=True, weighted=True, min_weight=1, max_weight=10):
    """
    Generates a random adjacency matrix for a graph with given vertices and edges.
    
    Args:
        n_vertices (int): Number of vertices (nodes)
        n_edges (int): Number of edges
        directed (bool): If True, generates a directed graph (default)
        weighted (bool): If True, assigns random weights (default)
        min_weight (int): Minimum edge weight (default=1)
        max_weight (int): Maximum edge weight (default=10)
        
    Returns:
        np.ndarray: Adjacency matrix of shape (n_vertices, n_vertices)
    """
    # Initialize adjacency matrix with zeros
    adj_matrix = np.zeros((n_vertices, n_vertices), dtype=int)
    
    # Maximum possible edges
    max_possible_edges = n_vertices * (n_vertices - 1) if directed else n_vertices * (n_vertices - 1) // 2
    
    # Ensure requested edges don't exceed maximum possible
    if n_edges > max_possible_edges:
        raise ValueError(f"Maximum possible edges for {n_vertices} vertices is {max_possible_edges}")
    
    # Generate edges
    edges_added = 0
    while edges_added < n_edges:
        u, v = np.random.randint(0, n_vertices, 2)  # Random vertices
        if u != v and adj_matrix[u][v] == 0:  # No self-loops or duplicate edges
            weight = np.random.randint(min_weight, max_weight + 1) if weighted else 1
            adj_matrix[u][v] = weight
            if not directed:  # If undirected, mirror the edge
                adj_matrix[v][u] = weight
            edges_added += 1
    
    return adj_matrix

import math

def convert_adj_matrix_to_vertices_edges(adj_matrix):
    n = len(adj_matrix)
    vertices = []
    edges = []

    for i in range(n):
        row_vertices = []
        row_edges = []
        for j in range(n):
            if adj_matrix[i][j] != 0 and not math.isinf(adj_matrix[i][j]):
                row_vertices.append(1)
                row_edges.append(adj_matrix[i][j])
            else:
                row_vertices.append(0)
                row_edges.append(0)
        vertices.append(row_vertices)
        edges.append(row_edges)
    
    return vertices, edges
import random

def generate_random_updates(adj_matrix, n):
    updates = []
    num_vertices = len(adj_matrix)

    for _ in range(n):
        # Randomly pick two distinct vertices
        u, v = random.sample(range(num_vertices), 2)
        
        # Ensure there is an edge between u and v (adjacency check)
        if adj_matrix[u][v] != float('inf'):
            # Choose randomly whether to increase or decrease the edge weight
            current_weight = adj_matrix[u][v]
            change = random.choice([-1, 1])  # Decrease (-1) or Increase (+1) the weight
            max_change = 10  # Adjust for a reasonable change range
            new_weight = current_weight + change * random.uniform(0, max_change)

            # Ensure weight is non-negative
            new_weight = max(0, new_weight)

            # Apply the update
            adj_matrix[u][v] = new_weight
            adj_matrix[v][u] = new_weight  # Assuming undirected graph, if directed, skip this line
            
            updates.append((u, v, new_weight))  # Save update
        else:
            # No edge between u and v, generate an edge with random weight
            new_weight = random.uniform(1, 10)  # Random weight between 1 and 10
            adj_matrix[u][v] = new_weight
            adj_matrix[v][u] = new_weight  # Assuming undirected graph

            updates.append((u, v, new_weight))

    return updates

numvertices = 5000
numedges = 500
numupdates = 200

adj = generate_adjacency_matrix(numvertices,numedges)
vertices, edges = convert_adj_matrix_to_vertices_edges(adj)
updates = generate_random_updates(adj,numupdates)


import time
starttime = time.time()
distances = dijkstra_with_updates(vertices, edges, updates)
endtime = time.time()
print(endtime-starttime)
# for i, d in enumerate(distances):
#     print(f"Distance from 'a' to {chr(ord('a') + i)}: {d}")

# vertices = [
#     [0, 1, 1, 1, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ]

# edges = [
#     [0, 2, 5, 4, 0, 0, 0, 0],
#     [0, 0, 2, 4, 7, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 4, 3, 0],
#     [0, 0, 0, 0, 0, 0, 0, 5],
#     [0, 0, 0, 0, 0, 0, 0, 7],
#     [0, 0, 0, 0, 0, 0, 0, 3],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ]

# updates = [
#     (0, 3, 100.0),  # Increased weight
#     (1, 4, 1.0),    # Decreased weight
#     (3, 5, 100.0),  # Increased weight
#     (6, 7, 100.0),  # Increased weight
#     (0, 2, 1.0),    # Decreased weight
#     (3, 6, 100.0),  # Increased weight
#     (1, 2, 0.5),    # Decreased weight to prioritize this path
#     (2, 3, 10.0),   # Increased weight to discourage this path
#     (4, 7, 2.0),    # Decreased weight to make this path more favorable
#     (5, 7, 20.0),   # Increased weight to make this path worse
#     (0, 1, 3.0),    # Adjusted weight slightly up
#     (1, 3, 10.0),   # Increased to change route preference
# ]
