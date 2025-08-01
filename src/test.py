import math
import random
import time
import heapq
from dynamic import DynamicDijkstra, generate_adjacency_matrix, generate_random_updates

# Standard Dijkstra's Algorithm with priority queue
def dijkstra(graph, source):
    num_vertices = len(graph)
    dist = [math.inf] * num_vertices  # Distance from source to each vertex
    pred = [None] * num_vertices
    dist[source] = 0
    pq = [(0, source)]     #priority queue 
    visited = set()
    # print(num_vertices)
    
    while pq:
        current_dist, u = heapq.heappop(pq)       #get vertices with smallest dist
        if u in visited:
            continue
        visited.add(u)
        for v in range(num_vertices):
            if graph[u][v] != math.inf:
                alt = current_dist + graph[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    pred[v] = u
                    heapq.heappush(pq, (alt, v))
    return dist, pred

# dijkstra without pq - slower 
def dijkstra_without_pq(graph, source):
    num_vertices = len(graph)
    dist = [math.inf] * num_vertices
    pred = [None] * num_vertices
    dist[source] = 0
    array_ = [(0, source)]       #array instead of pq
    visited = set()
    
    while len(array_)>0:
        # min_index = 0       #actually tarversing array to find min
        min_value = math.inf
        # print(array_[0])
        for i in range(0, len(array_)):
            if array_[i][0] < min_value:
                min_value = array_[i][0]
                min_index = i

        current_dist, u = array_.pop(min_index)
        if u in visited:
            continue
        visited.add(u)
        for v in range(num_vertices):
            if graph[u][v] != math.inf:
                alt = current_dist + graph[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    pred[v] = u
                    array_.append((alt, v))
    return dist, pred


# Bellman-Ford Algorithm - not optimized version
def bellend_ford(graph, source):
    num_vertices = len(graph)
    dist = [math.inf] * num_vertices
    pred = [None] * num_vertices
    dist[source] = 0
    # print(num_vertices)
    
    for _ in range(num_vertices - 1):
        for u in range(num_vertices):
            for v in range(num_vertices):
                if graph[u][v] != math.inf:
                    if dist[u] + graph[u][v] < dist[v]:
                        dist[v] = dist[u] + graph[u][v]
                        pred[v] = u
    
    # Check for negative cycles (though not expected in this context)
    for u in range(num_vertices):
        for v in range(num_vertices):
            if graph[u][v] != math.inf and dist[u] + graph[u][v] < dist[v]:
                raise ValueError("Graph contains a negative cycle")
    
    return dist, pred



# bellman fors optimized - early termination 
def bellman_ford(graph, source):
    num_vertices = len(graph)
    dist = [math.inf] * num_vertices
    pred = [None] * num_vertices
    dist[source] = 0
    
    # Create edge list from adjacency matrix
    edges = []
    for u in range(num_vertices):
        for v in range(num_vertices):
            if graph[u][v] != math.inf and u != v:
                edges.append((u, v, graph[u][v]))
    
    # Optimized Bellman-Ford with early termination
    for _ in range(num_vertices - 1):
        changed = False
        for u, v, weight in edges:
            if dist[u] != math.inf and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                pred[v] = u
                changed = True
        if not changed:
            break  # Early termination if no updates
    
    return dist, pred

# main comparison func - dijkstra vs bellman ford 
def compare_algorithms():
    # Parameters matching the original code

    # small
    numvertices = 200
    numedges = 500
    numupdates = 200

    # med

    numvertices = 5000
    numedges = 5000
    numupdates = 50

    # big
    
    # numvertices = 20000
    # numedges = 2000
    # numupdates = 200

    # Generate initial graph
    adj = generate_adjacency_matrix(numvertices, numedges)
    
    # Generate updates without modifying the original graph
    # Note: We redefine generate_random_updates to not modify adj in place
    def generate_random_updates_no_modify(adj_matrix, n):
        updates = []
        num_vertices = len(adj_matrix)
        i=0
        while i<n:
            u, v = random.sample(range(num_vertices), 2)
            if adj_matrix[u][v] == math.inf:
                continue
            if adj_matrix[u][v] != 0:
                current_weight = adj_matrix[u][v]
                change = random.choice([-1, 1])
                max_change = 10
                new_weight = current_weight + change * random.uniform(0, max_change)
                new_weight = max(0, new_weight)
                updates.append((u, v, new_weight))
            else:
                new_weight = random.uniform(1, 10)
                updates.append((u, v, new_weight))
            i+=1
        return updates
    
    updates = generate_random_updates_no_modify(adj, numupdates)
    source_node = 0
    
    # --- Dynamic Algorithm (Original) ---
    dd = DynamicDijkstra([row[:] for row in adj])
    dd.initial_dijkstra(source_node)
    starttime_dynamic = time.time()
    for u, v, weight in updates:
        dd.update_edge(u, v, weight)
    endtime_dynamic = time.time()
    time_dynamic = endtime_dynamic - starttime_dynamic
    print(f"Dynamic Algorithm Update Time: {time_dynamic:.4f} seconds")
    # --- Standard Dijkstra's ---
    graph_dijkstra = [row[:] for row in adj]
    dist_dijkstra, pred_dijkstra = dijkstra(graph_dijkstra, source_node)
    time_dijkstra = 0
    # time_dijkstra_without_pq = 0
    # print(len(updates))
    for i, (u, v, weight) in enumerate(updates):
        # print(i)
        graph_dijkstra[u][v] = weight  # Directed graph, so only u->v
        start = time.time()
        # print("her")
        dist_dijkstra, pred_dijkstra = dijkstra(graph_dijkstra, source_node)
        end = time.time()
        # start_wopq = time.time()
        # # print("her")
        # dist_dijkstra_wopq, pred_dijkstra_wopq = dijkstra_without_pq(graph_dijkstra, source_node)
        # end_wopq = time.time()
        # if dist_dijkstra!=dist_dijkstra_wopq or pred_dijkstra!=pred_dijkstra_wopq:
        #     print("Dijkstra without and with PQ are not giving same results!")
        #     break
        time_dijkstra += end - start
        # time_dijkstra_without_pq += end_wopq-start_wopq
        if i == 499:
            # print("bah")
            break
    
    # # --- Bellman-Ford ---
    graph_bf = [row[:] for row in adj]
    dist_bf, pred_bf = bellman_ford(graph_bf, source_node)
    # print("guh")
    time_bf = 0
    for i, (u, v, weight) in enumerate(updates):
        # print(i)
        graph_bf[u][v] = weight  # Directed graph, so only u->v
        start = time.time()
        dist_bf, pred_bf = bellman_ford(graph_bf, source_node)
        end = time.time()
        time_bf += end - start
        if i == 199:
            # print("dah")
            break
    
    # Print results

    print(f"Standard Dijkstra's with PQ Update Time: {time_dijkstra:.4f} seconds")
    
    print(f"Bellman-Ford Update Time: {time_bf:.4f} seconds")

if __name__ == "__main__":
    compare_algorithms()