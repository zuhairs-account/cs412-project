import heapq

class RetroactivePriorityQueue:
    def __init__(self):
        self.queue = []
        self.entry_finder = {}  # Maps vertices to their queue entries
        self.REMOVED = '<removed>'  # Placeholder for a removed node
        self.counter = 0  # A counter to break ties in priority

    def push(self, vertex, priority=0):
        """Push a new vertex into the queue."""
        if vertex in self.entry_finder:
            self.remove(vertex)
        count = next(self.counter)  # ensures a unique sequence for each push
        entry = [priority, count, vertex]
        self.entry_finder[vertex] = entry
        heapq.heappush(self.queue, entry)

    def remove(self, vertex):
        """Mark an existing vertex as REMOVED."""
        entry = self.entry_finder.pop(vertex)
        entry[-1] = self.REMOVED

    def pop(self):
        """Remove and return the vertex with the lowest priority."""
        while self.queue:
            priority, count, vertex = heapq.heappop(self.queue)
            if vertex is not self.REMOVED:
                del self.entry_finder[vertex]
                return vertex, priority
        raise KeyError('pop from an empty priority queue')

    def update(self, vertex, priority):
        """Update the priority of a vertex."""
        self.remove(vertex)
        self.push(vertex, priority)


class DynamicDijkstra:
    def __init__(self, graph, source):
        self.graph = graph  # Adjacency list representation of the graph
        self.source = source
        self.dist = {v: float('inf') for v in self.graph}  # Initial distances
        self.dist[source] = 0
        self.pred = {v: None for v in self.graph}  # Predecessors for path reconstruction
        self.rpq = RetroactivePriorityQueue()
        self.rpq.push(source, 0)

    def update_edge_weight(self, u, v, new_weight):
        """Update the weight of the edge (u, v) in the graph."""
        if u in self.graph and v in self.graph[u]:
            self.graph[u][v] = new_weight
        # In a real implementation, we would also need to handle updates to the priority queue

    def run(self):
        """Run the dynamic Dijkstra's algorithm."""
        while self.rpq.queue:
            u, _ = self.rpq.pop()
            for v, weight in self.graph.get(u, {}).items():
                new_dist = self.dist[u] + weight
                if new_dist < self.dist[v]:
                    self.dist[v] = new_dist
                    self.pred[v] = u
                    self.rpq.push(v, new_dist)
# Define the graph as an adjacency list (dictionary of dictionaries)
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

# Set the source vertex
source = 'A'

# Create an instance of DynamicDijkstra
dijkstra = DynamicDijkstra(graph, source)

# Run the algorithm to find the shortest paths
dijkstra.run()

# Print the shortest distances from source to each vertex
print("Shortest distances from source:", dijkstra.dist)

# Print the predecessors for each vertex in the shortest path tree
print("Predecessors for the shortest path:", dijkstra.pred)

# Dynamically update the weight of an edge
dijkstra.update_edge_weight('B', 'C', 1)  # Change the weight of edge (B, C)

# Re-run the algorithm after the edge update
dijkstra.run()

# Print the updated shortest distances
print("Updated shortest distances after edge update:", dijkstra.dist)
