# import heapq

# # Node class to represent each vertex in the priority queue
# class Node:
#     def __init__(self, ver=None, dist=None, pred=None, del_time=None, color='R', valid=True):
#         self.ver = ver           # Vertex
#         self.dist = dist         # Distance from source
#         self.pred = pred         # Predecessor node
#         self.del_time = del_time # Deletion time (for retroactive operations)
#         self.color = color       # Color for Red-Black tree
#         self.valid = valid       # Whether the node is valid or deleted

    # def __lt__(self, other):
    #     # This method defines the comparison for the priority queue (min-heap).
    #     return self.dist < other.dist

# # Retroactive Priority Queue using Red-Black Trees
# class RetroactivePriorityQueue:
#     def __init__(self):
#         self.Tins = []  # Inserted nodes (priority queue)
#         self.Td_m = []  # Del-min nodes
#         self.time = 0  # To keep track of operation time

#     def invoke_insert(self, x, t):
#         """
#         Performs the insert operation at time t.
#         """
#         x.del_time = None  # Initially, no delete time
#         heapq.heappush(self.Tins, (x.dist, t, x))  # Add to RPQ based on distance

#     def invoke_del_min(self, t):
#         """
#         Removes the minimum item at time t from the RPQ.
#         """
#         if self.Tins:
#             min_node = heapq.heappop(self.Tins)[2]  # Get the minimum node
#             min_node.del_time = t  # Set the delete time
#             self.Td_m.append(min_node)  # Store in Td_m (del-min tree)
#             return min_node
#         return None

#     def revoke_insert(self, t):
#         """
#         Undo the insert operation at time t.
#         """
#         if self.Td_m:
#             min_node = self.Td_m.pop()  # Revert the last operation (soft delete)
#             min_node.valid = False  # Mark as invalid
#             return min_node
#         return None

#     def revoke_del_min(self, t):
#         """
#         Undo the del-min operation at time t.
#         """
#         if self.Td_m:
#             node = self.Td_m.pop()  # Undo delete-min
#             heapq.heappush(self.Tins, (node.dist, t, node))  # Reinsert back into Tins
#             return node
#         return None

#     def find_min(self, t):
#         """
#         Find the minimum element at time t.
#         """
#         if self.Tins:
#             return self.Tins[0][2]  # Return the min node (smallest distance)
#         return None

# # Dynamic Dijkstra Algorithm using Retroactive Priority Queue
# class DynamicDijkstra:
#     def __init__(self, graph, source):
#         """
#         Initializes the Dynamic Dijkstra algorithm.
#         The graph is represented as an adjacency matrix.
#         The source is the starting vertex for the algorithm.
#         """
#         self.graph = graph  # Graph as an adjacency matrix
#         self.rpq = RetroactivePriorityQueue()  # Retroactive Priority Queue (RPQ)
#         self.dist = {}  # Holds the shortest distances from the source vertex
#         self.pred = {}  # Holds the predecessors for the shortest path
#         self.source = source  # The source node for Dijkstra's algorithm

#         # Initialize the distances and predecessors
#         for node in self.graph:
#             self.dist[node] = float('inf')  # Initially set to infinity
#             self.pred[node] = None  # No predecessor at the start

#         self.dist[source] = 0  # Set the distance to the source node as 0

#         # Insert the source node into the Retroactive Priority Queue
#         self.rpq.invoke_insert(Node(ver=source, dist=0, pred=None), 0)

#     def update_edge_weight(self, u, v, new_weight):
#         """
#         Updates the weight of an edge (u, v) in the graph and triggers necessary updates in the priority queue.
#         """
#         # Update the graph with the new edge weight
#         self.graph[u][v] = new_weight
#         self.graph[v][u] = new_weight  # Assuming undirected graph for simplicity
        
#         # Update the retroactive priority queue to propagate the changes in shortest paths
#         self.dijkstra_update(u, v)

#     def dijkstra_update(self, u, v):
#         """
#         Updates the shortest paths using Dijkstra's algorithm with retroactive updates.
#         This method handles the propagation of changes due to edge weight updates.
#         """
#         # Step 1: Ensure that the affected node `u` is in the RPQ
#         min_node = self.rpq.find_min(self.rpq.time)
        
#         # Step 2: Propagate the changes if the node is valid and the update affects the shortest paths
#         while min_node:
#             self._relax(min_node, v)  # Relax the node's distance to the affected vertex 'v'
#             min_node = self.rpq.find_min(self.rpq.time)

#     def _relax(self, node, v):
#         """
#         Performs the relaxation step in Dijkstra's algorithm to update the shortest path estimate.
#         """
#         u = node.ver  # Get the vertex from the current node
#         # If the new distance is shorter, update the distance and predecessor
#         if self.dist[v] > self.dist[u] + self.graph[u][v]:
#             self.dist[v] = self.dist[u] + self.graph[u][v]
#             self.pred[v] = u

#             # Insert the updated node back into the Retroactive Priority Queue
#             self.rpq.invoke_insert(Node(ver=v, dist=self.dist[v], pred=u), self.rpq.time)

#     def find_shortest_paths(self):
#         """
#         Executes the Dijkstra algorithm using the Retroactive Priority Queue.
#         This method calculates the shortest paths from the source node to all other nodes.
#         """
#         while True:
#             min_node = self.rpq.find_min(self.rpq.time)  # Get the minimum node from the queue
#             if not min_node:
#                 break  # Exit if no more nodes to process

#             u = min_node.ver  # Get the vertex of the current minimum node

#             # For each adjacent node 'v' in the graph, relax the edges and update the distances
#             for v in self.graph[u]:
#                 self._relax(min_node, v)

#             # After processing the node, remove it from the Retroactive Priority Queue
#             self.rpq.invoke_del_min(self.rpq.time)

# # Example Usage

# # Example graph initialization (adjacency matrix form)
# graph = {
#     'A': {'B': 2, 'C': 4},
#     'B': {'A': 2, 'C': 1, 'D': 7},
#     'C': {'A': 4, 'B': 1, 'D': 3},
#     'D': {'B': 7, 'C': 3}
# }

# # Initialize Dynamic Dijkstra with the source node 'A'
# dijkstra = DynamicDijkstra(graph, source='A')

# # Example: Update edge weight and recalculate shortest paths
# dijkstra.update_edge_weight('B', 'C', 5)

# # Print updated shortest paths and predecessors
# print(dijkstra.dist)  # Shortest distances from source
# print(dijkstra.pred)  # Predecessors for the shortest paths

class Node:
    def __init__(self, ver=None, dist=None, pred=None, del_time=None, color='R', valid=True):
        self.ver = ver           # Vertex
        self.dist = dist         # Distance from source
        self.pred = pred         # Predecessor node
        self.del_time = del_time # Deletion time (for retroactive operations)
        self.color = color       # Color for Red-Black tree
        self.valid = valid       # Whether the node is valid or deleted


class RedBlackTree:
    def __init__(self):
        self.TNULL = Node(key=None, color='B')  # Sentinel leaf node (black)
        self.root = self.TNULL
    
    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
    
    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
    
    def insert(self, key):
        node = Node(key)
        node.left = self.TNULL
        node.right = self.TNULL
        node.parent = None
        
        y = None
        x = self.root
        
        while x != self.TNULL:
            y = x
            if node.key < x.key:
                x = x.left
            else:
                x = x.right
        
        node.parent = y
        if y is None:
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node
        
        node.color = 'R'
        node.left = self.TNULL
        node.right = self.TNULL
        
        self.fix_insert(node)

    def fix_insert(self, k):
        while k.parent and k.parent.color == 'R':
            if k.parent == k.parent.parent.left:
                u = k.parent.parent.right
                if u.color == 'R':
                    u.color = 'B'
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    self.right_rotate(k.parent.parent)
            else:
                u = k.parent.parent.left
                if u.color == 'R':
                    u.color = 'B'
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    self.left_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 'B'

    def delete(self, node):
        # Placeholder for delete operation to remove a node
        pass
    
    def search(self, key):
        node = self.root
        while node != self.TNULL:
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None
class RetroactivePriorityQueue:
    def __init__(self):
        self.Tins = RedBlackTree()  # Tree for inserted nodes (priority queue)
        self.Td_m = RedBlackTree()  # Tree for del-min nodes
        self.time = 0  # To keep track of operation time
    
    def invoke_insert(self, x, t):
        """
        Perform insert operation at time t.
        """
        x['del_time'] = None  # Initially no delete time
        self.Tins.insert(x)  # Insert into the RPQ (Tins)
    
    def invoke_del_min(self, t):
        """
        Removes the minimum item from the priority queue at time t.
        """
        min_node = self.Tins.root  # Get the minimum node (root of the tree)
        if min_node:
            self.Tins.delete(min_node)  # Delete from Tins
            min_node.del_time = t  # Mark the time of deletion
            self.Td_m.insert(min_node)  # Insert into Td_m (del-min tree)
            return min_node
        return None
    
    def revoke_insert(self, t):
        """
        Undo the insert operation at time t.
        """
        # Revert the last insert operation by marking the node as invalid in Tins
        min_node = self.Td_m.root  # Fetch the last operation in Td_m
        if min_node:
            min_node.valid = False  # Mark as invalid
            self.Tins.delete(min_node)  # Remove it from Tins
        return min_node
    
    def revoke_del_min(self, t):
        """
        Undo the del-min operation at time t.
        """
        # Undo the delete-min operation by re-inserting the node back into Tins
        node = self.Td_m.root  # Fetch the last deleted node from Td_m
        if node:
            self.Tins.insert(node)  # Re-insert into Tins
        return node
    
    def find_min(self, t):
        """
        Returns the minimum element at time t.
        """
        # Return the current minimum node in Tins
        return self.Tins.root

# Dynamic Dijkstra Algorithm using Retroactive Priority Queue
class DynamicDijkstra:
    def __init__(self, graph, source):
        """
        Initializes the Dynamic Dijkstra algorithm.
        The graph is represented as an adjacency matrix.
        The source is the starting vertex for the algorithm.
        """
        self.graph = graph  # Graph as an adjacency matrix
        self.rpq = RetroactivePriorityQueue()  # Retroactive Priority Queue (RPQ)
        self.dist = {}  # Holds the shortest distances from the source vertex
        self.pred = {}  # Holds the predecessors for the shortest path
        self.source = source  # The source node for Dijkstra's algorithm

        # Initialize the distances and predecessors
        for node in self.graph:
            self.dist[node] = float('inf')  # Initially set to infinity
            self.pred[node] = None  # No predecessor at the start

        self.dist[source] = 0  # Set the distance to the source node as 0
        x = {'ver':source, 'dist':0, 'pred':None, 'valid':True,'del_time':None}
        # Insert the source node into the Retroactive Priority Queue
        self.rpq.invoke_insert(x, 0)

    def update_edge_weight(self, u, v, new_weight):
        """
        Updates the weight of an edge (u, v) in the graph and triggers necessary updates in the priority queue.
        """
        # Update the graph with the new edge weight
        self.graph[u][v] = new_weight
        self.graph[v][u] = new_weight  # Assuming undirected graph for simplicity
        
        # Update the retroactive priority queue to propagate the changes in shortest paths
        self.dijkstra_update(u, v)

    def dijkstra_update(self, u, v):
        """
        Updates the shortest paths using Dijkstra's algorithm with retroactive updates.
        This method handles the propagation of changes due to edge weight updates.
        """
        # Step 1: Ensure that the affected node `u` is in the RPQ
        min_node = self.rpq.find_min(self.rpq.time)
        
        # Step 2: Propagate the changes if the node is valid and the update affects the shortest paths
        while min_node:
            self._relax(min_node, v)  # Relax the node's distance to the affected vertex 'v'
            min_node = self.rpq.find_min(self.rpq.time)

    def _relax(self, node, v):
        """
        Performs the relaxation step in Dijkstra's algorithm to update the shortest path estimate.
        """
        u = node.ver  # Get the vertex from the current node
        # If the new distance is shorter, update the distance and predecessor
        if self.dist[v] > self.dist[u] + self.graph[u][v]:
            self.dist[v] = self.dist[u] + self.graph[u][v]
            self.pred[v] = u

            # Insert the updated node back into the Retroactive Priority Queue
            self.rpq.invoke_insert(Node(ver=v, dist=self.dist[v], pred=u), self.rpq.time)

    def find_shortest_paths(self):
        """
        Executes the Dijkstra algorithm using the Retroactive Priority Queue.
        This method calculates the shortest paths from the source node to all other nodes.
        """
        while True:
            min_node = self.rpq.find_min(self.rpq.time)  # Get the minimum node from the queue
            if not min_node:
                break  # Exit if no more nodes to process

            u = min_node.ver  # Get the vertex of the current minimum node

            # For each adjacent node 'v' in the graph, relax the edges and update the distances
            for v in self.graph[u]:
                self._relax(min_node, v)

            # After processing the node, remove it from the Retroactive Priority Queue
            self.rpq.invoke_del_min(self.rpq.time)

# Example Usage

# Example graph initialization (adjacency matrix form)
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'C': 1, 'D': 7},
    'C': {'A': 4, 'B': 1, 'D': 3},
    'D': {'B': 7, 'C': 3}
}

# Initialize Dynamic Dijkstra with the source node 'A'
dijkstra = DynamicDijkstra(graph, source='A')

# Example: Update edge weight and recalculate shortest paths
dijkstra.update_edge_weight('B', 'C', 5)

# Print updated shortest paths and predecessors
print(dijkstra.dist)  # Shortest distances from source
print(dijkstra.pred)  # Predecessors for the shortest paths