import heapq
import time



class Node:
    # def __init__(self, ver=None, ins_time=None, dist=None, pred=None, del_time=None, color='R', left=None, right=None, valid=None):
    #     self.ins_time = ins_time # Insertion time
    #     self.color = color       # Color ('R' for red, 'B' for black)
    #     self.left = left         # Left child
    #     self.right = right       # Right child
    #     self.parent = pred     # Parent node
    #     self.value = {'vertex': ver,'ins_time':ins_time, 'dist': dist, 'pred': pred, 'del_time': del_time, 'valid':valid}         # Next node in the list

    # simulated
    def __init__(self, x):
        self.value=x         # Next node in the list

class RedBlackTree:
    def __init__(self):
        # self.root = None

        #simulated
        self.queue = []

# as per the pseudocode, a an entry in queue has (ver,ins_time,dist,pred,del_time)
    def insert(self, tree, x):
        new_node = Node(x)        
        # if not self.root:
        #     self.root = new_node
        # else:
        #     self._insert_node(self.root, new_node, tree)
        
        # # Fix the red-black tree properties after insertion
        # self._fix_insert(new_node)
        #simulated
        self.queue.append(new_node)


    def _insert_node(self, root, node, tree):
        # Standard BST insert with priority given to node.dist first, then node.time
        if tree=='T_ins':
            if node.value.dist < root.dist or (node.value.dist == root.value.dist and node.value.time < root.value.time):
                if root.left is None:
                    root.left = node
                    node.parent = root
                else:
                    self._insert_node(root.left, node)
            else:
                if root.right is None:
                    root.right = node
                    node.parent = root
                else:
                    self._insert_node(root.right, node)
        else:
            if node.value.del_time < root.value.del_time:
                if root.left is None:
                    root.left = node
                    node.parent = root
                else:
                    self._insert_node(root.left, node)
            else:
                if root.right is None:
                    root.right = node
                    node.parent = root
                else:
                    self._insert_node(root.right, node)


    def _fix_insert(self, node):
        # Fix the tree after insertion to ensure red-black properties
        while node != self.root and node.parent.color == 'R':
            # Case 1: Parent is red, uncle is also red
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle and uncle.color == 'R':
                    # Case 1a: Uncle is red
                    node.parent.color = 'B'
                    uncle.color = 'B'
                    node.parent.parent.color = 'R'
                    node = node.parent.parent
                else:
                    # Case 1b: Uncle is black
                    if node == node.parent.right:
                        # Left rotate at parent
                        node = node.parent
                        self._rotate_left(node)
                    # Right rotate at grandparent
                    node.parent.color = 'B'
                    node.parent.parent.color = 'R'
                    self._rotate_right(node.parent.parent)
            else:
                # Symmetric to Case 1 (right-side case)
                uncle = node.parent.parent.left
                if uncle and uncle.color == 'R':
                    node.parent.color = 'B'
                    uncle.color = 'B'
                    node.parent.parent.color = 'R'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._rotate_right(node)
                    node.parent.color = 'B'
                    node.parent.parent.color = 'R'
                    self._rotate_left(node.parent.parent)

        self.root.color = 'B'  # Ensure root is always black

    def _rotate_left(self, x):
        # Left rotation
        y = x.right
        x.right = y.left
        if y.left:
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

    def _rotate_right(self, x):
        # Right rotation
        y = x.left
        x.left = y.right
        if y.right:
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

    # def _find_node_by_t(self, root, t):
    #     # Find the node with the given key (vertex)
    #     if root is None or root.value['del_time'] == t:
    #         return root
    #     if key < root.vertex:
    #         return self._find_node(root.left, key)
    #     return self._find_node(root.right, key)
    # def _find_node_by_t_x(self, root, key):
    #     # Find the node with the given key (vertex)
    #     if root is None or root.vertex == key:
    #         return root
    #     if key < root.vertex:
    #         return self._find_node(root.left, key)
    #     return self._find_node(root.right, key)
    
    # def Search_Min(self,tree, t):
    #     # Find the minimum node (leftmost node)
    #     if tree == 'T_d_m':
    #         current = self.root
    #         while current.left and current.value['del_time']>t:
    #             current = current.left
    #         return current

    #     else:
    #         current = self.root
    #         while current.left and current.value['del_time']>t:
    #             current = current.left
    #         return current
    # def Min(self):
    #     current = self.root
    #     while current.left:
    #         current = current.left
    #     return current
        
    # def Search_Min(self, t=None):
    #     # Find the minimum node (leftmost node)
    #     if not t:       #insert is calling
    #         current = self.root
    #         while current.left:
    #             current = current.left
    #         return current
    #     else:
    #         current = self.root
    #         while current.left and current.value['del_time']>t:
    #             current = current.left
    #         return current

def Search_Min_T_dm_after_t(T_dm,t):
    # current = T_dm.root
    # while current.right and current.value['del_time']>t:
    #     current = current.left
    #     return current
    # simulated
    if len(T_dm.queue)==0:
        print("T_d_m queue is empty.")
        return None

    T_dm.queue.sort(key=lambda node: node.value['del_time'])
    for a in T_dm.queue:
        if a.value['del_time'] > t:
            return a
def Search_element_del_before_t_T_dm(T_dm,t):
    # current = T_dm.root
    # while current.right and current.value['del_time']>t:
    #     current = current.left
    #     return current
    # simulated
    if len(T_dm.queue)==0:
        print("T_d_m queue is empty.")
        return None
    T_dm.queue.sort(key=lambda node: node.value['del_time'])
    for a in T_dm.queue:
        if a.value['del_time'] < t:
            return a
def Search_element_dist_gt_kprime_T_Ins(T_ins,k):
    # current = T_dm.root
    # while current.right and current.value['del_time']>t:
    #     current = current.left
    #     return current
    # simulated
    if len(T_ins.queue)==0:
        print("T_Ins queue is empty.")
        return None
    T_ins.queue.sort(key=lambda node: node.value['dist'])
    maxk = 0
    for a in T_ins.queue:
        if a.value['dist'] > k:
            return a
def find_min_T_ins(T_ins):
    T_dm.queue.sort(key=lambda node: node.value['dist'])
    return T_dm.queue[0]


T_dm = RedBlackTree()
T_dm.insert('T_dm',{'vertex':'A','ins_time': 4, 'del_time': 6, 'dist': 1})
T_dm.insert('T_dm',{'vertex':'A','ins_time': 5, 'del_time': 7, 'dist': 4})
print(T_dm.queue[0].value)
res  = Search_element_del_before_t_T_dm(T_dm,6)
if res:
    print(res[0].value)

class RetroactivePriorityQueue:
    def __init__(self):
        self.T_ins = RedBlackTree()  # Insertion tree
        self.T_d_m = RedBlackTree()  # Deletion tree
        self.time = 0  # Current time step

    def invoke_insert(self, vertex, dist, pred):
        self.time += 1
        # self.Tins.insert('T_ins',x)
        P = Search_Min_T_dm_after_t(self.T_d_m, self.time)
        if P:
            return P #send inconsistencies
        self.T_ins.insert({'vertex':vertex,'ins_time': self.time, 'dist':dist, 'pred':pred, 'del_time': None, 'valid': True})
        return None #no inconsistencies
        # # x ={'vertex': ver,'ins_time':ins_time, 'dist': dist, 'pred': pred, 'del_time': del_time, 'valid':valid} 
        # # node = Node(ver=vertex, ins_time=self.time, dist=dist, pred=pred, del_time=None, valid=True)  # Create Node        self.time += 1
        # self.Tins.insert('T_ins', t)
        # P = Search_Min(self.T_d_m,self.time)

    def invoke_del_min(self):
        self.time += 1
        P = Search_element_del_before_t_T_dm(self.T_d_m, self.time)
        if P is None: 
            N = find_min_T_ins(self.T_ins)
            node_to_del = N
            # self.Tins.delete(min_item[0])       #passing key
            # node = Node(ver=min_item[0], ins_time=min_item[1][0], dist=min_item[1][1], pred=min_item[1][1], del_time=self.time, valid=True)  # Create Node        self.time += 1
            # self.Td_m.insert(min_item[0], node)
            # return min_item
        else:
            K = P.value
            node_to_del = Search_element_dist_gt_kprime_T_Ins(K['dist'])
        node_to_del.value['valid'] = False
        del_node_vals = node_to_del.value
        del_node_vals['del_time'] = self.time
        self.T_d_m.insert('T_dm',del_node_vals)
        return del_node_vals.copy()

        # return None

    def revoke_insert(self):
        min_del_item = self.Td_m.Search_Min()
        if min_del_item:
            self.Td_m.delete(min_del_item[0])
            self.Tins.insert(min_del_item[0], (min_del_item[1][0], min_del_item[1][1], self.time))

    def revoke_del_min(self):
        self.time+=1
        last_del_item = self.Td_m.Search_Min((self.time,True))
        if last_del_item:
            self.Td_m.delete(last_del_item[0])
            self.Tins.insert(last_del_item[0], (last_del_item[1][0], last_del_item[1][1], self.time))

    def find_min(self):
        return self.Tins.find_min()


class DynamicDijkstra:
    def __init__(self, graph):
        self.graph = graph
        self.num_vertices = len(graph)
        self.dists = [float('inf')] * self.num_vertices
        self.predecessors = [None] * self.num_vertices
        self.queue = RetroactivePriorityQueue()

    def relax(self, u, v, weight):
        if self.dists[v] > self.dists[u] + weight:
            self.dists[v] = self.dists[u] + weight
            self.predecessors[v] = u
            self.queue.invoke_insert(v, self.dists[v], u)

    def dynamic_dijkstra(self, source):
        self.dists[source] = 0
        self.queue.invoke_insert(source, 0, None)
        
        while True:
            min_vertex = self.queue.find_min()
            if min_vertex is None:
                break
            u, (dist_u, pred_u, _) = min_vertex  # Unpack correctly to get distance and predecessor
            self.queue.invoke_del_min()

            for v, weight in enumerate(self.graph[u]):
                if weight > 0:
                    self.relax(u, v, weight)

    def update_edge_weight(self, u, v, new_weight):
        # Adjust graph edge weight and trigger dynamic updates
        self.graph[u][v] = new_weight
        self.graph[v][u] = new_weight  # Assuming undirected graph for simplicity
        self.dynamic_dijkstra(u)

# Sample graph represented as an adjacency matrix
# List of vertices in order: A, B, C, D, E, F, O, T
vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'O', 'T']

# Initialize the adjacency matrix
graph = [
    # A  B  C  D  E  F  O  T
    [ 0, 2, 0, 4, 7,12, 2, 0],  # A
    [ 2, 0, 1, 3, 4, 0, 5, 0],  # B
    [ 0, 1, 0, 0, 4, 0, 4, 0],  # C
    [ 4, 3, 0, 0, 4, 5, 0, 5],  # D
    [ 7, 4, 4, 4, 0, 0, 0, 7],  # E
    [12, 0, 0, 5, 0, 0, 0, 3],  # F
    [ 2, 5, 4, 0, 0, 0, 0, 0],  # O
    [ 0, 0, 0, 5, 7, 3, 0, 0],  # T
]


    # def delete(self, key):
    #     # Find the node to delete
    #     node_to_delete = self._find_node(self.root, key)
    #     if node_to_delete:
    #         self._delete_node(node_to_delete)
    #     else:
    #         print(f"Node with vertex {key} not found!")

    # def _find_node(self, root, key):
    #     # Find the node with the given key (vertex)
    #     if root is None or root.vertex == key:
    #         return root
    #     if key < root.vertex:
    #         return self._find_node(root.left, key)
    #     return self._find_node(root.right, key)

    # def _delete_node(self, node):
    #     # Standard delete for a binary search tree
    #     if node.left is None or node.right is None:
    #         # Node has at most one child
    #         if node.left:
    #             child = node.left
    #         else:
    #             child = node.right

    #         if child:
    #             child.parent = node.parent

    #         if node.parent is None:
    #             self.root = child
    #         elif node == node.parent.left:
    #             node.parent.left = child
    #         else:
    #             node.parent.right = child

    #         if node.color == 'B':
    #             self._fix_delete(child)

    #         node.left = node.right = node.parent = None

    # def _fix_delete(self, node):
    #     # Fix red-black tree properties after deletion
    #     while node != self.root and node.color == 'B':
    #         if node == node.parent.left:
    #             sibling = node.parent.right
    #             if sibling.color == 'R':
    #                 sibling.color = 'B'
    #                 node.parent.color = 'R'
    #                 self._rotate_left(node.parent)
    #                 sibling = node.parent.right

    #             if sibling.left.color == 'B' and sibling.right.color == 'B':
    #                 sibling.color = 'R'
    #                 node = node.parent
    #             else:
    #                 if sibling.right.color == 'B':
    #                     sibling.left.color = 'B'
    #                     sibling.color = 'R'
    #                     self._rotate_right(sibling)
    #                     sibling = node.parent.right

    #                 sibling.color = node.parent.color
    #                 node.parent.color = 'B'
    #                 sibling.right.color = 'B'
    #                 self._rotate_left(node.parent)
    #                 node = self.root

    #     node.color = 'B'  # Ensure node is black


# class RedBlackTree:
#     # Red-Black Tree using Node class for each node
#     def __init__(self):
#         self.tree = []

#     def insert(self, key, value):
#         # Insert a node into the red-black tree (simplified)
#         # Ensure value is a Node instance
#         if isinstance(value, Node):
#             self.tree[key] = value
#         else:
#             print(f"Error: Attempting to insert invalid value format for vertex {key}: {value}")

#     def delete(self, key):
#         # Mark the node as deleted by clearing its deletion time
#         if key in self.tree:
#             self.tree[key].del_time = self.tree[key].ins_time # Setting del_time to insertion time (indicating deletion)
#             self.tree[key].valid = False

#     def find_min(self):
#         # Find the node with the minimum distance (or any other appropriate comparison criterion)
#         return min(self.tree.items(), key=lambda x: x[1].dist, default=None) # Use .dist to compare nodes



# class Node:
#     def __init__(self, ver=None, ins_time=None, dist=None, pred=None, del_time=None, valid=None):
#         self.vertex = ver        # Vertex
#         self.ins_time = ins_time # Insertion time
#         self.dist = dist         # Distance
#         self.pred = pred         # Predecessor
#         self.del_time = del_time # Deletion time
#         self.valid = valid         # Next node in the list

    # def _insert_node(self, root, node, tree):
    #     # Standard BST insert
    #     if node.value.vertex < root.vertex:
    #         if root.left is None:
    #             root.left = node
    #             node.parent = root
    #         else:
    #             self._insert_node(root.left, node)
    #     else:
    #         if root.right is None:
    #             root.right = node
    #             node.parent = root
    #         else:
    #             self._insert_node(root.right, node)


# # Create an instance of Dynamic Dijkstra
# dd = DynamicDijkstra(graph)


# # Run dynamic Dijkstra starting from vertex 0
# dd.dynamic_dijkstra(0)

# # Update edge weight and rerun the Dijkstra process dynamically
# dd.update_edge_weight(1, 2, 3)  # Update edge weight between vertex 1 and 2

# # Display results
# print("Distances:", dd.dists)
# print("Predecessors:", dd.predecessors)


# ---------------


# import heapq
# import time

# class RedBlackTree:
#     # Basic structure for the Red-Black Tree
#     def __init__(self):
#         self.tree = {}

#     def insert(self, key, value):
#         # Insert into the red-black tree (simplified)
#         if len(value) == 3:  # Ensure the value is a tuple (dist, pred, time)
#             self.tree[key] = value
#         else:
#             print(f"Error: Attempting to insert invalid value format for vertex {key}: {value}")

#     def delete(self, key):
#         if key in self.tree:
#             del self.tree[key]

#     def find_min(self):
#         return min(self.tree.items(), key=lambda x: x[1], default=None)

# class RetroactivePriorityQueue:
#     def __init__(self):
#         self.Tins = RedBlackTree()  # Insertion tree
#         self.Td_m = RedBlackTree()  # Deletion tree
#         self.time = 0  # Current time step

#     def invoke_insert(self, vertex, dist, pred):
#         # Insert with correct format (dist, pred, time)
#         self.time += 1
#         self.Tins.insert(vertex, (dist, pred, self.time))

#     def invoke_del_min(self):
#         self.time += 1
#         min_item = self.Tins.find_min()
#         if min_item:
#             self.Tins.delete(min_item[0])
#             self.Td_m.insert(min_item[0], self.time)
#             return min_item
#         return None

#     def revoke_insert(self):
#         # Retrieve the latest inserted item in Td_m (deletion tree)
#         min_del_item = self.Td_m.find_min()
        
#         if min_del_item:
#             vertex = min_del_item[0]  # Get the vertex
#             # Ensure we unpack correctly, should be a tuple (dist, pred, time)
#             if isinstance(min_del_item[1], tuple) and len(min_del_item[1]) == 3:  # Check if it's in the correct format
#                 dist, pred, _ = min_del_item[1]  # Unpack the tuple
#                 # Reinserting into Tins (insertion tree) with the updated time
#                 self.Tins.insert(vertex, (dist, pred, self.time))
#             else:
#                 print("Error: Invalid format in Td_m for vertex", vertex)

#     def revoke_del_min(self):
#         # Retrieve the latest deleted item in Td_m (deletion tree)
#         last_del_item = self.Td_m.find_min()
        
#         if last_del_item:
#             vertex = last_del_item[0]  # Get the vertex
#             # Ensure we unpack correctly, should be a tuple (dist, pred, time)
#             if isinstance(last_del_item[1], tuple) and len(last_del_item[1]) == 3:  # Check if it's in the correct format
#                 dist, pred, _ = last_del_item[1]  # Unpack the tuple
#                 # Reinserting into Tins (insertion tree) with the updated time
#                 self.Tins.insert(vertex, (dist, pred, self.time))
#             else:
#                 print("Error: Invalid format in Td_m for vertex", vertex)

#     def find_min(self):
#         return self.Tins.find_min()


# class DynamicDijkstra:
#     def __init__(self, graph):
#         self.graph = graph
#         self.num_vertices = len(graph)
#         self.dists = [float('inf')] * self.num_vertices
#         self.predecessors = [None] * self.num_vertices
#         self.queue = RetroactivePriorityQueue()

#     def relax(self, u, v, weight):
#         if self.dists[v] > self.dists[u] + weight:
#             self.dists[v] = self.dists[u] + weight
#             self.predecessors[v] = u
#             self.queue.invoke_insert(v, (self.dists[v], u, self.time))  # Correct format: (dist, pred, time)

#     def dynamic_dijkstra(self, source):
#         self.dists[source] = 0
#         self.queue.invoke_insert(source, 0, None)
        
#         while True:
#             min_vertex = self.queue.find_min()
#             if min_vertex is None:
#                 break
#             u, (dist_u, pred_u, _) = min_vertex  # Unpack correctly to get distance and predecessor
#             self.queue.invoke_del_min()

#             for v, weight in enumerate(self.graph[u]):
#                 if weight > 0:
#                     self.relax(u, v, weight)

#     def update_edge_weight(self, u, v, new_weight):
#         # Adjust graph edge weight and trigger dynamic updates
#         self.graph[u][v] = new_weight
#         self.graph[v][u] = new_weight  # Assuming undirected graph for simplicity
        
#         # Revoke any invalidation from previous operations (insertions or deletions)
#         self.queue.revoke_insert()  # Undo any invalid insertions
#         self.queue.revoke_del_min()  # Undo any invalid deletions
        
#         # Rerun the Dijkstra process with updated edge weights
#         self.dynamic_dijkstra(u)

# # Sample graph represented as an adjacency matrix
# # List of vertices in order: A, B, C, D, E, F, O, T
# vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'O', 'T']

# # Initialize the adjacency matrix
# graph = [
#     # A  B  C  D  E  F  O  T
#     [ 0, 2, 0, 4, 7,12, 2, 0],  # A
#     [ 2, 0, 1, 3, 4, 0, 5, 0],  # B
#     [ 0, 1, 0, 0, 4, 0, 4, 0],  # C
#     [ 4, 3, 0, 0, 4, 5, 0, 5],  # D
#     [ 7, 4, 4, 4, 0, 0, 0, 7],  # E
#     [12, 0, 0, 5, 0, 0, 0, 3],  # F
#     [ 2, 5, 4, 0, 0, 0, 0, 0],  # O
#     [ 0, 0, 0, 5, 7, 3, 0, 0],  # T
# ]


# # Create an instance of Dynamic Dijkstra
# dd = DynamicDijkstra(graph)

# # Run dynamic Dijkstra starting from vertex 0
# dd.dynamic_dijkstra(0)
# # Display results
# print("Before Update \n Distances:", dd.dists)
# print("Predecessors:", dd.predecessors)

# # Update edge weight and rerun the Dijkstra process dynamically
# dd.update_edge_weight(1, 2, 3)  # Update edge weight between vertex 1 and 2

# # Display results
# print("After Update \n Distances:", dd.dists)
# print("Predecessors:", dd.predecessors)

# dd.update_edge_weight(1, 2, 2)  # Update edge weight between vertex 1 and 2

# # Display results
# print("After Rollback \n Distances:", dd.dists)
# print("Predecessors:", dd.predecessors)

# Create an instance of Dynamic Dijkstra
# dd = DynamicDijkstra(graph)

# # Run dynamic Dijkstra starting from vertex 0
# dd.dynamic_dijkstra(0)

# # Update edge weight and rerun the Dijkstra process dynamically
# dd.update_edge_weight(1, 2, 3)  # Update edge weight between vertex 1 and 2

# # Display results
# print("Distances:", dd.dists)
# print("Predecessors:", dd.predecessors)