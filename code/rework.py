import math
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Enum for node types in the RPQ
class NodeType(Enum):
    INS = 1
    DEL = 2

# Simple Red-Black Tree implementation for RPQ
@dataclass
class RBNode:
    key: tuple
    value: Any
    left: Optional['RBNode'] = None
    right: Optional['RBNode'] = None
    color: str = 'RED'  # RED or BLACK

class RBTree:
    def __init__(self):
        self.NIL = RBNode(key=(math.inf, math.inf), value=None, color='BLACK')
        self.root = self.NIL

    def search(self, key):
        node = self.root
        while node != self.NIL and node.key != key:
            if key < node.key:
                node = node.left
            else:
                node = node.right
        return node if node != self.NIL else None

    def insert(self, key, value):
        new_node = RBNode(key=key, value=value, left=self.NIL, right=self.NIL)
        if self.root == self.NIL:
            self.root = new_node
            self.root.color = 'BLACK'
            return new_node
        node = self.root
        parent = None
        while node != self.NIL:
            parent = node
            if key < node.key:
                node = node.left
            else:
                node = node.right
        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        return new_node

    def minimum(self, start_node=None):
        node = start_node if start_node else self.root
        if node == self.NIL:
            return None
        while node.left != self.NIL:
            node = node.left
        return node if node.value['valid'] else self.successor(node)

    def successor(self, node):
        if node.right != self.NIL:
            return self.minimum(node.right)
        parent = node.parent
        while parent and node == parent.right:
            node = parent
            parent = parent.parent
        return parent if parent and parent != self.NIL and parent.value['valid'] else None

    def delete(self, key):
        node = self.search(key)
        if node and node != self.NIL:
            node.value['valid'] = False
        return node

# Retroactive Priority Queue implementation
class RPQ:
    def __init__(self):
        self.T_ins = RBTree()  # Insertions: key = (distance, time)
        self.T_d_m = RBTree()  # Deletions: key = time
        self.node_map: Dict[int, list] = {}  # vertex -> list of nodes

    def invoke_insert(self, vertex: int, time: float, dist: float, pred: int) -> 'PQNode':
        node = PQNode(vertex=vertex, time=time, dist=dist, pred=pred, valid=True)
        key = (dist, time)
        rb_node = self.T_ins.insert(key, {'node': node, 'valid': True})
        if vertex not in self.node_map:
            self.node_map[vertex] = []
        self.node_map[vertex].append(rb_node)
        return node

    def invoke_del_min(self, time: float) -> Optional[dict]:
        min_node = self.find_min()
        if not min_node:
            return None
        min_node.value['valid'] = False
        self.T_d_m.insert(time, {'node': min_node, 'valid': True})
        return {'vertex': min_node.value['node'].vertex, 'dist': min_node.value['node'].dist, 'pred': min_node.value['node'].pred}

    def find_min(self) -> Optional['RBNode']:
        min_node = self.T_ins.minimum()
        return min_node if min_node and min_node.value['valid'] else None

    def get_vertex_node(self, vertex: int, active_only: bool = True) -> Optional['RBNode']:
        if vertex not in self.node_map:
            return None
        nodes = self.node_map[vertex]
        for node in reversed(nodes):  # Most recent first
            if node.value['valid'] and (not active_only or node.value['node'].valid):
                return node
        return None if active_only else nodes[-1] if nodes else None

    def revoke_del_min(self, time: float):
        del_node = self.T_d_m.search(time)
        if del_node and del_node != self.T_d_m.NIL and del_node.value['valid']:
            del_node.value['node'].value['valid'] = True
            del_node.value['valid'] = False

# Node for the priority queue
@dataclass
class PQNode:
    vertex: int
    time: float
    dist: float
    pred: int
    valid: bool

# Dynamic Dijkstra implementation
class DynamicDijkstra:
    def __init__(self, graph: list[list[float]]):
        self.graph = graph
        self.num_vertices = len(graph)
        self.dist = [math.inf] * self.num_vertices
        self.pred = [None] * self.num_vertices
        self.rpq = RPQ()
        self.current_time = 0.0
        self.deletion_times = {}

    def initial_dijkstra(self, source: int):
        self.dist[source] = 0
        self.rpq.invoke_insert(source, self.current_time, 0, None)
        self.current_time += 1
        self.process_rpq()

    def process_rpq(self):
        processed = set()
        while True:
            deleted_val = self.rpq.invoke_del_min(self.current_time)
            if not deleted_val:
                print(f"[{self.current_time}] RPQ is empty. Ending Dijkstra.")
                break
            u = deleted_val['vertex']
            dist_u = deleted_val['dist']
            print(f"[{self.current_time}] Popped node {u} with dist = {dist_u}")
            if u in processed:
                print(f"[{self.current_time}] Node {u} already processed. Skipping.")
                self.current_time += 1
                continue
            if dist_u != self.dist[u]:            #popped distance is not the current best, skip it
                print(f"[{self.current_time}] Outdated distance for node {u} (expected {self.dist[u]}, got {dist_u}). Skipping.")
                self.current_time += 1
                continue
            processed.add(u)
            self.deletion_times[u] = self.current_time
            for v in range(self.num_vertices):
                weight = self.graph[u][v]
                if weight != math.inf and u != v:
                    new_dist = self.dist[u] + weight
                    if new_dist < self.dist[v]:
                        print(f"[{self.current_time}] Updating node {v} from dist {self.dist[v]} to {new_dist} via {u}")
                        self.dist[v] = new_dist
                        self.pred[v] = u
                        self.rpq.invoke_insert(v, self.current_time, new_dist, u)
            self.current_time += 1
    def dynamic_dijkstra(self, et: int, eh: int, new_weight: float):
        N = self.rpq.get_vertex_node(eh, active_only=True)
        if N is None: #supposed to have no change
            new_dist = self.dist[et] + new_weight
            if new_dist < self.dist[eh]:
                self.dist[eh] = new_dist
                self.pred[eh] = et
                self.rpq.invoke_insert(eh, self.current_time, new_dist, et)
                self.current_time += 1
            return
        if N.value['node'].valid:
            if self.pred[eh] == et:
                new_dist = self.dist[et] + new_weight
                self.dist[eh] = new_dist
                self.rpq.invoke_insert(eh, self.current_time, new_dist, et)
                self.current_time += 1
        else:
            del_time = self.deletion_times.get(eh)
            if del_time is not None:
                self.rpq.revoke_del_min(del_time)
                if self.pred[eh] == et:
                    new_dist = self.dist[et] + new_weight
                    self.dist[eh] = new_dist
                    self.rpq.invoke_insert(eh, self.current_time, new_dist, et)
                    self.current_time += 1
        processed = set()
        while True:
            n = self.rpq.invoke_del_min(self.current_time)
            if not n:
                break
            u = n['vertex']
            if u in processed:
                self.current_time += 1
                continue
            processed.add(u)
            self.deletion_times[u] = self.current_time
            if n['dist'] != self.dist[u]:      
                self.current_time += 1
                continue
            for v in range(self.num_vertices):
                weight = self.graph[u][v]
                if weight != math.inf and u != v:
                    if self.pred[v] == eh:
                        self.dist[v] = math.inf
                        self.pred[v] = None
                    new_dist = self.dist[u] + weight
                    if new_dist < self.dist[v]:
                        self.dist[v] = new_dist
                        self.pred[v] = u
                        self.rpq.invoke_insert(v, self.current_time, new_dist, u)
                        self.current_time += 1
            self.current_time += 1
    # def dynamic_dijkstra(self, et: int, eh: int, new_weight: float):
    #     N = self.rpq.get_vertex_node(eh, active_only=True)
    #     if N is None: #supposed to have no change
    #         return
    #     if N.value['node'].valid:
    #         if self.pred[eh] == et:
    #             new_dist = self.dist[et] + new_weight
    #             self.dist[eh] = new_dist
    #             self.rpq.invoke_insert(eh, self.current_time, new_dist, et)
    #             self.current_time += 1
    #         else:
    #             new_dist = self.dist[et] + new_weight
    #             if new_dist < self.dist[eh]:#et becomes eh
    #                 self.dist[eh] = new_dist
    #                 self.pred[eh] = et
    #                 self.rpq.invoke_insert(eh, self.current_time, new_dist, et)
    #                 self.current_time += 1
    #     else:
    #         del_time = self.deletion_times.get(eh)
    #         if del_time is not None:
    #             self.rpq.revoke_del_min(del_time)
    #             if self.pred[eh] == et:
    #                 new_dist = self.dist[et] + new_weight
    #                 self.dist[eh] = new_dist
    #                 self.rpq.invoke_insert(eh, self.current_time, new_dist, et)
    #                 self.current_time += 1
    #     processed = set()
    #     while True:
    #         n = self.rpq.invoke_del_min(self.current_time)
    #         if not n:
    #             break
    #         u = n['vertex']
    #         if u in processed:
    #             self.current_time += 1
    #             continue
    #         processed.add(u)
    #         self.deletion_times[u] = self.current_time
    #         if n['dist'] != self.dist[u]:      
    #             self.current_time += 1
    #             continue
    #         for v in range(self.num_vertices):
    #             weight = self.graph[u][v]
    #             if weight != math.inf and u != v:
    #                 if self.pred[v] == eh:
    #                     self.dist[v] = math.inf
    #                     self.pred[v] = None
    #                 new_dist = self.dist[u] + weight
    #                 if new_dist < self.dist[v]:
    #                     self.dist[v] = new_dist
    #                     self.pred[v] = u
    #                     self.rpq.invoke_insert(v, self.current_time, new_dist, u)
    #                     self.current_time += 1
    #         self.current_time += 1

    def update_edge(self, u: int, v: int, new_weight: float):
        old_weight = self.graph[u][v]
        self.graph[u][v] = new_weight
        # self.graph[v][u] = new_weight  # Ensure undirected
        if new_weight < old_weight:
            self.dynamic_dijkstra(u, v, new_weight)
        elif new_weight > old_weight:
            self.dynamic_dijkstra(u, v, new_weight)

# Example usage
if __name__ == "__main__":
    inf = math.inf
    # graph = [
    #     [0,   2,   5,   4,   inf, inf, inf, inf],#0
    #     [2,   0,   2,   4,   7,   inf, 12,  inf],#1
    #     [5,   2,   0,   1,   4,   3,   inf, inf],#2
    #     [4,   4,   1,   0,   inf, 4,   inf, 7],  #3
    #     [inf, 7,   4,   inf, 0,   4,   inf, 5],  #4
    #     [inf, inf, 3,   4,   4,   0,   inf, 7],  #5
    #     [inf, 12,  inf, inf, inf, inf, 0,   3],  #6
    #     [inf, inf, inf, 7,   5,   7,   3,   0]   #7
    # ]
    graph = [
        [0,   2,   5,   4,   inf, inf, inf, inf],  # Node 0
        [inf, 0,   2,   4,   7,   inf, inf,  inf],  # Node 1
        [inf, inf, 0,   1,   inf, inf, inf, inf],  # Node 2
        [inf, inf, inf, 0,   inf,   4,   3, inf],  # Node 3
        [inf, inf, inf, inf, 0, inf, inf,   5],  # Node 4
        [inf, inf, inf, inf, inf, 0, inf,   7],  # Node 5
        [inf, inf, inf, inf, inf, inf, 0,   3],  # Node 6
        [inf, inf, inf, inf, inf, inf, inf, 0]  # Node 7
    ]

    dd = DynamicDijkstra(graph)
    dd.initial_dijkstra(0)
    print("Initial distances:", [round(d, 1) for d in dd.dist])
    print("Predecessors:", dd.pred)

    dd.update_edge(5, 7, 1)
    print("After update (2,4) to 5:", [round(d, 1) for d in dd.dist])
    print("Predecessors:", dd.pred)


    dd.update_edge(5, 7, 11)
    print("After update (2,4) to 5:", [round(d, 1) for d in dd.dist])
    print("Predecessors:", dd.pred)

    # dd.update_edge(1, 2, 1)
    # print("After update (1,2) to 1:", [round(d, 1) for d in dd.dist])
    # print("Predecessors:", dd.pred)

    # dd.update_edge(0, 3, 1)
    # print("After update (0,3) to 1:", [round(d, 1) for d in dd.dist])
    # print("Predecessors:", dd.pred)