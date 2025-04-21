import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import os
from collections import deque, defaultdict

if not os.path.exists('plots'):
    os.makedirs('plots')

@dataclass
class PQNode:
    vertex: int
    time: float
    dist: float
    pred: Optional[int]
    valid: bool = True
    deleted_by_node: Optional['RBNode'] = None
    tins_node: Optional['RBNode'] = field(default=None, repr=False)

@dataclass
class RBNode:
    key: tuple
    value: Any
    left: Optional['RBNode'] = None
    right: Optional['RBNode'] = None
    color: str = 'RED'
    parent: Optional['RBNode'] = None

    def __lt__(self, other):
        if not isinstance(other, RBNode):
            return NotImplemented
        if self.key is None or other.key is None:
            return False
        return self.key < other.key

class RBTree:
    def __init__(self):
        self.NIL = RBNode(key=(math.inf,), value=None, color='BLACK')
        self.NIL.left = self.NIL
        self.NIL.right = self.NIL
        self.NIL.parent = self.NIL
        self.root = self.NIL

    def search(self, key: tuple) -> Optional[RBNode]:
        node = self.root
        while node != self.NIL and node.key != key:
            if node.key is None or key < node.key:
                node = node.left
            else:
                node = node.right
        return node if node != self.NIL else None

    def insert(self, key: tuple, value: Any) -> RBNode:
        if key is None:
            raise ValueError("Cannot insert node with None key")
        new_node = RBNode(key=key, value=value, left=self.NIL, right=self.NIL, parent=self.NIL)
        if isinstance(value, PQNode):
            value.tins_node = new_node
        parent = self.NIL
        current = self.root
        while current != self.NIL:
            parent = current
            if current.key is None or new_node.key < current.key:
                current = current.left
            else:
                current = current.right
        new_node.parent = parent
        if parent == self.NIL:
            self.root = new_node
        elif parent.key is None or new_node.key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        new_node.color = 'RED'
        self.root.color = 'BLACK'
        return new_node

    def minimum(self, start_node: Optional[RBNode] = None) -> Optional[RBNode]:
        node = start_node if start_node else self.root
        if node is None or node == self.NIL:
            return None
        while node.left != self.NIL:
            node = node.left
        return node if node != self.NIL else None

    def successor(self, node: Optional[RBNode]) -> Optional[RBNode]:
        if node is None or node == self.NIL:
            return None
        if node.right != self.NIL:
            succ = self.minimum(node.right)
            return succ
        parent = node.parent
        while parent != self.NIL and node == parent.right:
            node = parent
            parent = parent.parent
        return parent if parent != self.NIL else None

    def remove_node(self, key_to_remove: tuple) -> Optional[RBNode]:
        node = self.search(key_to_remove)
        if not node:
            return None
        original_node = node
        if node.left == self.NIL:
            transplant_target = node.right
            self._transplant_tree(node, node.right)
        elif node.right == self.NIL:
            transplant_target = node.left
            self._transplant_tree(node, node.left)
        else:
            succ = self.minimum(node.right)
            if succ is None:
                print(f"ERROR: Successor not found for node {node.key} during remove!")
                return None
            if succ.parent != node:
                self._transplant_tree(succ, succ.right)
                if node.right != self.NIL:
                    succ.right = node.right
                    succ.right.parent = succ
            self._transplant_tree(node, succ)
            if node.left != self.NIL:
                succ.left = node.left
                succ.left.parent = succ
        return original_node

    def _transplant_tree(self, u, v):
        if u.parent == self.NIL:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v != self.NIL:
            v.parent = u.parent

    def inorder(self) -> list[tuple[tuple, Any]]:
        result = []
        def _inorder(node):
            if node != self.NIL:
                _inorder(node.left)
                result.append((node.key, node.value))
                _inorder(node.right)
        _inorder(self.root)
        return result

    def search_min_greater_equal(self, key: tuple) -> Optional[RBNode]:
        node = self.root
        result = None
        while node != self.NIL:
            if node.key is not None and node.key >= key:
                result = node
                node = node.left
            else:
                node = node.right
        return result

    def search_max_less(self, key: tuple) -> Optional[RBNode]:
        node = self.root
        result = None
        while node != self.NIL:
            if node.key is not None and node.key < key:
                result = node
                node = node.right
            else:
                node = node.left
        return result

class RPQ:
    def __init__(self):
        self.T_ins = RBTree()
        self.T_d_m = RBTree()

    def find_first_inconsistent_delete(self, new_node_key: tuple) -> Optional[RBNode]:
        insert_time = new_node_key[1]
        potential_del_node = self.T_d_m.search_min_greater_equal((insert_time,))
        while potential_del_node:
            deleted_pq_node = potential_del_node.value
            if deleted_pq_node and deleted_pq_node.tins_node and deleted_pq_node.tins_node.key is not None:
                if new_node_key is not None and new_node_key < deleted_pq_node.tins_node.key:
                    return potential_del_node
            elif not deleted_pq_node:
                print(f"RPQ WARNING: T_d_m node {potential_del_node.key} has None value!")
            elif not deleted_pq_node.tins_node:
                print(f"RPQ WARNING: T_d_m value {deleted_pq_node} lacks .tins_node link!")
            elif deleted_pq_node.tins_node.key is None:
                print(f"RPQ WARNING: T_ins node linked from T_d_m node {potential_del_node.key} has None key!")
            potential_del_node = self.T_d_m.successor(potential_del_node)
        return None

    def invoke_insert(self, vertex: int, time: float, dist: float, pred: Optional[int]) -> Optional[RBNode]:
        node = PQNode(vertex=vertex, time=time, dist=dist, pred=pred, valid=True)
        key = (dist, time)
        rb_node_ins = self.T_ins.insert(key, node)
        inconsistent_del_node = self.find_first_inconsistent_delete(key)
        if inconsistent_del_node:
            print(f"RPQ: Insertion at time {time:.1f} causes inconsistency with deletion at {inconsistent_del_node.key[0]:.1f}")
            return inconsistent_del_node
        else:
            return None

    def invoke_del_min(self, time: float) -> Optional[PQNode]:
        min_rb_node = self.find_valid_min_ins()
        if not min_rb_node:
            print(f"RPQ: T_ins is empty or has no valid nodes.")
            return None
        min_pq_node = min_rb_node.value
        if not min_pq_node:
            print(f"RPQ ERROR: Valid min RBNode {min_rb_node.key} has None value!")
            return None
        min_pq_node.valid = False
        print(f"RPQ: Marking T_ins node invalid: Key={min_rb_node.key}, Value={min_pq_node}")
        del_key = (time,)
        del_rb_node = self.T_d_m.insert(del_key, min_pq_node)
        if del_rb_node and del_rb_node != self.T_d_m.NIL:
            min_pq_node.deleted_by_node = del_rb_node
        return min_pq_node

    def find_valid_min_ins(self) -> Optional[RBNode]:
        node = self.T_ins.minimum()
        while node:
            if node.value and isinstance(node.value, PQNode) and node.value.valid:
                return node
            node = self.T_ins.successor(node)
        return None

    def revoke_insert(self, key: tuple) -> Optional[RBNode]:
        print(f"RPQ: Revoke Insert for key={key}")
        node_to_revoke = self.T_ins.search(key)
        if not node_to_revoke:
            print(f"RPQ: Revoke Insert failed: Node with key {key} not found in T_ins.")
            return None
        revoked_pq_node = node_to_revoke.value
        if not isinstance(revoked_pq_node, PQNode):
            return None
        if not revoked_pq_node.valid:
            print(f"RPQ: Revoke Insert: Node {key} value was already invalid.")
        print(f"RPQ: Marking T_ins node {key} as invalid.")
        revoked_pq_node.valid = False
        insert_time = key[1]
        potential_del_node = self.T_d_m.search_min_greater_equal((insert_time,))
        while potential_del_node:
            deleted_pq_node = potential_del_node.value
            if deleted_pq_node == revoked_pq_node:
                print(f"RPQ: Revoked insert {key} invalidates deletion at {potential_del_node.key[0]:.1f}")
                return potential_del_node
            potential_del_node = self.T_d_m.successor(potential_del_node)
        return None

    def revoke_del_min(self, time: float, mark_tins_valid: bool = True) -> Optional[RBNode]:
        print(f"RPQ: Revoke Del_Min for time={time:.1f} (Mark T_ins Valid: {mark_tins_valid})")
        del_key = (time,)
        node_in_tdm = self.T_d_m.search(del_key)
        if not node_in_tdm:
            print(f"RPQ: Revoke Del_Min failed: No deletion recorded at time {time:.1f}")
            return None
        pq_node_to_revive = node_in_tdm.value
        if not isinstance(pq_node_to_revive, PQNode):
            print(f"RPQ ERROR: T_d_m node {del_key} has non-PQNode value: {type(pq_node_to_revive)}")
            self.T_d_m.remove_node(del_key)
            return None
        original_tins_node = pq_node_to_revive.tins_node
        if original_tins_node and original_tins_node != self.T_ins.NIL:
            tins_node_key_str = str(original_tins_node.key)
            if mark_tins_valid:
                if not pq_node_to_revive.valid:
                    pq_node_to_revive.valid = True
                    print(f"RPQ: Marked original T_ins node {tins_node_key_str} back to valid.")
                else:
                    print(f"RPQ: Warning - PQNode {pq_node_to_revive} was already marked valid?")
                pq_node_to_revive.deleted_by_node = None
            else:
                print(f"RPQ: Revoke Del_Min: Skipping marking T_ins node {tins_node_key_str} as valid.")
                pq_node_to_revive.deleted_by_node = None
        else:
            print(f"RPQ: Revoke Del_Min Error/Warning: Could not find original T_ins node link for PQNode {pq_node_to_revive}")
            if pq_node_to_revive:
                pq_node_to_revive.deleted_by_node = None
        if mark_tins_valid:
            revived_node_key = None
            if original_tins_node and original_tins_node.key is not None:
                revived_node_key = original_tins_node.key
            if revived_node_key is not None:
                potential_del_node = self.T_d_m.search_min_greater_equal((time + 1e-9,))
                while potential_del_node:
                    deleted_pq_node = potential_del_node.value
                    if (deleted_pq_node and deleted_pq_node.tins_node and
                        deleted_pq_node.tins_node.key is not None):
                        if revived_node_key < deleted_pq_node.tins_node.key:
                            print(f"RPQ: Revoked deletion at {time:.1f} (revived {revived_node_key}) invalidates deletion at {potential_del_node.key[0]:.1f}")
                            return potential_del_node
                    potential_del_node = self.T_d_m.successor(potential_del_node)
            else:
                print(f"RPQ: Skipping inconsistency check after revoke; revived node key unavailable.")
        return None

    def find_vertex_rbnode_in_tins(self, vertex: int, active_only: bool = True) -> Optional[RBNode]:
        latest_node: Optional[RBNode] = None
        latest_time = -1.0
        nodes = self.T_ins.inorder()
        for key, pq_node in nodes:
            if isinstance(pq_node, PQNode) and pq_node.vertex == vertex:
                is_candidate = (not active_only) or pq_node.valid
                if is_candidate and pq_node.time > latest_time:
                    latest_time = pq_node.time
                    if pq_node.tins_node and pq_node.tins_node != self.T_ins.NIL:
                        latest_node = pq_node.tins_node
                    else:
                        print(f"RPQ WARNING: PQNode for vertex {vertex} at time {pq_node.time} has missing tins_node link.")
                        latest_node = None
        return latest_node

    def print_state(self, time_step):
        print(f"\n=== RPQ STATE at Time Step {time_step:.1f} ===")
        print("\n-- T_ins (Insertions: Key=(dist, time), Value=PQNode) --")
        valid_count_ins = 0
        nodes_ins = self.T_ins.inorder()
        if not nodes_ins: print("  (empty)")
        for key, pq_node in nodes_ins:
            if isinstance(pq_node, PQNode):
                valid_str = "VALID" if pq_node.valid else "INVALID"
                del_by_key_str = "N/A"
                if pq_node.deleted_by_node and pq_node.deleted_by_node.key is not None:
                    del_by_key_str = f"{pq_node.deleted_by_node.key[0]:.1f}"
                key_str = f"({key[0]:.1f}, {key[1]:.1f})" if isinstance(key, tuple) and len(key) == 2 else str(key)
                print(f"  Key: {key_str}, Vtx: {pq_node.vertex}, Pred: {pq_node.pred}, Status: {valid_str}, Del@: {del_by_key_str}")
                if pq_node.valid: valid_count_ins += 1
            else:
                print(f"  Key: {key}, Value Type: {type(pq_node)} (Expected PQNode)")
        print(f"  Total T_ins Nodes: {len(nodes_ins)}, Valid: {valid_count_ins}")
        print("\n-- T_d_m (Deletions: Key=(del_time,), Value=PQNode Deleted) --")
        nodes_del = self.T_d_m.inorder()
        if not nodes_del: print("  (empty)")
        for key, pq_node_deleted in nodes_del:
            if isinstance(pq_node_deleted, PQNode):
                orig_ins_key_str = "?"
                if pq_node_deleted.tins_node and pq_node_deleted.tins_node.key is not None:
                    tins_key = pq_node_deleted.tins_node.key
                    orig_ins_key_str = f"({tins_key[0]:.1f}, {tins_key[1]:.1f})" if isinstance(tins_key, tuple) and len(tins_key) == 2 else str(tins_key)
                del_time_str = f"{key[0]:.1f}" if isinstance(key, tuple) and len(key) > 0 else str(key)
                print(f"  Del Time: {del_time_str}, Deleted-> Vtx: {pq_node_deleted.vertex}, Dist: {pq_node_deleted.dist:.1f}, Pred: {pq_node_deleted.pred}, Orig Ins Key: {orig_ins_key_str}")
            else:
                print(f"  Key: {key}, Value Type: {type(pq_node_deleted)} (Expected PQNode)")
        print(f"  Total T_d_m Nodes: {len(nodes_del)}")
        print("=" * 30)

class DynamicDijkstra:
    def __init__(self, graph: list[list[float]]):
        self.graph = [row[:] for row in graph]
        self.num_vertices = len(graph)
        self.dist = [math.inf] * self.num_vertices
        self.pred: List[Optional[int]] = [None] * self.num_vertices
        self.rpq = RPQ()
        self.current_time = 0.0
        self.processed_times: Dict[int, List[float]] = {}
        self.update_counter = 0
        self.G = nx.DiGraph()
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if self.graph[i][j] != math.inf:
                    self.G.add_edge(i, j, weight=self.graph[i][j])
        self.pos = nx.spring_layout(self.G)

    def _increment_time(self):
        self.current_time += 1.0

    def initial_dijkstra(self, source: int):
        print(f"\n--- Initial Dijkstra from source {source} ---")
        if source < 0 or source >= self.num_vertices:
            print(f"ERROR: Invalid source node {source}")
            return
        self.dist[source] = 0.0
        self.rpq.invoke_insert(source, self.current_time, 0.0, None)
        self._increment_time()
        self.propagate_changes()
        print(f"--- Initial Dijkstra Complete ---")
        self.plot_graph(title="After initial Dijkstra")

    def plot_graph(self, highlight_edge=None, title=""):
        plt.figure(figsize=(10,8))
        tree_edges = [(self.pred[v], v) for v in range(self.num_vertices) if self.pred[v] is not None and self.G.has_edge(self.pred[v], v)]
        all_edges = list(self.G.edges())
        non_tree_edges = [e for e in all_edges if e not in tree_edges]
        nx.draw_networkx_edges(self.G, self.pos, edgelist=non_tree_edges, edge_color='gray', arrows=True)
        nx.draw_networkx_edges(self.G, self.pos, edgelist=tree_edges, edge_color='blue', arrows=True)
        if highlight_edge and self.G.has_edge(*highlight_edge):
            nx.draw_networkx_edges(self.G, self.pos, edgelist=[highlight_edge], edge_color='red', arrows=True)
        nx.draw_networkx_nodes(self.G, self.pos, node_color='lightblue')
        node_labels = {i: f"{i}\n({self.dist[i]:.1f})" if self.dist[i] != math.inf else f"{i}\n(inf)" for i in range(self.num_vertices)}
        nx.draw_networkx_labels(self.G, self.pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels)
        plt.title(title)
        file_name = title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace('.', '') + '.png'
        plt.savefig(f"plots/{file_name}")
        plt.close()

    def propagate_changes(self, start_time: Optional[float] = None):
        time_to_process = start_time if start_time is not None else self.current_time
        processed_in_this_run = set()
        while True:
            min_rb_node = self.rpq.find_valid_min_ins()
            if not min_rb_node:
                print(f"[{time_to_process:.1f}] RPQ has no more valid nodes. Propagation ends.")
                break
            min_pq_node = min_rb_node.value
            if not isinstance(min_pq_node, PQNode):
                print(f"ERROR [{time_to_process:.1f}]: Invalid value type {type(min_pq_node)} found in T_ins for key {min_rb_node.key}. Stopping propagation.")
                break
            u = min_pq_node.vertex
            dist_u_rpq = min_pq_node.dist
            if u in processed_in_this_run:
                print(f"WARNING [{time_to_process:.1f}]: Node {u} encountered again in this propagation run. Breaking loop.")
                break
            processed_in_this_run.add(u)
            deleted_node = self.rpq.invoke_del_min(time_to_process)
            if not deleted_node or deleted_node != min_pq_node:
                print(f"WARNING [{time_to_process:.1f}]: invoke_del_min did not return the expected node {min_pq_node}. Proceeding cautiously.")
            if self.dist[u] < dist_u_rpq:
                print(f"[{time_to_process:.1f}] Stale RPQ entry for {u}. Known dist={self.dist[u]:.1f} < RPQ dist={dist_u_rpq:.1f}. Skipping relaxation.")
                self._increment_time()
                time_to_process = self.current_time
                processed_in_this_run.clear()
                continue
            if u not in self.processed_times:
                self.processed_times[u] = [time_to_process]
            else:
                self.processed_times[u].append(time_to_process)
            print(f"[{time_to_process:.1f}] Processing node {u} (Dist={dist_u_rpq:.1f}).")
            if dist_u_rpq < self.dist[u]:
                print(f"[{time_to_process:.1f}] Updating main dist for {u} from {self.dist[u]:.1f} to {dist_u_rpq:.1f}")
                self.dist[u] = dist_u_rpq
                self.pred[u] = min_pq_node.pred
            for v in range(self.num_vertices):
                if u < self.num_vertices and v < self.num_vertices:
                    weight = self.graph[u][v]
                    if weight != math.inf and u != v:
                        if self.dist[u] != math.inf:
                            new_dist_v = self.dist[u] + weight
                            if new_dist_v < self.dist[v]:
                                self.dist[v] = new_dist_v
                                self.pred[v] = u
                                inconsistent_del = self.rpq.invoke_insert(v, time_to_process, new_dist_v, u)
                                if inconsistent_del:
                                    print(f"[!ALERT {time_to_process:.1f}!] Insertion for {v} conflicts with Del_Min at {inconsistent_del.key[0]:.1f}")
                        else:
                            print(f"[{time_to_process:.1f}] Skipping relaxation from {u}; its distance is infinity.")
                else:
                    print(f"ERROR [{time_to_process:.1f}]: Invalid vertex index during relaxation ({u}, {v})")
            self._increment_time()
            time_to_process = self.current_time
            processed_in_this_run.clear()

    def handle_edge_update(self, u: int, v: int, new_weight: float):
        print(f"\n--- Handling Edge Update ({u}, {v}) to New Weight={new_weight:.1f} at Time={self.current_time:.1f} ---")
        self.graph[u][v] = new_weight
        latest_node_v_rb = self.rpq.find_vertex_rbnode_in_tins(v, active_only=False)
        latest_pq_node_v = latest_node_v_rb.value if latest_node_v_rb else None

        if not latest_pq_node_v:
            print(f"[{self.current_time:.1f}] Vertex {v} not found in RPQ history.")
            if self.dist[u] != math.inf:
                potential_new_dist_v = self.dist[u] + new_weight
                if potential_new_dist_v < self.dist[v]:
                    print(f"[{self.current_time:.1f}] New edge creates better path to {v}.")
                    self.dist[v] = potential_new_dist_v
                    self.pred[v] = u
                    self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
                    self.propagate_changes(self.current_time)
            return

        if latest_pq_node_v.valid:
            print(f"[{self.current_time:.1f}] Vertex {v} is ACTIVE in RPQ.")
            new_dist_v = self.dist[u] + new_weight if self.dist[u] != math.inf else math.inf
            if self.pred[v] == u:
                print(f"[{self.current_time:.1f}] Path to {v} is via {u}. Updating.")
                if latest_node_v_rb:
                    self.rpq.revoke_insert(latest_node_v_rb.key)
                self.dist[v] = new_dist_v
                self.rpq.invoke_insert(v, self.current_time, new_dist_v, u)
                self.propagate_changes(self.current_time)
            else:
                if new_dist_v < self.dist[v]:
                    print(f"[{self.current_time:.1f}] New path to {v} via ({u},{v}) is better.")
                    self.dist[v] = new_dist_v
                    self.pred[v] = u
                    self.rpq.invoke_insert(v, self.current_time, new_dist_v, u)
                    self.propagate_changes(self.current_time)
            return

        print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
        deletion_times = sorted(self.processed_times.get(v, []), reverse=True)
        print("Processedtimes: ", self.processed_times)
        if not deletion_times:
            print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion times recorded!")
            return
        potential_new_dist_v = self.dist[u] + new_weight if self.dist[u] != math.inf else math.inf
        path_via_u_used = (self.pred[v] == u)
        updated = False
        newdist = potential_new_dist_v

        if path_via_u_used and potential_new_dist_v < self.dist[v]:
            print(f"[{self.current_time:.1f}] Processed path to {v} was via {u} and is still small.Updating")
            self.dist[v] = potential_new_dist_v
            self.pred[v] = u
            self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
            updated = True
        else:
            print(f"[{self.current_time:.1f}] Path used to process {v} was NOT via {u} or it may not be the best one")
            print(f"[{self.current_time:.1f}] Checking historical paths to {v}...")
            best_historical_dist = math.inf
            best_historical_pred = None
            best_del_time = None
            for del_time in deletion_times:
                rb_node = self.rpq.T_d_m.search((del_time,))
                if rb_node and rb_node.value:
                    historical_node = rb_node.value
                    if historical_node.pred != u:
                        if best_historical_pred:
                            if historical_node.dist < best_historical_dist and self.dist[best_historical_pred] == self.dist[best_historical_pred]:
                                best_historical_dist = historical_node.dist
                                best_historical_pred = historical_node.pred
                                best_del_time = del_time
            if best_historical_dist < potential_new_dist_v:
                print(f"[{self.current_time:.1f}] Found better historical path to {v} via {best_historical_pred} (dist {best_historical_dist:.1f})")
                print(f"[{self.current_time:.1f}] Revoking deletion at {best_del_time:.1f}")
                inconsistent_del = self.rpq.revoke_del_min(best_del_time)
                if inconsistent_del:
                    print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")
                self.dist[v] = best_historical_dist
                self.pred[v] = best_historical_pred
                newdist = best_historical_dist
                self.rpq.invoke_insert(v, self.current_time, best_historical_dist, best_historical_pred)
                self.propagate_changes(self.current_time)
                updated = True
            else:
                print(f"[{self.current_time:.1f}] No better path found for {v}. Maintaining new distance.")
                self.dist[v] = potential_new_dist_v
                self.pred[v] = u
                newdist = potential_new_dist_v
                self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
                updated = True

        print(f"#propagating changes to ancestors of {v}")
        q = deque([[v, newdist]])
        children = defaultdict(list)
        graph = self.graph
        for i in range(len(graph)):
            for j in range(len(graph[i])):
                if graph[i][j] != math.inf and graph[i][j] != 0:
                    children[i].append((j, graph[i][j]))
                elif i not in children:
                    children[i] = []
        vertices = list(children.keys())
        while q:
            currents = q.popleft()
            print(f"Processing: {currents}")
            distrn = currents[1]
            checked = []
            for child, weight in children[currents[0]]:
                print(f"Checking child {child} from parent {currents[0]} with weight {weight}")
                new_dist_c = math.inf
                best_historical_dist = math.inf
                best_historical_pred = None
                best_del_time = None
                for deletion_time in self.processed_times.get(child, []):
                    rb_node = self.rpq.T_d_m.search((deletion_time,))
                    if rb_node.value.pred and rb_node.value.pred not in checked:
                        checked.append(rb_node.value.pred)
                    if rb_node and rb_node.value:
                        historical_node = rb_node.value
                        best_historical_pred = historical_node.pred
                        if historical_node.pred != currents[0]:
                            if best_historical_pred:
                                if (historical_node.dist < best_historical_dist or historical_node.dist < self.graph[currents[0]][child] + distrn) and (historical_node.dist - graph[best_historical_pred][child] == self.dist[best_historical_pred]):
                                    best_historical_dist = historical_node.dist
                                    best_historical_pred = historical_node.pred
                                    best_del_time = deletion_time
                if best_historical_dist != math.inf:
                    print(f"[{self.current_time:.1f}] Better historical path to {child} via {best_historical_pred} (dist {best_historical_dist:.1f})")
                    self.dist[child] = best_historical_dist
                    self.pred[child] = best_historical_pred
                    new_dist_c = best_historical_dist
                    if best_del_time is not None:
                        print(f"[{self.current_time:.1f}] Revoking deletion at {best_del_time:.1f}")
                        inconsistent_del = self.rpq.revoke_del_min(best_del_time)
                        if inconsistent_del:
                            print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")
                else:
                    bestpred = None
                    bestdist = math.inf
                    for parent in vertices:
                        if parent not in checked:
                            if self.dist[parent] + self.graph[parent][child] < self.graph[currents[0]][child] + distrn:
                                bestpred = parent
                                bestdist = self.dist[parent] + graph[parent][child]
                    if bestdist < self.graph[currents[0]][child] + distrn:
                        self.dist[child] = bestdist
                        self.pred[child] = bestpred
                        self.rpq.invoke_insert(child, self.current_time, bestdist, bestpred)
                        self.propagate_changes(self.current_time)
                        new_dist_c = bestdist
                    else:
                        print(f"[{self.current_time:.1f}] No better historical path for {child}. Updating normally.")
                        new_dist_c = self.graph[currents[0]][child] + distrn
                        self.dist[child] = new_dist_c
                        self.pred[child] = currents[0]
                        self.rpq.invoke_insert(child, self.current_time, new_dist_c, currents[0])
                        self.propagate_changes(self.current_time)
                q.append([child, new_dist_c])
        self.propagate_changes(self.current_time)

    def update_edge(self, u: int, v: int, new_weight: float):
        if u < 0 or u >= self.num_vertices or v < 0 or v >= self.num_vertices:
            print(f"Error: Invalid vertices ({u}, {v})")
            return
        if new_weight < 0:
            print("Error: Negative weights not supported.")
            return
        if new_weight == math.inf and self.graph[u][v] == math.inf:
            print(f"Edge ({u}, {v}) already non-existent. No update.")
            return
        print(f"\n>>> Updating Edge ({u}, {v}) from {self.graph[u][v]:.1f} to {new_weight:.1f}")
        self.update_counter += 1
        title_before = f"Before update {self.update_counter}: edge ({u},{v}) to {new_weight:.1f}"
        self.plot_graph(highlight_edge=(u,v), title=title_before)
        old_weight = self.graph[u][v]
        if old_weight == math.inf and new_weight != math.inf:
            self.G.add_edge(u, v, weight=new_weight)
        elif old_weight != math.inf and new_weight == math.inf:
            self.G.remove_edge(u, v)
        else:
            self.G[u][v]['weight'] = new_weight
        self.graph[u][v] = new_weight
        self.handle_edge_update(u, v, new_weight)
        title_after = f"After update {self.update_counter}: edge ({u},{v}) to {new_weight:.1f}"
        self.plot_graph(highlight_edge=(u,v), title=title_after)

if __name__ == "__main__":
    inf = math.inf
    graph = [
        [inf, 2, 5, 4, inf, inf, inf, inf],
        [inf, inf, 2, 4, 7, inf, inf, inf],
        [inf, inf, inf, 1, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, 4, 3, inf],
        [inf, inf, inf, inf, inf, inf, inf, 5],
        [inf, inf, inf, inf, inf, inf, inf, 7],
        [inf, inf, inf, inf, inf, inf, inf, 3],
        [inf, inf, inf, inf, inf, inf, inf, inf]
    ]
    dd = DynamicDijkstra(graph)
    source_node = 0
    dd.initial_dijkstra(source_node)
    print("\nFINAL STATE AFTER INITIAL DIJKSTRA:")
    print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    print("Predecessors:", dd.pred)
    print("-" * 50)
    dd.update_edge(5, 7, 1.0)
    print("\nFINAL STATE AFTER UPDATE (5, 7) -> 1:")
    print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    print("Predecessors:", dd.pred)
    print("-" * 50)
    dd.update_edge(3, 6, 1.5)

    print("\nFINAL STATE AFTER UPDATE (1, 2) -> 0.5:")
    print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    print("Predecessors:", dd.pred)
    print("-" * 50)