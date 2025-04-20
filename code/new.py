import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

# --- Data Structures ---

@dataclass
class PQNode:
    vertex: int
    time: float  # Insertion time into the conceptual PQ
    dist: float
    pred: Optional[int]
    valid: bool = True # Is this specific entry currently considered valid?
    deleted_by_node: Optional['RBNode'] = None # Link to T_d_m node that invalidated this
    tins_node: Optional['RBNode'] = field(default=None, repr=False) # Link back to RBNode in T_ins

@dataclass
class RBNode:
    key: tuple
    value: Any # PQNode for T_ins, PQNode (deleted one) for T_d_m
    left: Optional['RBNode'] = None
    right: Optional['RBNode'] = None
    color: str = 'RED'
    parent: Optional['RBNode'] = None

    def __lt__(self, other):
        if not isinstance(other, RBNode):
            return NotImplemented        # Handle potential None keys during initial NIL setup if necessary
        if self.key is None or other.key is None:              # Or raise an error, depends on how NIL key is handled
             return False
        return self.key < other.key

class RBTree:
    def __init__(self):
        self.NIL = RBNode(key=(math.inf,), value=None, color='BLACK')        # Sentinel NIL node - Ensure key is comparable if used in comparisons
        self.NIL.left = self.NIL
        self.NIL.right = self.NIL
        self.NIL.parent = self.NIL
        self.root = self.NIL

    def search(self, key: tuple) -> Optional[RBNode]:
        node = self.root
        while node != self.NIL and node.key != key:            # Added check for node.key being None defensively
            if node.key is None or key < node.key:
                node = node.left
            else:
                node = node.right
        return node if node != self.NIL else None        # Return None if not found (node is NIL)


    def insert(self, key: tuple, value: Any) -> RBNode:
        if key is None:        # Ensure key is not None before insertion
            raise ValueError("Cannot insert node with None key")
        new_node = RBNode(key=key, value=value, left=self.NIL, right=self.NIL, parent=self.NIL)
        if isinstance(value, PQNode):
             value.tins_node = new_node
        parent = self.NIL
        current = self.root
        while current != self.NIL:
            parent = current            # Added check for current.key being None defensively
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
        new_node.color = 'RED' # Basic coloring
        # No fixup implemented
        self.root.color = 'BLACK' # Ensure root is black
        return new_node

    def minimum(self, start_node: Optional[RBNode] = None) -> Optional[RBNode]:
        node = start_node if start_node else self.root
        # Check if start_node is None or NIL
        if node is None or node == self.NIL:
            return None
        while node.left != self.NIL:
            node = node.left
        # Return node itself, could be NIL if tree was initially empty
        # but caught by the initial check if start_node was root.
        # If start_node was provided, it might point to NIL.
        return node if node != self.NIL else None

    def successor(self, node: Optional[RBNode]) -> Optional[RBNode]:
        if node is None or node == self.NIL:        # Check input node
             return None
        if node.right != self.NIL:            # Minimum of right subtree exists
            succ = self.minimum(node.right)            # minimum() already handles returning None if right subtree is empty/NIL
            return succ
        parent = node.parent        # Go up until we are a left child
        while parent != self.NIL and node == parent.right:
            node = parent
            parent = parent.parent
        return parent if parent != self.NIL else None        # Return parent if it's not NIL, otherwise None

    def remove_node(self, key_to_remove: tuple) -> Optional[RBNode]:
        node = self.search(key_to_remove)        # Check if node was found
        if not node: # Handles None case from search
            return None
        original_node = node # Keep track of the node being removed
        # --- Basic BST Deletion (No RB Fixup) ---
        if node.left == self.NIL:
            transplant_target = node.right
            self._transplant_tree(node, node.right)
        elif node.right == self.NIL:
            transplant_target = node.left
            self._transplant_tree(node, node.left)
        else:
            succ = self.minimum(node.right)
            # Check if successor was found
            if succ is None: # Should not happen if node.right != NIL, but defensive
                 # Handle error or unexpected state
                 print(f"ERROR: Successor not found for node {node.key} during remove!")
                 return None # Or raise exception

            # No need to store transplant_target explicitly here for simple BST delete
            if succ.parent != node:
                self._transplant_tree(succ, succ.right)
                # Check if node.right is NIL before assigning parent
                if node.right != self.NIL:
                     succ.right = node.right
                     succ.right.parent = succ # Assign parent only if not NIL
            self._transplant_tree(node, succ)
            # Check if node.left is NIL before assigning parent
            if node.left != self.NIL:
                 succ.left = node.left
                 succ.left.parent = succ # Assign parent only if not NIL
        return original_node         # Return the node that contained the original key/value

    def _transplant_tree(self, u, v):        # Assume u is not NIL based on how remove_node calls this
        if u.parent == self.NIL:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v        # Assign parent to v only if v is not NIL
        if v != self.NIL:
            v.parent = u.parent

    def inorder(self) -> list[tuple[tuple, Any]]:
        result = []
        def _inorder(node):
            # Check node against NIL
            if node != self.NIL:
                _inorder(node.left)                # Check value before adding? Depends if None value is possible
                result.append((node.key, node.value))
                _inorder(node.right)
        _inorder(self.root)
        return result

    def search_min_greater_equal(self, key: tuple) -> Optional[RBNode]:
        node = self.root
        result = None # Initialize result to None
        while node != self.NIL:
            # Defensive check for node.key
            if node.key is not None and node.key >= key:
                result = node
                node = node.left
            else:
                node = node.right
        # result will be None if no suitable node found
        return result

    def search_max_less(self, key: tuple) -> Optional[RBNode]:
        node = self.root
        result = None # Initialize result to None
        while node != self.NIL:
            # Defensive check for node.key
            if node.key is not None and node.key < key:
                result = node
                node = node.right
            else:
                node = node.left
        # result will be None if no suitable node found
        return result

# --- Retroactive Priority Queue ---

class RPQ:
    def __init__(self):
        self.T_ins = RBTree()
        self.T_d_m = RBTree()

    def find_first_inconsistent_delete(self, new_node_key: tuple) -> Optional[RBNode]:
         insert_time = new_node_key[1]
         potential_del_node = self.T_d_m.search_min_greater_equal((insert_time,))
         while potential_del_node: # Already checks if potential_del_node is not None
             deleted_pq_node = potential_del_node.value
             # Check if deleted_pq_node and its tins_node exist before comparing keys
             if deleted_pq_node and deleted_pq_node.tins_node and deleted_pq_node.tins_node.key is not None:
                 # Ensure new_node_key is also valid before comparison
                 if new_node_key is not None and new_node_key < deleted_pq_node.tins_node.key:
                     return potential_del_node             # Optional: Add debug prints for missing links if needed
             elif not deleted_pq_node:
                 print(f"RPQ WARNING: T_d_m node {potential_del_node.key} has None value!")
             elif not deleted_pq_node.tins_node:
                 print(f"RPQ WARNING: T_d_m value {deleted_pq_node} lacks .tins_node link!")
             elif deleted_pq_node.tins_node.key is None:
                  print(f"RPQ WARNING: T_ins node linked from T_d_m node {potential_del_node.key} has None key!")             # Get next node using successor, which now returns None if no successor
             potential_del_node = self.T_d_m.successor(potential_del_node)             # Loop condition 'while potential_del_node' handles the None case
         return None

    def invoke_insert(self, vertex: int, time: float, dist: float, pred: Optional[int]) -> Optional[RBNode]:
        # print(f"RPQ: Invoke Insert vertex={vertex}, time={time:.1f}, dist={dist:.1f}, pred={pred}")
        node = PQNode(vertex=vertex, time=time, dist=dist, pred=pred, valid=True)
        key = (dist, time)
        rb_node_ins = self.T_ins.insert(key, node)        # node.tins_node should be set within insert now
        inconsistent_del_node = self.find_first_inconsistent_delete(key)        # Check inconsistent delete
        if inconsistent_del_node:
             print(f"RPQ: Insertion at time {time:.1f} causes inconsistency with deletion at {inconsistent_del_node.key[0]:.1f}")
             return inconsistent_del_node
        else:
             return None

    def invoke_del_min(self, time: float) -> Optional[PQNode]:
        # print(f"RPQ: Invoke Del_Min at time={time:.1f}")
        min_rb_node = self.find_valid_min_ins()
        if not min_rb_node:
            print(f"RPQ: T_ins is empty or has no valid nodes.")
            return None
        min_pq_node = min_rb_node.value        # *** Added Check (although find_valid_min_ins should guarantee value exists) ***
        if not min_pq_node:
             print(f"RPQ ERROR: Valid min RBNode {min_rb_node.key} has None value!")
             return None # Should not happen if find_valid_min_ins is correct
        min_pq_node.valid = False
        print(f"RPQ: Marking T_ins node invalid: Key={min_rb_node.key}, Value={min_pq_node}")
        del_key = (time,)        # Pass the valid min_pq_node to insert
        del_rb_node = self.T_d_m.insert(del_key, min_pq_node)        # Check if insert returned a node before linking
        if del_rb_node and del_rb_node != self.T_d_m.NIL:
             min_pq_node.deleted_by_node = del_rb_node
        else:
            pass
            #  print(f"RPQ WARNING: Failed to get RBNode when inserting deletion record {del_key} into T_d_m.")
        # print(f"RPQ: Added deletion record to T_d_m: Key={del_key}, Deleted PQNode={min_pq_node}")
        return min_pq_node

    def find_valid_min_ins(self) -> Optional[RBNode]:
        node = self.T_ins.minimum() # Returns None if tree is empty
        while node: # Check if node is not None
            if node.value and isinstance(node.value, PQNode) and node.value.valid:            # Check node.value and node.value.valid
                return node
            node = self.T_ins.successor(node) # successor returns None if no more nodes
        return None

    def revoke_insert(self, key: tuple) -> Optional[RBNode]:
        print(f"RPQ: Revoke Insert for key={key}")
        node_to_revoke = self.T_ins.search(key) # Returns None if not found
        # *** Added Check ***
        if not node_to_revoke:
             print(f"RPQ: Revoke Insert failed: Node with key {key} not found in T_ins.")
             return None
        revoked_pq_node = node_to_revoke.value        # Check value before accessing valid attribute
        if not isinstance(revoked_pq_node, PQNode):
             return None # Cannot proceed logically
        if not revoked_pq_node.valid:
             print(f"RPQ: Revoke Insert: Node {key} value was already invalid.")
             pass # Fall through to inconsistency check

        print(f"RPQ: Marking T_ins node {key} as invalid.")
        revoked_pq_node.valid = False        # --- Check for inconsistency ---
        insert_time = key[1]
        potential_del_node = self.T_d_m.search_min_greater_equal((insert_time,))
        while potential_del_node: # Checks not None            # Check potential_del_node.value before comparison
            deleted_pq_node = potential_del_node.value
            if deleted_pq_node == revoked_pq_node: # Safe comparison by object identity
                print(f"RPQ: Revoked insert {key} invalidates deletion at {potential_del_node.key[0]:.1f}")
                return potential_del_node
            potential_del_node = self.T_d_m.successor(potential_del_node) # Returns None eventually
        return None

    # Add mark_tins_valid parameter with default True for backward compatibility
    def revoke_del_min(self, time: float, mark_tins_valid: bool = True) -> Optional[RBNode]:
        print(f"RPQ: Revoke Del_Min for time={time:.1f} (Mark T_ins Valid: {mark_tins_valid})") # Log modification
        del_key = (time,)
        node_in_tdm = self.T_d_m.search(del_key)
        if not node_in_tdm:
            print(f"RPQ: Revoke Del_Min failed: No deletion recorded at time {time:.1f}")
            return None
        pq_node_to_revive = node_in_tdm.value
        if not isinstance(pq_node_to_revive, PQNode):
             print(f"RPQ ERROR: T_d_m node {del_key} has non-PQNode value: {type(pq_node_to_revive)}")# Attempt cleanup before returning
             self.T_d_m.remove_node(del_key)
             return None
        # print(f"RPQ: Found deletion record for: {pq_node_to_revive}")
        tins_node_key_str = "N/A" # For logging
        # --- Conditional Marking ---
        # Check if the link to T_ins node exists before potentially using it
        original_tins_node = pq_node_to_revive.tins_node
        if original_tins_node and original_tins_node != self.T_ins.NIL:
            tins_node_key_str = str(original_tins_node.key) # Capture key for log
            if mark_tins_valid: # Check the new parameter
                 if not pq_node_to_revive.valid:
                      pq_node_to_revive.valid = True
                      print(f"RPQ: Marked original T_ins node {tins_node_key_str} back to valid.")
                 else:
                      print(f"RPQ: Warning - PQNode {pq_node_to_revive} was already marked valid?")
                 pq_node_to_revive.deleted_by_node = None # Clear link only if revived
            else:
                 print(f"RPQ: Revoke Del_Min: Skipping marking T_ins node {tins_node_key_str} as valid.")                 # Even if not marking valid, clear the deletion link from PQNode
                 pq_node_to_revive.deleted_by_node = None
        else:
             print(f"RPQ: Revoke Del_Min Error/Warning: Could not find original T_ins node link for PQNode {pq_node_to_revive}")             # Still clear the deletion link if the PQNode exists
             if pq_node_to_revive:
                  pq_node_to_revive.deleted_by_node = None
        # removed_node = self.T_d_m.remove_node(del_key)        # Physically remove from T_d_m
        # if removed_node:
        #      print(f"RPQ: Removed node {del_key} from T_d_m.")
        # else:
        #      print(f"RPQ: Revoke Del_Min Warning: Failed to remove node {del_key} from T_d_m (was already removed?).")
        # --- Inconsistency Check ---        # Only perform check if we actually marked the node valid,        # otherwise the premise (revived node competes) is false.
        if mark_tins_valid:
            revived_node_key = None
            if original_tins_node and original_tins_node.key is not None:
                 revived_node_key = original_tins_node.key
            if revived_node_key is not None:
                potential_del_node = self.T_d_m.search_min_greater_equal((time + 1e-9,))
                while potential_del_node:
                    deleted_pq_node = potential_del_node.value                    # Check deleted node and its T_ins link exist before comparison
                    if (deleted_pq_node and deleted_pq_node.tins_node and
                        deleted_pq_node.tins_node.key is not None):
                        if revived_node_key < deleted_pq_node.tins_node.key:
                            print(f"RPQ: Revoked deletion at {time:.1f} (revived {revived_node_key}) invalidates deletion at {potential_del_node.key[0]:.1f}")
                            return potential_del_node
                    potential_del_node = self.T_d_m.successor(potential_del_node)
            else:
                 print(f"RPQ: Skipping inconsistency check after revoke; revived node key unavailable.")
        return None # No immediate inconsistency found or check skipped/failed

    def find_vertex_rbnode_in_tins(self, vertex: int, active_only: bool = True) -> Optional[RBNode]:
         latest_node: Optional[RBNode] = None
         latest_time = -1.0
         nodes = self.T_ins.inorder()
         for key, pq_node in nodes:
             # Check if pq_node is valid PQNode object before accessing vertex
             if isinstance(pq_node, PQNode) and pq_node.vertex == vertex:
                  is_candidate = (not active_only) or pq_node.valid
                  if is_candidate and pq_node.time > latest_time:
                       latest_time = pq_node.time
                       # Check if tins_node link exists
                       if pq_node.tins_node and pq_node.tins_node != self.T_ins.NIL:
                           latest_node = pq_node.tins_node
                       else:
                           # This indicates an issue, maybe log it
                           print(f"RPQ WARNING: PQNode for vertex {vertex} at time {pq_node.time} has missing tins_node link.")
                           latest_node = None # Cannot return the RBNode if link is broken
         return latest_node

    def print_state(self, time_step):
        print(f"\n=== RPQ STATE at Time Step {time_step:.1f} ===")
        print("\n-- T_ins (Insertions: Key=(dist, time), Value=PQNode) --")
        valid_count_ins = 0
        nodes_ins = self.T_ins.inorder()
        if not nodes_ins: print("  (empty)")
        for key, pq_node in nodes_ins:
            # Check pq_node type before accessing attributes
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
        # self.processed_at_time: Dict[int, float] = {}
        self.processed_times: Dict[int, List[float]] = {}  # All deletion times for each vertex

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


    def propagate_changes(self, start_time: Optional[float] = None):
        time_to_process = start_time if start_time is not None else self.current_time
        # print(f"\n--- Propagating Changes starting from Time={time_to_process:.1f} ---")
        processed_in_this_run = set() # Basic cycle detection within propagation
        while True:
            # self.rpq.print_state(time_to_process)
            min_rb_node = self.rpq.find_valid_min_ins()
            if not min_rb_node:
                print(f"[{time_to_process:.1f}] RPQ has no more valid nodes. Propagation ends.")
                break
            min_pq_node = min_rb_node.value
            if not isinstance(min_pq_node, PQNode):
                 print(f"ERROR [{time_to_process:.1f}]: Invalid value type {type(min_pq_node)} found in T_ins for key {min_rb_node.key}. Stopping propagation.")
                 break # Cannot proceed
            u = min_pq_node.vertex
            dist_u_rpq = min_pq_node.dist
            if u in processed_in_this_run:            # Prevent infinite loops in case of weird state
                 print(f"WARNING [{time_to_process:.1f}]: Node {u} encountered again in this propagation run. Breaking loop.")
                 break
            processed_in_this_run.add(u)
            # print(f"[{time_to_process:.1f}] Found potential min: Vtx={u} with Dist={dist_u_rpq:.1f} (InsTime={min_pq_node.time:.1f})")
            # Simulate Del_Min
            deleted_node = self.rpq.invoke_del_min(time_to_process)
            # Check if deletion was successful (deleted_node is the PQNode)
            if not deleted_node or deleted_node != min_pq_node:
                 print(f"WARNING [{time_to_process:.1f}]: invoke_del_min did not return the expected node {min_pq_node}. Proceeding cautiously.")

            # Basic Dijkstra Checks
            # No need to check u in self.processed_at_time here, done implicitly by RPQ valid state? Revisit if needed.
            # Check for stale entry
            if self.dist[u] < dist_u_rpq:
                print(f"[{time_to_process:.1f}] Stale RPQ entry for {u}. Known dist={self.dist[u]:.1f} < RPQ dist={dist_u_rpq:.1f}. Skipping relaxation.")
                self._increment_time()
                time_to_process = self.current_time
                processed_in_this_run.clear() # Reset for next potential start
                continue
            if u not in self.processed_times:            # Mark as processed *at this time*
                self.processed_times[u] = [time_to_process]
            else:
                self.processed_times[u].append(time_to_process)            # self.processed_at_time[u] = time_to_process
            print(f"[{time_to_process:.1f}] Processing node {u} (Dist={dist_u_rpq:.1f}).")
            if dist_u_rpq < self.dist[u]:            # Update main distance if necessary
                 print(f"[{time_to_process:.1f}] Updating main dist for {u} from {self.dist[u]:.1f} to {dist_u_rpq:.1f}")
                 self.dist[u] = dist_u_rpq
                 self.pred[u] = min_pq_node.pred # Use pred from the PQNode being processed
            for v in range(self.num_vertices):            # Relax outgoing edges
                # Check graph bounds and edge existence
                if u < self.num_vertices and v < self.num_vertices:
                     weight = self.graph[u][v]
                     if weight != math.inf and u != v:
                          # Check if dist[u] is valid before adding
                          if self.dist[u] != math.inf:
                              new_dist_v = self.dist[u] + weight
                              if new_dist_v < self.dist[v]:
                                #   print(f"[{time_to_process:.1f}] Relaxing edge ({u},{v}): Found shorter path to {v}. Dist {self.dist[v]:.1f} -> {new_dist_v:.1f}")
                                  self.dist[v] = new_dist_v
                                  self.pred[v] = u
                                  # Insert into RPQ. Check result for inconsistency.
                                  inconsistent_del = self.rpq.invoke_insert(v, time_to_process, new_dist_v, u)
                                  if inconsistent_del:
                                      print(f"[!ALERT {time_to_process:.1f}!] Insertion for {v} conflicts with Del_Min at {inconsistent_del.key[0]:.1f}")
                                      # Basic handling: just print alert
                          else:
                               print(f"[{time_to_process:.1f}] Skipping relaxation from {u}; its distance is infinity.")
                else:
                     print(f"ERROR [{time_to_process:.1f}]: Invalid vertex index during relaxation ({u}, {v})")
            # Finished processing for this time step
            self._increment_time()
            time_to_process = self.current_time
            processed_in_this_run.clear() # Reset for next minimum find

    # def handle_edge_update(self, u: int, v: int, new_weight: float):        
    #     print(f"\n--- Handling Edge Update ({u}, {v}) to New Weight={new_weight:.1f} at Time={self.current_time:.1f} ---")
    #     # self.rpq.print_state(self.current_time)
    #     # Update edge weight in the graph
    #     self.graph[u][v] = new_weight
    #     latest_node_v_rb = self.rpq.find_vertex_rbnode_in_tins(v, active_only=False)
    #     latest_pq_node_v = latest_node_v_rb.value if latest_node_v_rb else None

    #     if not latest_pq_node_v:        # Case 1: 'v' not in RPQ at all
    #         print(f"[{self.current_time:.1f}] Vertex {v} not found in RPQ history.")
    #         if self.dist[u] != math.inf:
    #             potential_new_dist_v = self.dist[u] + new_weight
    #             if potential_new_dist_v < self.dist[v]:
    #                 print(f"[{self.current_time:.1f}] New edge creates better path to {v}.")
    #                 self.dist[v] = potential_new_dist_v
    #                 self.pred[v] = u
    #                 self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
    #                 self.propagate_changes(self.current_time)
    #         return

    #     if latest_pq_node_v.valid:        # Case 2: 'v' is currently active in the RPQ
    #         print(f"[{self.current_time:.1f}] Vertex {v} is ACTIVE in RPQ.")
    #         new_dist_v = self.dist[u] + new_weight if self.dist[u] != math.inf else math.inf
    #         if self.pred[v] == u:
    #             print(f"[{self.current_time:.1f}] Path to {v} is via {u}. Updating.")
    #             if latest_node_v_rb:
    #                 self.rpq.revoke_insert(latest_node_v_rb.key)
    #             self.dist[v] = new_dist_v
    #             self.rpq.invoke_insert(v, self.current_time, new_dist_v, u)
    #             self.propagate_changes(self.current_time)
    #         else:
    #             if new_dist_v < self.dist[v]:
    #                 print(f"[{self.current_time:.1f}] New path to {v} via ({u},{v}) is better.")
    #                 self.dist[v] = new_dist_v
    #                 self.pred[v] = u
    #                 self.rpq.invoke_insert(v, self.current_time, new_dist_v, u)
    #                 self.propagate_changes(self.current_time)
    #         return
    #     # Case 3: 'v' has been processed previously - Enhanced handling
    #     print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
    #     # Get all deletion times for this vertex (sorted newest first)
    #     deletion_times = sorted(self.processed_times.get(v, []), reverse=True)
    #     print("Processedtimes: ", self.processed_times)
    #     # print("REAL DELS: ", deletion_times)
    #     if not deletion_times:
    #         print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion times recorded!")
    #         return
    #     potential_new_dist_v = self.dist[u] + new_weight if self.dist[u] != math.inf else math.inf
    #     path_via_u_used = (self.pred[v] == u)
    #     if potential_new_dist_v < self.dist[v]:

    #         if path_via_u_used:
    #             # Edge was part of the final path - need to invalidate all relevant deletions
    #             # pass
    #             print(f"[{self.current_time:.1f}] Processed path to {v} was via {u}. Checking all deletions.")
    #             # Find all deletions where the path was via u
    #             relevant_deletions = []
    #             rbnodes = []
    #             for del_time in deletion_times:
    #                 rb_node = self.rpq.T_d_m.search((del_time,))
    #                 if rb_node and rb_node.value and rb_node!=u:
    #                     rbnodes.append(rb_node)
    #                     relevant_deletions.append(del_time)
    #             if len(rbnodes)>0:
    #                 distcheck = math.inf                    
    #                 for rbn in rbnodes:
    #                     if rbn.value.dist<distcheck:
    #                         bestnode = rbn
    #                         distcheck = rbn.value.dist
    #                 inconsistent_del = self.rpq.revoke_del_min(bestnode.value.time)

    #             if not relevant_deletions:
    #                 print(f"[{self.current_time:.1f}] No relevant deletions found via {u}. No action needed.")
    #                 return

    #             print(f"[{self.current_time:.1f}] Found {len(relevant_deletions)} relevant deletions via {u}")
                
    #             # Revoke all relevant deletions (from newest to oldest)
    #             for del_time in relevant_deletions:
    #                 print(f"[{self.current_time:.1f}] Revoking deletion at {del_time:.1f}")
    #                 inconsistent_del = self.rpq.revoke_del_min(del_time)
    #                 if inconsistent_del:
    #                     print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

    #             # Clear processed status
    #             if v in self.processed_times:
    #                 del self.processed_times[v]

    #             # Invalidate current path
    #             self.dist[v] = math.inf
    #             self.pred[v] = None

    #             # If the new weight still provides a valid path, consider it
    #             if potential_new_dist_v < math.inf:
    #                 print(f"[{self.current_time:.1f}] Reinserting {v} with potential new path.")
    #                 self.dist[v] = potential_new_dist_v
    #                 self.pred[v] = u
    #                 self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)

    #             self.propagate_changes(self.current_time)

    #     else:
    #         # Path wasn't via u - check if new path is better or if we should restore an old path
    #         print(f"[{self.current_time:.1f}] Path used to process {v} was mehhhh thru {u}.")
            
    #         # Check if new path via u is better than current
    #         if potential_new_dist_v < self.dist[v]:
    #             print(f"[{self.current_time:.1f}] New path via ({u},{v}) is better. Reactivating {v}.")
                
    #             # Revoke only the most recent deletion
    #             latest_del_time = deletion_times[0]
    #             print(f"[{self.current_time:.1f}] Revoking latest deletion at {latest_del_time:.1f}")
    #             inconsistent_del = self.rpq.revoke_del_min(latest_del_time)
    #             if inconsistent_del:
    #                 print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

    #             # Update processed times
    #             # if v in self.processed_times and deletion_times:
    #             #     self.processed_times[v].remove(latest_del_time)
    #             #     if not self.processed_times[v]:
    #             #         del self.processed_times[v]

    #             self.dist[v] = potential_new_dist_v
    #             self.pred[v] = u
    #             self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
    #             self.propagate_changes(self.current_time)
    #         else:
    #             # New path via u isn't better, but check if we should restore an old path
    #             print(f"[{self.current_time:.1f}] Checking historical paths to {v}...")
                
    #             # Find the best historical path that doesn't use (u,v)
    #             best_historical_dist = math.inf
    #             best_historical_pred = None
    #             best_del_time = None
                
    #             for del_time in deletion_times:
    #                 rb_node = self.rpq.T_d_m.search((del_time,))
    #                 if rb_node and rb_node.value:
    #                     historical_node = rb_node.value
    #                     # Only consider paths that don't go through the updated edge (u,v)
    #                     if historical_node.pred != u:
    #                         if historical_node.dist < best_historical_dist:
    #                             best_historical_dist = historical_node.dist
    #                             best_historical_pred = historical_node.pred
    #                             best_del_time = del_time
                
    #             # If found a better historical path, restore it
    #             if best_historical_dist < potential_new_dist_v:
    #                 print(f"[{self.current_time:.1f}] Found better historical path to {v} via {best_historical_pred} (dist {best_historical_dist:.1f})")
                    
    #                 # Revoke the deletion where this path was recorded
    #                 print(f"[{self.current_time:.1f}] Revoking deletion at {best_del_time:.1f}")
    #                 inconsistent_del = self.rpq.revoke_del_min(best_del_time)
    #                 if inconsistent_del:
    #                     print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

    #                 # Update processed times
    #                 # if v in self.processed_times:
    #                 #     self.processed_times[v].remove(best_del_time)
    #                 #     if not self.processed_times[v]:
    #                 #         del self.processed_times[v]

    #                 self.dist[v] = best_historical_dist
    #                 self.pred[v] = best_historical_pred
    #                 self.rpq.invoke_insert(v, self.current_time, best_historical_dist, best_historical_pred)
    #                 self.propagate_changes(self.current_time)
    #             else:
    #                 print(f"[{self.current_time:.1f}] No better path found for {v}. Maintaining current distance.")
    #     # self.rpq.print_state(self.current_time)
    def handle_edge_update(self, u: int, v: int, new_weight: float):
            
        print(f"\n--- Handling Edge Update ({u}, {v}) to New Weight={new_weight:.1f} at Time={self.current_time:.1f} ---")
        # self.rpq.print_state(self.current_time)
        
        # Update edge weight in the graph
        # old_weight = self.graph[u][v]
        self.graph[u][v] = new_weight
        latest_node_v_rb = self.rpq.find_vertex_rbnode_in_tins(v, active_only=False)
        latest_pq_node_v = latest_node_v_rb.value if latest_node_v_rb else None

        # Case 1: 'v' not in RPQ at all
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

        # Case 2: 'v' is currently active in the RPQ
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
        # Case 3: 'v' has been processed previously - Enhanced handling
        print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
        # Get all deletion times for this vertex (sorted newest first)
        deletion_times = sorted(self.processed_times.get(v, []), reverse=True)
        print("Processedtimes: ", self.processed_times)
        # print("REAL DELS: ", deletion_times)
        if not deletion_times:
            print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion times recorded!")
            return
        potential_new_dist_v = self.dist[u] + new_weight if self.dist[u] != math.inf else math.inf
        path_via_u_used = (self.pred[v] == u)
        findnew = False
        newdist = potential_new_dist_v
        # path_via_u_used = False\
        updated = False
        # oldpreds = self.pred.copy()
        if path_via_u_used and potential_new_dist_v < self.dist[v]:#most recent pred is u
        # if u was alreadu the best path, and now its even smaller than that it means theres no 
        # need to check other preds of v

            print(f"[{self.current_time:.1f}] Processed path to {v} was via {u} and is still small.Updating")            
            # Find all deletions where the path was via u
            # if potential_new_dist_v < self.dist[v]:
                # inconsistent_del = self.rpq.revoke_del_min(bestnode.value.time)
            # print(f"[{self.current_time:.1f}] It is better!")            
            self.dist[v] = potential_new_dist_v
            self.pred[v] = u
            self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
            # self.propagate_changes(self.current_time)
            # findnew = False
            updated = True
            # relevant_deletions = []
            # rbnodes = []
            # for del_time in deletion_times:
            #     rb_node = self.rpq.T_d_m.search((del_time,))
            #     if rb_node and rb_node.value and rb_node==u:
            #         rbnodes.append(rb_node)
            #         relevant_deletions.append(del_time)
            # distcheck = math.inf
            # if len(rbnodes)>0:
            #     bestnode = rbnodes[0]
            #     for rbn in rbnodes:
            #         if self.graph[rbn.value.vertex][v]+new_weight<distcheck:
            #             bestnode = rbn
            #             distcheck = self.graph[rbn.value.vertex][v]+new_weight
            #     if distcheck==math.inf:
            #         print("No old path is good")
            #         inconsistent_del = self.rpq.revoke_del_min(bestnode.value.time)
            #         self.dist[v] = new_dist_v
            #         self.pred[v] = u


            # if not relevant_deletions:
            #     print(f"[{self.current_time:.1f}] No relevant deletions found via {u}. No action needed.")
            #     return

            # print(f"[{self.current_time:.1f}] Found {len(relevant_deletions)} relevant deletions via {u}")
            
            # # Revoke all relevant deletions (from newest to oldest)
            # for del_time in relevant_deletions:
            #     print(f"[{self.current_time:.1f}] Revoking deletion at {del_time:.1f}")
            #     inconsistent_del = self.rpq.revoke_del_min(del_time)
            #     if inconsistent_del:
            #         print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

            # Clear processed status
            # if v in self.processed_times:
            #     del self.processed_times[v]

            # Invalidate current path

            # If the new weight still provides a valid path, consider it
            # if potential_new_dist_v < self.dist[v]:
            #     print(f"[{self.current_time:.1f}] Reinserting {v} with potential new path.")
            #     self.dist[v] = potential_new_dist_v
            #     self.pred[v] = u
            #     self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)

            # self.propagate_changes(self.current_time)

        else:
            # Path wasn't via u - check if new path is better or if we should restore an old path
            print(f"[{self.current_time:.1f}] Path used to process {v} was NOT via {u} or it may not be the best one")
            
            # # Check if new path via u is better than current
            # if potential_new_dist_v < new_weight+self.dist[u]:
            #     print(f"[{self.current_time:.1f}] New path via ({u},{v}) is better. Reactivating {v}.")
                
            #     # Revoke only the most recent deletion
            #     latest_del_time = deletion_times[0]
            #     print(f"[{self.current_time:.1f}] Revoking latest deletion at {latest_del_time:.1f}")
            #     inconsistent_del = self.rpq.revoke_del_min(latest_del_time)
            #     if inconsistent_del:
            #         print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

            #     # Update processed times
            #     # if v in self.processed_times and deletion_times:
            #     #     self.processed_times[v].remove(latest_del_time)
            #     #     if not self.processed_times[v]:
            #     #         del self.processed_times[v]

            #     self.dist[v] = potential_new_dist_v
            #     self.pred[v] = u
            #     self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
            #     self.propagate_changes(self.current_time)
            # else:
                # New path via u isn't better, but check if we should restore an old path
            print(f"[{self.current_time:.1f}] Checking historical paths to {v}...")
            
            # Find the best historical path that doesn't use (u,v)
            best_historical_dist = math.inf
            best_historical_pred = None
            best_del_time = None
            
            for del_time in deletion_times:
                rb_node = self.rpq.T_d_m.search((del_time,))
                if rb_node and rb_node.value:
                    historical_node = rb_node.value
                    # if historical_node.pred != u and self.dist[best_historical_pred]==self.dist[best_historical_pred]:
                    #     if historical_node.dist < best_historical_dist:
                    #         best_historical_dist = historical_node.dist
                    #         best_historical_pred = historical_node.pred
                    #         best_del_time = del_time

                    if historical_node.pred != u:# Avoid updated path
                        if best_historical_pred:
                            if historical_node.dist < best_historical_dist and self.dist[best_historical_pred]==self.dist[best_historical_pred]:
                                best_historical_dist = historical_node.dist
                                best_historical_pred = historical_node.pred
                                best_del_time = del_time

            
            # If found a better historical path, restore it
            print("Best historical dist:",best_historical_dist)
            if best_historical_dist < potential_new_dist_v:
                print(f"[{self.current_time:.1f}] Found better historical path to {v} via {best_historical_pred} (dist {best_historical_dist:.1f})")
                
                # Revoke the deletion where this path was recorded
                print(f"[{self.current_time:.1f}] Revoking deletion at {best_del_time:.1f}")
                inconsistent_del = self.rpq.revoke_del_min(best_del_time)
                if inconsistent_del:
                    print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

                # Update processed times
                # if v in self.processed_times:
                #     self.processed_times[v].remove(best_del_time)
                #     if not self.processed_times[v]:
                #         del self.processed_times[v]

                self.dist[v] = best_historical_dist
                self.pred[v] = best_historical_pred
                newdist = best_historical_dist
                self.rpq.invoke_insert(v, self.current_time, best_historical_dist, best_historical_pred)
                self.propagate_changes(self.current_time)
                updated = True
            else:
                
                print(f"[{self.current_time:.1f}] No better path found for {v}. Maintaining new  distance.")
                self.dist[v] = potential_new_dist_v
                self.pred[v] = u
                newdist = potential_new_dist_v
                self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
                updated = True
        # if updated:
        from collections import deque   #propagating changes to its ancestors
        print(f"#propagating changes to ancestors of {v}")
        q = deque([[v,newdist]])
        globalpred = v
        distrn = potential_new_dist_v
        paths = {}
        from collections import defaultdict, deque
        print(self.graph)
        children = defaultdict(list)
        graph  = self.graph
        
        for i in range(len(graph)):
            for j in range(len(graph[i])):
                if graph[i][j] != math.inf and graph[i][j] != 0:
                    children[i].append((j, graph[i][j]))
                elif i not in children:
                    children[i]=[]
        print(children)
        vertices = keys_list = list(children.keys())

        while q:
            currents = q.popleft()
            print(f"Processing: {currents}")
            distrn = currents[1]
            # Go from current → its children
            checked = []
            for child, weight in children[currents[0]]:
                checked = []
                print(f"Checking child {child} from parent {currents[0]} with weight {weight}")
                new_dist_c = math.inf                
                # Revoke any deletion events for child if needed
                best_historical_dist = math.inf
                best_historical_pred = None
                best_del_time = None
                for deletion_time in self.processed_times.get(child, []):
                    rb_node = self.rpq.T_d_m.search((deletion_time,))
                    print(self.dist)
                    print(f"rbnodes: {rb_node.value.vertex} rbdist: {rb_node.value.dist} rbpred: {rb_node.value.pred}")
                    if rb_node.value.pred and rb_node.value.pred not in checked:
                        checked.append(rb_node.value.pred)
                    if rb_node and rb_node.value:
                        historical_node = rb_node.value
                        best_historical_pred = historical_node.pred
                        # print()
                        if historical_node.pred != currents[0]:# Avoid updated path
                            if best_historical_pred:
                                # print("Best hsitorical pred dist:", best_historical.dist)
                                print(self.dist)
                                if (historical_node.dist < best_historical_dist or historical_node.dist<self.graph[currents[0]][child]+distrn) and (historical_node.dist-graph[best_historical_pred][child]==self.dist[best_historical_pred]):
                                    best_historical_dist = historical_node.dist
                                    best_historical_pred = historical_node.pred
                                    best_del_time = deletion_time
                                    print("here with",best_historical_dist)
                # print(checked)
                if best_historical_dist != math.inf:
                    print(self.dist)
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
                    #no shortest paths before worked 
                    # either this actually si the best path to v or we have to recompute.
                    # if we have to recompute, we have to recompute all the way up to th
                    # only need to check all other parents of v and check
                    bestpred = None
                    bestdist = math.inf
                    for parent in vertices:
                        if parent not in checked:
                            if self.dist[parent] + self.graph[parent][child] < self.graph[currents[0]][child]+distrn:
                                bestpred = parent
                                bestdist = self.dist[parent] + graph[parent][child]
                    if bestdist<self.graph[currents[0]][child]+distrn:
                        print(checked)
                        print(f"[{self.current_time:.1f}] Found new shortest path thorugh {bestpred} for {child} with dist {bestdist}")
                        self.dist[child] = bestdist
                        self.pred[child] = bestpred
                        self.rpq.invoke_insert(child, self.current_time, bestdist, bestpred)
                        self.propagate_changes(self.current_time)
                        new_dist_c = bestdist

                    else:
                        print(f"[{self.current_time:.1f}] No better historical path for {child}. Updating normally.")
                        new_dist_c = self.graph[currents[0]][child]+distrn
                        self.dist[child] = new_dist_c
                        self.pred[child] = currents[0]
                        self.rpq.invoke_insert(child, self.current_time, new_dist_c, currents[0])
                        self.propagate_changes(self.current_time)
                q.append([child,new_dist_c])  # Continue BFS to child


        # for i in range(0,len(graph)):
        #     for j in range(1,len(graph[i])):
        #         if graph[i][j] != math.inf and graph[i][j] != 0:
        #             if j not in paths:
        #                 paths[j] = [[i,graph[i][j]]]
        #             else:
        #                 paths[j].append([i,graph[i][j]])
        # while len(q)!=0:
        #     print(q)
        #     current = q.popleft()
        #     print(f"here curr: {current}")
        #     # if current==v:
        #     #     distrn = potential_new_dist_v
        #     #     break
        #     # else:
        #     #     distrn = current.value.dist
        #     print(paths)
        #     for c, preds in paths.items():
        #         q.append(c)  # Continue BFS through descendants

        #         for pred in preds:
        #             weight = pred[1]
        #             pred = pred[0]
        #             print(c, pred)
        #             if c==7:
        #                 # print(f"here pred: {pred}")
        #                 break
        #             if pred == current:
        #                 print(f"here pred: {pred}")
        #                 # Revoke any deletion events for vertex v
        #                 best_historical_dist = math.inf     #best deletion time for the ancestors
        #                 best_historical_pred = None
        #                 best_del_time = None
        #                 # globalpred = 
        #                 for deletion_time in self.processed_times.get(c, []):# fidning other better preds that may exist
        #                     rb_node = self.rpq.T_d_m.search((deletion_time,))
        #                     if rb_node and rb_node.value:
        #                         historical_node = rb_node.value
        #                         # Only consider paths that don't go through the updated edge (u,v)
        #                         if historical_node.pred != pred:          #not looking at path thru v
        #                             if historical_node.dist < best_historical_dist:
        #                                 best_historical_dist = historical_node.dist
        #                                 best_historical_pred = historical_node.pred
        #                                 best_del_time = deletion_time
        #                 if best_historical_dist < distrn+self.graph[pred][c]:
        #                     print(f"[{self.current_time:.1f}] Found better historical path to {c} via {best_historical_pred} (dist {best_historical_dist:.1f})")
        #                     self.dist[c] = best_historical_dist
        #                     self.pred[c] = best_historical_pred                                    
        #                     # Revoke the deletion where this path was recorded
        #                     print(f"[{self.current_time:.1f}] Revoking deletion at {best_del_time:.1f}")
        #                     inconsistent_del = self.rpq.revoke_del_min(best_del_time)
        #                     if inconsistent_del:
        #                         print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")
        #                     # del_node = self.rpq.T_d_m.search((deletion_time,))
        #                     # if del_node and del_node != self.rpq.T_d_m.NIL and del_node.value:
        #                     #     self.rpq.revoke_del_min(del_node.value.time)
        #                 else:
        #                     print(f"[{self.current_time:.1f}] No better historical path found for {c}")
        #                     self.dist[c] = distrn+self.graph[pred][c]
        #                     self.pred[c] = pred
        #                     self.rpq.invoke_insert(c, self.current_time, distrn+self.graph[pred][c], pred)
        #                 globalpred = c
        #                 distrn = distrn+self.dist[c]
        #             # break
        #     break

        self.propagate_changes(self.current_time)


        # self.rpq.print_state(self.current_time)
        # # Case 3: 'v' has been processed previously - Enhanced handling
        # print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
        # # Get all deletion times for this vertex (sorted newest first)
        # deletion_times = sorted(self.processed_times.get(v, []), reverse=True)
        # print("Processedtimes: ", self.processed_times)
        # # print("REAL DELS: ", deletion_times)
        # if not deletion_times:
        #     print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion times recorded!")
        #     return
        # potential_new_dist_v = self.dist[u] + new_weight if self.dist[u] != math.inf else math.inf
        # path_via_u_used = (self.pred[v] == u)

        # if path_via_u_used:
        #     # Edge was part of the final path - need to invalidate all relevant deletions
        #     print(f"[{self.current_time:.1f}] Processed path to {v} was via {u}. Checking all deletions.")
            
        #     # Find all deletions where the path was via u
        #     relevant_deletions = []
        #     rbnodes = []
        #     bestdist = math.inf
        #     for del_time in deletion_times:
        #         rb_node = self.rpq.T_d_m.search((del_time,))
        #         if rb_node and rb_node.value and rb_node.value.vertex!=u:
        #             rbnodes.append(rb_node)
        #             relevant_deletions.append(rb_node.value.time)
        #     if len(rbnodes)>0:
        #         for rbn in rbnodes:
        #             if rbn.value.dist+self.graph[rbn.value.vertex][v]<bestdist:
        #                 bestnode = rbn
        #                 bestdist = rbn.value.dist+self.graph[rbn.value.vertex][v]
        #         inconsistent_del = self.rpq.revoke_del_min(bestnode.value.time)

        #     # if not relevant_deletions:
        #     #     print(f"[{self.current_time:.1f}] No relevant deletions found via {u}. No action needed.")
        #     #     return

        #     # print(f"[{self.current_time:.1f}] Found {len(relevant_deletions)} relevant deletions via {u}")
            
        #     # # Revoke all relevant deletions (from newest to oldest)
        #     # for del_time in relevant_deletions:
        #     #     print(f"[{self.current_time:.1f}] Revoking deletion at {del_time:.1f}")
        #     #     inconsistent_del = self.rpq.revoke_del_min(del_time)
        #     #     if inconsistent_del:
        #     #         print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

        #     # Clear processed status
        #     # if v in self.processed_times:
        #     #     del self.processed_times[v]

        #     # Invalidate current path
        #     self.dist[v] = math.inf
        #     self.pred[v] = None

        #     # If the new weight still provides a valid path, consider it
        #     if potential_new_dist_v < bestdist:
        #         print(f"[{self.current_time:.1f}] Reinserting {v} with potential new path.")
        #         self.dist[v] = potential_new_dist_v
        #         self.pred[v] = u
        #         self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)

        #     self.propagate_changes(self.current_time)

        # else:
        #     # Path wasn't via u - check if new path is better or if we should restore an old path
        #     print(f"[{self.current_time:.1f}] Path used to process {v} was NOT via {u}.")
            
        #     # Check if new path via u is better than current
        #     if potential_new_dist_v < self.dist[v]:
        #         print(f"[{self.current_time:.1f}] New path via ({u},{v}) is better. Reactivating {v}.")
                
        #         # Revoke only the most recent deletion
        #         latest_del_time = deletion_times[0]
        #         print(f"[{self.current_time:.1f}] Revoking latest deletion at {latest_del_time:.1f}")
        #         inconsistent_del = self.rpq.revoke_del_min(latest_del_time)
        #         if inconsistent_del:
        #             print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

        #         # Update processed times
        #         # if v in self.processed_times and deletion_times:
        #         #     self.processed_times[v].remove(latest_del_time)
        #         #     if not self.processed_times[v]:
        #         #         del self.processed_times[v]

        #         self.dist[v] = potential_new_dist_v
        #         self.pred[v] = u
        #         self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
        #         self.propagate_changes(self.current_time)
        #     else:
        #         # New path via u isn't better, but check if we should restore an old path
        #         print(f"[{self.current_time:.1f}] Checking historical paths to {v}...")
                
        #         # Find the best historical path that doesn't use (u,v)
        #         best_historical_dist = math.inf
        #         best_historical_pred = None
        #         best_del_time = None
                
        #         for del_time in deletion_times:
        #             rb_node = self.rpq.T_d_m.search((del_time,))
        #             if rb_node and rb_node.value:
        #                 historical_node = rb_node.value
        #                 # Only consider paths that don't go through the updated edge (u,v)
        #                 p = historical_node.pred
        #                 if p is not None and p != u and self.dist.get(p, math.inf) < math.inf:
        #                     edge_weight = self.get_edge_weight(p, v)
        #                     if edge_weight is None:
        #                         continue  # Edge no longer exists

        #                     expected_dist = self.dist[p] + edge_weight
        #                     # Accept only if the path is still consistent
        #                     if abs(expected_dist - historical_node.dist) < 1e-6:
        #                         if expected_dist < best_historical_dist:
        #                             best_historical_dist = expected_dist
        #                             best_historical_pred = p
        #                             best_del_time = del_time
                
        #         # If found a better historical path, restore it
        #         if best_historical_dist < self.dist[v]:
        #             print(f"[{self.current_time:.1f}] Found better historical path to {v} via {best_historical_pred} (dist {best_historical_dist:.1f})")
                    
        #             # Revoke the deletion where this path was recorded
        #             print(f"[{self.current_time:.1f}] Revoking deletion at {best_del_time:.1f}")
        #             inconsistent_del = self.rpq.revoke_del_min(best_del_time)
        #             if inconsistent_del:
        #                 print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

        #             # Update processed times
        #             # if v in self.processed_times:
        #             #     self.processed_times[v].remove(best_del_time)
        #             #     if not self.processed_times[v]:
        #             #         del self.processed_times[v]

        #             self.dist[v] = best_historical_dist
        #             self.pred[v] = best_historical_pred
        #             self.rpq.invoke_insert(v, self.current_time, best_historical_dist, best_historical_pred)
        #             self.propagate_changes(self.current_time)
        #         else:
        #             print(f"[{self.current_time:.1f}] No better path found for {v}. Maintaining current distance.")
        # self.rpq.print_state(self.current_time)


    def update_edge(self, u: int, v: int, new_weight: float):
        """Public method to update an edge weight."""
        if u < 0 or u >= self.num_vertices or v < 0 or v >= self.num_vertices:
            print(f"Error: Invalid vertices ({u}, {v})")
            return
        if new_weight < 0:
            print("Error: Negative weights not supported.")
            return
        if new_weight == math.inf and self.graph[u][v] == math.inf:
             print(f"Edge ({u}, {v}) already non-existent. No update.")
             return # No change needed

        print(f"\n>>> Updating Edge ({u}, {v}) from {self.graph[u][v]:.1f} to {new_weight:.1f}")
        # Update graph *before* handling, so handle_edge_update sees the new weight
        self.graph[u][v] = new_weight
        self.handle_edge_update(u, v, new_weight)


# --- Example Usage --- (Remains the same)
if __name__ == "__main__":
    inf = math.inf
    graph = [ # Directed
        [inf,   2,   5,   4,   inf, inf, inf, inf],  # Node 0
        [inf, inf,   2,   4,   7,   inf, inf,  inf],  # Node 1
        [inf, inf,inf,   1,   inf, inf, inf, inf],  # Node 2
        [inf, inf, inf, inf,   inf,   4,   3, inf],  # Node 3
        [inf, inf, inf, inf, inf, inf, inf,   5],  # Node 4
        [inf, inf, inf, inf, inf, inf, inf,   7],  # Node 5
        [inf, inf, inf, inf, inf, inf, inf,   3],  # Node 6
        [inf, inf, inf, inf, inf, inf, inf, inf]  # Node 7
    ]
    #          A    B    C    D    E    F
    # graph = [
    #     [0,   2,   4,   math.inf, math.inf, math.inf],  # A
    #     [2,   0,   1,   4,    7,    math.inf],  # B 
    #     [4,   1,   0,   math.inf, 3,    math.inf],  # C
    #     [math.inf, 4,   math.inf, 0,    2,    1],  # D
    #     [math.inf, 7,   3,    2,    0,    5],  # E
    #     [math.inf, math.inf, math.inf, 1,    5,    0]   # F
    # ]

    dd = DynamicDijkstra(graph)
    source_node = 0
    dd.initial_dijkstra(source_node)

    print("\nFINAL STATE AFTER INITIAL DIJKSTRA:")
    print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    print("Predecessors:", dd.pred)
    print("-" * 50)


    # --- Test Edge Updates ---
    dd.update_edge(5, 7, 1.0) # Decrease weight (7 -> 1)

    print("\nFINAL STATE AFTER UPDATE (5, 7) -> 1:")
    print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    print("Predecessors:", dd.pred)
    # dd.rpq.print_state(dd.current_time) # Print final state if needed
    print("-" * 50)

    dd.update_edge(5, 7, 11.0) # Increase weight (1 -> 11)

    print("\nFINAL STATE AFTER UPDATE (5, 7) -> 11:")
    print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    print("Predecessors:", dd.pred)
    # dd.rpq.print_state(dd.current_time)
    print("-" * 50)

    dd.update_edge(3, 6, 1.5) # Decrease weight (2 -> 0.5
    dd.update_edge(3, 6, 11.5) # Decrease weight (2 -> 0.5
    dd.update_edge(5, 7, 1.0) # Increase weight (1 -> 11)
    dd.update_edge(3, 6, 1.5) # Decrease weight (2 -> 0.5
    dd.update_edge(3, 6, 11.5) # Decrease weight (2 -> 0.5
    # dd.update_edge(3, 6, 1.5) # Decrease weight (2 -> 0.5
    # dd.update_edge(0, 3, 88.5) # Decrease weight (2 -> 0.5

    print("\nFINAL STATE AFTER UPDATE (1, 2) -> 0.5:")
    print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    print("Predecessors:", dd.pred)
    # dd.rpq.print_state(dd.current_time)
    print("-" * 50)
    # dd.update_edge(5, 7, 0.5) # Decrease weight (2 -> 0.5
    # dd.update_edge(0, 3, 0.5) # Decrease weight (2 -> 0.5
    # dd.update_edge(3, 5, 0.5) # Decrease weight (2 -> 0.5
    # # dd.update_edge(5, 7, 18.5) # Decrease weight (2 -> 0.5
    # dd.update_edge(0, 3, .5) # Decrease weight (2 -> 0.5
    # dd.update_edge(5, 7, 18.5) # Decrease weight (2 -> 0.5
    

    # print("\nFINAL STATE AFTER UPDATE (1, 2) -> 0.5:")
    # print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    # print("Predecessors:", dd.pred)
    # # dd.rpq.print_state(dd.current_time)
    # print("-" * 50)
    # # dd.update_edge(0, 3, 1.0) # Decrease weight (4 -> 1)

    # # print("\nFINAL STATE AFTER UPDATE (0, 3) -> 1:")
    # # print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    # # print("Predecessors:", dd.pred)
    # # # dd.rpq.print_state(dd.current_time)
    # # print("-" * 50)

    # graph = [
    #     [0,   5,   inf, 3,   inf, inf, inf, inf],  # 0
    #     [inf, 0,   2,   inf, 6,   inf, inf, inf],  # 1
    #     [inf, inf, 0,   inf, inf, inf, 7,   inf],  # 2
    #     [inf, inf, 1,   0,   inf, inf, inf, inf],  # 3
    #     [inf, inf, inf, inf, 0,   4,   inf, 9],   # 4
    #     [inf, inf, inf, inf, inf, 0,   1,   inf],  # 5
    #     [inf, inf, inf, inf, inf, inf, 0,   inf],  # 6
    #     [inf, inf, inf, inf, inf, inf, 1,   0]    # 7
    # ]
    
    # dd = DynamicDijkstra(graph)
    # source_node = 0
    # dd.initial_dijkstra(source_node)

    # print("\nFINAL STATE AFTER INITIAL DIJKSTRA:")
    # print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    # print("Predecessors:", dd.pred)
    # print("-" * 50)
    # # Expected:
    # # Distances: ['0.0', '5.0', '4.0', '3.0', '11.0', '15.0', '16.0', '20.0']
    # # Predecessors: [None, 0, 3, 0, 1, 4, 5, 4]

    # # Test Case 1: Decrease edge (5,7) from 9 to 1
    # dd.handle_edge_update(5, 7, 1.0)
    
    # print("\nFINAL STATE AFTER UPDATE (5, 7) -> 1:")
    # print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    # print("Predecessors:", dd.pred)
    # print("-" * 50)
    # # Expected:
    # # Distances: ['0.0', '5.0', '4.0', '3.0', '11.0', '15.0', '16.0', '16.0']
    # # Predecessors: [None, 0, 3, 0, 1, 4, 5, 5]

    # # Test Case 2: Increase edge (5,7) from 1 to 11
    # dd.handle_edge_update(5, 7, 11.0)
    
    # print("\nFINAL STATE AFTER UPDATE (5, 7) -> 11:")
    # print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    # print("Predecessors:", dd.pred)
    # print("-" * 50)
    # # Expected (should revert to original path):
    # # Distances: ['0.0', '5.0', '4.0', '3.0', '11.0', '15.0', '16.0', '20.0']
    # # Predecessors: [None, 0, 3, 0, 1, 4, 5, 4]

    # # Test Case 3: Update edge (3,2) from 1 to 4
    # dd.handle_edge_update(3, 2, 4.0)
    
    # print("\nFINAL STATE AFTER UPDATE (3, 2) -> 4:")
    # print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    # print("Predecessors:", dd.pred)
    # print("-" * 50)
    # # Expected changes:
    # # Node 2 (distance increases from 4.0 to 7.0 via 0→1→2)
    # # Downstream nodes (6,7) may also update

    # # Test Case 4: Make edge (1,4) unreachable
    # dd.handle_edge_update(1, 4, inf)
    
    # print("\nFINAL STATE AFTER UPDATE (1, 4) -> inf:")
    # print("Distances:", [f"{d:.1f}" if d != inf else "inf" for d in dd.dist])
    # print("Predecessors:", dd.pred)
    # print("-" * 50)
    # # Expected:
    # # Node 4 becomes unreachable (inf)
    # # Nodes 5,6,7 that depended on 4 also become unreachable