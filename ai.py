
        # else:
        #     # Path wasn't via u - check if new path is better
        #     print(f"[{self.current_time:.1f}] Path used to process {v} was NOT via {u}.")
            
        #     if potential_new_dist_v < self.dist[v]:
        #         print(f"[{self.current_time:.1f}] New path via ({u},{v}) is better. Reactivating {v}.")
                
        #         # Revoke only the most recent deletion
        #         latest_del_time = deletion_times[0]
        #         print(f"[{self.current_time:.1f}] Revoking latest deletion at {latest_del_time:.1f}")
        #         inconsistent_del = self.rpq.revoke_del_min(latest_del_time)
        #         if inconsistent_del:
        #             print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

        #         # Update processed times
        #         if v in self.processed_times and deletion_times:
        #             self.processed_times[v].remove(latest_del_time)
        #             if not self.processed_times[v]:
        #                 del self.processed_times[v]

        #         self.dist[v] = potential_new_dist_v
        #         self.pred[v] = u
        #         self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
        #         self.propagate_changes(self.current_time)
        #     else:
        #         print(f"[{self.current_time:.1f}] No change needed for {v}. New path not better.")
# gpt
    # def handle_edge_update(self, u: int, v: int, new_weight: float):
    #     print(f"\n--- Handling Edge Update ({u}, {v}) to New Weight={new_weight:.1f} at Time={self.current_time:.1f} ---")

    #     # Update edge weight in the graph
    #     self.graph[u][v] = new_weight

    #     latest_node_v_rb = self.rpq.find_vertex_rbnode_in_tins(v, active_only=False)
    #     latest_pq_node_v = latest_node_v_rb.value if latest_node_v_rb else None

    #     # Case 1: 'v' not in RPQ at all
    #     if not latest_pq_node_v:
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

    #     # Case 2: 'v' is currently active in the RPQ
    #     if latest_pq_node_v.valid:
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

    #     # Case 3: 'v' has been processed previously
    #     print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
    #     deletion_time = self.processed_at_time.get(v)

    #     if deletion_time is None:
    #         print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion time recorded!")
    #         return

    #     path_via_u_used_when_processed = (self.pred[v] == u)
    #     potential_new_dist_v = self.dist[u] + new_weight if self.dist[u] != math.inf else math.inf

    #     if path_via_u_used_when_processed:
    #         print(f"[{self.current_time:.1f}] Processed path to {v} was via {u}. Invalidating.")
    #         inconsistent_del = self.rpq.revoke_del_min(deletion_time)
    #         if inconsistent_del:
    #             print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

    #         # Only invalidate if current edge weight actually changed
    #         self.dist[v] = math.inf
    #         self.pred[v] = None
    #         if v in self.processed_at_time:
    #             del self.processed_at_time[v]

    #         self.propagate_changes(self.current_time)

    #         # After propagation, if the same edge is still best, rediscover it
    #         if self.dist[u] != math.inf and (self.dist[u] + new_weight < self.dist[v]):
    #             print(f"[{self.current_time:.1f}] Edge ({u},{v}) still best after invalidation. Reinserting.")
    #             self.dist[v] = self.dist[u] + new_weight
    #             self.pred[v] = u
    #             self.rpq.invoke_insert(v, self.current_time, self.dist[v], u)
    #             self.propagate_changes(self.current_time)

    #     else:
    #         print(f"[{self.current_time:.1f}] Path used to process {v} was NOT via {u}.")
    #         if potential_new_dist_v < self.dist[v]:
    #             print(f"[{self.current_time:.1f}] New path via ({u},{v}) is better. Reactivating {v}.")
    #             inconsistent_del = self.rpq.revoke_del_min(deletion_time)
    #             if inconsistent_del:
    #                 print(f"[!ALERT!] Conflict with Del Min at {inconsistent_del.key[0]:.1f}")

    #             if v in self.processed_at_time:
    #                 del self.processed_at_time[v]

    #             self.dist[v] = potential_new_dist_v
    #             self.pred[v] = u
    #             self.rpq.invoke_insert(v, self.current_time, self.dist[v], u)
    #             self.propagate_changes(self.current_time)
    #         else:
    #             print(f"[{self.current_time:.1f}] No change needed for {v}. New path not better.")

    # def handle_edge_update(self, u: int, v: int, new_weight: float):
    #     """
    #     Handles the update of an edge weight (u, v) and triggers recalculations,
    #     designed to be more robust against multiple updates.
    #     """
    #     print(f"\n--- Handling Edge Update ({u}, {v}) to New Weight={new_weight:.1f} at Time={self.current_time:.1f} ---")

    #     # Find the latest RBNode associated with v in T_ins (could be valid or invalid)
    #     latest_node_v_rb = self.rpq.find_vertex_rbnode_in_tins(v, active_only=False)
    #     latest_pq_node_v = latest_node_v_rb.value if latest_node_v_rb else None

    #     # Calculate the potential distance via the updated edge, if u is reachable
    #     potential_new_dist_v = math.inf
    #     if self.dist[u] != math.inf:
    #         potential_new_dist_v = self.dist[u] + new_weight

    #     if not latest_pq_node_v:
    #         # --- CASE: Vertex 'v' not found in RPQ history ---
    #         print(f"[{self.current_time:.1f}] Vertex {v} not found in RPQ history.")
    #         # Check if the new edge provides a path (better than current self.dist[v], which might be inf)
    #         if potential_new_dist_v < self.dist[v]:
    #             print(f"[{self.current_time:.1f}] New edge creates path to {v}. Dist {self.dist[v]:.1f} -> {potential_new_dist_v:.1f}")
    #             self.dist[v] = potential_new_dist_v
    #             self.pred[v] = u
    #             self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
    #             self.propagate_changes(self.current_time) # Propagate potential effects
    #         else:
    #             print(f"[{self.current_time:.1f}] New edge doesn't improve path to {v} (or {u} unreachable). No action.")

    #     elif latest_pq_node_v.valid:
    #         # --- CASE: Vertex 'v' is ACTIVE in RPQ ---
    #         print(f"[{self.current_time:.1f}] Vertex {v} is ACTIVE in RPQ (Key={latest_node_v_rb.key}).")

    #         # Regardless of the current predecessor, the path via (u,v) might have changed.
    #         # We should ensure the RPQ considers the path with the new weight.
    #         # Strategy: Revoke the current active entry found and insert the new potential path.
    #         # (Note: A more complex strategy might involve finding *all* active entries for v,
    #         #  but revoking the latest and inserting the new one is a reasonable simplification).

    #         # 1. Revoke the latest active entry for v
    #         key_to_revoke = latest_node_v_rb.key
    #         print(f"[{self.current_time:.1f}] Revoking active entry {key_to_revoke} for {v}.")
    #         revoke_result = self.rpq.revoke_insert(key_to_revoke)
    #         if revoke_result: # Check if revoke caused inconsistency
    #              print(f"[!ALERT!] Revoke Insert for active {v} conflicts with Del Min at {revoke_result.key[0]:.1f}")

    #         # 2. Insert the new potential path if u is reachable
    #         if potential_new_dist_v != math.inf:
    #             print(f"[{self.current_time:.1f}] Inserting new potential path for active {v} via {u} with dist {potential_new_dist_v:.1f}")
    #             # Update main distance optimistically - propagation will correct if needed
    #             # Only update if potentially better or if the revoked one was the only path?
    #             # Let's update if it seems better than current known *best* distance,
    #             # RPQ handles multiple entries correctly.
    #             if potential_new_dist_v < self.dist[v]:
    #                  self.dist[v] = potential_new_dist_v
    #                  self.pred[v] = u

    #             insert_result = self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
    #             if insert_result: # Check if insert caused inconsistency
    #                  print(f"[!ALERT!] Insert for active {v} conflicts with Del Min at {insert_result.key[0]:.1f}")

    #             # 3. Trigger propagation because the state for active 'v' changed
    #             self.propagate_changes(self.current_time)
    #         else:
    #             print(f"[{self.current_time:.1f}] Not inserting path for active {v} via {u} because {u} is unreachable.")
    #             # Still propagate if the revoke might have changed the minimum
    #             self.propagate_changes(self.current_time)


    #     else: # --- CASE: Vertex 'v' was PROCESSED ---
    #         print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
    #         deletion_time = self.processed_at_time.get(v)

    #         if deletion_time is None:
    #             print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion time recorded!")
    #             return # Cannot proceed safely

    #         processed_dist = self.dist[v] # The distance v was finalized with
    #         processed_pred = self.pred[v] # The predecessor v was finalized with
    #         print(f"[{self.current_time:.1f}] Vertex {v} processed at time {deletion_time:.1f} with dist {processed_dist:.1f} via pred {processed_pred}.")

    #         # Condition 1: New path via 'u' is strictly better than the processed distance
    #         if potential_new_dist_v < processed_dist:
    #             print(f"[{self.current_time:.1f}] New path via ({u},{v}) ({potential_new_dist_v:.1f}) is better than processed dist ({processed_dist:.1f}). Reactivating {v}.")

    #             # Reactivate 'v' with this better path
    #             # 1. Revoke the original deletion
    #             inconsistent_del = self.rpq.revoke_del_min(deletion_time)
    #             if inconsistent_del: print(f"[!ALERT!] Revoke Del Min for {v} conflicts...") # Abbreviated

    #             # 2. Update 'v' with the new better path
    #             self.dist[v] = potential_new_dist_v
    #             self.pred[v] = u
    #             if v in self.processed_at_time:
    #                 del self.processed_at_time[v]
    #                 print(f"[{self.current_time:.1f}] Cleared processed status for {v}.")

    #             # 3. Re-insert 'v' into RPQ
    #             insert_result = self.rpq.invoke_insert(v, self.current_time, self.dist[v], self.pred[v])
    #             if insert_result: print(f"[!ALERT!] Insert for reactivated {v} conflicts...") # Abbreviated

    #             # 4. Trigger propagation
    #             print(f"[{self.current_time:.1f}] Re-inserted {v} with better path. Triggering propagation.")
    #             self.propagate_changes(self.current_time)

    #         # Condition 2: Path used for processing was via 'u' AND the new path isn't an improvement
    #         elif processed_pred == u and potential_new_dist_v >= processed_dist:
    #              # Cost basis is wrong due to edge (u,v) change (likely increase). Must invalidate.
    #              print(f"[{self.current_time:.1f}] Path used to process {v} via edge ({u},{v}) cost changed negatively. Invalidating {v}.")

    #              # Invalidate 'v'
    #              # 1. Revoke deletion
    #              inconsistent_del = self.rpq.revoke_del_min(deletion_time)
    #              if inconsistent_del: print(f"[!ALERT!] Revoke Del Min for {v} conflicts...") # Abbreviated

    #              # 2. Reset distance/predecessor
    #              self.dist[v] = math.inf
    #              self.pred[v] = None
    #              if v in self.processed_at_time:
    #                 del self.processed_at_time[v]
    #                 print(f"[{self.current_time:.1f}] Cleared processed status for {v}.")

    #              # 3. DO NOT re-insert. Let propagation find the new best path TO v.

    #              # 4. Trigger propagation
    #              print(f"[{self.current_time:.1f}] Triggering propagation to re-evaluate paths.")
    #              self.propagate_changes(self.current_time)

    #         # Condition 3: Path used was not via 'u' AND new path isn't better
    #         else: # (processed_pred != u and potential_new_dist_v >= processed_dist)
    #              print(f"[{self.current_time:.1f}] Path used for {v} not via {u}, and new path via ({u},{v}) not better. No change needed for {v}.")
    #              # No direct action for 'v'.

        # Moved time increment and completion message outside this function
    # def handle_edge_update(self, u: int, v: int, new_weight: float):
    #     """
    #     Handles the update of an edge weight (u, v) and triggers recalculations,
    #     designed to be more robust against multiple updates.
    #     """
    #     print(f"\n--- Handling Edge Update ({u}, {v}) to New Weight={new_weight:.1f} at Time={self.current_time:.1f} ---")

    #     # Find the latest RBNode associated with v in T_ins (could be valid or invalid)
    #     latest_node_v_rb = self.rpq.find_vertex_rbnode_in_tins(v, active_only=False)
    #     latest_pq_node_v = latest_node_v_rb.value if latest_node_v_rb else None

    #     # Calculate the potential distance via the updated edge, if u is reachable
    #     potential_new_dist_v = math.inf
    #     if self.dist[u] != math.inf:
    #         potential_new_dist_v = self.dist[u] + new_weight

    #     if not latest_pq_node_v:
    #         # --- CASE: Vertex 'v' not found in RPQ history ---
    #         print(f"[{self.current_time:.1f}] Vertex {v} not found in RPQ history.")
    #         # Check if the new edge provides a path (better than current self.dist[v], which might be inf)
    #         if potential_new_dist_v < self.dist[v]:
    #             print(f"[{self.current_time:.1f}] New edge creates path to {v}. Dist {self.dist[v]:.1f} -> {potential_new_dist_v:.1f}")
    #             self.dist[v] = potential_new_dist_v
    #             self.pred[v] = u
    #             self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
    #             self.propagate_changes(self.current_time) # Propagate potential effects
    #         else:
    #             print(f"[{self.current_time:.1f}] New edge doesn't improve path to {v} (or {u} unreachable). No action.")

    #     elif latest_pq_node_v.valid:
    #         # --- CASE: Vertex 'v' is ACTIVE in RPQ ---
    #         print(f"[{self.current_time:.1f}] Vertex {v} is ACTIVE in RPQ (Key={latest_node_v_rb.key}).")

    #         # Regardless of the current predecessor, the path via (u,v) might have changed.
    #         # We should ensure the RPQ considers the path with the new weight.
    #         # Strategy: Revoke the current active entry found and insert the new potential path.
    #         # (Note: A more complex strategy might involve finding *all* active entries for v,
    #         #  but revoking the latest and inserting the new one is a reasonable simplification).

    #         # 1. Revoke the latest active entry for v
    #         key_to_revoke = latest_node_v_rb.key
    #         print(f"[{self.current_time:.1f}] Revoking active entry {key_to_revoke} for {v}.")
    #         revoke_result = self.rpq.revoke_insert(key_to_revoke)
    #         if revoke_result: # Check if revoke caused inconsistency
    #              print(f"[!ALERT!] Revoke Insert for active {v} conflicts with Del Min at {revoke_result.key[0]:.1f}")

    #         # 2. Insert the new potential path if u is reachable
    #         if potential_new_dist_v != math.inf:
    #             print(f"[{self.current_time:.1f}] Inserting new potential path for active {v} via {u} with dist {potential_new_dist_v:.1f}")
    #             # Update main distance optimistically - propagation will correct if needed
    #             # Only update if potentially better or if the revoked one was the only path?
    #             # Let's update if it seems better than current known *best* distance,
    #             # RPQ handles multiple entries correctly.
    #             if potential_new_dist_v < self.dist[v]:
    #                  self.dist[v] = potential_new_dist_v
    #                  self.pred[v] = u

    #             insert_result = self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
    #             if insert_result: # Check if insert caused inconsistency
    #                  print(f"[!ALERT!] Insert for active {v} conflicts with Del Min at {insert_result.key[0]:.1f}")

    #             # 3. Trigger propagation because the state for active 'v' changed
    #             self.propagate_changes(self.current_time)
    #         else:
    #             print(f"[{self.current_time:.1f}] Not inserting path for active {v} via {u} because {u} is unreachable.")
    #             # Still propagate if the revoke might have changed the minimum
    #             self.propagate_changes(self.current_time)


    #     else: # --- CASE: Vertex 'v' was PROCESSED ---
    #         print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
    #         deletion_time = self.processed_at_time.get(v)

    #         if deletion_time is None:
    #             print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion time recorded!")
    #             return # Cannot proceed safely

    #         processed_dist = self.dist[v] # The distance v was finalized with
    #         processed_pred = self.pred[v] # The predecessor v was finalized with
    #         print(f"[{self.current_time:.1f}] Vertex {v} processed at time {deletion_time:.1f} with dist {processed_dist:.1f} via pred {processed_pred}.")

    #         # Condition 1: New path via 'u' is strictly better than the processed distance
    #         if potential_new_dist_v < processed_dist:
    #             print(f"[{self.current_time:.1f}] New path via ({u},{v}) ({potential_new_dist_v:.1f}) is better than processed dist ({processed_dist:.1f}). Reactivating {v}.")

    #             # Reactivate 'v' with this better path
    #             # 1. Revoke the original deletion
    #             inconsistent_del = self.rpq.revoke_del_min(deletion_time)
    #             if inconsistent_del: print(f"[!ALERT!] Revoke Del Min for {v} conflicts...") # Abbreviated

    #             # 2. Update 'v' with the new better path
    #             self.dist[v] = potential_new_dist_v
    #             self.pred[v] = u
    #             if v in self.processed_at_time:
    #                 del self.processed_at_time[v]
    #                 print(f"[{self.current_time:.1f}] Cleared processed status for {v}.")

    #             # 3. Re-insert 'v' into RPQ
    #             insert_result = self.rpq.invoke_insert(v, self.current_time, self.dist[v], self.pred[v])
    #             if insert_result: print(f"[!ALERT!] Insert for reactivated {v} conflicts...") # Abbreviated

    #             # 4. Trigger propagation
    #             print(f"[{self.current_time:.1f}] Re-inserted {v} with better path. Triggering propagation.")
    #             self.propagate_changes(self.current_time)

    #         # Condition 2: Path used for processing was via 'u' AND the new path isn't an improvement
    #         elif processed_pred == u and potential_new_dist_v >= processed_dist:
    #              # Cost basis is wrong due to edge (u,v) change (likely increase). Must invalidate.
    #              print(f"[{self.current_time:.1f}] Path used to process {v} via edge ({u},{v}) cost changed negatively. Invalidating {v}.")

    #              # Invalidate 'v'
    #              # 1. Revoke deletion
    #              inconsistent_del = self.rpq.revoke_del_min(deletion_time)
    #              if inconsistent_del: print(f"[!ALERT!] Revoke Del Min for {v} conflicts...") # Abbreviated

    #              # 2. Reset distance/predecessor
    #              self.dist[v] = math.inf
    #              self.pred[v] = None
    #              if v in self.processed_at_time:
    #                 del self.processed_at_time[v]
    #                 print(f"[{self.current_time:.1f}] Cleared processed status for {v}.")

    #              # 3. DO NOT re-insert. Let propagation find the new best path TO v.

    #              # 4. Trigger propagation
    #              print(f"[{self.current_time:.1f}] Triggering propagation to re-evaluate paths.")
    #              self.propagate_changes(self.current_time)

    #         # Condition 3: Path used was not via 'u' AND new path isn't better
    #         else: # (processed_pred != u and potential_new_dist_v >= processed_dist)
    #              print(f"[{self.current_time:.1f}] Path used for {v} not via {u}, and new path via ({u},{v}) not better. No change needed for {v}.")
    #              # No direct action for 'v'.

        # Moved time increment and completion message outside this function
    # def handle_edge_update(self, u: int, v: int, new_weight: float):
    #     print(f"\n--- Handling Edge Update ({u}, {v}) to New Weight={new_weight:.1f} at Time={self.current_time:.1f} ---")
    #     # Assume graph update happens outside or before this function now
    #     # old_weight = self.graph[u][v]
    #     # self.graph[u][v] = new_weight

    #     latest_node_v_rb = self.rpq.find_vertex_rbnode_in_tins(v, active_only=False)
    #     latest_pq_node_v = latest_node_v_rb.value if latest_node_v_rb else None

    #     if not latest_pq_node_v: # Includes case where latest_node_v_rb is None
    #          print(f"[{self.current_time:.1f}] Vertex {v} not found in RPQ history.")
    #          # Check potential new path only if 'u' is reachable
    #          if self.dist[u] != math.inf:
    #              potential_new_dist_v = self.dist[u] + new_weight
    #              if potential_new_dist_v < self.dist[v]:
    #                   print(f"[{self.current_time:.1f}] New edge creates better path to {v}. Dist {self.dist[v]:.1f} -> {potential_new_dist_v:.1f}")
    #                   self.dist[v] = potential_new_dist_v
    #                   self.pred[v] = u
    #                   self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
    #                   self.propagate_changes(self.current_time) # Start propagation immediately
    #              else:
    #                   print(f"[{self.current_time:.1f}] New edge doesn't improve path to unreachable {v}. No action.")
    #          else:
    #              print(f"[{self.current_time:.1f}] Source vertex {u} is unreachable. No path improvement possible.")


    #     elif latest_pq_node_v.valid:
    #          print(f"[{self.current_time:.1f}] Vertex {v} is ACTIVE in RPQ.")
    #          # Check if path being updated is the current best for v
    #          if self.pred[v] == u:
    #              print(f"[{self.current_time:.1f}] Current path to active {v} is via {u}. Updating.")
    #              new_dist_v = self.dist[u] + new_weight
    #              # Safely get the key of the RBNode to revoke
    #              key_to_revoke = latest_node_v_rb.key if latest_node_v_rb else None
    #              if key_to_revoke:
    #                   self.rpq.revoke_insert(key_to_revoke)
    #              else:
    #                   print(f"ERROR [{self.current_time:.1f}]: Cannot revoke, RBNode link missing for active {v}")

    #              self.dist[v] = new_dist_v
    #              self.rpq.invoke_insert(v, self.current_time, new_dist_v, u)
    #              self.propagate_changes(self.current_time)
    #          else:
    #              # Check if u->v is now better
    #              print(f"[{self.current_time:.1f}] Current path to active {v} NOT via {u}. Check if ({u},{v}) better.")
    #              if self.dist[u] != math.inf: # Can we reach u?
    #                   potential_new_dist_v = self.dist[u] + new_weight
    #                   if potential_new_dist_v < self.dist[v]:
    #                        print(f"[{self.current_time:.1f}] Path via ({u},{v}) now better for active {v}. Updating.")
    #                        self.dist[v] = potential_new_dist_v
    #                        self.pred[v] = u
    #                        # Insert the better path - RPQ handles finding minimum later
    #                        self.rpq.invoke_insert(v, self.current_time, potential_new_dist_v, u)
    #                        # Propagate needed as this might be the new minimum overall
    #                        self.propagate_changes(self.current_time)
    #                   else:
    #                        print(f"[{self.current_time:.1f}] Path via ({u},{v}) not better for active {v}.")
    #              else:
    #                   print(f"[{self.current_time:.1f}] Cannot check path via ({u},{v}) as {u} is unreachable.")
 
    #     else: # latest_pq_node_v exists but is not valid -> Vertex 'v' was processed
    #         print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
    #         deletion_time = self.processed_at_time.get(v)

    #         # *** Check if deletion time exists ***
    #         if deletion_time is None:
    #             print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion time recorded!")
    #             # Attempting recovery might be complex, safer to return/raise here
    #             # For now, just log and return
    #             return

    #         print(f"[{self.current_time:.1f}] Vertex {v} was processed at time {deletion_time:.1f} with dist {self.dist[v]:.1f} via pred {self.pred[v]}.")

    #         # --- CRITICAL CHECK for INCREASED WEIGHT ---
    #         # We only need to take action if the path used to process 'v'
    #         # *was* the one coming from 'u', because that path's cost is now invalid.
    #         # We use self.pred[v] which stores the predecessor used when v was finalized.
    #         path_via_u_used_when_processed = (self.pred[v] == u)

    #         if path_via_u_used_when_processed:
    #             # The cost basis for processing 'v' and potentially its downstream nodes is now wrong.
    #             print(f"[{self.current_time:.1f}] Path to processed {v} used edge ({u},{v}) which increased weight. Must re-evaluate paths from {v} onwards.")

    #             # --- Invalidation and Re-evaluation Strategy ---
    #             # 1. Revoke the original deletion of 'v'. Makes 'v' conceptually active again,
    #             #    allowing it to be re-processed if a valid path is found.
    #             inconsistent_del = self.rpq.revoke_del_min(deletion_time)
    #             if inconsistent_del:
    #                 print(f"[!ALERT!] Revoke Del Min for {v} at time {deletion_time:.1f} conflicts with a later Del Min at {inconsistent_del.key[0]:.1f}")
    #                 # More robust handling (e.g., revoking the conflicting one too) might be needed.

    #             # 2. Invalidate the current known shortest path to 'v'.
    #             #    Set distance to infinity to force rediscovery of the best path *to* v.
    #             print(f"[{self.current_time:.1f}] Invalidating current path to {v}. Setting dist=inf.")
    #             self.dist[v] = math.inf
    #             self.pred[v] = None # Remove predecessor link

    #             # 3. Clear the processed time for 'v'. It needs to be re-processed
    #             #    if a new path is found.
    #             if v in self.processed_at_time:
    #                 del self.processed_at_time[v]
    #                 print(f"[{self.current_time:.1f}] Cleared processed status for {v}.")

    #             # 4. DO NOT re-insert 'v' immediately. We need the propagation
    #             #    to find the *new* best path *to* v first. If a node 'x'
    #             #    can now reach 'v' cheaper than any previous path (including the
    #             #    now-more-expensive u->v path), the relaxation of 'x' will
    #             #    insert 'v' into the RPQ. If the u->v path (with the new weight)
    #             #    happens to be the best again, the relaxation from 'u' (if 'u'
    #             #    is processed) will re-insert 'v'.

    #             # 5. Trigger propagation. This is essential to find the new best path
    #             #    to 'v' (if one exists) and update all nodes downstream from the
    #             #    original processing of 'v'.
    #             print(f"[{self.current_time:.1f}] Triggering propagation to re-evaluate paths potentially affected by {v}.")
    #             self.propagate_changes(self.current_time) # Start propagation immediately
    #         else:
    #             # CASE 2: Path used for processing 'v' was NOT via 'u'.
    #             # The original processing of 'v' remains valid based on its path cost.
    #             # BUT, the updated edge (u,v) might offer a *new, shorter* path now.
    #             print(f"[{self.current_time:.1f}] Path used to process {v} was NOT via {u}.")

    #             # Check if u is reachable and calculate potential new distance
    #             potential_new_dist_v = math.inf
    #             if self.dist[u] != math.inf:
    #                 potential_new_dist_v = self.dist[u] + new_weight

    #             # Compare with the distance 'v' was originally processed with
    #             if potential_new_dist_v < self.dist[v]:
    #                 # The new path via 'u' is strictly better than the path used to process 'v'.
    #                 print(f"[{self.current_time:.1f}] New path via ({u},{v}) ({potential_new_dist_v:.1f}) is better than processed dist ({self.dist[v]:.1f}). Reactivating {v}.")

    #                 # Need to reactivate 'v' with this better path. Steps are similar to CASE 1's reactivation.
    #                 # 1. Revoke the original deletion
    #                 inconsistent_del = self.rpq.revoke_del_min(deletion_time)
    #                 if inconsistent_del:
    #                     print(f"[!ALERT!] Revoke Del Min for {v} conflicts with Del Min at {inconsistent_del.key[0]:.1f}")

    #                 # 2. Update 'v' with the new better path
    #                 self.dist[v] = potential_new_dist_v
    #                 self.pred[v] = u
    #                 if v in self.processed_at_time:
    #                     del self.processed_at_time[v] # Clear processed status
    #                     print(f"[{self.current_time:.1f}] Cleared processed status for {v}.")


    #                 # 3. Re-insert 'v' into RPQ with the new better distance/time
    #                 self.rpq.invoke_insert(v, self.current_time, self.dist[v], self.pred[v])

    #                 # 4. Trigger propagation to update downstream nodes based on v's new (better) distance.
    #                 print(f"[{self.current_time:.1f}] Re-inserted {v} with better path. Triggering propagation.")
    #                 self.propagate_changes(self.current_time)
    #             else:
    #                 # The new path via (u,v) is not better than the path 'v' was processed with.
    #                 print(f"[{self.current_time:.1f}] New path via ({u},{v}) ({potential_new_dist_v:.1f}) is not better than processed dist ({self.dist[v]:.1f}). No change needed for {v}.")
    #                 # No action needed for 'v' itself. The change to (u,v) doesn't affect it directly here.

        # # --- End of handling processed node 'v' ---
        # else: # latest_pq_node_v is not valid -> Vertex 'v' was processed
        #     print(f"[{self.current_time:.1f}] Vertex {v} was already PROCESSED.")
        #     deletion_time = self.processed_at_time.get(v)

        #     # *** Check if deletion time exists ***
        #     if deletion_time is None:
        #         print(f"ERROR [{self.current_time:.1f}]: Processed vertex {v} has no deletion time recorded!")
        #         return # Cannot proceed

        #     print(f"[{self.current_time:.1f}] Vertex {v} was processed at time {deletion_time:.1f}.")

        #     # Check if path via u was used (approximation) OR if u->v is now better
        #     path_via_u_used = (self.pred[v] == u)
        #     new_path_better = False
        #     if self.dist[u] != math.inf:
        #          potential_new_dist_v = self.dist[u] + new_weight
        #          if potential_new_dist_v < self.dist[v]:
        #               new_path_better = True

        #     if path_via_u_used or new_path_better:
        #          if path_via_u_used:
        #               print(f"[{self.current_time:.1f}] Path to processed {v} was via {u}. Reactivating.")
        #          if new_path_better:
        #               print(f"[{self.current_time:.1f}] Path via ({u},{v}) now better than processed path. Reactivating.")

        #          # --- Retroactive Action ---
        #          # 1. Revoke deletion
        #          inconsistent_del = self.rpq.revoke_del_min(deletion_time)
        #          if inconsistent_del:
        #              print(f"[!ALERT!] Revoke Del Min for {v} conflicts with Del Min at {inconsistent_del.key[0]:.1f}")
        #              # Add more robust handling if needed

        #          # 2. Update distance based on the better path (which must be u->v if new_path_better)
        #          new_dist_v = self.dist[u] + new_weight
        #          self.dist[v] = new_dist_v
        #          self.pred[v] = u # Predecessor is now u

        #          # 3. Re-insert 'v'
        #          self.rpq.invoke_insert(v, self.current_time, new_dist_v, u)
        #          # Clear processed time only if revoke succeeded? Maybe always clear?
        #          if v in self.processed_at_time:
        #               del self.processed_at_time[v]

        #          # 4. Propagate changes
        #          print(f"[{self.current_time:.1f}] Re-inserted {v}. Propagating changes.")
        #          self.propagate_changes(self.current_time)
        #     else:
        #           print(f"[{self.current_time:.1f}] Path to processed {v} not via {u}, and new path not better. No change.")

        # self._increment_time() # Increment time after handling the update
        # print(f"--- Edge Update ({u}, {v}) Handling Complete at Time={self.current_time:.1f} ---")

    # _increment_time() is now called outside handle_edge_update in update_edge
    # print(f"--- Edge Update ({u}, {v}) Handling Complete ---") # Moved outside


    # def revoke_del_min(self, time: float) -> Optional[RBNode]:
    #     print(f"RPQ: Revoke Del_Min for time={time:.1f}")
    #     del_key = (time,)
    #     node_in_tdm = self.T_d_m.search(del_key) # Returns None if not found
    #     # *** Added Check ***
    #     if not node_in_tdm:
    #         print(f"RPQ: Revoke Del_Min failed: No deletion recorded at time {time:.1f}")
    #         return None
    #     pq_node_to_revive = node_in_tdm.value
    #     # *** Added Check ***
    #     if not isinstance(pq_node_to_revive, PQNode):
    #          print(f"RPQ ERROR: T_d_m node {del_key} has non-PQNode value: {type(pq_node_to_revive)}")
    #          # Cannot proceed without the PQNode
    #          # Try removing from T_d_m anyway? Or just return?
    #          node_in_tdm.value.valid = False # Mark as invalid
    #         #  self.T_d_m.remove_node(del_key) # Attempt cleanup
    #          return None
    #     print(f"RPQ: Found deletion record for: {pq_node_to_revive}")
    #     # Check tins_node link before using it
    #     if pq_node_to_revive.tins_node and pq_node_to_revive.tins_node != self.T_ins.NIL:
    #          if not pq_node_to_revive.valid:
    #               pq_node_to_revive.valid = True
    #               print(f"RPQ: Marked original T_ins node {pq_node_to_revive.tins_node.key} back to valid.")
    #          else:
    #               print(f"RPQ: Warning - PQNode {pq_node_to_revive} was already marked valid?")
    #          pq_node_to_revive.deleted_by_node = None
    #     else:
    #          print(f"RPQ: Revoke Del_Min Error: Could not find original T_ins node link for PQNode {pq_node_to_revive}")
    #          # Cannot mark original node valid, inconsistency remains
    #          # Still remove from T_d_m but maybe return error indicator?

    #     # Physically remove from T_d_m
    #     node_in_tdm.value.valid = False # Mark as invalid
    #     # removed_node = self.T_d_m.remove_node(del_key)
    #     # if removed_node:
    #     #      print(f"RPQ: Removed node {del_key} from T_d_m.")
    #     # else:
    #     #      print(f"RPQ: Revoke Del_Min Warning: Failed to remove node {del_key} from T_d_m (was already removed?).")

    #     # --- Check for inconsistency ---
    #     # Need the revived node's key safely
    #     revived_node_key = None
    #     if pq_node_to_revive.tins_node and pq_node_to_revive.tins_node.key is not None:
    #         revived_node_key = pq_node_to_revive.tins_node.key
    #     else:
    #         print(f"RPQ WARNING: Cannot check inconsistency after revoke; revived node key unavailable.")
    #         return None # Cannot proceed with check
    #     potential_del_node = self.T_d_m.search_min_greater_equal((time + 1e-9,)) # Search strictly AFTER time
    #     while potential_del_node: # Checks not None
    #          deleted_pq_node = potential_del_node.value
    #          # Check both nodes and keys exist before comparing
    #          if (deleted_pq_node and deleted_pq_node.tins_node and
    #              deleted_pq_node.tins_node.key is not None and revived_node_key is not None):
    #              if revived_node_key < deleted_pq_node.tins_node.key:
    #                    print(f"RPQ: Revoked deletion at {time:.1f} (revived {revived_node_key}) invalidates deletion at {potential_del_node.key[0]:.1f}")
    #                    return potential_del_node # Report inconsistency
    #          # Optional: Debug prints for missing info during check

    #          potential_del_node = self.T_d_m.successor(potential_del_node)

    #     return None
