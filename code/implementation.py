import sys
import heapq # Used for standard Dijkstra comparison if needed, not for RPQ

# --- Node Class for Red-Black Tree ---
# --- Node Class for Red-Black Tree ---
class Node:
    def __init__(self, key=None, value_dict=None, color='R', parent=None, left=None, right=None):
        self.key = key # The key used for comparison in this specific tree
        self.value = value_dict if value_dict is not None else {} # Holds all associated data
        self.color = color
        self.parent = parent
        self.left = left
        self.right = right
        # --- RPQ Specific Fields ---
        self.valid = True # Is this node currently considered active in T_ins?
        self.t_ins_link = None # Link from T_d_m node to the corresponding T_ins node
        self.t_dm_link = None # Link from T_ins node to the T_d_m node that deleted it (if any)

    def __lt__(self, other):
        # Comparison primarily based on the key
        if other is None:
            return False # Nothing is less than None
        # Handle comparison with TNULL correctly
        if self.key is None and other.key is None: return False
        if self.key is None: return True # None key is considered smaller
        if other.key is None: return False
        return self.key < other.key

    def __gt__(self, other):
       if other is None:
           return True # Everything is greater than None
       # Handle comparison with TNULL correctly
       if self.key is None and other.key is None: return False
       if self.key is None: return False
       if other.key is None: return True # None key is considered smaller
       return self.key > other.key

    def __eq__(self, other):
        if other is None:
            return False
        # Equality check considers the full value dictionary for uniqueness if keys match
        return self.key == other.key #and self.value == other.value # Value check might be too strict

    def __str__(self):
        # Basic string representation for debugging
        dist_val = self.value.get('dist', 'N/A')
        # Conditionally format distance only if it's a number
        dist_str = f"{dist_val:.1f}" if isinstance(dist_val, (int, float)) else str(dist_val)

        val_str = f"V:{self.value.get('vertex', 'N/A')}, " \
                  f"D:{dist_str}, " \
                  f"IT:{self.value.get('ins_time', 'N/A')}, " \
                  f"P:{self.value.get('pred', 'N/A')}, " \
                  f"DT:{self.value.get('del_time', 'N/A')}"
        link_info = f" DML:{self.t_dm_link is not None} INSL:{self.t_ins_link is not None}" if self.t_dm_link or self.t_ins_link else ""
        # Handle printing TNULL node key
        key_str = "None" if self.key is None else str(self.key)
        return f"K:{key_str} ({self.color},{'V' if self.valid else 'I'})[{val_str}]{link_info}"
# --- Red-Black Tree Implementation ---
class RedBlackTree:
    def __init__(self, tree_type='T_ins'):
        """
        Initializes a Red-Black Tree.

        Args:
            tree_type: 'T_ins' or 'T_d_m' to determine key structure and comparison.
        """
        self.TNULL = Node(color='B', key=None, value_dict={}) # Sentinel Node
        self.root = self.TNULL
        self.tree_type = tree_type
        # Optional: Keep track of nodes by a unique identifier if needed for fast lookup
        # self.node_map = {} # e.g., {vertex_id: node_instance} or {ins_time: node_instance}

    def _get_key(self, value_dict):
        """Helper to extract the correct key based on tree type."""
        if self.tree_type == 'T_ins':
            # Key for T_ins: (distance, insertion_time)
            dist = value_dict.get('dist', float('inf'))
            ins_time = value_dict.get('ins_time', -1)
            return (dist, ins_time)
        elif self.tree_type == 'T_d_m':
            # Key for T_d_m: deletion_time
            return value_dict.get('del_time', -1)
        else:
            raise ValueError("Invalid tree_type")

    def minimum(self, node):
        """Finds the node with the minimum key in the subtree rooted at node."""
        while node.left != self.TNULL:
            node = node.left
        return node

    def maximum(self, node):
        """Finds the node with the maximum key in the subtree rooted at node."""
        while node.right != self.TNULL:
            node = node.right
        return node

    def successor(self, x):
        """Finds the successor of a given node x."""
        if x.right != self.TNULL:
            return self.minimum(x.right)
        y = x.parent
        while y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    def predecessor(self, x):
        """Finds the predecessor of a given node x."""
        if x.left != self.TNULL:
            return self.maximum(x.left)
        y = x.parent
        while y != self.TNULL and x == y.left:
            x = y
            y = y.parent
        return y

    def search(self, key, node=None):
        """Searches for a node with the exact key."""
        if node is None:
            node = self.root

        current = node
        while current != self.TNULL and key != current.key:
            if key < current.key:
                current = current.left
            else:
                current = current.right
        return current # Returns TNULL if not found

    def search_le(self, key, node=None):
        """Finds the node with the largest key less than or equal to the given key."""
        if node is None:
            node = self.root
        result = self.TNULL
        current = node
        while current != self.TNULL:
            if current.key == key:
                return current
            elif current.key < key:
                result = current  # Potential candidate
                current = current.right
            else:
                current = current.left
        return result

    def search_ge(self, key, node=None):
        """Finds the node with the smallest key greater than or equal to the given key."""
        if node is None:
            node = self.root
        result = self.TNULL
        current = node
        # print(f"In search ge, root is {node.key, node.value}")

        while current != self.TNULL:
            if current.key == key:
                return current
            elif current.key > key:
                result = current # Potential candidate
                current = current.left
            else:
                current = current.right
        # print(f"returning {result.key, result.value}")        
        return result

    def _left_rotate(self, x):
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

    def _right_rotate(self, x):
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

    def _insert_fixup(self, k):
        """Maintains Red-Black properties after insertion."""
        while k.parent is not None and k.parent.color == 'R':
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left  # uncle
                if u.color == 'R':
                    # Case 3.1: Uncle is Red
                    u.color = 'B'
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    k = k.parent.parent
                else:
                    # Case 3.2: Uncle is Black
                    if k == k.parent.left:
                        # Case 3.2.2: Node is left child (requires right rotation)
                        k = k.parent
                        self._right_rotate(k)
                    # Case 3.2.1: Node is right child (requires left rotation)
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    self._left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right  # uncle
                if u.color == 'R':
                    # Case 3.1 (symmetric): Uncle is Red
                    u.color = 'B'
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    k = k.parent.parent
                else:
                     # Case 3.2 (symmetric): Uncle is Black
                    if k == k.parent.right:
                         # Case 3.2.2 (symmetric)
                        k = k.parent
                        self._left_rotate(k)
                    # Case 3.2.1 (symmetric)
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    self._right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 'B' # Root must always be black

    def insert(self, value_dict):
        """Inserts a value dictionary into the RBT."""
        key = self._get_key(value_dict)
        node = Node(key=key, value_dict=value_dict, parent=None, color='R',
                    left=self.TNULL, right=self.TNULL)

        y = None
        x = self.root

        while x != self.TNULL:
            y = x
            # Use __lt__ defined in Node for comparison
            if node < x:
                x = x.left
            else:
                x = x.right

        node.parent = y
        if y is None:
            self.root = node
        elif node < y:
            y.left = node
        else:
            y.right = node

        # If new node is root node, simply set color to black
        if node.parent is None:
            node.color = 'B'
            return node

        # If grandparent is null, simply return
        if node.parent.parent is None:
            return node

        # Fix the tree
        self._insert_fixup(node)
        return node # Return the newly inserted node

    def _rb_transplant(self, u, v):
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v is not None: # Check if v is TNULL
             v.parent = u.parent


    def _delete_fixup(self, x):
        """Maintains Red-Black properties after deletion."""
        if x is None: return # Handle case where x might become None

        while x != self.root and x.color == 'B':
            if x.parent is None: break # Should not happen if x is not root, safety check

            if x == x.parent.left:
                s = x.parent.right
                if s is None: break # Safety check

                if s.color == 'R':
                    # Case 3.1
                    s.color = 'B'
                    x.parent.color = 'R'
                    self._left_rotate(x.parent)
                    s = x.parent.right
                    if s is None: break # s might change after rotation

                # Now s must be black
                if (s.left is None or s.left.color == 'B') and \
                   (s.right is None or s.right.color == 'B'):
                    # Case 3.2: Both children of s are black
                    s.color = 'R'
                    x = x.parent
                else:
                    if s.right is None or s.right.color == 'B':
                        # Case 3.3: s.left is red, s.right is black
                        if s.left is not None: s.left.color = 'B'
                        s.color = 'R'
                        self._right_rotate(s)
                        s = x.parent.right
                        if s is None: break # s might change

                    # Case 3.4: s.right is red
                    s.color = x.parent.color
                    x.parent.color = 'B'
                    if s.right is not None: s.right.color = 'B'
                    self._left_rotate(x.parent)
                    x = self.root # Terminate loop
            else:
                 # Symmetric case: x is right child
                s = x.parent.left
                if s is None: break

                if s.color == 'R':
                    # Case 3.1 (symmetric)
                    s.color = 'B'
                    x.parent.color = 'R'
                    self._right_rotate(x.parent)
                    s = x.parent.left
                    if s is None: break

                # Now s must be black
                if (s.right is None or s.right.color == 'B') and \
                   (s.left is None or s.left.color == 'B'):
                     # Case 3.2 (symmetric)
                    s.color = 'R'
                    x = x.parent
                else:
                    if s.left is None or s.left.color == 'B':
                        # Case 3.3 (symmetric)
                        if s.right is not None: s.right.color = 'B'
                        s.color = 'R'
                        self._left_rotate(s)
                        s = x.parent.left
                        if s is None: break

                     # Case 3.4 (symmetric)
                    s.color = x.parent.color
                    x.parent.color = 'B'
                    if s.left is not None: s.left.color = 'B'
                    self._right_rotate(x.parent)
                    x = self.root
        if x is not None:
            x.color = 'B'


    def _delete_node_helper(self, node, key):
        """Helper function to find and delete a node with the given key."""
        z = self.search(key, node) # Find the node to delete
        if z == self.TNULL:
            # print(f"Warning: Key {key} not found in tree {self.tree_type} for deletion.")
            return # Key not found

        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            if x is None: x = self.TNULL # Ensure x is never Python None
            self._rb_transplant(z, z.right)
        elif z.right == self.TNULL:
            x = z.left
            if x is None: x = self.TNULL
            self._rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if x is None: x = self.TNULL

            if y.parent == z:
                 if x != self.TNULL: x.parent = y # Check if x is TNULL
            else:
                self._rb_transplant(y, y.right)
                y.right = z.right
                if y.right != self.TNULL: y.right.parent = y # Check if y.right is TNULL

            self._rb_transplant(z, y)
            y.left = z.left
            if y.left != self.TNULL: y.left.parent = y # Check if y.left is TNULL
            y.color = z.color

        if y_original_color == 'B':
             if x != self.TNULL: # Only fixup if x is not the sentinel
                  self._delete_fixup(x)


    def delete(self, key):
        """Deletes the node with the given key."""
        self._delete_node_helper(self.root, key)

    # --- Tree Traversal for Debugging/Verification ---
    def inorder(self, node=None):
        """Performs inorder traversal, returning list of (key, value['vertex'])."""
        if node is None:
            node = self.root
        result = []
        if node != self.TNULL:
            result.extend(self.inorder(node.left))
            result.append(node) # Append the whole node for detailed inspection
            result.extend(self.inorder(node.right))
        return result

    def find_min_valid(self, node=None):
        """Finds the minimum valid node in T_ins."""
        if self.tree_type != 'T_ins':
            raise TypeError("find_min_valid only applicable to T_ins")
        if node is None:
            node = self.root

        min_valid_node = self.TNULL
        current = node
        stack = []

        while stack or current != self.TNULL:
            while current != self.TNULL:
                stack.append(current)
                current = current.left

            current = stack.pop()

            # Check if this node is valid and smaller than current min_valid_node
            if current.valid:
                if min_valid_node == self.TNULL or current.key < min_valid_node.key:
                     min_valid_node = current
                     # Optimization: In T_ins, once we find the first valid node
                     # during inorder traversal, it must be the minimum valid.
                     return min_valid_node


            current = current.right # Move to the right subtree

        return min_valid_node # Return TNULL if no valid node found

    def find_min_valid_greater_than(self, threshold_key, node=None):
        """Finds the minimum valid node in T_ins with key > threshold_key."""
        if self.tree_type != 'T_ins':
            raise TypeError("find_min_valid_greater_than only applicable to T_ins")
        if node is None:
            node = self.root

        min_node = self.TNULL
        current = node
        stack = []

        while stack or current != self.TNULL:
            while current != self.TNULL:
                stack.append(current)
                current = current.left

            current = stack.pop()

            # Check validity and if greater than threshold
            if current.valid and current.key > threshold_key:
                 if min_node == self.TNULL or current.key < min_node.key:
                     min_node = current
                     # Optimization: Since we are doing inorder, the first
                     # valid node > threshold encountered is the smallest one.
                     return min_node # Found the smallest valid node > threshold

            # If current node key is <= threshold, we still need to check its right subtree
            # If current node key > threshold, we explored it, move right
            current = current.right

        return min_node # Return TNULL if no such node found


# --- Retroactive Priority Queue (RPQ) ---# --- Retroactive Priority Queue (RPQ) ---
class RPQ:
    def __init__(self):
        self.T_ins = RedBlackTree(tree_type='T_ins')
        self.T_d_m = RedBlackTree(tree_type='T_d_m')
        # Optional: Direct mapping for faster vertex lookup in T_ins
        self.vertex_nodes = {} # {vertex_id: [list of T_ins nodes for this vertex]}

    def _add_vertex_node(self, vertex, node):
        if vertex not in self.vertex_nodes:
            self.vertex_nodes[vertex] = []
        self.vertex_nodes[vertex].append(node)

    def _remove_vertex_node(self, vertex, node):
         if vertex in self.vertex_nodes:
            try:
                self.vertex_nodes[vertex].remove(node)
                if not self.vertex_nodes[vertex]: # Remove entry if list becomes empty
                    del self.vertex_nodes[vertex]
            except ValueError:
                 pass # Node wasn't in the list, ignore

    # def invoke_insert(self, vertex, time, dist, pred):
    #     """
    #     Performs Insert(x, t) operation.
    #     Args:
    #         vertex: The vertex being inserted.
    #         time: The insertion time.
    #         dist: Estimated distance.
    #         pred: Predecessor vertex.
    #     Returns:
    #         The T_d_m node of the first inconsistent Del_Min operation (if any), otherwise None.
    #     """
    #     # 1. Check for future inconsistencies in T_d_m
    #     # Find the smallest deletion time >= t
    #     inconsistent_del_node = self.T_d_m.search_ge(time)
    #     # Use T_d_m's TNULL for comparison
    #     if inconsistent_del_node != self.T_d_m.TNULL:
    #         # print(f"RPQ Warn: Inserting ({vertex},{time},{dist}) conflicts with future Del_Min at {inconsistent_del_node.key}")
    #         return inconsistent_del_node # Return the node representing the inconsistent delete

    #     # 2. Perform the insertion in T_ins
    #     value_dict = {'vertex': vertex, 'ins_time': time, 'dist': dist, 'pred': pred, 'del_time': None}
    #     new_node = self.T_ins.insert(value_dict)
    #     self._add_vertex_node(vertex, new_node)
    #     # print(f"RPQ Insert: {value_dict} -> Node {new_node}")
    #     return None # No inconsistency found
    def invoke_insert(self, vertex, time, dist, pred):
        """
        Performs Insert(x, t) operation.
        Args:
            vertex: The vertex being inserted.
            time: The insertion time.
            dist: Estimated distance.
            pred: Predecessor vertex.
        Returns:
            The T_d_m node of the first inconsistent Del_Min operation (if any), otherwise None.
        """
        # 1. Check for future inconsistencies in T_d_m
        inconsistent_del_node = self.T_d_m.search_ge(time)
        # Use T_d_m's TNULL for comparison
        if inconsistent_del_node != self.T_d_m.TNULL:
            # Handle future inconsistencies more robustly
            print(f"Inconsistent Del_Min found at time {time} for vertex {vertex}")
            return inconsistent_del_node  # Return the node representing the inconsistent delete

        # 2. Perform the insertion in T_ins
        value_dict = {'vertex': vertex, 'ins_time': time, 'dist': dist, 'pred': pred, 'del_time': None}
        new_node = self.T_ins.insert(value_dict)
        self._add_vertex_node(vertex, new_node)
        # print(f"RPQ Insert: {value_dict} -> Node {new_node}")
        return None  # No inconsistency found

    # def invoke_del_min(self, time):
    #     """
    #     Performs Del_Min(t) operation.
    #     Args:
    #         time: The time at which Del_Min is invoked.
    #     Returns:
    #         The value dictionary {'vertex': ..., 'dist': ...} of the deleted element, or None if PQ is empty or becomes inconsistent.
    #     Raises:
    #         ValueError: If the operation causes inconsistency with a future operation.
    #     """
    #     # Check for future inconsistencies (e.g., another operation at the exact same time)
    #     # This simple check might need refinement based on exact tie-breaking rules.
    #     # A stricter check might look for any operation >= time.
    #     future_op_node = self.T_d_m.search_ge(time)
    #     # Use T_d_m's TNULL for comparison
    #     if future_op_node != self.T_d_m.TNULL and future_op_node.key == time:
    #          raise ValueError(f"invoke_del_min({time}) conflicts with existing operation at the same time.")

    #     # 1. Find the latest Del_Min operation *before* time `t`
    #     pred_del_node = self.T_d_m.search_le(time - 1e-9) # Search for max key < time

    #     # Use T_ins's TNULL for nodes originating from T_ins
    #     node_to_delete = self.T_ins.TNULL
    #     # Use T_d_m's TNULL for nodes originating from T_d_m
    #     if pred_del_node == self.T_d_m.TNULL:
    #         # No previous deletions, find the overall minimum valid node in T_ins
    #         node_to_delete = self.T_ins.find_min_valid()
    #         # print(f"InvokeDelMin({time}): No prev delete. Min valid in T_ins: {node_to_delete}")
    #     else:
    #         # Find the minimum valid node in T_ins *strictly greater* than the previously deleted key
    #         prev_deleted_ins_node = pred_del_node.t_ins_link
    #         if prev_deleted_ins_node is None:
    #              # This case should ideally not happen if links are maintained correctly
    #              print(f"RPQ Error: T_d_m node at {pred_del_node.key} has no T_ins link!")
    #              node_to_delete = self.T_ins.find_min_valid() # Fallback: find overall minimum
    #         # Use T_ins's TNULL when checking the linked node
    #         elif prev_deleted_ins_node == self.T_ins.TNULL:
    #             print(f"RPQ Error: T_d_m node at {pred_del_node.key} links to T_ins TNULL!")
    #             node_to_delete = self.T_ins.find_min_valid() # Fallback
    #         else:
    #             threshold_key = prev_deleted_ins_node.key
    #             # print(f"InvokeDelMin({time}): Prev delete at {pred_del_node.key}, threshold key {threshold_key}")
    #             node_to_delete = self.T_ins.find_min_valid_greater_than(threshold_key)
    #             # print(f"InvokeDelMin({time}): Min valid > threshold: {node_to_delete}")


    #     # Use T_ins's TNULL for comparison
    #     if node_to_delete == self.T_ins.TNULL:
    #         # print(f"RPQ Warn: invoke_del_min({time}) - No valid element found to delete.")
    #         return None # Priority queue effectively empty at this point

    #     # 2. Mark the T_ins node as invalid
    #     node_to_delete.valid = False
    #     # print(f"RPQ DelMin: Marking T_ins node invalid: {node_to_delete}")


    #     # 3. Insert a record into T_d_m
    #     del_value_dict = {'del_time': time, 'vertex': node_to_delete.value['vertex']} # Store minimal info needed
    #     del_node = self.T_d_m.insert(del_value_dict)
    #     # print(f"RPQ DelMin: Inserted into T_d_m: {del_node}")


    #     # 4. Establish links
    #     node_to_delete.t_dm_link = del_node
    #     del_node.t_ins_link = node_to_delete

    #     # Return the value of the deleted element
    #     return node_to_delete.value.copy() # Return a copy

    def invoke_del_min(self, time):
        """
        Performs Del_Min(t) operation.
        Args:
            time: The time at which Del_Min is invoked.
        Returns:
            The value dictionary {'vertex': ..., 'dist': ...} of the deleted element, or None if PQ is empty or becomes inconsistent.
        """
        # Check for future inconsistencies (e.g., another operation at the exact same time)
        # This simple check might need refinement based on exact tie-breaking rules.
        future_op_node = self.T_d_m.search_ge(time)
        # Use T_d_m's TNULL for comparison
        if future_op_node != self.T_d_m.TNULL and future_op_node.key == time:
            raise ValueError(f"invoke_del_min({time}) conflicts with existing operation at the same time.")

        # 1. Find the latest Del_Min operation *before* time `t`
        pred_del_node = self.T_d_m.search_le(time - 1e-9)  # Search for max key < time

        node_to_delete = self.T_ins.TNULL
        if pred_del_node == self.T_d_m.TNULL:
            # No previous deletions, find the overall minimum valid node in T_ins
            node_to_delete = self.T_ins.find_min_valid()
        else:
            # Find the minimum valid node in T_ins *strictly greater* than the previously deleted key
            prev_deleted_ins_node = pred_del_node.t_ins_link
            if prev_deleted_ins_node == self.T_ins.TNULL:
                print(f"RPQ Error: T_d_m node at {pred_del_node.key} links to TNULL!")
                node_to_delete = self.T_ins.find_min_valid()  # Fallback
            else:
                threshold_key = prev_deleted_ins_node.key
                node_to_delete = self.T_ins.find_min_valid_greater_than(threshold_key)

        # Ensure the node to delete is valid before marking it invalid
        if node_to_delete != self.T_ins.TNULL:
            node_to_delete.valid = False  # Mark as invalid after deletion
        return node_to_delete.value if node_to_delete != self.T_ins.TNULL else None

    def revoke_insert(self, time):
        """
        Performs Revoke(Insert(t)) operation.
        Args:
            time: The time of the insertion to revoke.
        Returns:
            The T_d_m node of the first inconsistent Del_Min operation (if any), otherwise None.
        """
         # 1. Check for future inconsistencies in T_d_m
        inconsistent_del_node = self.T_d_m.search_ge(time)
        # Use T_d_m's TNULL
        if inconsistent_del_node != self.T_d_m.TNULL:
            # Revoking insert might affect subsequent deletes
            # print(f"RPQ Warn: Revoking Insert at {time} conflicts with future Del_Min at {inconsistent_del_node.key}")
            return inconsistent_del_node

        # 2. Find the node(s) inserted at exactly 'time' in T_ins
        # ... (rest of the logic is likely okay, assuming vertex_nodes helps find the node) ...
        nodes_at_time = []
        for v_nodes in self.vertex_nodes.values():
            for node in v_nodes:
                if node.value.get('ins_time') == time:
                     nodes_at_time.append(node)

        if not nodes_at_time:
             # print(f"RPQ Warn: revoke_insert({time}) - No node found with this insertion time.")
             return None # Or raise error

        # Assuming only one node should match typically in Dijkstra context per vertex update
        node_to_revoke = nodes_at_time[0] # Take the first match
        if len(nodes_at_time) > 1:
            print(f"RPQ Warn: revoke_insert({time}) - Multiple nodes found. Revoking first match: {node_to_revoke}")


        # 3. Mark the node as invalid (or delete if preferred)
        if node_to_revoke.valid:
             node_to_revoke.valid = False
             # print(f"RPQ RevokeInsert({time}): Marked invalid: {node_to_revoke}")
        else:
            # print(f"RPQ Warn: revoke_insert({time}) - Node already invalid: {node_to_revoke}")
            pass # Already invalid, do nothing

        return None # No inconsistency


    def revoke_del_min(self, time):
        """
        Performs Revoke(Del_Min(t)) operation.
        Args:
            time: The time of the Del_Min operation to revoke.
        Returns:
            The T_d_m node of the first inconsistent *later* Del_Min operation (if any), otherwise None.
        """
        # 1. Find the Del_Min operation node in T_d_m at exactly time `t`.
        del_node = self.T_d_m.search(time)
        # Use T_d_m's TNULL
        if del_node == self.T_d_m.TNULL:
            # print(f"RPQ Warn: revoke_del_min({time}) - No Del_Min operation found at this exact time.")
            return None # Or raise error

        # 2. Check for inconsistencies *after* this revoked operation.
        # Find the smallest deletion time > t
        next_del_node = self.T_d_m.successor(del_node)
        # Use T_d_m's TNULL
        if next_del_node != self.T_d_m.TNULL:
             # print(f"RPQ Warn: Revoking Del_Min at {time} might conflict with next Del_Min at {next_del_node.key}")
             return next_del_node

        # 3. Find the corresponding T_ins node
        ins_node = del_node.t_ins_link
        if ins_node is None:
             print(f"RPQ Error: revoke_del_min({time}) - T_d_m node has no T_ins link!")
             return None
        # Use T_ins's TNULL
        elif ins_node == self.T_ins.TNULL:
             print(f"RPQ Error: revoke_del_min({time}) - T_d_m node links to TNULL in T_ins!")
             return None

        # 4. Mark the T_ins node as valid again
        ins_node.valid = True
        ins_node.t_dm_link = None
        # print(f"RPQ RevokeDelMin({time}): Marked valid again: {ins_node}")

        # 5. Remove the Del_Min record from T_d_m
        self.T_d_m.delete(time)
        # print(f"RPQ RevokeDelMin({time}): Removed node from T_d_m.")

        return None # No inconsistency


    def find_min(self, time):
        """
        Performs Find_Min(t) operation.
        Args:
            time: The time at which to find the minimum element.
        Returns:
            The value dictionary {'vertex': ..., 'dist': ...} of the minimum element at time t, or None if PQ is empty.
        """
        # 1. Find the latest Del_Min operation *before* time `t`
        pred_del_node = self.T_d_m.search_le(time - 1e-9) # max key < time

        # Use T_ins's TNULL
        min_valid_node = self.T_ins.TNULL
        # Use T_d_m's TNULL
        if pred_del_node == self.T_d_m.TNULL:
            # No deletions before time t, find the overall minimum valid node in T_ins
            min_valid_node = self.T_ins.find_min_valid()
            # print(f"FindMin({time}): No prev delete. Min valid: {min_valid_node}")
        else:
            # Find the minimum valid node in T_ins *strictly greater* than the previously deleted key
            prev_deleted_ins_node = pred_del_node.t_ins_link
            if prev_deleted_ins_node is None:
                 print(f"RPQ Error: find_min({time}) - T_d_m node at {pred_del_node.key} has no T_ins link!")
                 min_valid_node = self.T_ins.find_min_valid() # Fallback
            # Use T_ins's TNULL
            elif prev_deleted_ins_node == self.T_ins.TNULL:
                  print(f"RPQ Error: find_min({time}) - T_d_m node links to TNULL!")
                  min_valid_node = self.T_ins.find_min_valid() # Fallback
            else:
                threshold_key = prev_deleted_ins_node.key
                # print(f"FindMin({time}): Prev delete at {pred_del_node.key}, threshold key {threshold_key}")
                min_valid_node = self.T_ins.find_min_valid_greater_than(threshold_key)
                # print(f"FindMin({time}): Min valid > threshold: {min_valid_node}")

        # Use T_ins's TNULL
        if min_valid_node == self.T_ins.TNULL:
             # print(f"RPQ FindMin({time}): No valid element found.")
            return None
        else:
            return min_valid_node.value.copy() # Return a copy

    def get_vertex_node(self, vertex, active_only=True):
        """
        Finds the node(s) for a given vertex in T_ins.
        Returns the most relevant node (e.g., latest insertion, or currently valid one).
        This is a helper, potentially needing refinement based on exact needs.
        """
        if vertex not in self.vertex_nodes:
            return None

        relevant_node = None
        latest_time = -1

        for node in self.vertex_nodes[vertex]:
            if active_only and not node.valid:
                continue

            ins_time = node.value.get('ins_time', -1)
            if ins_time > latest_time:
                 if active_only and node.valid:
                      relevant_node = node
                      latest_time = ins_time
                 elif not active_only:
                      relevant_node = node
                      latest_time = ins_time

        # Fallback if no active node found but requested
        if active_only and relevant_node is None and vertex in self.vertex_nodes:
             for node in self.vertex_nodes[vertex]:
                 ins_time = node.value.get('ins_time', -1)
                 if ins_time > latest_time:
                     relevant_node = node
                     latest_time = ins_time

        return relevant_node

# --- Dynamic Dijkstra Algorithm ---
class DynamicDijkstra:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        # Using adjacency matrix as per paper's suggestion for simplicity
        # For large sparse graphs, adjacency list + dict lookup is better
        self.graph = [[float('inf')] * num_vertices for _ in range(num_vertices)]
        for i in range(num_vertices):
            self.graph[i][i] = 0 # Distance to self is 0

        self.dist = [float('inf')] * num_vertices
        self.pred = [None] * num_vertices
        self.rpq = RPQ()
        self.current_time = 0 # Global time counter

    def initialize(self, source):
        """Initializes distances and RPQ for the source vertex."""
        self.dist = [float('inf')] * self.num_vertices
        self.pred = [None] * self.num_vertices
        self.rpq = RPQ() # Reset RPQ

        self.dist[source] = 0
        self.current_time = 0
        # Insert source into RPQ at time 0
        inconsistency = self.rpq.invoke_insert(vertex=source, time=self.current_time,
                                               dist=0, pred=None)
        if inconsistency:
             print(f"Init Error: RPQ inconsistency: {inconsistency}") # Should not happen here

        self.current_time += 1 # Increment time after initial setup


    def run_static(self):
        """Runs Dijkstra using the RPQ until the queue is 'empty' for the current state."""
        print("\n--- Running Static Dijkstra using RPQ ---")
        processed_count = 0
        while processed_count < self.num_vertices:
            # Find the minimum element *at the current time*
            min_val = self.rpq.find_min(self.current_time)
            print(f"Time {self.current_time}: FindMin -> {min_val}")

            if min_val is None:
                 print(f"Time {self.current_time}: RPQ empty or no valid nodes. Stopping.")
                 break # No more reachable vertices

            # Perform the Del_Min operation for the found element's vertex at current time
            deleted_val = self.rpq.invoke_del_min(self.current_time)
            # print(f"Time {self.current_time}: InvokeDelMin -> {deleted_val}")


            if deleted_val is None:
                 # This might happen if find_min found something, but intervening
                 # operations (if any were allowed) made it invalid before del_min.
                 # Or simply if the queue became empty between find and delete.
                 print(f"Time {self.current_time}: Del_Min returned None. Continuing.")
                 self.current_time +=1 # Still advance time
                 continue


            u = deleted_val['vertex']
            u_dist = deleted_val['dist']

            # Check if we already processed a shorter path to u
            # Note: In classic Dijkstra, this check isn't needed because we always
            # extract the absolute minimum. With RPQ, it's safer to check.
            if u_dist > self.dist[u]:
                 # print(f"Time {self.current_time}: Stale entry for {u} ({u_dist} > {self.dist[u]}). Skipping.")
                 self.current_time +=1
                 continue

            processed_count += 1
            # print(f"Time {self.current_time}: Processing vertex {u} with dist {u_dist}")


            # Relax neighbors
            for v in range(self.num_vertices):
                edge_weight = self.graph[u][v]
                if edge_weight != float('inf') and u != v:
                    new_dist_v = self.dist[u] + edge_weight
                    # print(f"  Checking neighbor {v}: Current dist {self.dist[v]}, New potential dist {new_dist_v}")
                    if new_dist_v < self.dist[v]:
                        # print(f"    Relaxation: Found shorter path to {v} via {u}. Dist {new_dist_v}")
                        self.dist[v] = new_dist_v
                        self.pred[v] = u

                        # Update RPQ: Insert the new path info for v at current time
                        # We might need to revoke old inserts for v first if they exist and are now suboptimal.
                        # Simple approach: Just insert. RPQ's FindMin should handle finding the true minimum later.
                        # A better approach would use UpdateKey if RPQ supported it, or Revoke+Insert.
                        # For now, just InvokeInsert:
                        inconsistency = self.rpq.invoke_insert(vertex=v, time=self.current_time,
                                                               dist=new_dist_v, pred=u)
                        if inconsistency:
                             # Handle inconsistency - e.g., log, try recovery, or fail
                            print(f"RPQ Inconsistency during relaxation for {v}! Node: {inconsistency}. Halting Relaxation Loop for {u}.")
                            # This might require complex handling - for now, just log and continue time step
                            break # Stop relaxing for this u


            self.current_time += 1 # Increment time after processing a vertex

        print("--- Static Dijkstra Finished ---")
        print(f"Final Distances: {self.dist}")
        print(f"Final Predecessors: {self.pred}")


    def update_edge(self, u, v, new_weight):
        """
        Handles dynamic edge weight updates using the D_Dij logic sketch.
        NOTE: This implementation is a simplified interpretation of the paper's
              complex retroactive logic. Full retroactivity is hard. This focuses
              on adjusting the RPQ based on the change.
        """
        print(f"\n--- Updating Edge ({u}, {v}) to weight {new_weight} at time {self.current_time} ---")
        old_weight = self.graph[u][v]
        if old_weight == new_weight:
            print("  No weight change. Skipping.")
            return

        self.graph[u][v] = new_weight
        # self.graph[v][u] = new_weight # Assuming undirected for simplicity if needed

        # --- Simplified D_Dij Logic ---
        # The paper's logic is intricate, involving searching T_ins, checking predecessors,
        # and potentially moving back in time. A full implementation is complex.
        # Here's a more direct, forward-looking approach inspired by relaxation:

        # Case 1: Weight Decrease (Potential for shorter paths)
        if new_weight < old_weight:
             print(f"  Weight decreased. Potential shorter path from {u} to {v}.")
             # If the path through u *might* now be better for v
             new_dist_v = self.dist[u] + new_weight
             if new_dist_v < self.dist[v]:
                  print(f"    Found immediately shorter path to {v} ({new_dist_v} < {self.dist[v]}). Updating RPQ.")
                  # Update distance and predecessor estimation
                  self.dist[v] = new_dist_v
                  self.pred[v] = u
                  # Update RPQ: Add this new, better path possibility.
                  # Again, ideally use UpdateKey or Revoke+Insert. Using InvokeInsert for now.
                  inconsistency = self.rpq.invoke_insert(vertex=v, time=self.current_time,
                                                         dist=new_dist_v, pred=u)
                  if inconsistency:
                      print(f"    RPQ Inconsistency during decrease update for {v}! Node: {inconsistency}")
                      # Need strategy here: e.g., re-run Dijkstra?
                  else:
                      print(f"    RPQ updated for vertex {v} with new distance {new_dist_v}.")

                  # NOTE: This simple update doesn't automatically propagate the change
                  # to v's neighbors. A full solution would require re-running parts
                  # of Dijkstra or using a more sophisticated update propagation.

        # Case 2: Weight Increase (Potential for path invalidation)
        elif new_weight > old_weight:
             print(f"  Weight increased.")
             # If the shortest path to v *used* the edge (u, v)
             if self.pred[v] == u and self.dist[v] == self.dist[u] + old_weight:
                  print(f"    Increased edge ({u},{v}) was part of shortest path to {v}.")
                  # The current path to v is now potentially incorrect (cost is higher).
                  # We need to find an alternative path.

                  # --- How RPQ *should* handle this (conceptually) ---
                  # The existing entry for v (if inserted via u) in T_ins now represents
                  # an old, possibly suboptimal path cost. Future FindMin operations,
                  # when considering time > current_time, should automatically ignore
                  # this path if a better alternative (inserted earlier or later) becomes
                  # the minimum valid entry.

                  # --- Actions needed? ---
                  # 1. Invalidate the old path? Marking the T_ins node invalid might be too strong.
                  #    Perhaps just update the distance estimate?
                  # 2. Re-calculate dist[v]? This would involve finding the *new* best incoming edge.
                  # 3. Add v back to RPQ? If its path needs re-evaluation.

                  # --- Simplified approach ---
                  # For now, we don't explicitly modify the RPQ here for increases.
                  # We rely on subsequent FindMin/DelMin operations to naturally find
                  # the new correct path IF other path possibilities were already inserted
                  # or if we re-run parts of Dijkstra.
                  # A more robust solution might involve:
                  #   a) Setting self.dist[v] = infinity, self.pred[v] = None
                  #   b) Re-inserting v into RPQ if neighbors might provide a path.
                  #   c) Or, triggering a limited Dijkstra-like propagation from v.

                  # Let's try updating dist[v] and re-inserting it to trigger re-evaluation.
                  print(f"    Invalidating path to {v}. Setting dist to Inf and re-evaluating.")
                  # Find the node for v in RPQ that used u as predecessor (this requires better lookup)
                  v_node = self.rpq.get_vertex_node(v) # Simple lookup
                  if v_node and v_node.valid and v_node.value.get('pred') == u:
                       print(f"    Marking existing RPQ node for {v} invalid (due to increase).")
                       # Revoke the specific insert that represents the path through u
                       # This is complex. For now, just mark the latest node invalid.
                       v_node.valid = False # Mark it stale

                  # Reset v's distance temporarily, assume it needs recalculation
                  self.dist[v] = float('inf')
                  self.pred[v] = None

                  # We need to add v back to RPQ potentially, but with what distance?
                  # Maybe recalculate based on *other* neighbors?
                  min_neighbor_dist = float('inf')
                  best_neighbor_pred = None
                  for neighbor in range(self.num_vertices):
                       if self.graph[neighbor][v] != float('inf') and neighbor != v:
                            if self.dist[neighbor] + self.graph[neighbor][v] < min_neighbor_dist:
                                 min_neighbor_dist = self.dist[neighbor] + self.graph[neighbor][v]
                                 best_neighbor_pred = neighbor

                  if min_neighbor_dist != float('inf'):
                        print(f"    Found alternative path to {v} via {best_neighbor_pred} with cost {min_neighbor_dist}. Updating RPQ.")
                        self.dist[v] = min_neighbor_dist
                        self.pred[v] = best_neighbor_pred
                        inconsistency = self.rpq.invoke_insert(vertex=v, time=self.current_time,
                                                               dist=min_neighbor_dist, pred=best_neighbor_pred)
                        if inconsistency:
                             print(f"    RPQ Inconsistency during increase update for {v}! Node: {inconsistency}")


                  # This recalculation is still local. A full update might require broader propagation.

        # Increment time after the update operation
        self.current_time += 1
        print(f"--- Update Finished. Current time: {self.current_time} ---")
        # Optionally: Re-run parts of Dijkstra to propagate changes fully
        # self.run_static_from_state() # A hypothetical function to continue Dijkstra

# --- Example Usage ---
if __name__ == "__main__":
    # Example Graph (from Figure 9 in the paper, 0-indexed)
    # Vertices: O=0, A=1, B=2, C=3, D=4, E=5, F=6, T=7 (8 vertices)
    num_v = 8
    dd = DynamicDijkstra(num_v)
    print(f"Initialized DynamicDijkstra for {num_v} vertices.")

    # --- Define Initial Graph Edges (Based on Figure 9 BEFORE updates) ---
    # Add edges with their initial weights.
    # Using float('inf') for non-edges is handled by the class init.
    print("Defining initial graph edges...")
    dd.graph[0][1] = 2  # O-A
    dd.graph[0][2] = 5  # O-B
    dd.graph[1][6] = 12 # A-F
    dd.graph[1][2] = 7  # A-B
    dd.graph[1][3] = 4  # A-C
    dd.graph[2][4] = 1  # B-D (Initial weight is 1 before t=5 update)
    dd.graph[2][5] = 3  # B-E
    dd.graph[3][4] = 4  # C-D
    dd.graph[3][7] = 7  # C-T
    dd.graph[4][5] = 4  # D-E
    dd.graph[4][7] = 3  # D-T
    # Edge E-T weight seems missing in the figure text/diagram, let's assume 5
    dd.graph[5][7] = 5  # E-T

    # Optional: Make graph undirected if the example flow implies it
    print("Making graph undirected for example consistency...")
    for r in range(num_v):
        for c in range(r + 1, num_v):
             if dd.graph[r][c] != float('inf'):
                  dd.graph[c][r] = dd.graph[r][c]
             elif dd.graph[c][r] != float('inf'): # Ensure symmetry if only one direction was defined
                  dd.graph[r][c] = dd.graph[c][r]

    # --- Run Initial Static Dijkstra ---
    source_vertex = 0 # Start from vertex O
    # dd.graph.__repr__()
    print(f"\nInitializing Dijkstra from source vertex {source_vertex}...")
    dd.initialize(source_vertex)

    print("\nRunning initial static Dijkstra calculation...")
    dd.run_static() # Computes initial shortest paths using RPQ

    print("\n--- Initial Shortest Path Results ---")
    print(f"Distances: {dd.dist}")
    print(f"Predecessors: {dd.pred}")
    print(f"Current Algorithm Time: {dd.current_time}")

    # --- Apply Dynamic Updates ---
    print("\n--- Applying Dynamic Edge Updates ---")

    # Update 1: At time t=5 (in paper's example), edge B-D weight increases to 5
    # (B=2, D=4)
    print(f"\nUpdate 1 (Internal Time: {dd.current_time}): Increasing edge (2, 4) weight to 5")
    dd.update_edge(2, 4, 5)
    # Note: The simplified update_edge might not fully propagate changes.
    # We print the state immediately after the update attempt.
    print(f" State after Update 1:")
    print(f"  Distances: {dd.dist}")
    print(f"  Predecessors: {dd.pred}")

    # Update 2: At time t=7 (in paper's example), edge O-B weight increases to 7
    # (O=0, B=2)
    print(f"\nUpdate 2 (Internal Time: {dd.current_time}): Increasing edge (0, 2) weight to 7")
    dd.update_edge(0, 2, 7)
    print(f" State after Update 2:")
    print(f"  Distances: {dd.dist}")
    print(f"  Predecessors: {dd.pred}")


    # Update 3: At time t=9 (in paper's example), edge A-B weight decreases to 1
    # (A=1, B=2)
    print(f"\nUpdate 3 (Internal Time: {dd.current_time}): Decreasing edge (1, 2) weight to 1")
    dd.update_edge(1, 2, 1)
    print(f" State after Update 3:")
    print(f"  Distances: {dd.dist}")
    print(f"  Predecessors: {dd.pred}")


    # --- Final State ---
    # The distances/predecessors shown after updates reflect the immediate changes
    # made by the update_edge function. A full re-convergence might require
    # resuming the Dijkstra loop (which is not implemented in run_static/update_edge).
    print("\n--- Final State After Updates ---")
    print(f"Final Distances (may require further processing): {dd.dist}")
    print(f"Final Predecessors: {dd.pred}")
    print(f"Final Algorithm Time: {dd.current_time}")

    print("\n--- Final RPQ State ---")
    print("T_ins (Inorder Traversal):")
    for node in dd.rpq.T_ins.inorder():
        # Check if node is TNULL before printing
        if node != dd.rpq.T_ins.TNULL:
            print(f"  {node}")
        else:
            print("  Met TNULL in T_ins inorder") # Should not happen with proper traversal
    print("\nT_d_m (Inorder Traversal):")
    for node in dd.rpq.T_d_m.inorder():
         if node != dd.rpq.T_d_m.TNULL:
            print(f"  {node}")
         else:
             print("  Met TNULL in T_d_m inorder") # Should not happen

    # Example: Finding min at the very end
    final_min_val = dd.rpq.find_min(dd.current_time)
    print(f"\nFindMin at final time {dd.current_time}: {final_min_val}")