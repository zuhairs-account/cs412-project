def invoke_insert(T_ins, T_d_m, x, t):
    T_ins.insert(x, t)

    # Search in Td_m for minimum key value deleted after time t
    P = T_d_m.search_min_after(t)

    if P is not None:
        return P  # Return the first inconsistent operation
    else:
        return None


def invoke_del_min(T_ins, T_d_m, t):
    t_prime, P = T_d_m.search_latest_before(t)

    if P is None:
        # No del_min has been performed before time t
        N = T_ins.find_min()
        return N
    else:
        k_prime = P.data  # Max value deleted before time t
        k = T_ins.find_min_greater_than(k_prime)
        return k


def revoke_insert(T_ins, T_d_m, t):
    P = T_d_m.search_min_after(t)

    if P is not None:
        return P  # Return the first inconsistent operation
    else:
        P = T_ins.search_by_time(t)
        if P:
            P.valid = False

def revoke_del_min(t, x):
    # Search in tree T_dm for minimum key value deleted after time t
    P = search_min(T_dm, t)

    if P is not None:
        # Return the first inconsistent operation
        return P
    else:
        # Insert x into height-balanced tree T_ins at time t
        insert(T_ins, x, t)


def find_min(t):
    P = search_min(T_dm, t)

    if P is None:
        # Search in tree T_ins for minimum key at time t
        Q = search_min(T_ins, t)
        return Q.val
    else:
        return P.val
