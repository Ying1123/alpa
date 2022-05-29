"""ILP Solver"""
import numpy as np
import time
import warnings
import multiprocessing

import pulp

INFINITY_COST = 1e20

def call_solver(N, M, s_len, s_follow, E, A, L, c, d, m, r, v, s_init):
    """Serialize python lists to flatten numpy arraies and call solver"""
    # Serialize strategy lengths
    s_len_np = np.array(s_len, dtype=np.int32)
    s_follow_np = np.array(s_follow, dtype=np.int32)

    # Serialize edge set
    len_edges = len(E)
    E_np = np.empty((len_edges, 2), dtype=np.int32)
    for (idx, (i, j)) in enumerate(E):
        E_np[idx][:] = [i, j]

    # Serialize alias set
    len_aliases = len(A)
    A_np = np.empty((len_aliases, 2), dtype=np.int32)
    for (idx, (i, j)) in enumerate(A):
        A_np[idx][:] = [i, j]

    # Serialize liveness set
    len_liveness_set = N + sum(len(v) for v in L)
    L_np = np.empty((len_liveness_set,), dtype=np.int32)
    L_np[0:N] = [len(v) for v in L]
    L_np[N:] = [x for v in L for x in v]

    # Serialize node costs
    len_node_costs = sum(len(v) for v in c)
    c_np = np.empty((len_node_costs,), dtype=np.float32)
    d_np = np.empty((len_node_costs,), dtype=np.float32)
    m_np = np.empty((len_node_costs,), dtype=np.float32)
    c_np[:] = [x for v in c for x in v]
    d_np[:] = [x for v in d for x in v]
    m_np[:] = [x for v in m for x in v]

    # Serialize edge costs
    len_edge_costs = sum(len(vec) for vec in r)
    r_np = np.empty((len_edge_costs,), dtype=np.float32)
    r_np[:] = [x for vec in r for x in vec]

    # Serialize alias costs
    len_alias_costs = sum(len(vec) for vec in v)
    v_np = np.empty((len_alias_costs,), dtype=np.float32)
    v_np[:] = [x for vec in v for x in vec]

    # Serialize init value
    s_init_np = None

    return _call_solver_serialized_args(
        N, M, s_len_np, s_follow_np, E_np, A_np, L_np,
        c_np, d_np, m_np, r_np, v_np, s_init_np)


class CostGraph:
    def __init__(self, node_lens, edges, edge_costs, to_merge_pair):
        self.node_lens = node_lens
        self.adjacency = dict()   # map a node to its neighbors
        self.edge_costs = dict()  # map an edge to its cost matrix
        self.reindexing_vector = dict()  # map a node to its reindexing vector
        self.merged_to = dict()   # map an merged node to its destination
        self.to_merge_pair = to_merge_pair  # the input follow pairs

        for i in range(len(node_lens)):
            self.adjacency[i] = set()

        # For redundant edges, we will overwrite the results with
        # the last value
        for ((i, j), cost) in zip(edges, edge_costs):
            cost = np.reshape(cost, (self.node_lens[i], self.node_lens[j]))

            self.add_edge_cost(i, j, cost)

    def get_edge_cost(self, i, j):
        if i <= j:
            return self.edge_costs[(i, j)]
        else:
            return self.edge_costs[(j, i)].transpose()

    def add_edge_cost(self, i, j, cost):
        if i > j:
            i, j = j, i
            cost = cost.transpose()

        if (i, j) in self.edge_costs:
            assert i in self.adjacency[j]
            assert j in self.adjacency[i]
            self.edge_costs[(i, j)] += cost
        else:
            self.adjacency[i].add(j)
            self.adjacency[j].add(i)
            self.edge_costs[(i, j)] = cost

    def remove_edge(self, i, j):
        if i > j:
            i, j = j, i

        assert j in self.adjacency[i]
        assert i in self.adjacency[j]
        assert (i, j) in self.edge_costs

        self.adjacency[i].remove(j)
        self.adjacency[j].remove(i)
        del self.edge_costs[(i, j)]

    def merge_node(self, src, dst):
        """Merge node src to node dst"""
        print(f"merge {src} to {dst}")
        assert dst in self.adjacency[src]
        assert src in self.adjacency[dst]
        assert dst not in self.merged_to
        assert src != dst

        edge_cost = self.get_edge_cost(dst, src)

        # Find the strategy to follow greedily
        reindexing = []
        candidates = list(range(self.node_lens[src]))
        for i in range(self.node_lens[dst]):
            # Pick the strategy with the lowest cost to follow.
            # If there are multiple strategies with the same lowest costs,
            # prefer to follow "replicated", which has the largest index.
            keys = [(edge_cost[i][j], -j) for j in range(self.node_lens[src])]
            candidates.sort(key=lambda j: keys[j])
            reindexing.append(candidates[0])

        self.merged_to[src] = dst
        self.reindexing_vector[src] = reindexing

        # Merge edge cost matrix
        adj_list = list(self.adjacency[src])
        for adj in adj_list:
            if adj == dst:
                continue
            added_edge_cost = np.empty((self.node_lens[dst], self.node_lens[adj]))
            for i in range(self.node_lens[dst]):
                j = reindexing[i]
                edge_cost_src_adj = self.get_edge_cost(src, adj)
                for k in range(self.node_lens[adj]):
                    added_edge_cost[i][k] = edge_cost_src_adj[j][k] + edge_cost[i][j]

            self.add_edge_cost(dst, adj, added_edge_cost)

        # Remove edges
        for adj in adj_list:
            self.remove_edge(src, adj)

    def query_destination(self, node):
        if node in self.merged_to:
            old_dst = self.merged_to[node]
            new_dst = self.query_destination(old_dst)
            if old_dst != new_dst:
                # Compress path
                old_reindexing_vector = self.reindexing_vector[node]
                new_reindexing_vector = []
                for i in range(self.node_lens[new_dst]):
                    new_reindexing_vector.append(
                        old_reindexing_vector[self.reindexing_vector[old_dst][i]])

                self.reindexing_vector[node] = new_reindexing_vector
                self.merged_to[node] = new_dst
            return new_dst
        else:
            return node

    def simplify(self):
        for (src, dst) in self.to_merge_pair:
            assert src not in self.merged_to
            dst = self.query_destination(dst)
            if src != dst:
                self.merge_node(src, dst)

    def export_result(self):
        E = []
        r = []
        s_follow = []

        for i in range(len(self.node_lens)):
            if i in self.merged_to:
                s_follow.append(self.query_destination(i))
            else:
                s_follow.append(-1)

        for ((i, j), v) in self.edge_costs.items():
            v = v.reshape(-1)
            E.append((i, j))
            r.append(v)

            assert len(v) == self.node_lens[i] * self.node_lens[j]

        return s_follow, E, r, self.reindexing_vector

    def __str__(self):
        ret = ""
        for i in range(len(self.node_lens)):
            ret += f"Node {i}: {self.node_lens[i]}\n"

        edges = list(self.edge_costs.keys())
        edges.sort()

        for (i, j) in edges:
            ret += f"Edge {(i, j)}:\n"
            ret += str(self.edge_costs[(i, j)]) + "\n"

        return ret


class SolverOption:
    def __init__(self):
        self.force_batch_dim_to_mesh_dim = None

        self.forward_backward_sep_id = None
        self.force_all_reduce_cost = None
        self.force_all_gather_cost = None
        self.force_reduce_scatter_cost = None


def solve_auto_sharding(computation, cluster_env, solver_option=None):
    print("===== Hlo Computation =====")
    print(computation, "\n")

    print("===== Liveness Analysis =====")
    liveness_dict = computation.liveness_analysis()
    for i in range(len(computation.instructions)):
        names = [ins.name for ins in liveness_dict[i]]
        names.sort()
        print(f"Time: {i}, Live set: {names}")

    if solver_option is None:
        solver_option = SolverOption()

    # Build strategies and costs
    computation.build_strategy_and_cost(cluster_env, solver_option)

    # Build all constants for ILP
    N = len(computation.instructions)
    M = cluster_env.memory_per_device

    s_len = []
    follow_pair = []
    E = []
    A = []
    L = []
    c = []
    d = []
    m = []
    r = []
    v = []
    for i in range(N):
        ins = computation.instructions[i]
        s_len.append(len(ins.strategies))
        L.append([ins.index for ins in liveness_dict[i]])
        c.append(ins.compute_costs)
        d.append(ins.communication_costs)
        m.append(ins.memory_costs)

        if ins.follow_ins is not None:
            follow_pair.append((ins.index, ins.follow_ins.index))

        for op_idx, operand in enumerate(ins.operands):
            E.append((operand.index, i))

            src = operand.index
            dst = i

            #ins.resharding_costs  # [s_i, operand_idx, s_operand]
            cost = []
            for p in range(len(computation.instructions[src].strategies)):
                for q in range(len(computation.instructions[dst].strategies)):
                    cost.append(ins.resharding_costs[q][op_idx][p])
            r.append(cost)

    # Simplify the graph by merging nodes
    cost_graph = CostGraph(s_len, E, r, follow_pair)
    cost_graph.simplify()
    s_follow, E, r, reindexing_vector = cost_graph.export_result()

    for src, dst in enumerate(s_follow):
        if dst >= 0:
            s_len[src] = len(reindexing_vector[src])
            c[src] = np.array(c[src])[reindexing_vector[src]]
            d[src] = np.array(d[src])[reindexing_vector[src]]
            m[src] = np.array(m[src])[reindexing_vector[src]]

    # Deal with alias
    for ((ins_a, ins_b), cost_vector) in zip(computation.alias_list,
                                             computation.alias_cost_vector):

        idx_a, idx_b = ins_a.index, ins_b.index
        cost_vector = np.array(cost_vector).reshape(
            len(ins_a.strategies), len(ins_b.strategies))

        if s_follow[idx_a] >= 0:
            reindexing_a = reindexing_vector[idx_a]
            idx_a = s_follow[idx_a]
        else:
            reindexing_a = range(len(ins_a.strategies))

        if s_follow[idx_b] >= 0:
            reindexing_b = reindexing_vector[idx_b]
            idx_b = s_follow[idx_b]
        else:
            reindexing_b = range(len(ins_b.strategies))

        if idx_a != idx_b:
            A.append((idx_a, idx_b))
            new_cost_vector = []
            for i in reindexing_a:
                for j in reindexing_b:
                    new_cost_vector.append(cost_vector[i, j])
            v.append(new_cost_vector)

    s_val, e_val, objective, status = call_solver(N, M, s_len, s_follow, E, A, L,
                                                  c, d, m, r, v, s_init=None)

    if True:
        # Print sharding spec
        instructions = computation.instructions
        print("===== Sharding Strategy =====")
        for i in range(N):
            if s_follow[i] < 0:
                stra_idx = s_val[i]
                name = instructions[i].strategies[stra_idx].name
                follow_map = ""
                spec = instructions[i].strategies[stra_idx].output_spec
            else:
                dst = s_follow[i]
                stra_idx = reindexing_vector[i][s_val[i]]
                name = instructions[i].strategies[stra_idx].name + f" follow {dst}"
                spec = instructions[i].strategies[stra_idx].output_spec

                follow_map = ""
                for idx in range(len(reindexing_vector[i])):
                    stra_idx = reindexing_vector[i][idx]
                    follow_map += f"[{instructions[dst].strategies[idx].name} -> "\
                            f"{instructions[i].strategies[stra_idx].name}] "
            #print(f"Time {i:2d}: {computation.instructions[i]}  Strategy: {name} Spec: {spec}")
            print(f"Time {i:2d}: {computation.instructions[i]}  Strategy: {name}")
            #if follow_map:
            #    print(follow_map)

        # Print edge cost
        for (idx, (i, j)) in enumerate(E):
            if r[idx][e_val[idx]] > 0:
                print(f"Edge cost {(i, j)} : {r[idx][e_val[idx]]}")

        # Print peak memory
        print("===== Memory Usage =====")
        for t in range(N):
            mem = 0
            for i in L[t]:
                mem += m[i][s_val[i]]
            print(f"Time {t}, memory: {mem / 1024**2: .2f} MB")

    return objective


# The last solution vector of auto sharding.
last_s_val = None

# The last objective value of the best ILP solution.
last_objective = None



# pylint: disable=import-outside-toplevel
# noqa
def _call_solver_serialized_args(
        N,  # noqa
        M,
        s_len_np,
        s_follow_np,
        E_np,
        A_np,
        L_np,
        c_np,
        d_np,
        m_np,
        r_np,
        v_np,
        s_init_np=None):
    """Call the solver with serialized arguments."""
    global last_s_val, last_objective

    import pulp
    from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus
    tic = time.time()

    for x in [s_len_np, E_np, A_np, L_np, c_np, d_np, m_np, r_np, v_np]:
        assert isinstance(x, np.ndarray)
    assert len(s_len_np) == N, "s_len_np"

    # Dump arguments for re-solving
    # pickle.dump([N, M, s_len_np, s_follow_np, E_np, A_np, L_np,
    #              c_np, d_np, m_np, r_np, v_np, s_init_np],
    #              open("args.pkl", "wb"))
    # TODO(lmzheng): cache the ILP solution.

    def get_non_zero_index(binary_vector):
        """Get the index of non-zero item in a vector."""
        ct = 0
        ret = None
        for i, elem in enumerate(binary_vector):
            if pulp.value(elem):
                ret = i
                ct += 1

        assert ct == 1
        return ret

    # 0. Unpack flatten numpy arrays
    s_len = s_len_np
    s_follow = s_follow_np

    E = E_np.reshape((-1, 2))  # noqa
    r = []
    pt = 0
    edge_set = set()
    for (i, j) in E:
        prod_length = s_len[i] * s_len[j]

        if (i, j) in edge_set:
            raise ValueError(f"Duplicated edges: {(i, j)}")

        edge_set.add((i, j))
        r.append(r_np[pt:pt + prod_length])
        pt += prod_length
    assert pt == len(r_np)

    A = A_np.reshape((-1, 2))  # noqa
    v = []
    pt = 0
    for (i, j) in A:
        prod_length = s_len[i] * s_len[j]
        v.append(v_np[pt:pt + prod_length])
        pt += prod_length
    assert pt == len(v_np)

    L = []  # noqa
    pt = N
    for i in range(N):
        length = L_np[i]
        L.append(L_np[pt:pt + length])
        pt += length
    assert pt == len(L_np)

    c = []
    d = []
    m = []
    pt = 0
    for i in range(N):
        length = s_len[i]
        c.append(c_np[pt:pt + length])
        d.append(d_np[pt:pt + length])
        m.append(m_np[pt:pt + length])
        pt += length
    assert pt == len(c_np), f"{pt} == {len(c_np)}"
    assert pt == len(d_np), f"{pt} == {len(d_np)}"
    assert pt == len(m_np), f"{pt} == {len(m_np)}"

    # 1. Create variables
    s = []
    e = []

    num_nodes = 0
    reverse_follow_backpatch = []
    for i in range(N):
        if s_follow[i] < 0:
            if s_len[i] == 1:
                s.append([1])
            else:
                num_nodes += 1
                s.append(
                    LpVariable.matrix(f"s[{i}]", (range(s_len[i]),),
                                      cat="Binary"))
        else:
            if s_follow[i] < len(s):
                s.append(s[s_follow[i]])
            else:
                s.append(None)
                reverse_follow_backpatch.append(i)

    for i in reverse_follow_backpatch:
        s[i] = s[s_follow[i]]

    num_edges = 0
    for (idx, (i, j)) in enumerate(E):
        if len(s[i]) == 1:
            e.append(s[j])
        elif len(s[j]) == 1:
            e.append(s[i])
        else:
            num_edges += 1
            e.append(
                LpVariable.matrix(f"e[{i},{j}]",
                                  (range(len(s[i]) * len(s[j])),),
                                  cat="Binary"))
        assert len(e[idx]) == len(r[idx])

    # 2. Set initial value for warm start
    if s_init_np is not None:
        s_init = s_init_np.reshape((-1, 3))
        for (idx, value, fix) in s_init:
            for i in range(len(s[idx])):
                s[idx][i].setInitialValue(i == value)
                if fix:
                    s[idx][i].fixValue()

    # 3. Objective
    prob = LpProblem("myProblem", LpMinimize)
    # compute cost
    obj = 0
    for i in range(N):
        obj += lpDot(s[i], c[i]) + lpDot(s[i], d[i])

    # communication cost
    for i in range(len(E)):
        obj += lpDot(e[i], r[i])

    prob += obj

    # 4. Constraints
    # (a). specified by `cat="Binary"`

    # (b)
    for i in range(N):
        if s_follow[i] < 0:
            prob += lpSum(s[i]) == 1

    # (c)
    if M > 0:
        for t in range(N):
            mem = 0
            for i in L[t]:
                mem += lpSum(s[i][j] * m[i][j] for j in range(len(s[i])))
            prob += mem <= M

    # (d). specified by `cat="Binary"`

    for (idx, (i, j)) in enumerate(E):
        if s_len[i] == 1 or s_len[j] == 1:
            continue

        # (e)
        prob += lpSum(e[idx]) == 1

        # (f)
        for row in range(len(s[i])):
            C = len(s[j])  # noqa
            prob += lpSum(
                e[idx][row * C + col] for col in range(0, C)) <= s[i][row]

        # (g)
        for col in range(len(s[j])):
            R = len(s[i])  # noqa
            C = len(s[j])  # noqa
            prob += lpSum(
                e[idx][row * C + col] for row in range(0, R)) <= s[j][col]

    # (h)
    alias_set = set()
    for (idx, (i, j)) in enumerate(A):
        R = len(s[i])  # noqa
        C = len(s[j])  # noqa
        if (i, j) in alias_set:
            raise ValueError(f"Duplicated edges: {(i, j)}")

        alias_set.add((i, j))
        alias_set.add((j, i))

        for row in range(len(s[i])):
            for col in range(len(s[j])):
                if v[idx][row * C + col] > 0.5:
                    prob += s[i][row] + s[j][col] <= 1

    verbose = False

    msg = verbose
    time_limit = 600
    assert "COIN_CMD" in pulp.listSolvers(onlyAvailable=True), (
        "Please install ILP solvers by 'sudo apt install coinor-cbc'"
    )

    with warnings.catch_warnings():  # disable CBC warnings
        warnings.simplefilter("ignore")
        solver = pulp.COIN_CMD(mip=True,
                               msg=msg,
                               timeLimit=time_limit,
                               threads=multiprocessing.cpu_count())
        # solver = pulp.GLPK_CMD(mip=True, msg=msg, timeLimit=time_limit)
        prob.solve(solver)

    status = prob.status
    objective = pulp.value(prob.objective)
    objective = float(objective) if objective is not None else -1.0
    if verbose:
        print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
              f"Time: {time.time() - tic}")
        print(f"#nodes: {num_nodes},  #edges: {num_edges}")

    if prob.status in [pulp.LpStatusInfeasible]:
        raise RuntimeError(
            "Cannot run the function under the given memory budget. "
            "Please increase the memory budget.")

    # Get and check results
    s_val = np.full((N,), -1, dtype=np.int32)
    for i in range(N):
        s_val[i] = get_non_zero_index(s[i])

    e_val = np.full((len(E),), -1, dtype=np.int32)
    for (idx, (i, j)) in enumerate(E):
        e_val[idx] = get_non_zero_index(e[idx])
        i_spec_index = e_val[idx] // len(s[j])
        j_spec_index = e_val[idx] % len(s[j])
        assert i_spec_index == s_val[i], f"e_val[{i}][{j}]"
        assert j_spec_index == s_val[j], f"e_val[{i}][{j}]"
        if verbose and r[idx][e_val[idx]] > 0:
            print(f"Edge cost {(i, j)} : {r[idx][e_val[idx]]}")

    last_s_val = s_val
    last_objective = objective

    if objective > INFINITY_COST:
        warnings.warn("Detect unexpected behaviors in the auto-sharding pass.")

    return s_val, e_val, objective, status

