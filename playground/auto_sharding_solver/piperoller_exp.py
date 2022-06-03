from collections import defaultdict
import time
import multiprocessing

import graphviz

from hlo import *

FLOP_PER_SECOND = 10
IO_PER_SECOND = 1000000

def get_graph_demo(batch_size, input_dim, hidden_dim, output_dim):
    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w1 = HloParameter((input_dim, hidden_dim))
        w2 = HloParameter((hidden_dim, output_dim))

        ## forward
        h1 = HloDot(x, w1)
        h2 = HloDot(h1, w2)
        loss = HloSubtract(h2, y)

        ## backward
        coef = HloConstant(2 / batch_size / output_dim)
        coef = HloBroadcast(coef, (batch_size, output_dim))
        grad_loss = HloMutiply(loss, coef)

        grad_w2 = HloDot(h1, grad_loss,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w2 = HloSubtract(w2, grad_w2)
        grad_h1 = HloDot(grad_loss, w2,
                         lhs_contracting_dims=(1,),
                         rhs_contracting_dims=(1,),)

        grad_w1 = HloDot(x, grad_h1,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w1 = HloSubtract(w1, grad_w1)
        out = HloTuple((new_w1, new_w2))

        ## alias
        computation.set_alias([(w1, new_w1), (w2, new_w2)])

        """
         0: parameter.0 (128, 1024) = parameter()
         1: parameter.1 (128, 1024) = parameter()
         2: parameter.2 (1024, 1024) = parameter()
         3: parameter.3 (1024, 1024) = parameter()
         4: dot.0 (128, 1024) = dot(parameter.0, parameter.2)  lhs_con_dim=(1,), rhs_con_dim=(0,)
         5: dot.1 (128, 1024) = dot(dot.0, parameter.3)  lhs_con_dim=(1,), rhs_con_dim=(0,)
         6: subtract.0 (128, 1024) = subtract(dot.1, parameter.1)
         7: constant.0 () = constant(1.52587891e-05)
         8: broadcast.0 (128, 1024) = broadcast(constant.0)
         9: multiply.0 (128, 1024) = multiply(subtract.0, broadcast.0)
        10: dot.2 (1024, 1024) = dot(dot.0, multiply.0)  lhs_con_dim=(0,), rhs_con_dim=(0,)
        11: subtract.1 (1024, 1024) = subtract(parameter.2, dot.2)
        12: dot.3 (128, 1024) = dot(multiply.0, parameter.3)  lhs_con_dim=(1,), rhs_con_dim=(1,)
        13: dot.4 (1024, 1024) = dot(parameter.0, dot.3)  lhs_con_dim=(0,), rhs_con_dim=(0,)
        14: subtract.2 (1024, 1024) = subtract(parameter.2, dot.4)
        15: tuple.0 () = tuple('subtract.2', 'subtract.1') 
        """
    return computation

## hidden_dim should be a list with length = num_layer
def get_graph_naive(num_layer, num_batch, batch_size, input_dim, hidden_dim, output_dim):
    assert num_layer >= 1
    assert num_batch >= 1
    assert len(hidden_dim) >= 1
    assert len(hidden_dim) == num_layer

    computation = HloComputation()
    with computation:
        ## x, y and w
        x = [None] * num_batch
        y = [None] * num_batch
        for i in range(num_batch):
            x[i] = HloParameter((batch_size, input_dim), name=f"B{i}_x")
            y[i] = HloParameter((batch_size, output_dim), name=f"B{i}_y")
        w = [None] * (num_layer + 1) # the last one for output y'
        for i in range(num_layer):
            if i == 0:
                w[i] = HloParameter((input_dim, hidden_dim[i]), name=f"w_{i}")
            else:
                w[i] = HloParameter((hidden_dim[i - 1], hidden_dim[i]), name=f"w_{i}")
        w[num_layer] = HloParameter((hidden_dim[num_layer - 1], output_dim), name=f"w_{num_layer}")

        h = [[None for i in range(num_layer + 1)] for j in range(num_batch)]
        loss = [None] * num_batch
        grad_loss = [None] * num_batch
        grad_h = [[None for i in range(num_layer + 1)] for j in range(num_batch)]
        grad_w = [[None for i in range(num_layer + 1)] for j in range(num_batch)]
        for batch in range(num_batch):
            ## forward
            for layer in range(num_layer + 1):
                if layer == 0:
                    h[batch][layer] = HloDot(x[batch], w[layer], name=f"B{batch}_h_{layer}")
                else:
                    h[batch][layer] = HloDot(h[batch][layer - 1], w[layer], name=f"B{batch}_h_{layer}")
            assert h[batch][num_layer].shape == y[batch].shape
            loss[batch] = HloSubtract(h[batch][num_layer], y[batch], name=f"B{batch}_loss")
    
            ## backward
            coef = HloConstant(2 / batch_size / output_dim)
            coef = HloBroadcast(coef, (batch_size, output_dim))
            grad_loss[batch] = HloMutiply(loss[batch], coef, name=f"B{batch}_dloss")
    
            for layer in reversed(range(num_layer + 1)):
                if layer == num_layer:
                    grad_h[batch][layer] = grad_loss[batch]
                else:
                    grad_h[batch][layer] = HloDot(grad_h[batch][layer + 1], w[layer + 1],
                                                  lhs_contracting_dims=(1,),
                                                  rhs_contracting_dims=(1,),
                                                  name=f"B{batch}_dh_{layer}")
                if layer == 0:
                    grad_w[batch][layer] = HloDot(x[batch], grad_h[batch][layer],
                                                  lhs_contracting_dims=(0,),
                                                  rhs_contracting_dims=(0,),
                                                  name=f"B{batch}_dw_{layer}")
                else:
                    grad_w[batch][layer] = HloDot(h[batch][layer - 1], grad_h[batch][layer],
                                                  lhs_contracting_dims=(0,),
                                                  rhs_contracting_dims=(0,),
                                                  name=f"B{batch}_dw_{layer}")

        ## sum up grad_w
        if num_batch > 1:
            grad_w_sum = [[None for i in range(num_layer + 1)] for j in range(num_batch - 1)]
            grad_w_avg = [None] * (num_layer + 1)
            for layer in range(num_layer + 1):
                for batch in range(num_batch - 1):
                    if batch == 0:
                        grad_w_sum[batch][layer] = HloAdd(grad_w[batch][layer], grad_w[batch + 1][layer])
                    else:
                        grad_w_sum[batch][layer] = HloAdd(grad_w_sum[batch - 1][layer], grad_w[batch + 1][layer])
                coef = HloConstant(1 / num_batch)
                coef = HloBroadcast(coef, grad_w_sum[num_batch - 2][layer].shape)
                grad_w_avg[layer] = HloMutiply(grad_w_sum[num_batch - 2][layer], coef, name=f"dw_{layer}")
 
        new_w = [None] * (num_layer + 1)
        for layer in range(num_layer + 1):
            if num_batch == 1:
                new_w[layer] = HloSubtract(w[layer], grad_w[0][layer], name=f"new_w_{layer}")
            else:
                new_w[layer] = HloSubtract(w[layer], grad_w_avg[layer], name=f"new_w_{layer}")

        out = HloTuple(tuple(new_w))

        ## alias
        computation.set_alias([(w[i], new_w[i]) for i in range(num_layer + 1)])

    return computation

def get_latency(node):
    return node.get_flop_count() // FLOP_PER_SECOND

def get_comm_cost(node):
    return np.prod(node.shape) // IO_PER_SECOND

def call_ilp(graph, args):
    import pulp
    from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus, LpContinuous

    tic = time.time()

    ## initialization & preprocessing
    num_nodes = len(graph.instructions) - 1
    max_subgraphs = args['maxDevices'] * args['maxSubgraphsPerDevice']
    # compute latencyUpperBound
    latencyUpperBound = 0
    for node in graph.instructions[:-1]:
        latencyUpperBound += get_latency(node) + get_comm_cost(node) * num_nodes
    assert latencyUpperBound < 1e9
#    print('latencyUpperBound', latencyUpperBound)

    ## create variables
    # 1. Indicator for the placement of each node
    x = LpVariable.dicts('Indicator', (range(num_nodes), range(max_subgraphs)), cat='Binary')
    # 2. CommIn, CommOut
    comm_in = LpVariable.dicts('CommIn', (range(num_nodes), range(max_subgraphs)), 0, None, LpContinuous)
    comm_out = LpVariable.dicts('CommOut', (range(num_nodes), range(max_subgraphs)), 0, None, LpContinuous)
    # 3. Latency
    latency = LpVariable.dicts('Latency', (range(num_nodes)), 0, None, LpContinuous)
    # 4. Subgraph start, finish
    start = LpVariable.dicts('SubgraphStart', (range(max_subgraphs)), 0, None, LpContinuous)
    finish = LpVariable.dicts('SubgraphFinish', (range(max_subgraphs)), 0, None, LpContinuous)
    # 5. TotalLatency that we are minimizing
    total_latency = LpVariable('TotalLatency', 0, None, LpContinuous)

    ## objective
    prob = LpProblem("NaiveFormulation", LpMinimize)
    prob += total_latency

    ## constraints
    # 1. schedule every node on exactly one subgraph
    for i in range(num_nodes):
        prob += lpSum([x[i][subgraph] for subgraph in range(max_subgraphs)]) == 1
    # 2. total size of nodes on one device cannot exceed its size capability
    for device in range(args['maxDevices']):
        l = args['maxSubgraphsPerDevice'] * device
        r = l + args['maxSubgraphsPerDevice']
        size = []
        for subgraph in range(l, r):
            for node in graph.instructions[:-1]:
                if isinstance(node, HloParameter):
                    tmp_size = np.prod(node.shape)
                else:
                    tmp_size = 0
                size.append(tmp_size * x[node.index][subgraph])
        prob += lpSum(size) <= args['maxSizePerDevice']
    # 3. CommIn, CommOut
    for subgraph in range(max_subgraphs):
        for edge in graph.get_edge_set():
            source = edge[0]
            dest = edge[1]
            prob += (comm_in[source][subgraph] >= x[dest][subgraph] - x[source][subgraph])
            prob += (comm_out[source][subgraph] >= x[source][subgraph] - x[dest][subgraph])
    # 4. subgraph can't start before the incoming results are ready  
    for subgraph in range(max_subgraphs):
        for node_id in range(num_nodes):
            # quadratic constraint!
            # model.addConstr(start[subgraph] >= latency[node_id] * comm_in[node_id, subgraph])
            # rewrite it like so:
            prob += (start[subgraph] >= latency[node_id]
                     - (1 - comm_in[node_id][subgraph]) * latencyUpperBound)
    # 5. finish time of a subgraph
    for subgraph in range(max_subgraphs):
        load = []
        for node in graph.instructions[:-1]:
            load.append(get_latency(node) * x[node.index][subgraph])
            # model with "calls": communication NOT overlapped with compute
            # so we add communication here
            load.append(get_comm_cost(node) * comm_in[node.index][subgraph])
            load.append(get_comm_cost(node) * comm_out[node.index][subgraph])
        prob += (finish[subgraph] == start[subgraph] + lpSum(load))
    # 6. Latency for nodes on a subgraph
    for subgraph in range(max_subgraphs):
        for node_id in range(num_nodes):
            # quadratic constraint!
            # model.addConstr(latency[node_id] >= x[node_id][subgraph] * finish[subgraph])
            # rewrite it like so:
            prob += (latency[node_id] >= finish[subgraph]
                     - (1 - x[node_id][subgraph]) * latencyUpperBound)
    # 7. ordering of subgraphs assigned to the same device
    for device in range(args['maxDevices']):
        l = args['maxSubgraphsPerDevice'] * device
        r = l + args['maxSubgraphsPerDevice']
        for subgraph in range(l + 1, r):
            prob += (start[subgraph] >= finish[subgraph - 1])
    # 8. TotalLatency that we are minimizing
    for node_id in range(num_nodes):
        prob += (total_latency >= latency[node_id])

    ## send to solver
    time_limit = 60
    assert "COIN_CMD" in pulp.listSolvers(onlyAvailable=True), (
        "Please install ILP solvers by 'sudo apt install coinor-cbc'"
    )
    solver = pulp.COIN_CMD(mip=True,
                           msg=False,
                           timeLimit=time_limit,
                           threads=multiprocessing.cpu_count())
    prob.solve(solver)

    ## get results
    status = prob.status
    objective = pulp.value(prob.objective)
    objective = float(objective) if objective is not None else -1.0
    print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
          f"Time: {time.time() - tic}")

    if prob.status in [pulp.LpStatusInfeasible]:
        raise RuntimeError("Cannot run find any feasible solutions.")

    partition = [None] * max_subgraphs
    for subgraph in range(max_subgraphs):
        partition[subgraph] = []
        for node in graph.instructions[:-1]:
            if pulp.value(x[node.index][subgraph]) == 1:
                partition[subgraph].append(node.index)
#    assert np.sum([len(lst) for lst in partition]) == num_nodes

    # print start and finish time of each subgraph
    for subgraph in range(max_subgraphs):
        print(f'subgraph {subgraph} start {pulp.value(start[subgraph])} \
                finish {pulp.value(finish[subgraph])}')

    for node in graph.instructions[:-1]:
        print(node.name, 'latency', pulp.value(latency[node.index]))
        
    return objective, status, partition
 
def print_placement(partition, num_device):
    assert len(partition) % num_device == 0
    num_subgraph = len(partition) // num_device
    for device in range(num_device):
        print('device', device)
        for subgraph in range(num_subgraph):
            print(f'subgraph {subgraph}: {partition[num_subgraph * device + subgraph]}')

colors = ["green", "blue", "red", "gray", "pink", "yellow"]

def visualize(graph, partition):
    g = graphviz.Digraph("hlo_graph")

    group_by_batch = defaultdict(list)
    ins_id_to_partition_id = {}

    for partition_id in range(len(partition)):
        for ins_id in partition[partition_id]:
            ins_id_to_partition_id[ins_id] = partition_id

    for i, ins in enumerate(graph.instructions[:-1]):
        if ins.name[0] == "B":
            batch_id = int(ins.name.split("_")[0][1:])
            group_by_batch[batch_id].append(i)
        else:
            g.node(str(i), ins.name, color=colors[ins_id_to_partition_id[i]])

    for batch_id, ins_ids in group_by_batch.items():
        with g.subgraph(name=f'cluster_{batch_id}') as c:
            c.attr(label=f'batch {batch_id}')
            for i in ins_ids:
                name = graph.instructions[i].name
                c.node(str(i), name[name.index("_")+1:], color=colors[ins_id_to_partition_id[i]])

    for (src, dst) in graph.get_edge_set():
        g.edge(str(src), str(dst))

    g.render(directory='.')


if __name__ == "__main__":
    num_layer = 2 - 1
    hidden_dim = [1024] * num_layer
    graph = get_graph_naive(num_layer=num_layer, num_batch=2, batch_size=4, \
                            input_dim=1024, hidden_dim=hidden_dim, output_dim=1024)
    print(graph)

    args = {'maxDevices': 3, 'maxSubgraphsPerDevice': 2,
            'maxSizePerDevice': 1024 * 1024 * 1 + 4 * 1024 * 4 + 1000}
    objective, status, partition = call_ilp(graph, args)
    visualize(graph, partition)
