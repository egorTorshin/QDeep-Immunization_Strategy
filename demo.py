import argparse
import sys
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt

import dimod
import neal

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt


def read_in_args(args):
    """Read user-specified parameters for graph generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", default='karate',
                        choices=['karate', 'internet', 'rand-reg', 'ER', 'SF'],
                        help='Graph to partition (default: %(default)s)')
    parser.add_argument("-n", "--nodes",
                        help="Set graph size for graph (default: %(default)s)",
                        default=34, type=int)
    parser.add_argument("-d", "--degree",
                        help="Set node degree for a random regular graph (default: %(default)s)",
                        default=4, type=int)
    parser.add_argument("-p", "--prob",
                        help="Set graph edge probability for an ER graph. Must be between 0 and 1 (default: %(default)s)",
                        default=0.25, type=float)
    parser.add_argument("-e", "--new-edges",
                        help="Set the number of edges from a new node to existing nodes in an SF graph (default: %(default)s)",
                        default=4, type=int)
    return parser.parse_args(args)


def build_graph(args):
    """Builds a sample graph based on user input."""
    if args.graph == 'karate':
        print("\nReading in karate graph...")
        G = nx.karate_club_graph()
    elif args.graph == 'internet':
        print("\nReading in internet graph of size", args.nodes, "...")
        G = nx.random_internet_as_graph(args.nodes)
    elif args.graph == 'rand-reg':
        print("\nGenerating random regular graph...")
        G = nx.random_regular_graph(args.degree, args.nodes)
    elif args.graph == 'ER':
        print("\nGenerating Erdos-Renyi graph...")
        G = nx.erdos_renyi_graph(args.nodes, args.prob)
    elif args.graph == 'SF':
        print("\nGenerating Barabasi-Albert scale-free graph...")
        G = nx.barabasi_albert_graph(args.nodes, args.new_edges)
    else:
        print("\nReading in karate graph...")
        G = nx.karate_club_graph()
    return G


def visualize_input_graph(G):
    """Visualize the input graph."""
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='r', edgecolors='k')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), style='solid', edge_color='#808080')
    plt.draw()
    plt.savefig('input_graph.png')
    plt.close()
    print("Input graph visualization saved to 'input_graph.png'.")


def build_bqm(G, penalty=10.0):
    """
    Build a BQM with 'soft' penalty terms instead of hard constraints:

      - We have binary variables x_{v,0}, x_{v,1}, x_{v,2} for each vertex v
        representing membership in group 0, 1, or 2.
      - Objective: minimize the total number of vertices in group 2 (separator).
      - Constraints are turned into penalty terms:
          1) Discrete: x_{v,0} + x_{v,1} + x_{v,2} = 1
          2) |G1| = |G2|
          3) No edges between G1 and G2
        Each is encoded as a quadratic penalty added to the BQM.

    Returns a dimod.BinaryQuadraticModel.
    """

    bqm = dimod.BinaryQuadraticModel('BINARY')

    vertices = list(G.nodes())
    num_groups = 3

    # (A) First, explicitly add all variables with initial bias=0
    for v in vertices:
        for g in range(num_groups):
            var = f'x_{v}_{g}'
            bqm.add_variable(var, 0.0)

    # (1) Objective: minimize the number of vertices in group 2
    #     => add a linear bias +1 to x_{v,2}
    for v in vertices:
        var_2 = f'x_{v}_2'
        bqm.add_linear(var_2, 1.0)

    # (2) Penalty for the "discrete" constraint: (x_{v,0} + x_{v,1} + x_{v,2} - 1)^2
    #    Expand: (s - 1)^2 = s^2 - 2s + 1, where s = x_{v,0} + x_{v,1} + x_{v,2}.
    for v in vertices:
        var0 = f'x_{v}_0'
        var1 = f'x_{v}_1'
        var2 = f'x_{v}_2'

        # (A) s^2 = sum of linear terms plus 2 * cross terms
        bqm.add_linear(var0, penalty)
        bqm.add_linear(var1, penalty)
        bqm.add_linear(var2, penalty)

        bqm.add_quadratic(var0, var1, 2 * penalty)
        bqm.add_quadratic(var0, var2, 2 * penalty)
        bqm.add_quadratic(var1, var2, 2 * penalty)

        # (B) -2 * penalty * s => subtract 2*penalty from each x_{v,g}
        bqm.add_linear(var0, -2 * penalty)
        bqm.add_linear(var1, -2 * penalty)
        bqm.add_linear(var2, -2 * penalty)

        # (C) Add penalty for the constant term (1^2)
        bqm.offset += penalty

    # (3) Penalty for |G1| = |G2|: (sum_v x_{v,0} - sum_v x_{v,1})^2
    #    Expand: A^2 + B^2 - 2AB, where A = sum_v x_{v,0}, B = sum_v x_{v,1}
    penalty_eq = penalty

    g1_vars = [f'x_{v}_0' for v in vertices]
    g2_vars = [f'x_{v}_1' for v in vertices]

    # A^2 = sum_v x_{v,0} + 2 * sum_{v1 < v2} x_{v1,0} x_{v2,0}
    for i in range(len(vertices)):
        vi0 = g1_vars[i]
        bqm.add_linear(vi0, penalty_eq)
        for j in range(i + 1, len(vertices)):
            vj0 = g1_vars[j]
            bqm.add_quadratic(vi0, vj0, 2 * penalty_eq)

    # B^2 = sum_v x_{v,1} + 2 * sum_{v1 < v2} x_{v1,1} x_{v2,1}
    for i in range(len(vertices)):
        vi1 = g2_vars[i]
        bqm.add_linear(vi1, penalty_eq)
        for j in range(i + 1, len(vertices)):
            vj1 = g2_vars[j]
            bqm.add_quadratic(vi1, vj1, 2 * penalty_eq)

    # -2AB = -2 * sum_{i} sum_{j} x_{i,0} x_{j,1}
    for i in range(len(vertices)):
        vi0 = g1_vars[i]
        for j in range(len(vertices)):
            vj1 = g2_vars[j]
            bqm.add_quadratic(vi0, vj1, -2 * penalty_eq)

    # (4) Penalty for no edges between G1 and G2:
    #     penalty_cross * sum_{(a,b) in E}[ x_{a,0} x_{b,1} + x_{a,1} x_{b,0} ]
    penalty_cross = penalty
    for (a, b) in G.edges():
        if a != b:
            va0, vb1 = f'x_{a}_0', f'x_{b}_1'
            va1, vb0 = f'x_{a}_1', f'x_{b}_0'
            bqm.add_quadratic(va0, vb1, penalty_cross)
            bqm.add_quadratic(va1, vb0, penalty_cross)

    return bqm


def run_bqm_and_collect_solutions(bqm):
    """Solve the penalty-based BQM using SimulatedAnnealingSampler."""
    sampler = neal.SimulatedAnnealingSampler()
    print("\nSolving BQM with penalty-based constraints...")
    sampleset = sampler.sample(bqm, num_reads=200)
    best_sample = sampleset.first.sample  # the best (lowest-energy) sample
    return best_sample


def process_sample(G, sample):
    """
    Interpret the solution: for each vertex v, determine which group it belongs to.
    'sample' is a dict { 'x_v_g': 0 or 1 }.
    """
    group_1 = []
    group_2 = []
    sep_group = []

    for key, val in sample.items():
        if val == 1:
            # key like 'x_3_2' => vertex=3, group=2
            parts = key.split('_')
            v = int(parts[1])
            g = int(parts[2])
            if g == 0:
                group_1.append(v)
            elif g == 1:
                group_2.append(v)
            elif g == 2:
                sep_group.append(v)

    # Print results
    print("\nPartition Found:")
    print("\tGroup 1: \tSize", len(group_1))
    print("\tGroup 2: \tSize", len(group_2))
    print("\tSeparator: \tSize", len(sep_group))

    print("\nSeparator Fraction: \t", len(sep_group) / len(G.nodes()))

    # Check edges that cross between G1 and G2
    illegal_edges = []
    for (u, v) in G.edges():
        if sample.get(f'x_{u}_0', 0) * sample.get(f'x_{v}_1', 0) == 1 or \
           sample.get(f'x_{u}_1', 0) * sample.get(f'x_{v}_0', 0) == 1:
            illegal_edges.append((u, v))

    print("\nNumber of illegal edges:\t", len(illegal_edges))

    return group_1, group_2, sep_group, illegal_edges


def visualize_results(G, group_1, group_2, sep_group, illegal_edges):
    """Visualize the final partition."""
    print("\nVisualizing output...")

    G1 = G.subgraph(group_1)
    G2 = G.subgraph(group_2)
    SG = G.subgraph(sep_group)

    pos_1 = nx.random_layout(G1, center=(-5, 0))
    pos_2 = nx.random_layout(G2, center=(5, 0))
    pos_sep = nx.random_layout(SG, center=(0, 0))
    pos = {**pos_1, **pos_2, **pos_sep}

    nx.draw_networkx_nodes(G, pos_1, node_size=50, nodelist=group_1, node_color='#17bebb', edgecolors='k')
    nx.draw_networkx_nodes(G, pos_2, node_size=50, nodelist=group_2, node_color='#2a7de1', edgecolors='k')
    nx.draw_networkx_nodes(G, pos_sep, node_size=50, nodelist=sep_group, node_color='#f37820', edgecolors='k')

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), style='solid', edge_color='#808080')
    nx.draw_networkx_edges(G, pos, edgelist=illegal_edges, style='solid', edge_color='red')

    plt.draw()
    output_name = 'separator_penalty.png'
    plt.savefig(output_name)
    plt.close()
    print(f"\tOutput stored in {output_name}")


if __name__ == '__main__':
    # 1. Read command-line arguments
    args = read_in_args(sys.argv[1:])

    # 2. Generate or load the graph
    G = build_graph(args)

    # 3. Visualize the input graph
    visualize_input_graph(G)

    # 4. Build the BQM with penalty-based constraints
    bqm = build_bqm(G, penalty=10.0)

    # 5. Solve the BQM with simulated annealing
    best_sample = run_bqm_and_collect_solutions(bqm)

    # 6. Process the solution
    group_1, group_2, sep_group, illegal_edges = process_sample(G, best_sample)

    # 7. Visualize the result
    visualize_results(G, group_1, group_2, sep_group, illegal_edges)
