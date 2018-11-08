from copy import deepcopy
import numpy as np
from collections import namedtuple
# This is the file where should insert your own code.
#
# Author: Paul Hilt <mk197@uni-heidelberg.de>

# Theoretical Questions:
# Exercise 1.1:
# Formulate the MAP-inference problem for graphical models. How is the optimization
# objective defined? Which role play the unary and pairwise costs?
#
# The MAP-inference problem is derived from a graphical model, in which one specific label from a set of possible
# labels should be assigned to each node of an undirected graph. All label assignment come with a specific cost which
# which is composed of unary and pairwise costs. The unary costs depend on the underlying data. E.g. if an underlying
# image has a specific pixel value, the unary-cost function of the respective vertex of the graph will be low for one
# specific label. The pairwise cost function depends on the difference between each vertex and its neighbor. E.g.
# Labels of vertices that vary highly from their neighbors induce high costs. Tn optimization terms, the unary costs
# can be expressed as the data term and the pairwise costs as the regularization term. The objective function which is
# to be minimized over the set af all possible labelings, is the sum of unary and pairwise costs of all
# vertices. The MAP-inference problem is NP-hard, because Hamilton-Cycle which is NP-complete can be reduced to it in
# polynomial time.
#
# Exercise 1.2
# Evaluate the energy function for the given graphical model and parameter settings. What
# role does the parameter λ play for the “Potts” pairwise potential?
# The energy function penalizes unequal labelings of two neighbouring vertices. λ can be seen as regularization
# parameter. It determines, how much to trust the pairwise costs relative to the unaries.



Edge = namedtuple('Edge', 'left right costs')
# For exercise 1.2

def evaluate_energy(nodes, edges, assignment):
    e=0         # Total energy of graph
    for node in range(len(nodes)):
        e += getattr(nodes[node], 'costs')[assignment[node]]
    for edge in edges:
        e += getattr(edge, 'costs')[assignment[getattr(edge, 'left')], assignment[getattr(edge, 'right')]]
    return e

# For exercise 1.3
def bruteforce(nodes, edges):
    assignment = [0] * len(nodes)
    best_assignment = [0] * len(nodes)
    counter = 0
    energy = evaluate_energy(nodes, edges, assignment)
    l_nodes = len(nodes)

    def brute_force_recursion(node, label, assignment):
        nonlocal best_assignment
        nonlocal energy
        nonlocal counter
        if node == l_nodes or label == len(getattr(nodes[node], 'costs')):
            return
        counter += 1
        assignment[node] = label
        new_energy = evaluate_energy(nodes, edges, assignment)
        if new_energy < energy:
            energy = new_energy
            best_assignment = deepcopy(assignment)
        brute_force_recursion(node + 1, 0, assignment)
        return brute_force_recursion(node, label+1, assignment)

    brute_force_recursion(0, 0, assignment)
    return best_assignment, energy


def dynamic_programming(nodes, edges):
    F = []
    helper_1 = []
    helper_2 = []
    r_helper = []
    helper_1.extend([0 for cost in getattr(nodes[0], 'costs')])
    F.append([])
    F[-1].append(helper_1)
    F[-1].append(helper_1)
    helper_1 = []
    for i in range(1, len(nodes)):
        for s in range(len(getattr(nodes[i], 'costs'))):
            for t in range(len(getattr(nodes[i-1], 'costs'))):
                helper_1.append(F[i-1][0][t]
                                + getattr(nodes[i-1], 'costs')[t]
                                + get_edgeCosts(edges, i-1, i)[t, s])
            curr_min = min(helper_1)
            min_idx = np.argmin(helper_1)
            r_helper.append(min_idx)
            helper_2.append(curr_min)
            helper_1 = []
        F.append([])
        F[-1].append(helper_2)
        F[-1].append(r_helper)
        helper_2 = []
        r_helper = []
    return F

def compute_min_marginals(nodes, edges):
    l_nds = len(nodes)
    # reverse nodes and edges:
    rvsd_edges = []
    for edge in edges:
        fwd_costs = getattr(edge, 'costs')
        rvsd_costs = {}
        for key in list(fwd_costs.keys()):
            rvsd_costs[(key[1], key[0])] = fwd_costs[key]
        rvsd_edges.append(Edge(right=abs(getattr(edge, 'left')-(l_nds-1)),
                           left=abs(getattr(edge, 'right')-(l_nds-1)),
                           costs=rvsd_costs))
    rvsd_nodes = list(reversed(nodes))

    # execute forward and backward dynamic programming:
    fwd_intermediates = dynamic_programming(nodes, edges)
    bwd_intermediates = dynamic_programming(rvsd_nodes, rvsd_edges)
    # Make sure, both calculated the map solution
    assert backtrack(nodes, edges, *fwd_intermediates) == list(reversed(backtrack(rvsd_nodes, rvsd_edges, *bwd_intermediates)))

    # calc the node min marginals
    bwd_intermediates = list(reversed(bwd_intermediates))
    min_marginals = []
    for node, i in zip(nodes, range(l_nds)):
        min_marginals.append([fwd_intermediates[i][0][l]
                              + bwd_intermediates[i][0][l]
                              + getattr(node, 'costs')[l]
                              for l in range(len(getattr(node, 'costs')))])

    print(min_marginals)

    return min_marginals

def dynamic_programming_tree(nodes, edges):
    F_v = []
    F_indx = []
    node_list = [i for i in range(len(nodes))]
    edge_list = [i for i in range(len(edges))]

    for node in nodes:
        F_v.append(getattr(node,'costs'))
        F_indx.append([0 for cost in getattr(nodes[0], 'costs')])

    while len(node_list) > 1 :
        connect_list = [[] for node in range(len(nodes))]
        direction_list = [[] for node in range(len(nodes))]
        edge_kill_list = [[] for i in range(len(edges)+1)]

        for i, u in enumerate(node_list):
            for j, edge in enumerate(edge_list):
                if getattr(edges[edge],'left') == u:

                    connect_list[u] += [getattr(edges[edge], 'right')]
                    direction_list[u] = 'L-R'
                    edge_kill_list[u] = j

                if getattr(edges[edge],'right') == u:

                    connect_list[u] += [getattr(edges[edge], 'left')]
                    direction_list[u] = 'R-L'
                    edge_kill_list[u] = j

            #check if its a leave with only on connection
            if len(connect_list[u]) == 1:
                label_list = [0 for i in range(len(F_v[0]))]
                for label in range(len(F_v[0])):
                    for t in range(len(F_v[0])):
                        if direction_list[u] == 'L-R':
                            label_list[t] = (F_v[u][t] + get_edgeCosts(edges, u, connect_list[u][0])[label, t])
                        elif direction_list[u] == 'R-L':
                            label_list[t] = (F_v[u][t] + get_edgeCosts(edges, connect_list[u][0], u)[label, t])

                    min_cost = min(label_list)
                    min_idx = np.argmin(label_list)

                    F_v[connect_list[u][0]][label] += min_cost
                    F_indx[connect_list[u][0]][label] = min_idx

                #pop leave 
                node_list.pop(i)
                if len(edge_list)>0:
                    edge_list.pop(edge_kill_list[u])

    intermediates = []
    for i in range(len(F_v)):
        intermediates.append([F_v[i], F_indx[i]])

    print(intermediates)

    return intermediates
                    

def get_edgeCosts(edges, left, right):
    for edge in edges:
        if getattr(edge, 'left') == left and getattr(edge, 'right') == right:
            return getattr(edge, 'costs')
    return None

def backtrack(nodes, edges, *F):
    y = [0]*len(nodes)
    y[-1] = np.argmin([F[-1][0][s] + getattr(nodes[-1], 'costs')[s] for s in range(len(F[-1][0]))])
    for i in range(len(nodes)-1, 0, -1):
        y[i-1] = F[i][1][y[i]]
    return y
