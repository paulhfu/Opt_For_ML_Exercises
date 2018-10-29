from copy import deepcopy
import numpy as np
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
    print(counter)
    print(l_nodes)
    print(len(getattr(nodes[0], 'costs')))
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
        for t in range(len(getattr(nodes[i], 'costs'))):
            for s in range(len(getattr(nodes[i-1], 'costs'))):
                helper_1.append(F[i-1][0][s]
                                + getattr(nodes[i-1], 'costs')[s]
                                + get_edgeCosts(edges, i-1, i)[s, t])
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
    F = []
    helper_1 = []
    helper_2 = []
    helper_1.extend([0 for cost in getattr(nodes[0], 'costs')])
    F.append(helper_1)
    helper_1 = []
    for i in range(1, len(nodes)):
        for t in range(len(getattr(nodes[i], 'costs'))):
            for s in range(len(getattr(nodes[i-1], 'costs'))):
                helper_1.append(F[i-1][s]
                                + getattr(nodes[i-1], 'costs')[s]
                                + get_edgeCosts(edges, i-1, i)[s, t])
            curr_min = min(helper_1)
            helper_2.append(curr_min)
            helper_1 = []
        F.append(helper_2)
        helper_2 = []
    return F


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