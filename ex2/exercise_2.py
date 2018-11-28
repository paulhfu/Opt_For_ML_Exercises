# This is the file where should insert your own code.
#
# Author: Paul Hilt <mk197@uni-heidelberg.de>
# Author: Jonas Massa <massajonas@googlemail.com>

# Theoretical Questions:
# Exercise 2.2:
# The LP solution has no variables that are no integers. This is why the LP solution, the rounded solution and the ILP
# solution match exactly.
# Exercise 2.3:
# The LP variables for the nodes of the lp solutions have all the value 0.5. The np.argmax function returns the
# smallest index of the largest value in an array. This is why the labels of the rounded solution all result to be 0.
# Whereas the ilp solution is [1, 1, 0].

from pulp import *
from pulp.solvers import *
import fnmatch
import numpy as np

def convert_to_ilp(nodes, edges):
    prob = LpProblem("test1", LpMinimize)

    # Variables
    u_nodes = []
    objs = []
    bins = []
    for i in range(len(nodes)):
        for j in range(len(nodes[i].costs)):
            bins.append(LpVariable("u_n"+str(i)+","+str(j), 0, 1, cat=LpInteger))
            # Objective
            objs.append(bins[j] * nodes[i].costs[j])
        prob += lpSum(bins) == 1
        u_nodes.append(bins)
        bins = []

    u_edges = []
    bins = []
    for i in range(len(edges)):
        for key in edges[i].costs:
            bins.append(LpVariable("u_e" + str(edges[i].left)
                                   + "," + str(edges[i].right)
                                   + ";" + str(key[0]) + "," + str(key[1])
                                   , 0, 1, cat=LpInteger))
            # Objective
            objs.append(bins[-1] * edges[i].costs[key])
        bins2 = []
        for label in range(len(nodes[edges[i].left].costs)):
            for var in bins:
                if "u_e"+str(edges[i].left) + "," + str(edges[i].right)+";"+str(label) in var.name:
                    bins2.append(var)  # Collect here all right edges-variables per node and label
            # Pose constraint, that the sum of right edges per node and label equals to the value of the variable of
            # corresponding node and label. This together with the same constraint for the left labels guarantees, that
            # nodes and edges in the solution correspond to the same labels.
            prob += lpSum(bins2) == u_nodes[edges[i].left][label]
            bins2 = []
        for label in range(len(nodes[edges[i].right].costs)):
            for var in bins:
                if fnmatch.fnmatch(var.name, "u_e" + str(edges[i].left)
                                             + "," + str(edges[i].right)
                                             + ";" + "*," + str(label)):
                    bins2.append(var)
            prob += lpSum(bins2) == u_nodes[edges[i].right][label]
            bins2 = []
        u_edges.append(bins)
        bins = []

    prob += lpSum(objs)
    return prob

def ilp_to_labeling(nodes, edges, ilp):
    sol = [0] * len(nodes)
    for v in ilp.variables():
        if "u_n" in v.name and v.varValue == 1:
            sol[int(v.name[3])] = int(v.name[5])
    return sol

def convert_to_lp(nodes, edges):
    prob = LpProblem("test1", LpMinimize)

    # Variables
    u_nodes = []
    objs = []
    bins = []
    for i in range(len(nodes)):
        for j in range(len(nodes[i].costs)):
            bins.append(LpVariable("u_n"+str(i)+","+str(j), 0, 1, cat=LpContinuous))
            # Objective
            objs.append(bins[j] * nodes[i].costs[j])
        prob += lpSum(bins) == 1
        u_nodes.append(bins)
        bins = []

    u_edges = []
    bins = []
    for i in range(len(edges)):
        for key in edges[i].costs:
            bins.append(LpVariable("u_e" + str(edges[i].left)
                                   + "," + str(edges[i].right) + ";"
                                   + str(key[0]) + "," + str(key[1])
                                   , 0, 1, cat=LpContinuous))
            # Objective
            objs.append(bins[-1] * edges[i].costs[key])
        bins2 = []
        for label in range(len(nodes[edges[i].left].costs)):
            for var in bins:
                if "u_e"+str(edges[i].left) + "," \
                        + str(edges[i].right) + ";" \
                        + str(label) in var.name:
                    bins2.append(var)  # Collect here all right edges-variables per node and label
            # Pose constraint, that the sum of right edges per node and label equals to the value of the variable of
            # corresponding node and label. This together with the same constraint for the left labels guarantees, that
            # nodes and edges in the solution correspond to the same labels.
            prob += lpSum(bins2) == u_nodes[edges[i].left][label]
            bins2 = []
        for label in range(len(nodes[edges[i].right].costs)):
            for var in bins:
                if fnmatch.fnmatch(var.name, "u_e"+str(edges[i].left)
                                             + "," + str(edges[i].right)
                                             + ";" + "*," + str(label)):
                    bins2.append(var)

            prob += lpSum(bins2) == u_nodes[edges[i].right][label]
            bins2 = []
        u_edges.append(bins)
        bins = []

    prob += lpSum(objs)
    return prob

def lp_to_labeling(nodes, edges, lp):
    sol = [0] * len(nodes)
    for n in range(len(nodes)):
        bin = [0] * len(nodes[n].costs)
        for v in lp.variables():
            if "u_n" in v.name and int(v.name[3:v.name.find(',')]) == n:
                bin[int(v.name[v.name.find(',') + 1:len(v.name)])] = v.varValue  # Collect node variables for each node
        sol[n] = np.argmax(np.asarray(bin))  # the index of the largest variable corresponds to rounded labeling
    return sol
