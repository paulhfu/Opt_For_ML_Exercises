# This is the file where should insert your own code.
#
# Author: Paul Hilt <mk197@uni-heidelberg.de>
# Author: Jonas Massa <massajonas@googlemail.com>

from pulp import *
from pulp.solvers import *

def convert_to_ilp(nodes, edges):
    prob = LpProblem("test1", LpMinimize)
    solver = CPLEX_CMD(path="/home/paul/tools/ibm/ILOG/CPLEX_Studio128/cplex/bin/x86-64_linux/cplex")
    prob.setSolver(solver)

    # Variables
    u_nodes = []
    objs = []
    bins = []
    for i in range(len(nodes)):
        for j in range(len(getattr(nodes[i], "costs"))):
            bins.append(LpVariable("u_n"+str(i)+","+str(j), 0, 1, cat=LpInteger))
            # Objective
            objs.append(bins[j] * getattr(nodes[i], "costs")[j])
        prob += lpSum(bins) == 1
        u_nodes.append(bins)
        bins = []

    u_edges = []
    bins = []
    for i in range(len(edges)):
        for key in getattr(edges[i], "costs"):
            bins.append(LpVariable("u_e"+str(getattr(edges[i], 'left')) + "," + str(getattr(edges[i], 'right'))+";"+str(key), 0, 1, cat=LpInteger))
            # Objective
            objs.append(bins[-1] * getattr(edges[i], "costs")[key])
            # Constraints
            prob += bins[-1] - u_nodes[getattr(edges[i], 'left')][key[0]] == 0
            prob += bins[-1] - u_nodes[getattr(edges[i], 'right')][key[1]] == 0
        u_edges.append(bins)
        bins = []

    prob += lpSum(objs)
    print(prob)
    print(LpStatus[prob.status])
    prob.solve()
    print(LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    return prob

def ilp_to_labeling(nodes, edges, ilp):
    solution = ilp.solve
    print(LpStatus[ilp.status])
