from tsukuba_model import *
from grid import *
import random
import numpy as np
from copy import deepcopy

#3.1 :
#What type of algorithm is it ?
# ICM and Block-ICM are coordinate descent algorithms. They both apply the subgradient method with respect to one
# variable or to one block (acyclic induced subgraph) at a time. This induces, that they are not strictly non increasing
# and also they find fixed points that are local optima where thy stop changing the labels.

#Does the algorithm provide guarantees?
# ICM and Block-ICM are both finding a labeling with lower energy, but the
# ICM optimization depends highly on the label initialization and often finds only a local minimum 

#Time Complexity of my implementation:
# ICM: O(nodes * edges) BlockICM: O(grid.width * grid.height * nodes * labels) without parallelization

#Describe the quality of the output
# ICM: Energy of the labeling decreases slightly and varies with the initial labeling 
# Block-ICM: For Graph with induced subgraphs that are acyclic, Block-ICM most likely performs better than ICM,
# however there is no guarantee for that both algorithms are theoretically able to find the global optimum. They both
# find local optima where ICM finds local optima per node and Bloch-ICM per induced subgraph which it perfoms on.

#3.2 :
#What type of algorithm is it ?
# The subgradient method acting on the dual problem computes all subgradients with respect to each lagrange dual variable (one per step) and updates them.
# All except two of those subgradients are 0, meaning the label is locally optimal and does not need a change but still all
# subgradients have to be computed in each iteration. This, and the diminishing stepsize which is required for the convergence
# criterium make the algorithm slow.

#Does the algorithm provide guarantees?
# the algorithm guarantees convergence to the global optimum in a finite number of iterations. If the primal problem is
# dual optimal.

#3.3 :
#What type of algorithm is it ?
# It solves the primal problem by redestributing costs such that all unaries, are 0 and all costs are defined by the
# pairwise costs.
# In each iteration the locally optimal edge is selected per node.

#Does the algorithm provide guarantees?
# It enforces convergence to node edge agreement (0-subgradient).
# The number of locally optimal edges with node edge agreement is non decreasing. However there are cases
# (even for acyclic graphs) where this algorithm is very slow. This is overcome by anisotropic diffusion.

#3.4 :
#What type of algorithm is it ?


#Does the algorithm provide guarantees?


nodes, edges = load_downsampled_model(32)


def evaluate_energy(nodes, edges, assignment):
    e=0         # Total energy of graph
    for node in range(len(nodes)):
        e += getattr(nodes[node], 'costs')[assignment[node]]
    for edge in edges:
        e += getattr(edge, 'costs')[assignment[getattr(edge, 'left')], assignment[getattr(edge, 'right')]]
    return e

def ICM(nodes, edges):

	#initialize labeling
	labeling = []
	for _ in range(len(nodes)):
		labeling.append(random.randint(0,15))

	#energy after init
	print(evaluate_energy(nodes, edges, labeling))

	while True:

		old_labeling = deepcopy(labeling)

		for i, node in enumerate(nodes):
			for edge in edges:
				if edge.left == i:
					label_v = labeling[i]
					label_u = labeling[edge.right]

					if label_v != label_u:
						labeling[i] = label_v
					else:
						costs = []

						for label in range(len(node[0])-1):
							costs.append((node[0][label] + edge.costs[label, label_u]))

						s = np.argmin(costs)
						labeling[i] = s

		check_list = [i for i,j in zip(old_labeling, labeling) if i==j]
		if len(check_list) == len(labeling):
			break

	print(evaluate_energy(nodes, edges, labeling))


ICM(nodes, edges)

def Block_ICM(nodes, edges):

	grid = determine_grid(nodes, edges)	
	print(grid.height, grid.width)
	decompostion = row_column_decomposition(grid)

	label_list = []
	for _ in range(len(decompostion)):
		label_list.append([random.randint(0,15) for _ in range(len(decompostion[0]))])

	#row 
	for start in range(2): #loop for even and odd
		for n in range(start, grid.height, 2):
			for i, node in enumerate(range(len(decompostion[n]))):				
				cost = []

				for label in range(len(nodes[0][0])-1):
					if n==0:
						cost.append(decompostion[n+1][i].costs[label, label_list[n][i]])
					elif n==grid.height-1:
						cost.append(decompostion[n-1][i].costs[label, label_list[n][i]])
					else:
						pairwise = decompostion[n+1][i].costs[label, label_list[n][i]] + decompostion[n-1][i].costs[label, label_list[n][i]]
						cost.append(pairwise)

				label = np.argmin(cost)
				label_list[n][i] = label

	#column
	for start in range(2):
		for n in range(start+grid.height, len(decompostion), 2):
			for i, node in enumerate(range(len(decompostion[n]))):				
				cost = []

				for label in range(len(nodes[0][0])-1):
					if n==grid.height:
						cost.append(decompostion[n+1][i].costs[label, label_list[n][i]])
					elif n==len(decompostion)-1:
						cost.append(decompostion[n-1][i].costs[label, label_list[n][i]])
					else:
						pairwise = decompostion[n+1][i].costs[label, label_list[n][i]] + decompostion[n-1][i].costs[label, label_list[n][i]]
						cost.append(pairwise)

				label = np.argmin(cost)
				label_list[n][i] = label

	print(len(label_list))
	print(evaluate_energy(nodes, edges, label_list))

Block_ICM(nodes, edges)

