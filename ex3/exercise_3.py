from tsukuba_model import *
from grid import *
import random
import numpy as np
from copy import deepcopy


nodes, edges = load_downsampled_model(32)


def ICM(nodes, edges):

	#initialize labeling
	labeling = []
	for _ in range(len(nodes)):
		labeling.append(random.randint(0,15))


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

# ICM(nodes, edges)

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




Block_ICM(nodes, edges)