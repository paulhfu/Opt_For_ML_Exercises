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

ICM(nodes, edges)

def Block_ICM(nodes, edges):

	grid = determine_grid(nodes, edges)	
	decompostion = row_column_decomposition(grid)
	
