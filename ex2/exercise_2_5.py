from model_2_5 import *
import exercise_2 as student
from pulp import *
import time

# tsu_down_32: 12x9 = 108 pixel, each pixel has 16 labels -> total: 16^108 labeling. Number of variables in the 
# ILP presentation: 16x108 = 1728 labels. Number of edges = 195 --> 195 * 16^2 = 49920 total number of label pairs.
# O(49920) variables in the ILP respresentation. The total number of constraints has the same order O(49920) as the 
# number of variables

# tsu_down_16: 432 pixels. Number edges = 822 --> 822 * 16^2 = 210432  --> O(210432)
# tsu_down_8: 1728 pixels. Number edges = 3372 --> 3372 * 16^2 = 863232 --> O(863232)
# tsu_down_4: 6912 pixels. Number edges = 13656 --> 13656 * 16^2 = 3495936 --> O(3495936)
# tsu_down_2: 27648 pixels. Number edges = 54960 --> 54960 * 16^2 = 14069760 --> O(14069760)
# tsu_down_1: 110592 pixels. Number edges = 220512 --> 220512 * 16^2 = 56451072 --> O(56451072)

# For LP the order for the number of variables is the same but less constraints are used

# Inference Time 32: 5.134387016296387 seconds ILP 3.2923340797424316 seconds LP
# Inference Time 16: 28.932204961776733 seconds ILP 12.250999689102173 seconds LP
# Inference Time 8:  202.6747579574585 seconds ILP 56.20522689819336 seconds LP
# Inference Time 4:  2671.6575849056244 seconds ILP 477.34674096107483 seconds LP
# Inference Time 2:  too long, would probably exceed the deadline 
# Inference Time 1:  -----



for num in [32, 16, 8, 4, 2, 1]:

	print("Tsu_down_{}".format(num))

	model = load_downsampled_model(num)
	# print(len(model_32[1]))
	# print(len(model_32[1]) * 16**2)
	ilp = student.convert_to_ilp(model[0], model[1])
	start_time = time.time()
	assert(ilp.solve())
	print("--- %s seconds ILP---" % (time.time() - start_time))

	lp = student.convert_to_lp(model[0], model[1])
	start_time = time.time()
	assert(lp.solve())
	print("--- %s seconds LP---" % (time.time() - start_time))

