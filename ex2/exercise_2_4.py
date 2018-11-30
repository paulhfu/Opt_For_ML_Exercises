from model_2_4 import *
import exercise_2 as student
from pulp import *

# Acyclic Models     LP                    ILP                 
# model 1            -1652.1280753854446   -1652.1280753854446                            
# model 2            -1549.6171687052808   -1549.6171687052986         
# model 3            -1481.3946487904093   -1481.3946487904093 
# model 4            -1580.4022079676752   -1580.4022079676752 
# model 5            -1742.2031132608479   -1742.2031132608479

# Optima for ILP and LP is the same for acyclic models. Variable values of the 
# LP relaxation are also integer values. Its a good LP relaxation

# Cyclic Models      LP                    ILP
# model 1            -1245.7603539223276   -897.6670324970523
# model 2            -1341.8275658773414   -1057.279021838753
# model 3            -1291.550024505251    -999.251650373353
# model 4            -1190.8596833675488   -853.2618587226295
# model 5            -1311.4823238162048   -1034.1672284440253

# ILP and LP have different optima for cyclic models. The LP relaxation has
# fractional values and the optima has a lower bound. 
# The LP relaxation of ILP in cyclic models is not as good as for the acyclic model
# since the lower bound is not that close to the non-relaxed problem.

for i, (ac_m, cy_m) in enumerate(zip(ACYCLIC_MODELS, CYCLIC_MODELS)):
	
	print("Step {}".format(i))

	# Acyclic Model ILP
	print("Acyclic ILP")
	ilp_ac = student.convert_to_ilp(ac_m[0], ac_m[1])
	assert(ilp_ac.solve()) 
	print(tuple(student.ilp_to_labeling(ac_m[0], ac_m[1], ilp_ac)))
	print("Status:", LpStatus[ilp_ac.status])
	print(ilp_ac.objective.value())
	# for v in ilp_ac.variables():
	#     print(v.name, "=", v.varValue)

	# Acyclic Model LP
	print("Acyclic LP")
	lp_ac = student.convert_to_lp(ac_m[0], ac_m[1])
	assert(lp_ac.solve()) 
	print(tuple(student.lp_to_labeling(ac_m[0], ac_m[1], lp_ac)))
	print("Status:", LpStatus[lp_ac.status])
	print(lp_ac.objective.value())
	# for v in lp_ac.variables():
	#     print(v.name, "=", v.varValue)

	# Cyclic Model ILP
	print("Cyclic ILP")
	ilp_cy = student.convert_to_ilp(cy_m[0], cy_m[1])
	assert(ilp_cy.solve()) 
	print(tuple(student.ilp_to_labeling(cy_m[0], cy_m[1], ilp_cy)))
	print("Status:", LpStatus[ilp_cy.status])
	print(ilp_cy.objective.value())
    # for v in ilp_cy.variables():
	#     print(v.name, "=", v.varValue)

	# Cyclic Model LP
	print("Cyclic LP")
	lp_cy = student.convert_to_lp(cy_m[0], cy_m[1])
	assert(lp_cy.solve()) 
	print(tuple(student.lp_to_labeling(cy_m[0], cy_m[1], lp_cy)))
	print("Status:", LpStatus[lp_cy.status])
	print(lp_cy.objective.value())
	# for v in lp_cy.variables():
	#     print(v.name, "=", v.varValue)
		