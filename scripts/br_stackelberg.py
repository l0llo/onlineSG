
"""
Stackelberg Best Response
(in a 2 player, 2 targets SG)
This script finds with linear programming the maxmin solution to a 2 players-2
targets SG, which correspond to the best response to a Stackelberg adversary.
Analitical solution to this problem can be found here:
https://www.overleaf.com/read/xmcpdzwsmrty
"""
import sys
from gurobipy import *


"""
The approach used here is taken from "Introduction to Operative Research" 
by Hillier & Lieberman (section 14.5)
"""


def main():
    try:
        print(sys.argv)
        # Create a new model
        m = Model("SSG")

        # Create variables
        x1 = m.addVar(vtype=GRB.CONTINUOUS, name="x1")
        x2 = m.addVar(vtype=GRB.CONTINUOUS, name="x2")
        x3 = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x3")
        v_m = 5
        v_M = 6
        # Set objective
        m.setObjective(x3, GRB.MAXIMIZE)

        # Add constraint:
        m.addConstr(x1 + x2 == 1, "c0")
        m.addConstr(-v_m * x2 >= x3, "c1")
        m.addConstr(-v_M * x1 >= x3, "c2")

        m.optimize()

        for v in m.getVars():
            print(v.varName, v.x)

        print('Obj:', m.objVal)

    except GurobiError:
        print('Error reported')


    # We can compare these results with the analytical ones:

    # In[3]:

    analitycal_obj = -(v_M * v_m)/(v_m + v_M)
    analitycal_x1 = v_m / (v_m + v_M)
    analitycal_x2 = v_M / (v_m + v_M)
    print(analitycal_x1 ,analitycal_x2 , analitycal_obj)


def main2(values):
    values = [int(v) for v in values]
    m = Model("SSG")
    targets = list(range(len(values)))
    strategy = []
    for t in targets:
        strategy.append(m.addVar(vtype=GRB.CONTINUOUS, name="x"+str(t)))
    v = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")
    m.setObjective(v, GRB.MAXIMIZE)
    for t in targets:
        terms = [-values[t]*strategy[i] for i in targets if i != t]
        m.addConstr(sum(terms) - v >= 0, "c"+str(t))
    m.addConstr(sum(strategy) == 1, "c"+str(len(targets)))
    # m.params.outputflag = 0  # comment this line to have the details of the optimization
    m.optimize()
    for v in m.getVars():
        print(v.varName, v.x)

    print('Obj:', m.objVal)
    print([float(s.x) for s in strategy])

    m.optimize


if __name__ == '__main__':
    #main()
    main2(sys.argv[1:])
