from dataset.problems import SAMSPProblem
from solver.temp import TEMP as SAMSPSolver

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":
    problem = SAMSPProblem.from_file(r"./test.sm.txt")
    samsp_solver = SAMSPSolver(problem)

    # build the known solution for the problem
    # get_S_at_t = lambda itt, na, t: samsp_solver.S[problem.node_id[(itt, na)], t]

    # samsp_solver.add_clause([get_S_at_t(1, 10, 52)])
    # samsp_solver.add_clause([get_S_at_t(2, 10, 62)])

    # samsp_solver.add_clause([get_S_at_t(1, 5, 4)])
    # samsp_solver.add_clause([get_S_at_t(1, 6, 9)])
    # samsp_solver.add_clause([get_S_at_t(1, 3, 14)])
    # samsp_solver.add_clause([get_S_at_t(1, 4, 20)])
    # samsp_solver.add_clause([get_S_at_t(2, 5, 25)])
    # samsp_solver.add_clause([get_S_at_t(1, 7, 30)])
    # samsp_solver.add_clause([get_S_at_t(2, 6, 34)])
    # samsp_solver.add_clause([get_S_at_t(2, 3, 39)])
    # samsp_solver.add_clause([get_S_at_t(2, 4, 45)])
    # samsp_solver.add_clause([get_S_at_t(2, 7, 55)])
    
    # samsp_solver.add_clause([get_S_at_t(1, 2, 1)])
    # samsp_solver.add_clause([get_S_at_t(1, 1, 11)])
    # samsp_solver.add_clause([get_S_at_t(2, 2, 18)])
    # samsp_solver.add_clause([get_S_at_t(2, 1, 36)])
    # samsp_solver.add_clause([get_S_at_t(1, 9, 49)])
    # samsp_solver.add_clause([get_S_at_t(1, 11, 56)])
    # samsp_solver.add_clause([get_S_at_t(2, 9, 59)])
    # samsp_solver.add_clause([get_S_at_t(1, 8, 62)])
    # samsp_solver.add_clause([get_S_at_t(2, 11, 66)])
    # samsp_solver.add_clause([get_S_at_t(2, 8, 90)])
    # samsp_solver.add_clause([get_S_at_t(1, 12, 93)])
    # samsp_solver.add_clause([get_S_at_t(2, 12, 96)])

    samsp_solver.encode()
    samsp_solver.solve()

    model = samsp_solver.get_last_feasible_model()
    print("SAT" if model is not None else "UNSAT")
    print(f"Makespan: {samsp_solver.get_makespan()}")
    print(f"Scheduling: {samsp_solver.get_scheduling()}")