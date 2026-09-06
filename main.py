from solver.temp import TEMP as SAMSPSATSolver
# from solver.samsp_sat_solver import SAMSPSATSolver
from solver.samsp_cp_solver import SAMSPCPSolver
from solver.samsp_milp_solver import SAMSPMILPSolver
from dataset.problems import SAMSPProblem

import time

if __name__ == "__main__":
    problem = SAMSPProblem.from_file(r"./test.sm.txt")
    samsp_sat_solver = SAMSPSATSolver(problem)
    samsp_cp_solver = SAMSPCPSolver(problem)
    samsp_milp_solver = SAMSPMILPSolver(problem)

    print("Solving with SAT Solver...")
    # Encode and Solve
    start = time.time()
    print("Encoding the problem...")
    samsp_sat_solver.encode()
    end = time.time()
    print(f"Encoding time: {end - start:.2f} seconds")

    
    print("Solving...")
    start = time.time()
    samsp_sat_solver.solve()
    end = time.time()
    print(f"Solving time: {end - start:.2f} seconds")
    
    model = samsp_sat_solver.get_last_feasible_model()
    print("SAT" if model is not None else "UNSAT")
    
    if model:
        makespan = samsp_sat_solver.get_makespan()
        scheduling = samsp_sat_solver.get_scheduling()
        
        print(f"Makespan: {makespan}")
        print(f"Scheduling: {scheduling}")
        print(f"Stats: {samsp_sat_solver.get_statistics()}")

    print("\nSolving with CP Solver...")
    samsp_cp_solver.build_model()
    samsp_cp_solver.solve()


    print("\nSolving with MILP Solver...")
    samsp_milp_solver.build_model()
    samsp_milp_solver.solve()
