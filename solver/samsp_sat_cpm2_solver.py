from solver.solver import SATSolver, SOLVER_STATUS
from dataset.problems import SAMSPProblem

class SAMSPSAT_CPM2_Solver:
    def __init__(self, problem: SAMSPProblem, LB=None, UB=None):
        self.__solver = SATSolver()
        self.__problem = problem
        self.__LB = LB if LB is not None and LB > problem.LB else problem.LB
        self.__UB = UB if UB is not None and UB < problem.UB else problem.UB

        self.__preprocessing()

    def add_clause(self, clause):
        self.__solver.add_clause(clause)

    def encode(self):
        self.__consistency_clause()
        self.__start_clauses()
        self.__precedence_constraints()
        self.__renewable_resources_constraints()
        self.__cummulative_resources_constraints()

    def solve(self):
        low = self.__LB
        high = self.__UB
        best = None

        while low <= high:
            makespan = (low + high) // 2
            print(f"Trying makespan <= {makespan}")
            self.__solver.clear_assumptions()
            self.__solver.add_assumption(
                self.L[self.__problem.sink_id, makespan]
            )

            result = self.__solver.solve()
            if result is None:
                print("Solver returned UNKNOWN")
                return SOLVER_STATUS.UNKNOWN
            if result:
                best = makespan
                high = makespan - 1
            else:
                low = makespan + 1

        if best is None:
            return SOLVER_STATUS.UNSATISFIABLE

        print(f"Optimal feasible makespan: {best}")
        return SOLVER_STATUS.OPTIMAL

    def get_statistics(self):
        return self.__solver.get_statistics()

    def get_last_feasible_model(self):
        return self.__solver.get_last_feasible_model()

    def get_makespan(self):
        model = self.get_last_feasible_model()
        if model is None:
            return None
        sink = self.__problem.sink_id
        for t in range(self.__problem.ES[sink], self.__problem.UB + 1):
            if model[self.L[sink, t] - 1] > 0:
                return t
        return None

    def __preprocessing(self):
        # Auxiliary variables
        self.u = {}  # u(ai,t) activity i is running at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__problem.ES[i], self.__problem.LC[i] + 1):
                self.u[i, t] = self.__solver.create_new_variable()
        self.L = {}  # L(ai,t) activity i is started before or at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__problem.ES[i], self.__problem.UB + 1):
                self.L[i, t] = self.__solver.create_new_variable()
        self.register = {}  # R(Ai, s, t) activity i is started in time range [s, t]

    def __consistency_clause(self):
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__problem.ES[i], self.__problem.UB):
                self.__solver.add_clause([-self.L[i, t], self.L[i, t + 1]])

        for i in range(self.__problem.number_of_activities):
            for t in range(self.__problem.ES[i], self.__problem.LC[i]):
                self.__solver.add_clause([-self.u[i, t], self.L[i, t]])
                previous_time = t - self.__problem.durations[i]
                if previous_time < self.__problem.ES[i]:
                    # Before ES the activity has not started, so L is false.
                    self.__solver.add_clause([self.u[i, t], -self.L[i, t]])
                else:
                    self.__solver.add_clause([-self.u[i, t], -self.L[i, previous_time]])
                    self.__solver.add_clause([self.u[i, t], -self.L[i, t], self.L[i, previous_time]])
                  
    def __start_clauses(self):
        for i in range(self.__problem.number_of_activities):
            self.__solver.add_clause([self.L[i, self.__problem.LS[i]]])

    def __precedence_constraints(self):
        for predecessor, successor, data in self.__problem.precedence_graph.edges(data=True):
            weight = data.get("weight", 0)
            for t in range(self.__problem.ES[successor], self.__problem.LS[successor] + 1):
                if t - weight in range(self.__problem.ES[predecessor], self.__problem.LS[predecessor] + 1):
                    self.__solver.add_clause(
                        [-self.L[successor, t], self.L[predecessor, t - weight]]
                    )

    def __renewable_resources_constraints(self):
        for r in range(self.__problem.number_of_renewable_resources):
            for t in range(self.__UB):
                activities_using_r_at_t = []
                for i in range(self.__problem.number_of_activities):
                    if (
                        self.__problem.renewable_requests[i][r] > 0
                        and self.__problem.ES[i] <= t < self.__problem.LC[i]
                    ):
                        activities_using_r_at_t.append(self.u[i, t])

                if len(activities_using_r_at_t) > 1:
                    # AMO for activities using resource r at time t
                    R = {}
                    R[0] = self.__solver.create_new_variable()
                    self.add_clause([-activities_using_r_at_t[0], R[0]])
                    for k in range(1, len(activities_using_r_at_t) - 1):
                        R[k] = self.__solver.create_new_variable()
                        self.add_clause([-activities_using_r_at_t[k], R[k]])
                        self.add_clause([-R[k - 1], R[k]])

                        self.add_clause([-R[k - 1], -activities_using_r_at_t[k]])

                    self.add_clause(
                        [
                            -R[len(activities_using_r_at_t) - 2],
                            -activities_using_r_at_t[-1],
                        ]
                    )

    def __cummulative_resources_constraints(self):
        # usage at t of r <= capacity of r - ps of r
        for r in range(self.__problem.number_of_cumulative_resources):
            for t in range(self.__UB + 1):
                lits = []
                weights = []
                limit = self.__problem.cumulative_capacities[r] - self.__problem.PS[r, t]

                for i in range(self.__problem.number_of_activities):
                    c = self.__problem.cumulative_requests[i][r]
                    if c > 0:
                        if self.__problem.ES[i] <= t < self.__problem.LS[i]:
                            lits.append(self.L[i, t])
                            weights.append(c)
                    elif c < 0:
                        if self.__problem.EC[i] <= t < self.__problem.LC[i]:
                            w = -c
                            t_finish = t - self.__problem.durations[i]

                            lits.append(-self.L[i, t_finish])
                            weights.append(w)
                            limit += w

                if limit < 0:
                    raise ValueError(
                        f"Negative limit for cumulative resource {r} at time {t}: {limit}"
                    )

                if lits:
                    self.__solver.add_at_most_k(lits, weights, limit)

    # def __get_forward_staircase_register(
    #     self, job: int, start: int, end: int
    # ) -> int | None:
    #     """Get forward staircase register for a job for a range of time [start, end)"""
    #     # Check if current job with provided start and end time is already in the register
    #     if start >= end:
    #         return None
    #     temp = tuple(self.S[(job, s)] for s in range(start, end))
    #     if temp in self.register:
    #         return self.register[temp]
    #     # Store the start time of the job
    #     accumulative = []
    #     for s in range(start, end):
    #         accumulative.append(self.S[(job, s)])
    #         current_tuple = tuple(accumulative)
    #         # If the current tuple is not in the register, create a new variable
    #         if current_tuple not in self.register:
    #             # Create a new variable for the current tuple
    #             self.register[current_tuple] = self.__solver.create_new_variable()
    #             # Create constraint for staircase
    #             # If current tuple is true then the register associated with it must be true
    #             self.__solver.add_clause(
    #                 [-self.S[(job, s)], self.register[current_tuple]]
    #             )
    #             if s == start:
    #                 self.__solver.add_clause(
    #                     [self.S[(job, s)], -self.register[current_tuple]]
    #                 )
    #             else:
    #                 # Get the previous tuple
    #                 previous_tuple = tuple(accumulative[:-1])
    #                 # If previous tuple is true then the current register must be true
    #                 self.__solver.add_clause(
    #                     [
    #                         -self.register[previous_tuple],
    #                         self.register[current_tuple],
    #                     ]
    #                 )
    #                 # Both previous tuple and current variable is false then current tuple must be false
    #                 self.__solver.add_clause(
    #                     [
    #                         self.register[previous_tuple],
    #                         self.S[(job, s)],
    #                         -self.register[current_tuple],
    #                     ]
    #                 )
    #                 # Previous tuple and current variable must not be true at the same time
    #                 self.__solver.add_clause(
    #                     [-self.register[previous_tuple], -self.S[(job, s)]]
    #                 )
    #     return self.register[temp]
