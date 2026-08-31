from solver.solver import SATSolver
from dataset.problems import SAMSPProblem


class TEMP:
    def __init__(self):
        self.__solver = SATSolver()
        self.__problem = SAMSPProblem()

    def solve(self):
        self.__preprocessing()

        self.__first_activity()
        self.__activity_start_in_range()
        self.__precedence_constraints()

        return self.__solver.solve()

    def __preprocessing(self):
        # Preprocessing time
        self.__ES = []
        self.__LS = []
        self.__EC = []
        self.__LC = []

        # Calculate LB, UB
        self.__LB = 0
        self.__UB = 0

        self.S = {}  # S(ai,t) starting time of activity i at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i]):
                self.S[i, t] = self.__solver.create_new_variable()

        self.x = {}  # x(ai,t) activity i is running at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LC[i]):
                self.x[i, t] = self.__solver.create_new_variable()

        self.y = {}  # y(ai,t) activity i is started before or at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i]):
                self.y[i, t] = self.__solver.create_new_variable()

        self.register = {}  # R(Ai, s, t) activity i is started in time range [s, t]

    def __first_activity(self):
        # First activity is always started at time 0
        self.__solver.add_clause([self.S[0, 0]])

    def __activity_start_in_range(self):
        # sequential counter for activity start time in range [ES, LS]
        for i in range(self.__problem.number_of_activities):
            # AMO (sequential counter)
            if (i, self.__ES[i], self.__ES[i]) not in self.register:
                self.register[i, self.__ES[i], self.__ES[i]] = (
                    self.__solver.create_new_variable()
                )
            self.__solver.add_clause(
                [-self.S[i, self.__ES[i]], self.register[i, self.__ES[i], self.__ES[i]]]
            )

            for t in range(self.__ES[i] + 1, self.__LS[i] + 1):
                if (i, self.__ES[i], t) not in self.register:
                    self.register[i, self.__ES[i], t] = (
                        self.__solver.create_new_variable()
                    )
                self.__solver.add_clause(
                    [-self.S[i, t], self.register[i, self.__ES[i], t]]
                )
                self.__solver.add_clause(
                    [
                        -self.register[i, self.__ES[i], t - 1],
                        self.register[i, self.__ES[i], t],
                    ]
                )

                self.__solver.add_clause(
                    [-self.register[i, self.__ES[i], t - 1], -self.S[i, t]]
                )

            # ALO (Big OR)
            self.__solver.add_clause(
                [self.S[i, t] for t in range(self.__ES[i], self.__LS[i] + 1)]
            )

    def __precedence_constraints(self):
        # Sj - Si >= li,j for all (i, j, l) in graph
        for i, j in self.__problem.precedence_graph.edges():
            pass

    def __consistency_constraints(self):
        # x(ai,t) = 1 if Ai is running at time t ie xi,t <-> Si in [t-di+1, t]
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LC[i]):
                # x(ai,t) -> Si in [t-di+1, t]
                s_in_valid_range = [
                    self.S[i, curr_t]
                    for curr_t in range(
                        max(self.__ES[i], t - self.__problem.durations[i] + 1),
                        min(t, self.__LS[i]) + 1,
                    )
                ]
                self.__solver.add_clause([-self.x[i, t], *s_in_valid_range])

                # S(ai,t') -> x(ai,t) for t' in [t-di+1, t]
                for curr_t in range(
                    max(self.__ES[i], t - self.__problem.durations[i] + 1),
                    min(t, self.__LS[i]) + 1,
                ):
                    self.__solver.add_clause([-self.S[i, curr_t], self.x[i, t]])

        # y(ai,t) = 1 if and only if S(ai,t') = 1 for some t' in [ESi, t]
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i]):
                s_in_valid_range = [
                    self.S[i, curr_t] for curr_t in range(self.__ES[i], t + 1)
                ]
                self.__solver.add_clause([-self.y[i, t], *s_in_valid_range])
                
                for t_prime in range(self.__ES[i], t + 1):
                    self.__solver.add_clause([-self.S[i, t_prime], self.y[i, t]])

    def __renewable_resources_constraints(self):
        for r in range(self.__problem.number_of_renewable_resources):
            for t in range(self.__UB):
                activities_using_r_at_t = []
                for i in range(self.__problem.number_of_activities):
                    if (
                        self.__problem.renewable_requests[i][r] > 0
                        and self.__ES[i] <= t < self.__LC[i]
                    ):
                        activities_using_r_at_t.append(self.x[i, t])

                if len(activities_using_r_at_t) > 1:
                    # AMO for activities using resource r at time t
                    R = {}
                    R[0] = self.__solver.create_new_variable()
                    self.__solver.add_clause([-activities_using_r_at_t[0], R[0]])
                    for k in range(1, len(activities_using_r_at_t) - 1):
                        R[k] = self.__solver.create_new_variable()
                        self.__solver.add_clause([-activities_using_r_at_t[k], R[k]])
                        self.__solver.add_clause([-R[k - 1], R[k]])

                        self.__solver.add_clause(
                            [-R[k - 1], -activities_using_r_at_t[k]]
                        )

                    self.__solver.add_clause(
                        [
                            -R[len(activities_using_r_at_t) - 2],
                            -activities_using_r_at_t[-1],
                        ]
                    )
