from solver.solver import SATSolver, SOLVER_STATUS
from dataset.problems import SAMSPProblem

import networkx as nx

class TEMP:
    def __init__(self, problem: SAMSPProblem, LB=None, UB=None):
        self.__solver = SATSolver()
        self.__problem = problem
        self.__LB = LB
        self.__UB = UB

        self.__int_to_node = {v: k for k, v in self.__problem.node_id.items()}

        self.__preprocessing()

    def add_clause(self, clause):
        self.__solver.add_clause(clause)

    def encode(self):
        self.__first_activity()
        self.__activity_start_in_range()
        self.__precedence_constraints()
        self.__auxiliary_variables_constraints()
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
                [
                    -self.S[self.__problem.number_of_activities - 1, t]
                    for t in range(makespan + 1, self.__UB + 1)
                ]
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
        scheduling = self.get_scheduling()
        if scheduling is None:
            return None
        # Retrieve the scheduling time of the Super-Sink
        super_sink_tuple = self.__int_to_node[self.__problem.number_of_activities - 1]
        last_start_time = scheduling[super_sink_tuple]
        return last_start_time

    def get_scheduling(self):
        model = self.get_last_feasible_model()
        if not model:
            return None
        scheduling = {}
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i] + 1):
                if model[self.S[i, t] - 1] > 0:
                    # Look up the tuple key using O(1) dictionary instead of looping
                    tuple_key = self.__int_to_node[i]
                    scheduling[tuple_key] = t
                    break
        return scheduling

    def __preprocessing(self):
        # Build transitive closure, negate the weight, find shortest
        # negate again, get the longest path
        negated_graph = self.__problem.precedence_graph.copy()
        for _, _, data in negated_graph.edges(data=True):
            data["weight"] = -data["weight"]

        try:
            shortest_paths = nx.floyd_warshall(negated_graph, weight="weight")
        except nx.NetworkXError as e:
            print(f"Error computing shortest paths: {e}")
            raise e

        self.__extended_precedence_graph = nx.DiGraph()
        for u_tuple in shortest_paths:
            for v_tuple in shortest_paths[u_tuple]:
                if u_tuple != v_tuple and shortest_paths[u_tuple][v_tuple] != float(
                    "inf"
                ):
                    u_int = self.__problem.node_id[u_tuple]
                    v_int = self.__problem.node_id[v_tuple]
                    self.__extended_precedence_graph.add_edge(
                        u_int, v_int, weight=-shortest_paths[u_tuple][v_tuple]
                    )

        # Calculate LB, UB
        LB = self.__extended_precedence_graph[0][
            self.__problem.number_of_activities - 1
        ]["weight"]
        if self.__LB is None:
            self.__LB = LB

        UB = 0
        for i in range(self.__problem.number_of_activities):
            # Map index 'i' to tuple to fetch successors from original graph, then map back to int
            i_tuple = self.__int_to_node[i]
            successors_tuples = self.__problem.precedence_graph.successors(i_tuple)
            successors_int = [
                self.__problem.node_id[j_tuple] for j_tuple in successors_tuples
            ]

            UB += max(
                self.__problem.durations[i],
                *[
                    self.__extended_precedence_graph[i][j]["weight"]
                    for j in successors_int
                ]
                + [0],
            )

        if self.__UB is None or self.__UB > UB or self.__UB < self.__LB:
            self.__UB = UB

        print(f"LB: {self.__LB}, UB: {self.__UB}")

        # Preprocessing time
        self.__ES = [0]
        for i in range(1, self.__problem.number_of_activities):
            self.__ES.append(self.__extended_precedence_graph[0][i]["weight"])

        self.__LS = []
        for i in range(self.__problem.number_of_activities - 1):
            self.__LS.append(
                self.__UB
                - self.__extended_precedence_graph[i][
                    self.__problem.number_of_activities - 1
                ]["weight"]
            )
        self.__LS.append(self.__UB)

        self.__EC = [x + self.__problem.durations[i] for i, x in enumerate(self.__ES)]
        self.__LC = [x + self.__problem.durations[i] for i, x in enumerate(self.__LS)]
        print(
            f"ES: {self.__ES},\n LS: {self.__LS},\n EC: {self.__EC},\n LC: {self.__LC}"
        )

        # Auxiliary variables
        self.S = {}  # S(ai,t) starting time of activity i at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i] + 1):
                self.S[i, t] = self.__solver.create_new_variable()

        self.x = {}  # x(ai,t) activity i is running at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LC[i] + 1):
                self.x[i, t] = self.__solver.create_new_variable()

        self.y = {}  # y(ai,t) activity i is started before or at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i] + 1):
                self.y[i, t] = self.__solver.create_new_variable()

        self.register = {}  # R(Ai, s, t) activity i is started in time range [s, t]

        # Prefix sum of cumulative resources for resource r at time t.
        self.__PS = {}
        for r in range(self.__problem.number_of_cumulative_resources):
            usages = {}
            for i in range(self.__problem.number_of_activities):
                request = self.__problem.cumulative_requests[i][r]
                if request > 0:
                    usages[self.__LS[i]] = usages.get(self.__LS[i], 0) + request
                elif request < 0:
                    usages[self.__LC[i]] = usages.get(self.__LC[i], 0) + request

            running_total = 0
            for t in range(self.__UB + 1):
                running_total += usages.get(t, 0)
                self.__PS[r, t] = running_total
        # print(f"Prefix sums of cumulative resources: {self.__PS}")

    def __first_activity(self):
        # First activity is always started at time 0
        self.add_clause([self.S[0, 0]])

    def __activity_start_in_range(self):
        for i in range(1, self.__problem.number_of_activities):
                    self.__solver.add_clause(
                        [
                            self.__get_forward_staircase_register(
                                i, self.__ES[i], self.__LS[i] + 1
                            )
                        ]
                    )

    def __precedence_constraints(self):
        for predecessor in range(1, self.__problem.number_of_activities):
            for successor in self.__extended_precedence_graph.successors(predecessor):
                # Precedence constraint
                for k in range(self.__ES[successor], self.__LS[successor] + 1):
                    if (
                        k
                        - self.__extended_precedence_graph[predecessor][successor].get(
                            "weight"
                        )
                        + 1
                        >= self.__LS[predecessor] + 1
                    ):
                        continue
                    first_half = self.__get_forward_staircase_register(
                        successor, self.__ES[successor], k + 1
                    )
                    t = (
                        k
                        - self.__extended_precedence_graph[predecessor][successor].get(
                            "weight"
                        )
                        + 1
                    )
                    if t < self.__ES[predecessor]:
                        t = self.__ES[predecessor]
                    self.__solver.add_clause(
                        [-first_half, -self.S[predecessor, t]]
                    )
                for k in range(
                    self.__LS[successor]
                    - self.__extended_precedence_graph[predecessor][successor].get(
                        "weight"
                    )
                    + 2,
                    self.__LS[predecessor] + 1,
                ):
                    self.__solver.add_clause(
                        [
                            -self.__get_forward_staircase_register(
                                successor,
                                self.__ES[successor],
                                self.__LS[successor] + 1,
                            ),
                            -self.S[predecessor, k],
                        ]
                    )

    def __auxiliary_variables_constraints(self):
        # x(ai,t) = 1 if and only if S(ai,t') = 1 for some t' in [t - di + 1, t]
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i] + 1):
                self.add_clause(
                    [-self.x[i, t]]
                    + [
                        self.S[i, t_prime]
                        for t_prime in range(
                            max(self.__ES[i], t - self.__problem.durations[i] + 1),
                            min(t, self.__LS[i]) + 1,
                        )
                    ]
                )

                for t_prime in range(
                    max(self.__ES[i], t - self.__problem.durations[i] + 1),
                    min(t, self.__LS[i]) + 1,
                ):
                    self.add_clause([-self.S[i, t_prime], self.x[i, t]])

        # y(ai,t) = 1 if and only if S(ai,t') = 1 for some t' in [ESi, t]
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i] + 1):
                self.add_clause(
                    [-self.y[i, t]]
                    + [self.S[i, t_prime] for t_prime in range(self.__ES[i], t + 1)]
                )
                for t_prime in range(self.__ES[i], t + 1):
                    self.add_clause([-self.S[i, t_prime], self.y[i, t]])

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
                limit = self.__problem.cumulative_capacities[r] - self.__PS[r, t]

                for i in range(self.__problem.number_of_activities):
                    c = self.__problem.cumulative_requests[i][r]
                    if c > 0:
                        if self.__ES[i] <= t < self.__LS[i]:
                            lits.append(self.y[i, t])
                            weights.append(c)
                    elif c < 0:
                        if self.__EC[i] <= t < self.__LC[i]:
                            w = -c
                            t_finish = t - self.__problem.durations[i]

                            lits.append(-self.y[i, t_finish])
                            weights.append(w)
                            limit += w

                if limit < 0:
                    raise ValueError(
                        f"Negative limit for cumulative resource {r} at time {t}: {limit}"
                    )

                if lits:
                    self.__solver.add_at_most_k(lits, weights, limit)

    def __get_forward_staircase_register(
        self, job: int, start: int, end: int
    ) -> int | None:
        """Get forward staircase register for a job for a range of time [start, end)"""
        # Check if current job with provided start and end time is already in the register
        if start >= end:
            return None
        temp = tuple(self.S[(job, s)] for s in range(start, end))
        if temp in self.register:
            return self.register[temp]
        # Store the start time of the job
        accumulative = []
        for s in range(start, end):
            accumulative.append(self.S[(job, s)])
            current_tuple = tuple(accumulative)
            # If the current tuple is not in the register, create a new variable
            if current_tuple not in self.register:
                # Create a new variable for the current tuple
                self.register[current_tuple] = self.__solver.create_new_variable()
                # Create constraint for staircase
                # If current tuple is true then the register associated with it must be true
                self.__solver.add_clause(
                    [-self.S[(job, s)], self.register[current_tuple]]
                )
                if s == start:
                    self.__solver.add_clause(
                        [self.S[(job, s)], -self.register[current_tuple]]
                    )
                else:
                    # Get the previous tuple
                    previous_tuple = tuple(accumulative[:-1])
                    # If previous tuple is true then the current register must be true
                    self.__solver.add_clause(
                        [
                            -self.register[previous_tuple],
                            self.register[current_tuple],
                        ]
                    )
                    # Both previous tuple and current variable is false then current tuple must be false
                    self.__solver.add_clause(
                        [
                            self.register[previous_tuple],
                            self.S[(job, s)],
                            -self.register[current_tuple],
                        ]
                    )
                    # Previous tuple and current variable must not be true at the same time
                    self.__solver.add_clause(
                        [-self.register[previous_tuple], -self.S[(job, s)]]
                    )
        return self.register[temp]
