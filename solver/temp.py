from solver.solver import SATSolver, SOLVER_STATUS
from dataset.problems import SAMSPProblem

import networkx as nx

class TEMP:
    def __init__(self, problem: SAMSPProblem, LB=None, UB=None):
        self.__solver = SATSolver()
        self.__problem = problem
        self.__LB = LB
        self.__UB = UB

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
            self.__solver.add_assumption([-self.S[self.__problem.number_of_activities - 1, t] for t in range(makespan + 1, self.__UB + 1)])

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
        last_start_time = scheduling[0, 1]
        return last_start_time

    def get_scheduling(self):
        model = self.get_last_feasible_model()
        if not model:
            return None
        scheduling = {}
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i] + 1):
                if model[self.S[i, t] - 1] > 0:
                    for (session, activity), node in self.__problem.node_id.items():
                        if node == i:
                            scheduling[(session, activity)] = t
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
        for u in shortest_paths:
            for v in shortest_paths[u]:
                if u != v and shortest_paths[u][v] != float("inf"):
                    self.__extended_precedence_graph.add_edge(
                        u, v, weight=-shortest_paths[u][v]
                    )
        # print("Extended precedence graph edges with weights:")
        # for u, v, data in self.__extended_precedence_graph.edges(data=True):
        #     print(f"Edge from {u} to {v} with weight {data['weight']}")

        # Calculate LB, UB
        LB = self.__extended_precedence_graph[0][
            self.__problem.number_of_activities - 1
        ]["weight"]
        if self.__LB is None:
            self.__LB = LB

        UB = 0
        for i in range(self.__problem.number_of_activities):
            # max between duration of i or  max time lag of (i,j) for all j in successors of i
            UB += max(
                self.__problem.durations[i],
                *[
                    self.__extended_precedence_graph[i][j]["weight"]
                    for j in self.__problem.precedence_graph.successors(i)
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
        print(f"Prefix sums of cumulative resources: {self.__PS}")

    def __first_activity(self):
        # First activity is always started at time 0
        self.add_clause([self.S[0, 0]])
        # print("Added clause: S(0, 0) is True")

    def __activity_start_in_range(self):
        # sequential counter for activity start time in range [ES, LS]
        for i in range(self.__problem.number_of_activities):
            # AMO (sequential counter)
            if (i, self.__ES[i], self.__ES[i]) not in self.register:
                self.register[i, self.__ES[i], self.__ES[i]] = (
                    self.__solver.create_new_variable()
                )

            # Si,ESi => R(Ai, ESi, ESi)
            self.add_clause(
                [-self.S[i, self.__ES[i]], self.register[i, self.__ES[i], self.__ES[i]]]
            )

            # Si,t ^ R(Ai, ESi, t-1) => R(Ai, ESi, t), R(Ai, ESi, t-1) => ~Si,t
            for t in range(self.__ES[i] + 1, self.__LS[i] + 1):
                if (i, self.__ES[i], t) not in self.register:
                    self.register[i, self.__ES[i], t] = (
                        self.__solver.create_new_variable()
                    )
                self.add_clause(
                    [-self.S[i, t], self.register[i, self.__ES[i], t]]
                )
                self.add_clause(
                    [
                        -self.register[i, self.__ES[i], t - 1],
                        self.register[i, self.__ES[i], t],
                    ]
                )

                self.add_clause(
                    [-self.register[i, self.__ES[i], t - 1], -self.S[i, t]]
                )

            # ALO (Big OR)
            self.add_clause(
                [self.S[i, t] for t in range(self.__ES[i], self.__LS[i] + 1)]
            )

            # print(
            #     f"Added clauses for activity {i} to start in range [{self.__ES[i]}, {self.__LS[i]}]"
            # )

    def __precedence_constraints(self):
        # Sj - Si >= li,j for all (i, j, l) in graph
        for i, j in self.__extended_precedence_graph.edges():
            l = self.__extended_precedence_graph[i][j]["weight"]
            # print(f"Adding precedence constraint: S({j}) - S({i}) >= {l}")

            for t in range(self.__ES[j], self.__LS[j] + 1):
                if t - l < self.__ES[i]:
                    # then j cannot start as i cannot start
                    self.add_clause([-self.y[j, t]])
                elif t - l > self.__LS[i]:
                    # contraint alway true
                    continue
                else:
                    # if y(j, t) then y(i, t - l)
                    self.add_clause([-self.y[j, t], self.y[i, t - l]])

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
                # print(
                #     f"Added auxiliary variable constraints for x({i},{t}) = {self.x[i, t]}"
                # )

        # y(ai,t) = 1 if and only if S(ai,t') = 1 for some t' in [ESi, t]
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i], self.__LS[i] + 1):
                self.add_clause(
                    [-self.y[i, t]]
                    + [self.S[i, t_prime] for t_prime in range(self.__ES[i], t + 1)]
                )
                for t_prime in range(self.__ES[i], t + 1):
                    self.add_clause([-self.S[i, t_prime], self.y[i, t]])
                # print(
                #     f"Added auxiliary variable constraints for y({i},{t}) = {self.y[i, t]}"
                # )

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

                        self.add_clause(
                            [-R[k - 1], -activities_using_r_at_t[k]]
                        )

                    self.add_clause(
                        [
                            -R[len(activities_using_r_at_t) - 2],
                            -activities_using_r_at_t[-1],
                        ]
                    )
            # print(f"Added renewable resource constraints for resource {r}")

    def __cummulative_resources_constraints(self):
        # usage at t of r <= capacity of r - ps of r
        for r in range(self.__problem.number_of_cumulative_resources):
            # print(f"Adding cumulative resource constraints for resource {r}")

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
                            # !y(i, t - di) = True -> Ai still occupied
                            # weight = -c,
                            # if Ai not occupied at t, then limit += -c to increase the limit
                            # if Ai occupied at t, then Ai still take c resource at t, increase the left of inquation by c, balance the added limit
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
