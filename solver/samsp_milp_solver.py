from docplex.mp.model import Model


class SAMSPMILPSolver:
    def __init__(self, problem):
        self.problem = problem
        self.model = Model(name="SAMSP_MILP")
        self.solution = None
        self._model_built = False

        self.__preprocessing()

    def __preprocessing(self):
        self.x = self.model.binary_var_dict(
            keys=[
                (i, t)
                for i in range(self.problem.number_of_activities)
                for t in range(self.problem.ES[i], self.problem.LC[i] + 1)
            ],
            name="x",
        )

    def build_model(self):
        if self._model_built:
            return self.model

        # every activity must be scheduled exactly once
        for i in range(self.problem.number_of_activities):
            x_i = [self.x[(i, t)] for t in range(self.problem.ES[i], self.problem.LS[i] + 1)]
            self.model.add_constraint(self.model.sum(x_i) == 1)

        # Precedence constraints
        for u, v, data in self.problem.extended_precedence_graph.edges(data=True):
            lag = data.get("weight", 0)
            tx_u = [t * self.x[(u, t)] for t in range(self.problem.ES[u], self.problem.LS[u] + 1)]
            tx_v = [t * self.x[(v, t)] for t in range(self.problem.ES[v], self.problem.LS[v] + 1)]
            self.model.add_constraint(
                self.model.sum(tx_v) - self.model.sum(tx_u) >= lag
            )

        # Renewable resource constraints
        for r in range(self.problem.number_of_renewable_resources):
            B_r = 1
            for t in range(0, self.problem.UB + 1):
                activities_at_t = []
                for i in range(self.problem.number_of_activities):
                    if self.problem.ES[i] <= t < self.problem.LC[i] and self.problem.renewable_requests[i][r] > 0:
                        b_ir = self.problem.renewable_requests[i][r]
                        p_i = self.problem.durations[i]
                        for z in range(max(self.problem.ES[i], t - p_i + 1), t + 1):
                            activities_at_t.append(b_ir * self.x[(i, z)])

                if activities_at_t:
                    self.model.add_constraint(self.model.sum(activities_at_t) <= B_r)
                        

        # Cumulative resource constraints
        for r in range(self.problem.number_of_cumulative_resources):
            for t in range(self.problem.LB, self.problem.UB + 1):
                storage_occupy = []
                storage_release = []
                for i in range(self.problem.number_of_activities):
                    c_ir = self.problem.cumulative_requests[i][r]
                    if c_ir > 0 and self.problem.ES[i] <= t < self.problem.LS[i]:
                        for z in range(self.problem.ES[i], t + 1):
                            storage_occupy.append(c_ir * self.x[(i, z)])
                    elif c_ir < 0 and self.problem.EC[i] <= t < self.problem.LC[i]:
                        for z in range(self.problem.EC[i], t + 1):
                            storage_release.append(c_ir * self.x[(i, z - self.problem.durations[i])])

                if storage_occupy or storage_release:
                    self.model.add_constraint(
                        self.problem.PS[r, t] + self.model.sum(storage_occupy) + self.model.sum(storage_release) <= self.problem.cumulative_capacities[r]
                    )

        self.model.minimize(
            self.model.sum(
                t * self.x[(self.problem.number_of_activities - 1, t)]
                for t in range(
                    self.problem.ES[self.problem.number_of_activities - 1],
                    self.problem.LS[self.problem.number_of_activities - 1] + 1,
                )
            )
        )

        self._model_built = True

    def solve(self):
        self.solution = self.model.solve(log_output=True)

    def get_solution(self):
        return self.solution
