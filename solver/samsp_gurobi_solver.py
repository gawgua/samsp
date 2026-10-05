from gurobipy import GRB, Model, quicksum

from dataset.problems import SAMSPProblem


class SAMSPGurobiSolver:
    """Gurobi translation of SAMSPMILPSolver's time-indexed model."""

    def __init__(self, problem: SAMSPProblem):
        self.problem = problem
        self.model = Model(name="SAMSP_GUROBI")
        self._model_built = False

        self.x = self.model.addVars(
            [
                (i, t)
                for i in range(self.problem.number_of_activities)
                for t in range(self.problem.ES[i], self.problem.LC[i] + 1)
            ],
            vtype=GRB.BINARY,
            name="x",
        )

    def build_model(self):
        if self._model_built:
            return self.model

        # Every activity must be scheduled exactly once.
        for i in range(self.problem.number_of_activities):
            x_i = [self.x[(i, t)] for t in range(self.problem.ES[i], self.problem.LS[i] + 1)]
            self.model.addConstr(quicksum(x_i) == 1)

        # Precedence constraints.
        for u, v, data in self.problem.extended_precedence_graph.edges(data=True):
            lag = data.get("weight", 0)
            tx_u = [t * self.x[(u, t)] for t in range(self.problem.ES[u], self.problem.LS[u] + 1)]
            tx_v = [t * self.x[(v, t)] for t in range(self.problem.ES[v], self.problem.LS[v] + 1)]
            self.model.addConstr(quicksum(tx_v) - quicksum(tx_u) >= lag)

        # Renewable resource constraints.
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
                    self.model.addConstr(quicksum(activities_at_t) <= B_r)

        # Cumulative resource constraints.
        for r in range(self.problem.number_of_cumulative_resources):
            for t in range(0, self.problem.UB + 1):
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
                    self.model.addConstr(
                        self.problem.PS[r, t] + quicksum(storage_occupy) + quicksum(storage_release)
                        <= self.problem.cumulative_capacities[r]
                    )

        self.model.setObjective(
            quicksum(
                t * self.x[(self.problem.number_of_activities - 1, t)]
                for t in range(
                    self.problem.ES[self.problem.number_of_activities - 1],
                    self.problem.LS[self.problem.number_of_activities - 1] + 1,
                )
            ),
            GRB.MINIMIZE,
        )

        self._model_built = True
        return self.model

    def solve(self):
        self.model.optimize()

    def get_solution(self):
        if self.model.Status == GRB.OPTIMAL:
            return {variable.VarName: variable.X for variable in self.model.getVars()}
        return None

    def get_scheduling(self):
        if self.model.SolCount == 0:
            return None
        return {
            i: next(
                t
                for t in range(self.problem.ES[i], self.problem.LS[i] + 1)
                if self.x[i, t].X > 0.5
            )
            for i in range(self.problem.number_of_activities)
        }
