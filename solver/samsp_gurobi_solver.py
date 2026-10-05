from gurobipy import GRB, Model, quicksum

from dataset.problems import SAMSPProblem


class SAMSPGurobiSolver:
    """Compact time-indexed Gurobi model for SAMSP."""

    def __init__(self, problem: SAMSPProblem):
        self.problem = problem
        self.model = Model(name="SAMSP_model")
        self._model_built = False

        self.x = self.model.addVars(
            [
                (activity_id, start_time)
                for activity_id in self.problem.activity_ids
                for start_time in range(
                    self.problem.ES[activity_id],
                    self.problem.LS[activity_id] + 1,
                )
            ],
            vtype=GRB.BINARY,
            name="x",
        )
        self._storage_activities = [
            activity_id
            for activity_id in self.problem.activity_ids
            if any(self.problem.cumulative_requests[activity_id])
        ]
        self.y = self.model.addVars(
            [
                (activity_id, time_point)
                for activity_id in self._storage_activities
                for time_point in range(
                    self.problem.ES[activity_id],
                    self.problem.LS[activity_id],
                )
            ],
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS,
            name="started",
        )
        self.start_time = {
            activity_id: quicksum(
                start_time * self.x[activity_id, start_time]
                for start_time in range(
                    self.problem.ES[activity_id],
                    self.problem.LS[activity_id] + 1,
                )
            )
            for activity_id in self.problem.activity_ids
        }

    def build_model(self):
        if self._model_built:
            return self.model

        for activity_id in self.problem.activity_ids:
            self.model.addConstr(
                quicksum(
                    self.x[activity_id, start_time]
                    for start_time in range(
                        self.problem.ES[activity_id],
                        self.problem.LS[activity_id] + 1,
                    )
                )
                == 1,
                name=f"activity_{activity_id}_scheduled_once",
            )

        self.model.addConstr(
            self.start_time[self.problem.source_id] == 0,
            name="source_starts_at_zero",
        )
        self._add_started_by_constraints()

        for predecessor, successor, lag in self.problem.precedence_graph.edges(
            data="weight", default=0
        ):
            self.model.addConstr(
                self.start_time[successor] - self.start_time[predecessor] >= lag,
                name=f"lag_{predecessor}_{successor}",
            )

        for resource_id, capacity in enumerate(
            self.problem.renewable_capacities
        ):
            for time_point in range(self.problem.UB):
                active_terms = []
                for activity_id in self.problem.activity_ids:
                    demand = self.problem.renewable_requests[activity_id][resource_id]
                    if demand <= 0:
                        continue
                    duration = self.problem.durations[activity_id]
                    for start_time in range(
                        max(
                            self.problem.ES[activity_id],
                            time_point - duration + 1,
                        ),
                        min(time_point, self.problem.LS[activity_id]) + 1,
                    ):
                        active_terms.append(
                            demand * self.x[activity_id, start_time]
                        )
                if active_terms:
                    self.model.addConstr(
                        quicksum(active_terms) <= capacity,
                        name=f"renewable_{resource_id}_{time_point}",
                    )

        for storage_id, capacity in enumerate(
            self.problem.cumulative_capacities
        ):
            for time_point in range(self.problem.UB + 1):
                occupancy_terms = []
                for activity_id in self._storage_activities:
                    delta = self.problem.cumulative_requests[activity_id][storage_id]
                    if delta > 0:
                        occupancy_terms.append(
                            delta * self._started_by(activity_id, time_point)
                        )
                    elif delta < 0:
                        occupancy_terms.append(
                            delta
                            * self._started_by(
                                activity_id,
                                time_point - self.problem.durations[activity_id],
                            )
                        )
                if occupancy_terms:
                    occupancy = quicksum(occupancy_terms)
                    self.model.addConstr(
                        occupancy >= 0,
                        name=f"storage_lower_{storage_id}_{time_point}",
                    )
                    self.model.addConstr(
                        occupancy <= capacity,
                        name=f"storage_upper_{storage_id}_{time_point}",
                    )

        self.model.setObjective(
            self.start_time[self.problem.sink_id], GRB.MINIMIZE
        )
        self._model_built = True
        return self.model

    def _add_started_by_constraints(self):
        for activity_id in self._storage_activities:
            earliest = self.problem.ES[activity_id]
            latest = self.problem.LS[activity_id]
            if earliest >= latest:
                continue
            self.model.addConstr(
                self.y[activity_id, earliest]
                == self.x[activity_id, earliest],
                name=f"started_first_{activity_id}",
            )
            for time_point in range(earliest + 1, latest):
                self.model.addConstr(
                    self.y[activity_id, time_point]
                    == self.y[activity_id, time_point - 1]
                    + self.x[activity_id, time_point],
                    name=f"started_{activity_id}_{time_point}",
                )

    def _started_by(self, activity_id, time_point):
        if time_point < self.problem.ES[activity_id]:
            return 0
        if time_point >= self.problem.LS[activity_id]:
            return 1
        return self.y[activity_id, time_point]

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
            activity_id: next(
                start_time
                for start_time in range(
                    self.problem.ES[activity_id],
                    self.problem.LS[activity_id] + 1,
                )
                if self.x[activity_id, start_time].X > 0.5
            )
            for activity_id in self.problem.activity_ids
        }
