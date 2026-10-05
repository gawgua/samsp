from itertools import permutations

from dataset.problems import SAMSPProblem
import networkx as nx
from docplex.cp.model import CpoModel
from docplex.cp.modeler import (
    end_at_end,
    end_of,
    no_overlap,
    pulse,
    start_at_start,
    start_before_start,
    start_of,
    sum,
)


class SAMSPCPSolver:
    def __init__(self, problem: SAMSPProblem):
        self.problem = problem
        self.model = CpoModel(name="SAMSP_CP")
        self.solution = None
        self.tasks = {}
        self.occupancy = {}
        self.storage_events = []
        self.global_source = None
        self.global_sink = None
        self._model_built = False

        self.__preprocessing()

    def __preprocessing(self):
        """Pair storage gets and releases using the JSON activity IDs."""
        graph = self.problem.precedence_graph

        # Dataset tests are listed consecutively and finish with empty storage.
        groups = []
        indexes = []
        balance = [0] * self.problem.number_of_cumulative_resources
        for index in self.problem.activity_ids[1:-1]:
            indexes.append(index)
            for resource_index, delta in enumerate(self.problem.cumulative_requests[index]):
                balance[resource_index] += delta
            if not any(balance):
                groups.append(indexes)
                indexes = []
        if indexes:
            groups.append(indexes)

        for indexes in groups:
            for resource_index in range(self.problem.number_of_cumulative_resources):
                gets = [
                    index
                    for index in indexes
                    if self.problem.cumulative_requests[index][resource_index] > 0
                ]
                releases = [
                    index
                    for index in indexes
                    if self.problem.cumulative_requests[index][resource_index] < 0
                ]
                if len(gets) != len(releases):
                    raise ValueError(
                        f"Unmatched cumulative resource {resource_index + 1} "
                        f"events in activity group starting at {indexes[0]}"
                    )
                if not gets:
                    continue

                distances = {}
                for get in gets:
                    for release in releases:
                        try:
                            distances[get, release] = nx.shortest_path_length(
                                graph, get, release
                            )
                        except nx.NetworkXNoPath:
                            distances[get, release] = float("inf")

                pairing = min(
                    permutations(releases),
                    key=lambda candidate: sum(
                        distances[get, release]
                        for get, release in zip(gets, candidate)
                    ),
                )
                if any(
                    distances[get, release] == float("inf")
                    for get, release in zip(gets, pairing)
                ):
                    raise ValueError(
                        f"Could not match cumulative resource {resource_index + 1} "
                        f"events in activity group starting at {indexes[0]}"
                    )
                self.storage_events.extend(
                    (
                        get,
                        release,
                        resource_index + 1,
                    )
                    for get, release in zip(gets, pairing)
                )

    def build_model(self):
        if self._model_built:
            return self.model

        for index in range(self.problem.number_of_activities):
            self.tasks[index] = self.model.interval_var(
                size=self.problem.durations[index], name=f"Task_{index}"
            )

        source_index = self.problem.source_id
        sink_index = self.problem.sink_id
        self.global_source = self.tasks[source_index]
        self.global_sink = self.tasks[sink_index]
        self.model.add(start_of(self.global_source) == 0)

        for predecessor, successor, data in self.problem.precedence_graph.edges(data=True):
            self.model.add(
                start_before_start(
                    self.tasks[predecessor],
                    self.tasks[successor],
                    data.get("weight", 0),
                )
            )

        for resource_index in range(self.problem.number_of_renewable_resources):
            self.model.add(
                no_overlap(
                    self.tasks[index]
                    for index in self.tasks
                    if self.problem.renewable_requests[index][resource_index] > 0
                )
            )

        for start_index, end_index, resource_index in self.storage_events:
            occupancy = self.model.interval_var(
                size=(0, self._horizon()),
                name=f"Occupancy_{start_index}_{end_index}_{resource_index}",
            )
            self.occupancy[
                (start_index, end_index, resource_index)
            ] = occupancy
            self.model.add(start_at_start(occupancy, self.tasks[start_index]))
            self.model.add(end_at_end(occupancy, self.tasks[end_index]))

        for resource_index, capacity in enumerate(self.problem.cumulative_capacities, start=1):
            resource_usage = sum(
                pulse(occupancy, 1)
                for event, occupancy in self.occupancy.items()
                if event[2] == resource_index
            )
            self.model.add(resource_usage <= capacity)

        self.model.add(self.model.minimize(end_of(self.global_sink)))
        self._model_built = True
        return self.model

    def _horizon(self):
        return max(
            1,
            sum(max(0, duration) for duration in self.problem.durations)
            + sum(
                abs(data.get("weight", 0))
                for _, _, data in self.problem.precedence_graph.edges(data=True)
            ),
        )

    def solve(self):
        self.solution = self.model.solve(TimeLimit=120)

    def get_solution(self):
        return self.solution
