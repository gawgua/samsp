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
        """Pair cumulative-resource gets and releases for each session."""
        graph = self.problem.precedence_graph

        nodes_by_session = {}
        for index in range(self.problem.number_of_activities):
            node = self.problem.int_to_node(index)
            test_type, session, _ = node
            if session:
                nodes_by_session.setdefault((test_type, session), []).append(index)

        for (test_type, session), indexes in nodes_by_session.items():
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
                        f"events in session {session}"
                    )
                if not gets:
                    continue

                distances = {}
                for get in gets:
                    for release in releases:
                        try:
                            distances[get, release] = nx.shortest_path_length(
                                graph, self.problem.int_to_node(get), self.problem.int_to_node(release)
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
                        f"events in session {session}"
                    )
                self.storage_events.extend(
                    (
                        test_type,
                        session,
                        self.problem.int_to_node(get)[2],
                        self.problem.int_to_node(release)[2],
                        resource_index + 1,
                    )
                    for get, release in zip(gets, pairing)
                )

    def build_model(self):
        if self._model_built:
            return self.model

        for index in range(self.problem.number_of_activities):
            node = self.problem.int_to_node(index)
            self.tasks[index] = self.model.interval_var(
                size=self.problem.durations[index], name=f"Task_{node}"
            )

        source_index = self.problem.node_id_to_int((0, 0, 0))
        sink_index = self.problem.node_id_to_int((0, 0, 1))
        self.global_source = self.tasks[source_index]
        self.global_sink = self.tasks[sink_index]
        self.model.add(start_of(self.global_source) == 0)

        for predecessor, successor, data in self.problem.precedence_graph.edges(data=True):
            self.model.add(
                start_before_start(
                    self.tasks[self.problem.node_id_to_int(predecessor)],
                    self.tasks[self.problem.node_id_to_int(successor)],
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

        for test_type, session, start_job, end_job, resource_index in self.storage_events:
            start_index = self.problem.node_id_to_int((test_type, session, start_job))
            end_index = self.problem.node_id_to_int((test_type, session, end_job))
            occupancy = self.model.interval_var(
                size=(0, self._horizon()),
                name=f"Occupancy_{session}_{start_job}_{end_job}_{resource_index}",
            )
            self.occupancy[
                (test_type, session, start_job, end_job, resource_index)
            ] = occupancy
            self.model.add(start_at_start(occupancy, self.tasks[start_index]))
            self.model.add(end_at_end(occupancy, self.tasks[end_index]))

        for resource_index, capacity in enumerate(self.problem.cumulative_capacities, start=1):
            resource_usage = sum(
                pulse(occupancy, 1)
                for event, occupancy in self.occupancy.items()
                if event[4] == resource_index
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
