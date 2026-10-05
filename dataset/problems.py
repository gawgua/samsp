import json
from pathlib import Path

import networkx as nx
import numpy as np


class SAMSPProblem:
    """SAMSP JSON data, indexed by the activity IDs stored in the file."""

    def __init__(self, *, metadata=None, resources, storages, activities, lags):
        self.metadata = metadata if metadata is not None else {}
        self.resources = resources
        self.storages = storages
        self.activities = activities
        self.lags = lags
        self.file_path = None
        self.activity_ids = [activity["id"] for activity in activities]
        self.number_of_activities = len(activities)
        self.source_id = 0
        self.sink_id = self.number_of_activities - 1
        self.number_of_renewable_resources = len(resources)
        self.number_of_cumulative_resources = len(storages)
        self.durations = [activity["duration"] for activity in activities]
        self.renewable_requests = [activity["demands"] for activity in activities]
        self.cumulative_requests = [activity["storage_deltas"] for activity in activities]
        self.renewable_capacities = [resource["capacity"] for resource in resources]
        self.cumulative_capacities = [storage["capacity"] for storage in storages]
        self.precedence_graph = nx.DiGraph()
        self.precedence_graph.add_nodes_from(self.activity_ids)
        for arc in lags:
            previous = self.precedence_graph.get_edge_data(arc["from"], arc["to"])
            if previous is None or arc["lag"] > previous["weight"]:
                self.precedence_graph.add_edge(arc["from"], arc["to"], weight=arc["lag"])
        # Legacy SAT field names expose the same renewable data from JSON.
        self.number_of_resources = self.number_of_renewable_resources
        self.requests = self.renewable_requests
        self.capacities = self.renewable_capacities
        self._preprocess()

    def _preprocess(self):
        # JSON already includes symmetry breaking. Compute strongest implied lags.
        negated_graph = self.precedence_graph.copy()
        for _, _, data in negated_graph.edges(data=True):
            data["weight"] = -data["weight"]
        shortest_paths = nx.floyd_warshall_numpy(
            negated_graph, nodelist=self.activity_ids, weight="weight"
        )
        self.extended_precedence_graph = nx.DiGraph()
        self.extended_precedence_graph.add_nodes_from(self.activity_ids)
        for i in self.activity_ids:
            for j in np.flatnonzero(np.isfinite(shortest_paths[i])):
                if i != j:
                    self.extended_precedence_graph.add_edge(
                        i, self.activity_ids[j], weight=int(-shortest_paths[i, j])
                    )

        self.LB = self.extended_precedence_graph.get_edge_data(
            self.source_id, self.sink_id, default={"weight": 0}
        )["weight"]
        # This is a bounded search horizon, not a certified feasible incumbent.
        self.UB = max(self.LB, sum(
            max([self.durations[i], 0] + [
                self.extended_precedence_graph[i][j]["weight"]
                for j in self.precedence_graph.successors(i)
            ]) for i in self.activity_ids
        ))
        # Some JSON activities lack a path to a dummy. Assign finite horizon
        # windows without inserting precedence edges into the supplied graph.
        self.ES = [
            max(0, self.extended_precedence_graph.get_edge_data(
                self.source_id, i, default={"weight": 0}
            )["weight"]) for i in self.activity_ids
        ]
        self.LS = [
            self.UB - self.extended_precedence_graph.get_edge_data(
                i, self.sink_id, default={"weight": self.durations[i]}
            )["weight"] for i in self.activity_ids
        ]
        self.EC = [start + duration for start, duration in zip(self.ES, self.durations)]
        self.LC = [start + duration for start, duration in zip(self.LS, self.durations)]
        self.PS = {}
        for resource_id in range(self.number_of_cumulative_resources):
            events = {}
            for i in self.activity_ids:
                delta = self.cumulative_requests[i][resource_id]
                boundary = self.LS[i] if delta > 0 else self.LC[i]
                events[boundary] = events.get(boundary, 0) + delta
            occupancy = 0
            for time_point in range(self.UB + 1):
                occupancy += events.get(time_point, 0)
                self.PS[resource_id, time_point] = occupancy

    @classmethod
    def from_json(cls, file_path):
        path = Path(file_path)
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        problem = cls(metadata=data.get("metadata", {}), resources=data["resources"],
                      storages=data["storages"], activities=data["activities"], lags=data["lags"])
        problem.file_path = str(path)
        return problem

    @classmethod
    def from_file(cls, file_path):
        """Compatibility entry point for callers; input uses the JSON schema."""
        return cls.from_json(file_path)
