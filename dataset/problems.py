import json
from pathlib import Path

import numpy as np


class SAMSPProblem:
    """Shared NumPy SAMSP input, with graph views for legacy solvers.

    Arrays hold the direct arcs, max-plus closure, resource data and windows.
    NetworkX graphs and historical storage constants are created on demand.
    """

    def __init__(self, *, metadata=None, resources, storages, activities, lags):
        self.metadata = metadata if metadata is not None else {}
        self.resources = resources
        self.storages = storages
        self.activities = activities
        self.lags = lags
        self.file_path = None
        self.activity_ids = [activity["id"] for activity in activities]
        self.number_of_activities = len(activities)
        if self.activity_ids != list(range(self.number_of_activities)):
            raise ValueError("activities IDs must be ordered consecutively from zero")
        self.source_id = 0
        self.sink_id = self.number_of_activities - 1
        self.number_of_renewable_resources = len(resources)
        self.number_of_cumulative_resources = len(storages)
        self.durations = np.asarray([activity["duration"] for activity in activities], dtype=np.int64)
        self.renewable_requests = np.asarray(
            [activity["demands"] for activity in activities], dtype=np.int64)
        self.cumulative_requests = np.asarray(
            [activity["storage_deltas"] for activity in activities], dtype=np.int64)
        self.renewable_capacities = np.asarray([resource["capacity"] for resource in resources], dtype=np.int64)
        self.cumulative_capacities = np.asarray([storage["capacity"] for storage in storages], dtype=np.int64)
        # Preserve source/neighbor order and maximum-weight duplicate arcs.
        neighbors = [{} for _ in self.activity_ids]
        for arc in lags:
            source, target, weight = arc["from"], arc["to"], arc["lag"]
            previous = neighbors[source].get(target)
            if previous is None or weight > previous:
                neighbors[source][target] = weight
        self.precedence_arcs = np.asarray(
            [(source, target, weight) for source in self.activity_ids
             for target, weight in neighbors[source].items()], dtype=np.int64).reshape((-1, 3))
        self._precedence_graph = None
        self._extended_precedence_graph = None
        # Legacy SAT field names expose the same renewable arrays.
        self.number_of_resources = self.number_of_renewable_resources
        self.requests = self.renewable_requests
        self.capacities = self.renewable_capacities
        self._preprocess()

    def _preprocess(self):
        # JSON already includes symmetry breaking. Compute strongest implied lags.
        n = self.number_of_activities
        self.temporal_closure = np.full((n, n), -np.inf)
        np.fill_diagonal(self.temporal_closure, 0)
        for source, target, weight in self.precedence_arcs:
            self.temporal_closure[source, target] = max(self.temporal_closure[source, target], weight)
        for k in self.activity_ids:
            np.maximum(self.temporal_closure,
                       self.temporal_closure[:, k, None] + self.temporal_closure[None, k, :],
                       out=self.temporal_closure)
        value = self.temporal_closure[self.source_id, self.sink_id]
        self.LB = int(value) if np.isfinite(value) and self.source_id != self.sink_id else 0
        # This is a bounded search horizon, not a certified feasible incumbent.
        maxima = np.maximum(self.durations, 0)
        if len(self.precedence_arcs):
            sources, targets = self.precedence_arcs[:, :2].T
            np.maximum.at(maxima, sources, self.temporal_closure[sources, targets].astype(np.int64))
        self.UB = max(self.LB, int(np.sum(maxima)))
        self._extended_precedence_graph = None
        self.set_search_horizon(self.UB)

    @property
    def precedence_graph(self):
        """Materialize the original direct graph only for graph consumers."""
        if self._precedence_graph is None:
            import networkx as nx
            graph = nx.DiGraph()
            graph.add_nodes_from(self.activity_ids)
            graph.add_weighted_edges_from(tuple(map(int, arc)) for arc in self.precedence_arcs)
            self._precedence_graph = graph
        return self._precedence_graph

    @property
    def extended_precedence_graph(self):
        """Expose the historical closure graph, excluding diagonal edges."""
        if self._extended_precedence_graph is None:
            import networkx as nx
            graph = nx.DiGraph()
            graph.add_nodes_from(self.activity_ids)
            for source in self.activity_ids:
                for target in np.flatnonzero(np.isfinite(self.temporal_closure[source])):
                    if source != target:
                        graph.add_edge(source, int(target), weight=int(self.temporal_closure[source, target]))
            self._extended_precedence_graph = graph
        return self._extended_precedence_graph

    def set_search_horizon(self, upper_bound):
        """Recompute finite windows and invalidate derived storage constants."""
        if not isinstance(upper_bound, int) or isinstance(upper_bound, bool):
            raise ValueError("Search horizon must be an integer")
        if upper_bound < max(0, self.LB):
            raise ValueError("Search horizon must be at least the temporal lower bound")
        self.UB = upper_bound
        # Missing dummy paths retain the historical finite horizon fallbacks.
        heads = self.temporal_closure[self.source_id]
        self.ES = np.where(np.isfinite(heads), np.maximum(0, heads), 0).astype(np.int64)
        self.ES[self.source_id] = 0
        tails = self.temporal_closure[:, self.sink_id]
        tails = np.where(np.isfinite(tails), tails, self.durations).astype(np.int64)
        tails[self.sink_id] = self.durations[self.sink_id]
        self.LS = upper_bound - tails
        self.EC = self.ES + self.durations
        self.LC = self.LS + self.durations
        self._PS = None

    @property
    def PS(self):
        """Historical fixed storage usage, allocated only by solvers using it."""
        if self._PS is None:
            self._PS = np.zeros((self.number_of_cumulative_resources, self.UB + 1), dtype=np.int64)
            for resource in range(self.number_of_cumulative_resources):
                deltas = self.cumulative_requests[:, resource]
                boundaries = np.where(deltas > 0, self.LS, self.LC)
                active = (boundaries >= 0) & (boundaries <= self.UB)
                np.add.at(self._PS[resource], boundaries[active], deltas[active])
            np.cumsum(self._PS, axis=1, out=self._PS)
        return self._PS

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
