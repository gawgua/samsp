import networkx as nx
import math


class SAMSPProblem:
    def __init__(
        self,
        number_of_activities: int = None,
        number_of_renewable_resources: int = None,
        number_of_cumulative_resources: int = None,
        durations: list[int] = None,
        precedence_graph: nx.DiGraph = None,
        test_type_blueprints: dict[tuple, nx.DiGraph] = None,
        renewable_requests: list[list[int]] = None,
        cumulative_requests: list[list[int]] = None,
        cumulative_capacities: list[int] = None,
        node_id: dict = None,
    ):

        self.number_of_activities = number_of_activities
        self.number_of_renewable_resources = number_of_renewable_resources
        self.number_of_cumulative_resources = number_of_cumulative_resources
        self.durations = durations
        self.precedence_graph = (
            precedence_graph if precedence_graph is not None else nx.DiGraph()
        )
        self.test_type_blueprints = (
            test_type_blueprints if test_type_blueprints is not None else {}
        )
        self.renewable_requests = renewable_requests
        self.cumulative_requests = cumulative_requests
        self.cumulative_capacities = cumulative_capacities
        self.__node_id = node_id if node_id else {}
        self.__int_to_node = {v: k for k, v in self.__node_id.items()}

        self.__preprocessing()

    def int_to_node(self, i: int):
        return self.__int_to_node.get(i, None)

    def node_id_to_int(self, node_tuple):
        return self.__node_id.get(node_tuple, None)

    def __symmetry_breaking(self):
        # Resource symmetry breaking: Add edge to same activities but in different sessions of the same test type
        for u_tuple in self.precedence_graph.nodes():
            for v_tuple in self.precedence_graph.nodes():
                if (
                    u_tuple[0] == v_tuple[0]  # Same test type
                    and u_tuple[1] != v_tuple[1]  # Different sessions
                    and u_tuple[2] == v_tuple[2]  # Same activity ID
                ):
                    weight = (
                        self.durations[self.node_id_to_int(u_tuple)]
                        if sum(self.renewable_requests[self.node_id_to_int(u_tuple)])
                        > 0
                        else 0
                    )
                    if (
                        u_tuple[1] < v_tuple[1]
                    ):  # Order not matter but enforce for symmetry breaking
                        self.precedence_graph.add_edge(u_tuple, v_tuple, weight=weight)
                    else:
                        self.precedence_graph.add_edge(v_tuple, u_tuple, weight=weight)

        # Storage symmetry breaking: Add edge to same activities but in different sessions of the same test type
        min_occupy_time = {}
        processed_testtypes = set()
        for (test_type, _), g in self.test_type_blueprints.items():
            if test_type in processed_testtypes:
                continue

            processed_testtypes.add(test_type)
            extended_graph = self.__extended_precedence_graph(g)
            for u in extended_graph.nodes():
                for s in range(self.number_of_cumulative_resources):
                    for v in extended_graph.successors(u):
                        edge_data = extended_graph.get_edge_data(u, v)
                        if edge_data is not None:
                            w = edge_data.get("weight", 0)
                            if w > 0 and self.cumulative_requests[u][s] != 0:
                                min_occupy_time[(self.int_to_node(u)[2], s, test_type)] = min(
                                    min_occupy_time.get(
                                        (self.int_to_node(u)[2], s, test_type),
                                        float("inf"),
                                    ),
                                    w + self.durations[v],
                                )

        for (activity_id, s, test_type), occupy_time in min_occupy_time.items():
            for u_tuple in self.precedence_graph.nodes():
                for v_tuple in self.precedence_graph.nodes():
                    if (
                        u_tuple[0] != test_type
                        or v_tuple[0] != test_type
                        or u_tuple[2] != activity_id
                        or v_tuple[2] != activity_id
                    ):
                        continue

                    request = self.cumulative_requests[self.node_id_to_int(u_tuple)][s]
                    if (
                        u_tuple[1] == v_tuple[1] - math.floor(
                            self.cumulative_capacities[s]
                            / request
                        )
                    ):
                        self.precedence_graph.add_edge(
                            u_tuple, v_tuple, weight=occupy_time
                        )

    def __extended_precedence_graph(self, graph: nx.DiGraph):
        # Build transitive closure, negate the weight, find shortest
        # negate again, get the longest path
        negated_graph = graph.copy()
        for _, _, data in negated_graph.edges(data=True):
            data["weight"] = -data["weight"]

        try:
            shortest_paths = nx.floyd_warshall(negated_graph, weight="weight")
        except nx.NetworkXError as e:
            print(f"Error computing shortest paths: {e}")
            raise e

        extended_graph = nx.DiGraph()
        for u_tuple in shortest_paths:
            for v_tuple in shortest_paths[u_tuple]:
                if u_tuple != v_tuple and shortest_paths[u_tuple][v_tuple] != float(
                    "inf"
                ):
                    u_int = self.node_id_to_int(u_tuple)
                    v_int = self.node_id_to_int(v_tuple)
                    extended_graph.add_edge(
                        u_int, v_int, weight=-shortest_paths[u_tuple][v_tuple]
                    )

        return extended_graph

    def __calculate_time_bounds(self):
        # Calculate LB, UB
        self.LB = self.extended_precedence_graph[0][self.number_of_activities - 1][
            "weight"
        ]

        self.UB = 0
        for i in range(self.number_of_activities):
            # Map index 'i' to tuple to fetch successors from original graph, then map back to int
            i_tuple = self.int_to_node(i)
            successors_tuples = self.precedence_graph.successors(i_tuple)
            successors_int = [
                self.node_id_to_int(j_tuple) for j_tuple in successors_tuples
            ]

            self.UB += max(
                self.durations[i],
                *[
                    self.extended_precedence_graph[i][j]["weight"]
                    for j in successors_int
                ]
                + [0],
            )

        # Preprocessing time
        self.ES = [0]
        for i in range(1, self.number_of_activities):
            self.ES.append(self.extended_precedence_graph[0][i]["weight"])

        self.LS = []
        for i in range(self.number_of_activities - 1):
            self.LS.append(
                self.UB
                - self.extended_precedence_graph[i][self.number_of_activities - 1][
                    "weight"
                ]
            )
        self.LS.append(self.UB)

        self.EC = [x + self.durations[i] for i, x in enumerate(self.ES)]
        self.LC = [x + self.durations[i] for i, x in enumerate(self.LS)]

    def __calculate_cumulative_prefix_sums(self):
        # Prefix sum of cumulative resources for resource r at time t.
        self.PS = {}
        for r in range(self.number_of_cumulative_resources):
            usages = {}
            for i in range(self.number_of_activities):
                request = self.cumulative_requests[i][r]
                if request > 0:
                    usages[self.LS[i]] = usages.get(self.LS[i], 0) + request
                elif request < 0:
                    usages[self.LC[i]] = usages.get(self.LC[i], 0) + request

            running_total = 0
            for t in range(self.UB + 1):
                running_total += usages.get(t, 0)
                self.PS[r, t] = running_total
        # print(f"Prefix sums of cumulative resources: {self.PS}")

    def __preprocessing(self):
        self.__symmetry_breaking()
        self.extended_precedence_graph = self.__extended_precedence_graph(
            self.precedence_graph
        )
        self.__calculate_time_bounds()
        self.__calculate_cumulative_prefix_sums()

    @staticmethod
    def from_file(file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [
                line.strip() for line in f if line.strip() and not line.startswith("*")
            ]

        test_instances = {}
        blueprints = []
        test_type = None
        blueprint_temp_graph = None
        blueprint_activities = None
        capacities = []
        parse_mode = None

        # 1. Parse the Blueprint (Single Graph Template)
        iterator = iter(lines)
        for line in iterator:
            if line.startswith("INSTANCES"):
                next(iterator)
                parse_mode = "INSTANCES"
            elif line.startswith("TESTTYPE:"):
                if blueprint_temp_graph is not None:
                    blueprints.append(
                        {
                            "test_type": test_type,
                            "graph": blueprint_temp_graph,
                            "activities": blueprint_activities,
                        }
                    )
                raw_tt = line.split(":")[1].strip()
                test_type = int(raw_tt) if raw_tt.isdigit() else raw_tt
                blueprint_temp_graph = nx.DiGraph()
                blueprint_activities = {}
                parse_mode = None
            elif line.startswith("PRECEDENCE RELATIONS:"):
                next(iterator)
                parse_mode = "PRECEDENCE"
            elif line.startswith("TIME LAGS:"):
                next(iterator)
                parse_mode = "TIMELAGS"
            elif line.startswith("REQUESTS/DURATIONS:"):
                next(iterator)
                parse_mode = "DURATIONS"
            elif line.startswith("RESOURCEAVAILABILITIES:"):
                next(iterator)
                parse_mode = "RESOURCES"
            elif line.startswith("PROJECT INFORMATION:") or line.startswith(
                "RESOURCES"
            ):
                parse_mode = None
            else:
                var = line.split()
                if not var:
                    continue

                if parse_mode == "INSTANCES":
                    t_type = int(var[0]) if var[0].isdigit() else var[0]
                    test_instances[t_type] = int(var[1])
                elif parse_mode == "PRECEDENCE":
                    u = int(var[0])
                    for v in [int(x) for x in var[2 : 2 + int(var[1])]]:
                        blueprint_temp_graph.add_edge(u, v)
                elif parse_mode == "TIMELAGS":
                    if blueprint_temp_graph is None:
                        raise ValueError(
                            "TIMELAGS section found before any TESTTYPE section."
                        )

                    u, v, lag = int(var[0]), int(var[1]), int(var[2])
                    if blueprint_temp_graph.has_edge(u, v):
                        blueprint_temp_graph.edges[u, v]["weight"] = lag
                    else:
                        blueprint_temp_graph.add_edge(u, v, weight=lag)
                elif parse_mode == "DURATIONS":
                    if blueprint_activities is None:
                        raise ValueError(
                            "DURATIONS section found before any TESTTYPE section."
                        )

                    u, duration = int(var[0]), int(var[2])
                    blueprint_activities[u] = {
                        "duration": duration,
                        "renewable": [int(x) for x in var[3:6]],
                        "cumulative": [int(x) for x in var[6:10]],
                    }
                elif parse_mode == "RESOURCES":
                    capacities = [int(x) for x in var]
                    parse_mode = None

        if blueprint_temp_graph is not None:
            blueprints.append(
                {
                    "test_type": test_type,
                    "graph": blueprint_temp_graph,
                    "activities": blueprint_activities,
                }
            )

        # ---------------------------------------------------------
        # 2. Graph Expansion Logic (Using Tuples for Nodes/Edges)
        # ---------------------------------------------------------
        test_types_graphs = {}
        node_id_map = {}  # Maps (type, session, idx) -> flat list index

        # Define the global Super-Source and Super-Sink as (type=0, session=0, index)
        super_source = (0, 0, 0)
        super_sink = (0, 0, 1)

        # Initialize lists with Super-Source data at index 0
        full_durations = [0]
        full_ren = [[0, 0, 0]]
        full_cum = [[0, 0, 0, 0]]

        node_id_map[super_source] = 0
        current_list_idx = 1

        # Duplicate every blueprint graph for each of its instances/sessions.
        for blueprint in blueprints:
            test_type = blueprint["test_type"]
            blueprint_graph = blueprint["graph"]
            blueprint_activities = blueprint["activities"]
            num_instances = test_instances.get(test_type, 1)
            activity_ids = sorted(blueprint_activities.keys())

            for session in range(1, num_instances + 1):
                mapping = {u: (test_type, session, u) for u in blueprint_graph.nodes()}
                session_graph = nx.relabel_nodes(blueprint_graph, mapping)

                for u in activity_ids:
                    tuple_node = (test_type, session, u)
                    activity = blueprint_activities[u]
                    node_id_map[tuple_node] = current_list_idx
                    full_durations.append(activity["duration"])
                    full_ren.append(activity["renewable"])
                    full_cum.append(activity["cumulative"])
                    session_graph.nodes[tuple_node].update(
                        {
                            "node_id": tuple_node,
                            "activity_id": u,
                            "duration": activity["duration"],
                            "renewable": activity["renewable"],
                            "cumulative": activity["cumulative"],
                            "renewable_requests": activity["renewable"],
                            "cumulative_requests": activity["cumulative"],
                        }
                    )
                    current_list_idx += 1

                full_precedence_start = (test_type, session, activity_ids[0])
                full_precedence_end = (test_type, session, activity_ids[-1])
                session_graph.add_edge(super_source, full_precedence_start, weight=0)
                session_graph.add_edge(full_precedence_end, super_sink, weight=0)
                test_types_graphs[test_type, session] = session_graph

        # 3. Combine all graphs and add Super-Source & Super-Sink
        full_precedence_graph = (
            nx.compose_all(test_types_graphs.values())
            if test_types_graphs
            else nx.DiGraph()
        )

        # Add Super-Sink to lists
        node_id_map[super_sink] = current_list_idx
        full_durations.append(0)
        full_ren.append([0, 0, 0])
        full_cum.append([0, 0, 0, 0])
        full_precedence_graph.add_node(
            super_source,
            node_id=super_source,
            activity_id=None,
            duration=0,
            renewable=[0, 0, 0],
            cumulative=[0, 0, 0, 0],
            renewable_requests=[0, 0, 0],
            cumulative_requests=[0, 0, 0, 0],
        )
        full_precedence_graph.add_node(
            super_sink,
            node_id=super_sink,
            activity_id=None,
            duration=0,
            renewable=[0, 0, 0],
            cumulative=[0, 0, 0, 0],
            renewable_requests=[0, 0, 0],
            cumulative_requests=[0, 0, 0, 0],
        )

        return SAMSPProblem(
            number_of_activities=len(full_durations),
            number_of_renewable_resources=3,
            number_of_cumulative_resources=4,
            durations=full_durations,
            precedence_graph=full_precedence_graph,
            test_type_blueprints=test_types_graphs,
            renewable_requests=full_ren,
            cumulative_requests=full_cum,
            cumulative_capacities=capacities,
            node_id=node_id_map,
        )
