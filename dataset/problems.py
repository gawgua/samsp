import networkx as nx

class SAMSPProblem:
    def __init__(self, number_of_activities: int = None,
                 number_of_renewable_resources: int = None,
                 number_of_cumulative_resources: int = None,
                 durations: list[int] = None,
                 precedence_graph: nx.DiGraph = None,
                 renewable_requests: list[list[int]] = None,
                 cumulative_requests: list[list[int]] = None,
                 cumulative_capacities: list[int] = None,
                 node_id: dict = None):
        
        self.number_of_activities = number_of_activities
        self.number_of_renewable_resources = number_of_renewable_resources
        self.number_of_cumulative_resources = number_of_cumulative_resources
        self.durations = durations
        self.precedence_graph = precedence_graph if precedence_graph is not None else nx.DiGraph()
        self.renewable_requests = renewable_requests
        self.cumulative_requests = cumulative_requests
        self.cumulative_capacities = cumulative_capacities
        self.node_id = node_id if node_id else {}

    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('*')]

        test_instances = {}
        test_type = None
        blueprint_graph = nx.DiGraph()
        blueprint_activities = {}
        capacities = []
        parse_mode = None
        
        # 1. Parse the Blueprint (Single Graph Template)
        iterator = iter(lines)
        for line in iterator:
            if line.startswith("INSTANCES"):
                next(iterator)
                parse_mode = "INSTANCES"
            elif line.startswith("TESTTYPE:"):
                raw_tt = line.split(":")[1].strip()
                test_type = int(raw_tt) if raw_tt.isdigit() else raw_tt
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
            elif line.startswith("PROJECT INFORMATION:") or line.startswith("RESOURCES"):
                parse_mode = None 
            else:
                var = line.split()
                if not var: continue
                    
                if parse_mode == "INSTANCES":
                    t_type = int(var[0]) if var[0].isdigit() else var[0]
                    test_instances[t_type] = int(var[1])
                elif parse_mode == "PRECEDENCE":
                    u = int(var[0])
                    for v in [int(x) for x in var[2 : 2 + int(var[1])]]:
                        blueprint_graph.add_edge(u, v)
                elif parse_mode == "TIMELAGS":
                    u, v, lag = int(var[0]), int(var[1]), int(var[2])
                    if blueprint_graph.has_edge(u, v):
                        blueprint_graph[u][v]['weight'] = lag
                    else:
                        blueprint_graph.add_edge(u, v, weight=lag)
                elif parse_mode == "DURATIONS":
                    u, dur = int(var[0]), int(var[2])
                    blueprint_activities[u] = {
                        'duration': dur,
                        'renewable': [int(x) for x in var[3:6]],
                        'cumulative': [int(x) for x in var[6:10]]
                    }
                elif parse_mode == "RESOURCES":
                    capacities = [int(x) for x in var]
                    parse_mode = None

        # ---------------------------------------------------------
        # 2. Graph Expansion Logic (Using Tuples for Nodes/Edges)
        # ---------------------------------------------------------
        N = len(blueprint_activities)
        num_instances = test_instances.get(test_type, 1)

        test_types_graphs = []
        node_id_map = {} # Maps (type, session, idx) -> flat list index
        
        # Define the global Super-Source and Super-Sink as (type=0, session=0, index)
        super_source = (0, 0, 0)
        super_sink = (0, 0, 1)
        
        # Initialize lists with Super-Source data at index 0
        full_durations = [0]
        full_ren = [[0, 0, 0]]
        full_cum = [[0, 0, 0, 0]]
        
        node_id_map[super_source] = 0
        current_list_idx = 1

        # Duplicate the blueprint graph for each instance/session
        for session in range(1, num_instances + 1):
            
            # Map the blueprint integer nodes to tuple nodes: (test_type, session, node_id)
            mapping = {u: (test_type, session, u) for u in blueprint_graph.nodes()}
            session_graph = nx.relabel_nodes(blueprint_graph, mapping)
            test_types_graphs.append(session_graph)

            # Append the attributes for this session's nodes
            for u in sorted(blueprint_activities.keys()):
                tuple_node = (test_type, session, u)
                
                # Link tuple to its flat index so durations/requests can be looked up
                node_id_map[tuple_node] = current_list_idx
                full_durations.append(blueprint_activities[u]['duration'])
                full_ren.append(blueprint_activities[u]['renewable'])
                full_cum.append(blueprint_activities[u]['cumulative'])
                current_list_idx += 1

        # 3. Combine all graphs and add Super-Source & Super-Sink
        full_precedence_graph = nx.compose_all(test_types_graphs) if test_types_graphs else nx.DiGraph()
        
        # Add Super-Sink to lists
        node_id_map[super_sink] = current_list_idx
        full_durations.append(0)
        full_ren.append([0, 0, 0])
        full_cum.append([0, 0, 0, 0])

        # Connect Super-Source and Super-Sink to each session graph
        for session in range(1, num_instances + 1):
            bp_start = (test_type, session, 0)      # Activity 0 of the session
            bp_end = (test_type, session, N - 1)    # Activity N-1 of the session
            
            # Connect Global Start -> Session Start
            full_precedence_graph.add_edge(super_source, bp_start, weight=0)
            # Connect Session End -> Global End
            full_precedence_graph.add_edge(bp_end, super_sink, weight=0)

        return SAMSPProblem(
            number_of_activities=len(full_durations),
            number_of_renewable_resources=3,
            number_of_cumulative_resources=4,
            durations=full_durations,
            precedence_graph=full_precedence_graph,
            renewable_requests=full_ren,
            cumulative_requests=full_cum,
            cumulative_capacities=capacities,
            node_id=node_id_map
        )