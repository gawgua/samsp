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
             node_id: dict[tuple[int, int], int] = None):
        """
        Initialize the problem instance.
        Args:
            number_of_activities (int, optional): Number of activities (including dummy start and end activities).
            number_of_renewable_resources (int, optional): Number of renewable resources.
            number_of_cumulative_resources (int, optional): Number of cumulative resources.
            durations (list[int], optional): List of durations for each activity.
            precedence_graph (nx.DiGraph, optional): Directed graph representing the precedence relations between activities.
            renewable_requests (list[list[int]], optional): List of renewable resource requests for each activity.
            cumulative_requests (list[list[int]], optional): List of cumulative resource requests for each activity.
            cumulative_capacities (list[int], optional): List of capacities for each cumulative resource.
        """
        self.number_of_activities = number_of_activities
        self.number_of_renewable_resources = number_of_renewable_resources
        self.number_of_cumulative_resources = number_of_cumulative_resources
        self.durations = durations
        self.precedence_graph = precedence_graph if precedence_graph is not None else nx.DiGraph()
        self.renewable_requests = renewable_requests
        self.cumulative_requests = cumulative_requests
        self.cumulative_capacities = cumulative_capacities
        self.node_id = node_id

    @staticmethod
    def from_file(file_path: str):
        """
        Create a SAMSPProblem instance from a file.
        Args:
            file_path (str): Path to the input file.
        """

        sections = []

        with open(file_path, 'r', encoding='utf-8') as f:
            section_buffer = ""
            for line in f.readlines():
                if "*" in line:
                    sections.append(section_buffer)
                    section_buffer = ""
                    continue
                section_buffer += line     

        precedence_graph = nx.DiGraph()
        node_id = {} # (session, activity) -> node

        precedence = sections[4].splitlines()[2:]
        for line in precedence:
            var = line.split()
            i = int(var[0])
            precedence_graph.add_node(i)
            node_id[(int(var[1]), int(var[2]))] = i

        timelags = sections[5].splitlines()[2:]
        for line in timelags:
            var = line.split()
            i = int(var[0])
            j = int(var[1])
            timelag = int(var[2])
            precedence_graph.add_edge(i, j, weight = timelag)

        requests_duration = sections[6].splitlines()[2:]
        activities = []
        for line in requests_duration:
            var = line.split()
            i = int(var[0])
            duration = int(var[2])
            requests = list(map(int, var[3:]))
            activities.append((i, duration, *requests))

        resources_capacity = sections[7].splitlines()[2:][0]
        capacities = list(map(int, resources_capacity.split()))

        return SAMSPProblem(
            number_of_activities = len(activities),
            number_of_renewable_resources = 3,
            number_of_cumulative_resources = 4,
            durations = [activity[1] for activity in activities],
            precedence_graph = precedence_graph,
            renewable_requests = [list(activity[2:5]) for activity in activities],
            cumulative_requests = [list(activity[5:9]) for activity in activities],
            cumulative_capacities = list(capacities),
            node_id = node_id
        )