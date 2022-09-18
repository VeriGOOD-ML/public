from collections import defaultdict, deque, namedtuple
from dataclasses import dataclass, field
from typing import List, Dict
import networkx as nx


@dataclass
class Graph:
    graph: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    time: int = field(default=0)
    dfs_util_output: List[str] = field(default_factory=list)

    @property
    def V(self):
        return len(self.graph)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        if v not in self.graph:
            self.graph[v] = []

    def dfs_util(self, v, visited):
        # Mark the current node as visited and print it
        visited[v] = True
        self.dfs_util_output.append(v)
        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.dfs_util(i, visited)

    def fill_order(self, v, visited, stack):
        # Mark the current node as visited
        visited[v] = True
        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                stack = self.fill_order(i, visited, stack)
        stack.append(v)
        return stack
        # Function that returns reverse (or transpose) of this graph

    def get_transpose(self):
        g = Graph()

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph.keys():
            for j in self.graph[i]:
                g.add_edge(j, i)
        return g

    def scc_util(self, u, low, disc, stackMember, st):

        # Initialize discovery time and low value
        disc[u] = self.time
        low[u] = self.time
        self.time += 1
        stackMember[u] = True
        st.append(u)

        # Go through all vertices adjacent to this
        for v in self.graph[u]:

            # If v is not visited yet, then recur for it
            if disc[v] == -1:

                self.scc_util(v, low, disc, stackMember, st)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 (per above discussion on Disc and Low value)
                low[u] = min(low[u], low[v])

            elif stackMember[v] == True:

                '''Update low value of 'u' only if 'v' is still in stack 
                (i.e. it's a back edge, not cross edge). 
                Case 2 (per above discussion on Disc and Low value) '''
                low[u] = min(low[u], disc[v])

                # head node found, pop the stack and print an SCC
        w = -1  # To store stack extracted vertices
        temp_output = []
        if low[u] == disc[u]:
            while w != u:
                w = st.pop()
                temp_output.append(w)
                stackMember[w] = False

            self.dfs_util_output.append(temp_output)


def kosaraju(g):
    visited = {name: False for name in g.graph.keys()}
    stack = deque()

    for i in g.graph.keys():
        if visited[i] == False:
            stack = g.fill_order(i, visited, stack)

            # Create a reversed graph
    gr = g.get_transpose()

    # Mark all the vertices as not visited (For second DFS)
    visited = {name: False for name in gr.graph.keys()}

    # Now process all vertices in order defined by Stack
    groups = []
    while len(stack) > 0:
        i = stack.pop()
        if visited[i] == False:
            gr.dfs_util(i, visited)
            groups.append(gr.dfs_util_output)
            gr.dfs_util_output = []
    return groups

def tarjan(graph: Graph):
    disc = {k: -1 for k in graph.graph.keys()}
    low = {k: -1 for k in graph.graph.keys()}
    stack_member = {k: False for k in graph.graph.keys()}

    st = []

    # Call the recursive helper function
    # to find articulation points
    # in DFS tree rooted with vertex 'i'
    for i in graph.graph.keys():
        if disc[i] == -1:
            graph.scc_util(i, low, disc, stack_member, st)

    return graph.dfs_util_output


def compute_node_levels(nodes: dict, use_tarjan=True):
    graph = nx.MultiDiGraph()

    level_map = defaultdict(lambda: float('inf'))
    target_names = []
    for name, node in nodes.items():
        if node.node_type == "compute" and len(list(node._succs.keys()) + list(node._preds.keys())) > 0:
            target_names.append(name)

        for v in node._succs.values():
            graph.add_edge(name, v.name)

    for target in target_names:

        lengths = list(nx.single_target_shortest_path_length(graph, target))
        for name, length in lengths:
            if name == target or name in target_names:
                continue
            level_map[name] = min(level_map[name], length)
    level_map.update({t: 0 for t in target_names})
    max_val = int(max(level_map.values()))
    rev_map = {i: max_val - i for i in range(max_val+1)}
    node_levels = defaultdict(list)
    for k, v in level_map.items():
        node_levels[rev_map[v]].append(k)

    return node_levels

def get_shortest_paths(nodes: dict, src: str, dst: str):
    graph = nx.MultiDiGraph()

    for name, node in nodes.items():
        for v in node._succs.values():
            graph.add_edge(name, v.name)

    return nx.all_shortest_paths(graph, src, dst)





