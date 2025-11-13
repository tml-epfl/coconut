from typing import List, Optional, Union
import random, math, json
import traceback

import numpy as np


class DAG:
    @staticmethod
    def generate_layered_dag(
        num_nodes: int,
        num_layers: int,
        connection_probability: float,
        layer_probabilities: Optional[List[float]] = None,
    ):
        """
        Generates a Directed Acyclic Graph (DAG) with a layered structure.

        Args:
            num_nodes (int): The total number of nodes in the graph.
            num_layers (int): The number of layers to partition the nodes into. This
                            controls the depth of the graph. Must be at least 2.
            connection_probability (float): The probability (between 0.0 and 1.0) of
                                        creating an edge between nodes in
                                        subsequent layers.
            layer_probabilities (list[float], optional): A list of probabilities for
                                                        assigning a node to each layer.
                                                        The list length must equal
                                                        num_layers and its elements
                                                        must sum to 1. If None,
                                                        assignment is uniform.
                                                        Defaults to None.

        Returns:
            dict: An adjacency list representation of the DAG, where each key is a
                node and its value is a list of nodes it connects to.
            dict: A dictionary mapping each node to its assigned layer.
        """
        # --- Input Validation ---
        if num_nodes <= 0:
            raise ValueError("Number of nodes must be positive.")
        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2 for connections.")
        if not (0.0 <= connection_probability <= 1.0):
            raise ValueError("Connection probability must be between 0.0 and 1.0.")
        if layer_probabilities is not None:
            if len(layer_probabilities) != num_layers:
                raise ValueError(
                    "The length of layer_probabilities must be equal to num_layers."
                )
            if not all(p >= 0 for p in layer_probabilities):
                raise ValueError("All layer probabilities must be non-negative.")
            if not math.isclose(sum(layer_probabilities), 1.0):
                raise ValueError(
                    f"The sum of layer_probabilities must be 1, but it is {sum(layer_probabilities)}."
                )

        # --- Initialization ---
        graph = {i: [] for i in range(num_nodes)}
        node_layers = [-1 for i in range(num_nodes)]
        layers_of_nodes = list(range(num_layers))

        # --- Step 1: Assign each node to a layer ---
        # To ensure the graph can span all layers, we can optionally place the first
        # two nodes in the first and last layers respectively.
        if num_nodes > 1:
            node_layers[0] = 0
            node_layers[1] = num_layers - 1
            nodes_to_assign = range(2, num_nodes)
        else:  # Handle the single-node case
            if num_nodes == 1:
                node_layers[0] = 0
            nodes_to_assign = range(0)

        if layer_probabilities is None:
            # Uniformly assign the remaining nodes to any layer
            for node_idx in nodes_to_assign:
                node_layers[node_idx] = random.choice(layers_of_nodes)
        else:
            # Assign remaining nodes based on the provided probability distribution
            assigned_layers = random.choices(
                population=layers_of_nodes,
                weights=layer_probabilities,
                k=len(nodes_to_assign),
            )
            for i, node_idx in enumerate(nodes_to_assign):
                node_layers[node_idx] = assigned_layers[i]

        # --- Step 2: Create edges based on layers and probability ---
        nodes = list(range(num_nodes))
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue

                # An edge can only go from a lower layer to a higher layer
                if node_layers[u] < node_layers[v]:
                    # Add the edge with the given probability
                    if random.random() < connection_probability:
                        graph[u].append(v)

        return DAG(list(graph.keys()), node_layers, list(graph.values()))

    def __init__(
        self,
        nodes: List[int],
        layers: List[int],
        edges: List[List[int]],
    ):
        super(DAG, self).__init__()

        self.nodes = nodes
        self.layers = layers
        self.num_layers = max(self.layers) + 1
        self.layer_map = {
            idx: [node for node in self.nodes if self.layers[node] == idx]
            for idx in range(self.num_layers)
        }

        self.edges = edges
        self.parents = [
            [j for j in self.nodes if i in self.edges[j]] for i in self.nodes
        ]
        self.descendants = [None for _ in self.nodes]
        self.paths = [
            [None for _ in self.nodes] for _ in self.nodes
        ]  # computed in a lazy fashion

    def get_descendants(self, a: int) -> List[int]:
        if self.descendants[a] is None:
            self.descendants[a] = self.edges[a]
            for node in self.edges[a]:
                self.descendants[a].extend(self.get_descendants(node))
            self.descendants[a] = list(set(self.descendants[a]))

        return self.descendants[a]

    def get_paths_between(self, a: int, b: int) -> Optional[List[int]]:
        if a == b:
            return [b]

        if self.paths[a][b] is None:
            paths = []
            for descendant in self.get_descendants(a):
                paths.extend(
                    [[a] + path for path in self.get_paths_between(descendant, b)]
                )
            self.paths[a][b] = paths

        return self.paths[a][b]


def generate_query_from_dag(
    dag: DAG,
    entities: Optional[List[str]] = None,
    info_format: str = "{#1} is a {#2}.",
    question_format: str = "Is {#1} a {#2} or a {#3}?",
    answer_format: str = "{#1} is a {#2}.",
    prefix_non_roots: str = "Every ",
    verbose: bool = False,
) -> str:
    if entities is None:
        entities = [f"ent{i}" for i in range(len(dag.nodes))]
    entities = [
        entities[i].capitalize() if dag.layers[i] == 0 else entities[i]
        for i in range(len(dag.nodes))
    ]

    pairs = [(a, b) for a in dag.layer_map[0] for b in dag.nodes if dag.layers[b] != 0]
    random.shuffle(pairs)

    for a, b in pairs:
        try:
            # generate the path
            paths = dag.get_paths_between(a, b)
            path = paths[random.randint(0, len(paths) - 1)]

            # turn into query
            a, b, c = path[0], path[-1], -1
            descendants = dag.get_descendants(a)

            index = random.randint(
                0,
                len(dag.nodes) - len(descendants) - len(dag.layer_map[0]) - 2,
            )
            assert index >= 0

            while index > 0 or c == -1:
                while (c in descendants) or (dag.layers[c] == 0) or c == b or c == -1:
                    c += 1
                index -= 1

            # prepare the strings
            context = []
            for n1 in dag.nodes:
                for n2 in dag.edges[n1]:
                    context.append(
                        (prefix_non_roots if dag.layers[n1] > 0 else "")
                        + (
                            info_format.replace(
                                "{#1}",
                                entities[n1],
                            ).replace("{#2}", entities[n2])
                        )
                    )
            random.shuffle(context)
            context = " ".join(context)

            question = (
                question_format.replace("{#1}", entities[a])
                .replace("{#2}", entities[b])
                .replace("{#3}", entities[c])
            )

            chain = []
            for i in range(1, len(path)):
                chain.append(
                    info_format.replace("{#1}", entities[path[i - 1]]).replace(
                        "{#2}", entities[path[i]]
                    )
                )

            answer = answer_format.replace("{#1}", entities[a]).replace(
                "{#2}", entities[b]
            )

            return (a, b, c), context, question, chain, answer
        except Exception as e:
            if verbose:
                print(e)
                traceback.print_exc()
    return None


def sample_names_for_dag(
    dag: DAG,
    names: Union[str, List[str]],
    entities: Union[str, List[str]],
) -> List[str]:
    if isinstance(names, str):
        with open(names, "r") as file:
            names = file.readlines()
    if isinstance(entities, str):
        with open(entities, "r") as file:
            entities = file.readlines()

    n_root = len(dag.layer_map[0])
    names = random.sample(names, n_root)
    entities = names + random.sample(entities, len(dag.nodes) - n_root)
    entities = [entity.strip() for entity in entities]

    ind = 0
    result = [None for _ in range(len(dag.nodes))]
    for i in range(0, dag.num_layers):
        for j in range(len(dag.layer_map[i])):
            result[dag.layer_map[i][j]] = entities[ind]
            ind += 1

    return result


def get_names_and_entities(file: str) -> None:
    with open(file, "r") as file:
        # Load the JSON data from the file
        data = json.load(file)

    names, entities = set(), set()
    for query in data:
        symbols = query["idx_to_symbol"]
        names.update([symbol for symbol in symbols if symbol[0].isupper()])
        entities.update([symbol for symbol in symbols if (not symbol[0].isupper())])
    names, entities = list(names), list(entities)

    with open("names.txt", "w") as f:
        f.writelines([str(number) + "\n" for number in names])
    with open("entities.txt", "w") as f:
        f.writelines([str(number) + "\n" for number in entities])


def get_statistics(file: str) -> None:
    with open(file, "r") as file:
        data = json.load(file)

    num_nodes = [len(d["idx_to_symbol"]) for d in data]
    num_steps = [len(d["steps"]) for d in data]
    num_edges = [len(d["edges"]) for d in data]

    for name, data in {"nodes": num_nodes, "steps": num_steps, "edges": num_edges}:
        np_array = np.array(data)

        mean_value = np.mean(np_array)
        variance_value = np.var(np_array)
        print(f"\nResults for: {name}")
        print(f"Data: {data}")
        print(f"  -> Mean:     {mean_value:.4f}")  # Format to 4 decimal places
        print(f"  -> Variance: {variance_value:.4f}")
