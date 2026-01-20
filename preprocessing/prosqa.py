from typing import List, Optional, Union, Tuple
import random, math, json
import traceback

import numpy as np
import matplotlib.pyplot as plt


class DAG:
    @staticmethod
    def generate_layered_dag(
        num_nodes: int,
        num_layers: int,
        num_edges: int,
        layer_probabilities: Optional[List[float]] = None,
        strict: bool = True,
    ):
        """
        Generates a Directed Acyclic Graph (DAG) with a layered structure.

        Args:
            num_nodes (int): The total number of nodes in the graph.
            num_layers (int): The number of layers to partition the nodes into. This
                            controls the depth of the graph. Must be at least 2.
            num_edges (int): The total number of edges in the graph.
            layer_probabilities (list[float], optional): A list of probabilities for
                                                        assigning a node to each layer.
                                                        The list length must equal
                                                        num_layers and its elements
                                                        must sum to 1. If None,
                                                        assignment is uniform.
                                                        Defaults to None.
            strict (bool): Whether to force layer for each node

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
        if not (num_edges >= num_layers - 1):
            raise ValueError(
                "Number of edges must be at least as large as number of layers - 1"
            )
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
        assert (
            num_nodes >= num_layers
        ), "Number of nodes need to be larger than number of layers!"
        for i in range(num_layers):
            node_layers[i] = i

        if num_nodes > num_layers:
            nodes_to_assign = list(range(num_layers, num_nodes))

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
        node_pairs = []
        for a in range(num_nodes):
            for b in range(num_nodes):
                if node_layers[a] == node_layers[b] - 1:
                    node_pairs.append((a, b))

        if num_edges > len(node_pairs):
            num_edges = len(node_pairs)

        if strict:
            edges = []

            # select at least one edge per non-root nodes
            for b in range(num_nodes):
                node_pairs_b = [pair for pair in node_pairs if pair[-1] == b]
                if node_layers[b] > 0:
                    edges.append(
                        (
                            random.choice(node_pairs_b)[0],
                            b,
                        )
                    )

            # select the rest of the edges
            if num_edges - len(edges) > 0:
                edges.extend(
                    random.sample(
                        population=[pair for pair in node_pairs if not (pair in edges)],
                        k=num_edges - len(edges),
                    )
                )
        else:
            edges = random.sample(
                population=node_pairs,
                k=num_edges,
            )

        for edge in edges:
            graph[edge[0]].append(edge[1])
        return DAG(list(graph.keys()), node_layers, list(graph.values()))

    @staticmethod
    def generate_prosqa_dag(
        num_nodes: int,
        poisson_coeff: float = 1.5,
        prob_coeff: float = 0.35,
        depth_coeff: float = 1.5,
    ):
        """
        Generates a Directed Acyclic Graph (DAG) with a layered structure
        based on ProsQA pseudo code.

        Returns:
            Tuple[DAG, Dict[int, int]]: The DAG and a dictionary mapping node index
                to its family label (1 = descendant of node 0, 2 = descendant of node 1,
                3 = both, 0 = neither).
        """
        assert prob_coeff < 0.5, "`prob_coeff` needs to be smaller than 0.5"

        nodes = [0, 1]
        edges = {0: [], 1: []}
        labels = {0: 1, 1: 2}
        depth = {0: 0, 1: 0}
        groups = {0: [], 1: [0], 2: [1], 3: []}
        for i in range(2, num_nodes):
            num_parents = np.random.poisson(poisson_coeff)
            rand = np.random.random()

            if rand <= prob_coeff:
                candidates = groups[0] + groups[1]
            elif rand <= 2 * prob_coeff:
                candidates = groups[0] + groups[2]
            else:
                candidates = nodes

            # ensure we have candidates; if empty (e.g., early iterations), fall back to existing nodes
            if len(candidates) == 0:
                candidates = nodes

            num_parents = min(len(candidates), num_parents)
            weights = np.asarray([depth[c] * depth_coeff + 1 for c in candidates])
            parents = np.random.choice(
                candidates,
                num_parents,
                p=weights / np.sum(weights),
                replace=False,  # distinct parents, as in pseudo code
            )

            # compute the label
            _labels = [labels[parent] for parent in parents]
            _label = 0
            for _l in _labels:
                _label |= _l  # bitwise OR of parent labels

            # compute the depth
            _depths = [depth[parent] for parent in parents]
            _depth = 0 if len(_depths) == 0 else min(_depths) + 1

            # add edges
            for parent in parents:
                edges[parent].append(i)

            nodes.append(i)
            edges[i] = []
            depth[i] = _depth
            labels[i] = _label
            groups[_label].append(i)

        dag = DAG(
            list(edges.keys()),
            [depth[i] for i in range(num_nodes)],
            list(edges.values()),
        )
        # Return both the DAG and the node family labels
        return dag, labels

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
            self.descendants[a] = list(self.edges[a])
            for node in self.edges[a]:
                self.descendants[a].extend(self.get_descendants(node))
            self.descendants[a] = list(set(self.descendants[a]))

        return self.descendants[a]

    def get_paths_between(self, a: int, b: int) -> Optional[List[int]]:
        if a == b:
            return [[b]]

        if self.paths[a][b] is None:
            paths = []
            for c in self.edges[a]:
                paths.extend([[a] + path for path in self.get_paths_between(c, b)])
            self.paths[a][b] = paths

        return self.paths[a][b]


def generate_query_from_dag(
    dag: DAG,
    entities: Optional[List[str]] = None,
    node_labels: Optional[dict] = None,
    info_format: str = "{#1} is a {#2}.",
    question_format: str = "Is {#1} a {#2} or a {#3}?",
    answer_format: str = "{#1} is a {#2}.",
    prefix_non_roots: str = "Every ",
    length: int = -1,
    neg_length: int = -1,
    num_chains: int = -1,
    verbose: bool = False,
) -> str:
    """
    Generate a query from a DAG following the ProsQA paper specification.

    Args:
        dag: The directed acyclic graph.
        entities: List of entity/concept names for each node.
        node_labels: Dictionary mapping node index to family label
            (1 = family of node 0, 2 = family of node 1, 3 = both, 0 = neither).
            Required for ProsQA-style generation to select concepts correctly.
        info_format: Format string for context statements.
        question_format: Format string for the question.
        answer_format: Format string for the answer.
        prefix_non_roots: Prefix for non-root nodes in statements.
        length: Target path length for concept A (-1 for max depth).
        neg_length: Target path length for concept B (-1 for max depth).
        num_chains: Number of reasoning chains to generate (-1 for all).
        verbose: Whether to print debug information.

    Returns:
        Tuple of (nodes, context, question, chains, answer) or None if generation fails.
    """
    assert (
        num_chains == -1 or num_chains > 0
    ), "`num_chains` needs to be either -1 (all) or some positive integer"

    if entities is None:
        entities = [f"ent{i}" for i in range(len(dag.nodes))]
    entities = [
        entities[i].capitalize() if dag.layers[i] == 0 else entities[i]
        for i in range(len(dag.nodes))
    ]

    if length == -1:
        length = max(dag.layers)
    if neg_length == -1:
        neg_length = max(dag.layers)

    # Identify leaf nodes (nodes with no children)
    leaf_nodes = [n for n in dag.nodes if len(dag.edges[n]) == 0]

    # Per paper: entity is node 0, concept A is a leaf with label 1, concept B is a leaf with label 2
    if node_labels is not None:
        # ProsQA-style: use labels to select concepts
        # Entity is always node 0
        a = 0

        # Concept A candidates: leaf nodes with label 1 (family of node 0 only)
        concept_a_candidates = [n for n in leaf_nodes if node_labels.get(n) == 1]
        # Concept B candidates: leaf nodes with label 2 (family of node 1 only)
        concept_b_candidates = [n for n in leaf_nodes if node_labels.get(n) == 2]

        if not concept_a_candidates or not concept_b_candidates:
            if verbose:
                print("No valid concept candidates found with required labels")
            return None

        # Build pairs from valid candidates
        pairs = [(a, b, c) for b in concept_a_candidates for c in concept_b_candidates]
    else:
        # Fallback: original behavior based on layer depth
        pairs = [
            (a, b, c)
            for a in dag.layer_map[0]
            for b in dag.layer_map[length]
            for c in dag.layer_map[neg_length]
            if not (c in dag.get_descendants(a))
        ]
    random.shuffle(pairs)

    for a, b, c in pairs:
        try:
            assert a != b and b != c and c != a

            # generate the path
            paths = dag.get_paths_between(a, b)
            if num_chains != -1:
                paths = random.sample(paths, num_chains)
            assert len(paths) > 0

            # turn into query
            descendants = dag.get_descendants(a)
            assert not (c in descendants)

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

            # Per paper: randomly permute concept A and B to avoid positional bias
            if random.random() < 0.5:
                # Swap positions: concept B first, then concept A
                question = (
                    question_format.replace("{#1}", entities[a])
                    .replace("{#2}", entities[c])
                    .replace("{#3}", entities[b])
                )
            else:
                # Original order: concept A first, then concept B
                question = (
                    question_format.replace("{#1}", entities[a])
                    .replace("{#2}", entities[b])
                    .replace("{#3}", entities[c])
                )

            chains = [[] for _ in paths]
            for idx, path in enumerate(paths):
                for i in range(1, len(path)):
                    chains[idx].append(
                        (prefix_non_roots if i > 1 else "")
                        + (
                            info_format.replace("{#1}", entities[path[i - 1]]).replace(
                                "{#2}", entities[path[i]]
                            )
                        )
                    )

            # Answer always refers to the correct concept (b)
            answer = answer_format.replace("{#1}", entities[a]).replace(
                "{#2}", entities[b]
            )

            return (a, b, c), context, question, chains, answer
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

    def _integer_bins(values: List[int]) -> List[int]:
        """Return histogram bin edges aligned to integers."""
        if not values:
            return [0, 1]
        min_val, max_val = min(values), max(values)
        return list(range(min_val, max_val + 2))

    # Create three side-by-side histograms: nodes, steps, and edges.
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram for number of nodes
    axs[0].hist(
        num_nodes,
        bins=_integer_bins(num_nodes),
        color="tab:blue",
        edgecolor="black",
        alpha=0.7,
    )
    axs[0].set_title("Number of Nodes")
    axs[0].set_xlabel("Nodes")
    axs[0].set_ylabel("Frequency")

    # Histogram for number of steps
    axs[1].hist(
        num_steps,
        bins=_integer_bins(num_steps),
        color="tab:orange",
        edgecolor="black",
        alpha=0.7,
    )
    axs[1].set_title("Number of Steps")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Frequency")

    # Histogram for number of edges
    axs[2].hist(
        num_edges,
        bins=_integer_bins(num_edges),
        color="tab:green",
        edgecolor="black",
        alpha=0.7,
    )
    axs[2].set_title("Number of Edges")
    axs[2].set_xlabel("Edges")
    axs[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def read_names_and_entities(
    max_names: int = 0, max_entities: int = 0
) -> Tuple[List[str], List[str]]:
    with open("data/names.txt", "r") as file:
        names_list = file.readlines()
    with open("data/entities.txt", "r") as file:
        entities_list = file.readlines()

    if max_names > 0:
        names_list = names_list[:max_names]
    if max_entities > 0:
        entities_list = entities_list[:max_entities]

    return names_list, entities_list
