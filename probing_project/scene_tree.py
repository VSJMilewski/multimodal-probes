import logging
import pickle
import queue
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import h5py  # type: ignore
import numpy as np
from probing_project.tasks.depth_task import DepthTask
from probing_project.tasks.distance_task import DistanceTask
from probing_project.utils import (
    Observation,
    dist_matrix_to_edges,
    _text_edges2visual_edges,
)
from rich.progress import track

logger = logging.getLogger(__name__)


def create_scene_tree(
    conllx_observations: Dict[int, Observation],
    featurefile: str,
    output_file: str,
) -> None:
    """
    Given the intermediate processed sentences and images, it creates
    the scene tree structure over the regions in the image.
    Args:
        conllx_observations: The sentence conllx data stored in Observation tuples
        featurefile: The file with the processed embeddings and information
        output_file: Where to store the scene tree data
    """
    featuresdata = h5py.File(featurefile, "r")
    box_phrase_id = featuresdata["box_phrase_id"]
    num_boxes = featuresdata["num_boxes"]

    scene_tree_storage = []
    for index, ob in track(
        conllx_observations.items(), description="create scene tree"
    ):
        # first we remove all the words_phraseids that are 0,
        # since these don't belong to a box and cause errors
        words_phraseids = ["-1" if k == "0" else k for k in ob.pp_entity_id]
        phrase_ids = box_phrase_id[index]
        num_box = num_boxes[index]
        ob_dist = DistanceTask.compute_parse_distance_labels(ob)
        ob_depth = DepthTask.compute_parse_depth_labels(ob)
        text_edges = dist_matrix_to_edges(ob_dist, ob.xpos_sentence)
        _edges2directed_edges(text_edges, ob_depth)
        visual_edges, phrase2depth, phrase2text_idx = _text_edges2visual_edges(
            text_edges, ob_depth, words_phraseids
        )

        # add edges for boxes not in current sentence
        max_idx = np.max(list(phrase2text_idx.values()))
        for phrase_id in phrase_ids[:num_box]:
            if str(phrase_id) not in phrase2text_idx.keys():
                max_idx += 1
                visual_edges.append((phrase2text_idx["0"], max_idx))
                phrase2depth[str(phrase_id)] = -1
                phrase2text_idx[str(phrase_id)] = max_idx
        head_idx2phrase = {v: k for k, v in phrase2text_idx.items()}
        vertex_connections, idx2vert, vert2idx = _edges2vertex_connections(
            visual_edges, zero_range_verts=True
        )
        scene_tree_storage.append(
            {
                "visual_edges": visual_edges,
                "phrase2depth": phrase2depth,
                "phrase2text_idx": phrase2text_idx,
                "vertex_connections": vertex_connections,
                "head_idx2phrase": head_idx2phrase,
                "idx2vert": idx2vert,
                "vert2idx": vert2idx,
            }
        )
    featuresdata.close()

    with open(output_file, "wb") as f:
        pickle.dump(scene_tree_storage, f)


def find_node_distance(
    i: int, vertex_connections: Dict[int, List[int]], target: Optional[int] = None
):
    """
    Finds the distances from one node to all others.
    Args:
        i: starting vertex idx
        vertex_connections: expects the vertices to be in a range starting from zero.
        target:
    Returns: distance between vertex i and target, if target is not None.
             Otherwise the distance to every other node.
    """
    # visited[n] for keeping track of visited node in BFS
    visited = [False] * len(vertex_connections.keys())
    # Initialize distances as 0
    distance = [0] * len(vertex_connections.keys())
    # queue to do BFS
    queue_: queue.Queue = queue.Queue()
    distance[i] = 0
    queue_.put(i)
    visited[i] = True
    while not queue_.empty():
        x = queue_.get()
        for i in range(len(vertex_connections[x])):
            if visited[vertex_connections[x][i]]:
                continue
            # update distance for i
            distance[vertex_connections[x][i]] = distance[x] + 1
            queue_.put(vertex_connections[x][i])
            visited[vertex_connections[x][i]] = True
    if target is not None:
        return distance[target]
    else:
        return distance


def _edges2vertex_connections(
    edges: List[Tuple[int, int]], zero_range_verts: bool = False
) -> Tuple[DefaultDict[int, List[int]], Optional[List[int]], Optional[Dict[int, int]]]:
    """
    Changes list of edges into a graph. For each vertex a list of the connected vertices
    Args:
        edges:
        zero_range_verts: If true, all the vertices will become
                          enumerated starting from zero. Needed for BFS.
    Returns: The graph dict with all vertex connections.
             If zero_range_verts is true, also returns the vertices and the mapping
    """
    vertices: List[int] = sorted(list(set(np.array(edges).flatten())))
    vert2idx = None
    if zero_range_verts:
        vert2idx = {v: idx for idx, v in enumerate(vertices)}
    vertex_connections = defaultdict(list)
    for edge in edges:
        if vert2idx is not None:
            i = vert2idx[edge[0]]
            j = vert2idx[edge[1]]
        else:
            i = edge[0]
            j = edge[1]
        vertex_connections[i].append(j)
        vertex_connections[j].append(i)
    if zero_range_verts:
        return vertex_connections, vertices, vert2idx
    return vertex_connections, None, None


def _edges2directed_edges(edges, depths):
    for i, edge in enumerate(edges):
        # If the second element, is higher in the tree (lower depth), siwtch edge around
        if depths[edge[1]] < depths[edge[0]]:
            edges[i] = (edge[1], edge[0])

