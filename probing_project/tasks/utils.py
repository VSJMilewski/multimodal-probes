from typing import List

from probing_project.utils import Observation


def create_head_indices(observation: Observation) -> List[int]:
    """
    Args:
        observation:
    Returns:
    """
    head_indices = []
    number_of_underscores = 0
    for idx, elt in enumerate(observation.head_indices):
        if elt == "_":
            head_indices.append(0)
            number_of_underscores += 1
        else:
            head_indices.append(int(elt) + number_of_underscores)
    return head_indices
