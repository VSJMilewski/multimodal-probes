import logging
from typing import Set

from probing_project.utils import get_all_subclasses

from .base_task import TaskBase
from .depth_task import DepthTask
from .distance_task import DistanceTask
from .edge_position_control_task import EdgePositionControlTask
from .edge_position_task import EdgePositionTask
from .pos_control_task import POSControlTask
from .pos_task import POSTask
from .visual_depth_task import VisualDepthTask
from .visual_distance_task import VisualDistanceTask

logger = logging.getLogger(__name__)

# assumes everything inherits from TaskBase
name2subclass_map = {cls.__name__: cls for cls in get_all_subclasses(TaskBase)}
name2subclass_map[TaskBase.__name__] = TaskBase


def get_task_class(name: str):
    if name in name2subclass_map:
        return name2subclass_map[name]
    else:
        logger.error(f"Task class does not exist: {name}")
        raise ValueError(f"Task class does not exist: {name}")


def get_possible_tasks() -> Set[str]:
    return set(name2subclass_map.keys())
