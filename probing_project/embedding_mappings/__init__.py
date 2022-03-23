import logging
from typing import Set

from probing_project.utils import get_all_subclasses

from .base_mapping import MappingBase
from .decay_mapping import DecayMapping
from .disk_mapping import DiskMapping
from .projection_mapping import ProjectionMapping

logger = logging.getLogger(__name__)

# assumes everything inherits from TaskBase
name2subclass_map = {cls.__name__: cls for cls in get_all_subclasses(MappingBase)}


def get_mapping_class(name: str):
    if name in name2subclass_map.keys():
        return name2subclass_map[name]
    else:
        logger.error(f"Mapping class does not exist: '{name}'")
        raise ValueError(f"Mapping class does not exist: '{name}'")


def get_possible_mappings() -> Set[str]:
    return set(name2subclass_map.keys())
