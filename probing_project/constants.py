from typing import Final, Set

from probing_project.data.datasets import (
    get_possible_multimodal_datasets,
    get_possible_text_datasets,
)
from probing_project.embedding_mappings import get_possible_mappings
from probing_project.tasks import get_possible_tasks

# BERT MODEL OPTIONS
BERT_MODELS: Final[Set[str]] = {"base", "large"}

## DEFINE ALL OPTIONAL MODELS
TEXT_MODELS: Final[Set[str]] = {"BERT"}
MULTIMODAL_MODELS: Final[Set[str]] = {
    "ViLBERT",
    "LXMERT",
    "VL-BERT",
    "VisualBERT",
    "UNITER",
}
EMBEDDING_MODELS: Final[Set[str]] = set.union(TEXT_MODELS, MULTIMODAL_MODELS)

# VOLTA MULTIMODAL MODELS INFORMATION
MULTIMODAL2PRETRAINEDFILE = {
    "ViLBERT": "volta/save/aQCx8cLWK7",
    "LXMERT": "volta/save/Dp1g16DIA5",
    "VL-BERT": "volta/save/Dr8geMQyRd",
    "VisualBERT": "volta/save/GCBlzUuoJl",
    "UNITER": "volta/save/FeYIWpMSFg",
}
MULTIMODAL2CONFIGFILE = {
    "ViLBERT": "volta/config/ctrl_vilbert_base.json",
    "LXMERT": "volta/config/ctrl_lxmert.json",
    "VL-BERT": "volta/config/ctrl_vl-bert_base.json",
    "VisualBERT": "volta/config/ctrl_visualbert_base.json",
    "UNITER": "volta/config/ctrl_uniter_base.json",
}

# BASED ON EXISTING CLASSES CREATE THE OPTIONAL VALUES FOR ARGUMENTS
TEXT_DATASETS: Final[Set[str]] = get_possible_text_datasets()
MULTIMODAL_DATASETS: Final[Set[str]] = get_possible_multimodal_datasets()
DATASETS: Final[Set[str]] = set.union(TEXT_DATASETS, MULTIMODAL_DATASETS)
EMBEDDING_MAPPINGS: Final[Set[str]] = get_possible_mappings()
TASK_CLASSES: Final[Set[str]] = get_possible_tasks()
