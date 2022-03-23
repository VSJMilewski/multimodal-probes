import logging
import os
import pickle
from typing import Dict

import h5py  # type: ignore
from probing_project.losses import L1DistanceLoss
from probing_project.reporters import VisualWordPairReporter
from probing_project.scene_tree import create_scene_tree, find_node_distance
from probing_project.utils import Observation
from rich.progress import track

from .distance_task import DistanceTask

logger = logging.getLogger(__name__)


class VisualDistanceTask(DistanceTask):
    def __init__(self):
        super(VisualDistanceTask, self).__init__()
        self.label_name = "parse_visual_distance_labels"
        self.loss = L1DistanceLoss()
        self.reporter_class = VisualWordPairReporter

    def add_task_label_dataset(
        self,
        input_conllx_pkl: str,
        unformatted_output_h5: str,
        unformatted_feature_file: str,
        scene_tree_file: str,
        *_,
        **__,
    ) -> None:
        for split in ["train", "dev", "test"]:
            label_h5_file = unformatted_output_h5.format(split)
            featurefile = unformatted_feature_file.format(split)
            treefile = scene_tree_file.format(split)

            if os.path.exists(label_h5_file):
                logging.info(
                    f"File for label {self.label_name} exists for split {split}."
                )
                continue
            logger.info(f"Create label {self.label_name} file for split {split}")
            with open(input_conllx_pkl.format(split), "rb") as in_pkl:
                conllx_observations: Dict[int, Observation] = pickle.load(in_pkl)

            featuresdata = h5py.File(featurefile, "r")
            boxes_phrase_ids = featuresdata["box_phrase_id"]
            num_boxes = featuresdata["num_boxes"]

            if not os.path.exists(treefile):
                logger.info(
                    f"Scene tree file {treefile} does not exist yet, creating it."
                )
                create_scene_tree(conllx_observations, featurefile, treefile)
            with open(treefile, "rb") as f:
                scene_trees = pickle.load(f)

            num_lines = len(conllx_observations)
            max_label_len = boxes_phrase_ids.shape[1]
            with h5py.File(label_h5_file, "w", libver="latest") as f_out:
                dist_label = f_out.create_dataset(
                    self.label_name,
                    shape=(num_lines, max_label_len, max_label_len),
                    fillvalue=-1,
                    dtype="int",
                    chunks=(1, max_label_len, max_label_len),
                    compression=5,
                    shuffle=True,
                )

                for index, ob in track(
                    conllx_observations.items(), description="create VIS DIST"
                ):
                    tree = scene_trees[index]
                    phrase_ids = boxes_phrase_ids[index]
                    num_box = num_boxes[index]

                    # Fill distance labels
                    for idx, phrase_id in enumerate(phrase_ids[:num_box]):
                        if (
                            tree["phrase2text_idx"][str(phrase_id)]
                            not in tree["vert2idx"]
                        ):
                            continue
                        current_vert = tree["phrase2text_idx"][str(phrase_id)]
                        all_dists = find_node_distance(
                            tree["vert2idx"][current_vert], tree["vertex_connections"]
                        )
                        for dist_idx, dist in enumerate(all_dists):
                            target_phrase_id = tree["head_idx2phrase"][
                                tree["idx2vert"][dist_idx]
                            ]
                            target_indices = [
                                x
                                for x, p in enumerate(phrase_ids[:num_box])
                                if str(p) == target_phrase_id
                            ]
                            dist_label[index, idx, target_indices] = dist
            featuresdata.close()
        logger.info(f"All split files for label {self.label_name} ready!")
