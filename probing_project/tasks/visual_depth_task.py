import logging
import os
import pickle
from typing import Dict

import h5py  # type: ignore
from probing_project.losses import L1DepthLoss
from probing_project.reporters import VisualWordReporter
from probing_project.scene_tree import create_scene_tree
from probing_project.utils import Observation
from rich.progress import track

from .depth_task import DepthTask

logger = logging.getLogger(__name__)


class VisualDepthTask(DepthTask):
    def __init__(self):
        super(VisualDepthTask, self).__init__()
        self.label_name = "parse_visual_depth_labels"
        self.loss = L1DepthLoss()
        self.reporter_class = VisualWordReporter

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
                depth_label = f_out.create_dataset(
                    self.label_name,
                    shape=(num_lines, max_label_len),
                    fillvalue=-2,
                    dtype="int",
                    chunks=(1, max_label_len),
                    compression=5,
                    shuffle=True,
                )

                for index, ob in track(
                    conllx_observations.items(), description="create VIS DEPTH"
                ):
                    tree = scene_trees[index]
                    phrase_ids = boxes_phrase_ids[index]
                    num_box = num_boxes[index]

                    # Fill distance labels
                    for idx, phrase_id in enumerate(phrase_ids[:num_box]):
                        if phrase_id == -1:
                            continue
                        elif phrase_id == 0:
                            # it is the root node (full image)
                            phr_depth = 0
                        elif str(phrase_id) not in tree["phrase2depth"].keys():
                            # it is not in the sentence. So it is not part
                            # of dependency tree, but attached to root.
                            logger.error(
                                "never expecting this case actually. What is happening!"
                            )
                            phr_depth = -1
                        else:
                            phr_depth = tree["phrase2depth"][str(phrase_id)]
                        depth_label[index, idx] = phr_depth
                    if num_box == 0:
                        logger.warning(
                            "I'm suprised. Never expected the code to reach this..."
                        )
                        depth_label[index, 0] = 0
            featuresdata.close()
        logger.info(f"All split files for label {self.label_name} ready!")
