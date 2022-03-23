import base64
import csv
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional, Union

import h5py  # type: ignore
import numpy as np
import torch
from probing_project.constants import (
    MULTIMODAL2CONFIGFILE,
    MULTIMODAL2PRETRAINEDFILE,
    MULTIMODAL_DATASETS,
    MULTIMODAL_MODELS,
)
from probing_project.data.datasets import DatasetBase
from probing_project.data.probing_dataset import ProbingDataset
from probing_project.scene_tree import create_scene_tree
from probing_project.tasks import TaskBase
from probing_project.utils import track
from pytorch_pretrained_bert import BertModel, BertTokenizer  # type: ignore

sys.path.append("../volta")
sys.path.append("volta")
from volta.config import BertConfig  # type: ignore
from volta.encoders import BertForVLPreTraining  # type: ignore

from .base_module import ProbingModuleBase

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
csv.field_size_limit(sys.maxsize)


class ProbingModuleMultimodal(ProbingModuleBase):
    def __init__(self, only_text, *args, **kwargs) -> None:
        """
        Args:
            multimodal_model: Which multimodal should be used.
                              This is based on VOLTA and what is implemented there.
        """
        self.only_text = only_text
        super().__init__(*args, **kwargs)

        # when using multimodal embeddings, make sure a model is set. Also load the config
        self.mm_map_bert_layer = True
        assert self.embeddings_model in MULTIMODAL_MODELS, (
            "when the embeddings_model is set to 'multimodal_bert', "
            "make sure that the multimodal_model is set."
        )
        self.mm_config = BertConfig.from_json_file(
            MULTIMODAL2CONFIGFILE[self.embeddings_model]
        )

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Checks if all necesary data is created and downloaded.
        If something is missing, it will create it. If it is some file that cannot
        be created by the datamodule, it will provide instructions on what to do.
        """
        super().prepare_data(*args, **kwargs)

        for split in ['train', 'dev', 'test']:
            scene_tree_file = self.unformatted_scene_tree_file.format(split)
            if os.path.exists(scene_tree_file):
                logger.info(f"The intermediate scene tree file "
                            f"{scene_tree_file} already exists.")
                continue
            logger.info(f"creating scene trees for split {split}")
            with open(self.unformatted_conllx_pickle.format(split), 'rb') as f:
                observations = pickle.load(f)
            create_scene_tree(
                observations,
                self.unformatted_emb_h5_file.format(split),
                self.unformatted_scene_tree_file.format(split),
            )
        logger.info("All scene tree files ready!")

        # CHECK FOR LABEL HDF5
        self._create_label_h5(
            task=self.task,
            input_conllx_pkl=self.unformatted_conllx_pickle,
            output_file=self.unformatted_label_h5_file,
            unformatted_feature_file=self.unformatted_emb_h5_file,
            scene_tree_file=self.unformatted_scene_tree_file,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setups the datamodule with datasets for the different splits
        Args:
            stage: If 'None', load train, val, and test split.
                   If 'fit', load only the train and val split.
                   if 'test', load only the test split.
        Returns: None
        """
        logger.debug("Setting up Data")
        assert (
            self.task is not None
        ), "Preparing can be done without task, for setup it must be set"
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = ProbingDataset(
                h5_embeddings_file=self.unformatted_emb_h5_file.format("train"),
                h5_labels_file=self.unformatted_label_h5_file.format("train"),
                h5_label_length_file=self.unformatted_length_label_h5_file.format(
                    "train"
                ),
                mm_vision_layer="Visual" in self.task.__class__.__name__,
                layer_idx=self.bert_model_layer,
                task=self.task,
                mm_bert_layer=self.mm_map_bert_layer,
                mm_config=self.mm_config,
            )
            self.val_dataset = ProbingDataset(
                h5_embeddings_file=self.unformatted_emb_h5_file.format("dev"),
                h5_labels_file=self.unformatted_label_h5_file.format("dev"),
                h5_label_length_file=self.unformatted_length_label_h5_file.format(
                    "dev"
                ),
                mm_vision_layer="Visual" in self.task.__class__.__name__,
                layer_idx=self.bert_model_layer,
                task=self.task,
                sentence_file=self.unformatted_conllx_pickle.format("dev"),
                mm_bert_layer=self.mm_map_bert_layer,
                mm_config=self.mm_config,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ProbingDataset(
                h5_embeddings_file=os.path.join(
                    self.unformatted_emb_h5_file.format("test")
                ),
                h5_labels_file=self.unformatted_label_h5_file.format("test"),
                h5_label_length_file=self.unformatted_length_label_h5_file.format(
                    "test"
                ),
                mm_vision_layer="Visual" in self.task.__class__.__name__,
                layer_idx=self.bert_model_layer,
                task=self.task,
                sentence_file=self.unformatted_conllx_pickle.format("test"),
                mm_bert_layer=self.mm_map_bert_layer,
                mm_config=self.mm_config,
            )

    def _setup_data_specific_dirs_and_dirs(self):
        super()._setup_data_specific_dirs_and_dirs()
        self.unformatted_scene_tree_file = os.path.join(
            self.general_intermediate_path, self.dataset.filename + "-{}.tree.pkl"
        )
        if self.only_text:
            self.unformatted_emb_h5_file = (
                os.path.splitext(self.unformatted_emb_h5_file)[0] + ".only_text.h5"
            )

    def _check_and_create_embedding_features_file(self):
        logger.info("Checking files for multimodal DataModule")
        if self.dataset.__class__.__name__ in MULTIMODAL_DATASETS:
            with open(self.dataset.id_mappings_file, "rb") as id_file:
                id2cap_id_splits = pickle.load(id_file)
        else:
            id2cap_id_splits = None

        for split in ["train", "dev", "test"]:
            # check if the saved embeddings hdf5 files exist.
            # Otherwise extract the embeddings. TAKES LONG!
            outfile = self.unformatted_emb_h5_file.format(split)
            if not os.path.isfile(outfile):
                logger.info("Creating embeddings h5 file for split {}".format(split))
                ProbingModuleMultimodal._convert_conllx_to_bert(
                    input_conllx_pkl=self.unformatted_conllx_pickle.format(split),
                    output_file=outfile,
                    model_name=self.model_name,
                    bert_cased=self.bert_cased,
                    bert_model=self.bert_model,
                    multimodal_model=self.embeddings_model,
                    multimodal_config_file=MULTIMODAL2CONFIGFILE[self.embeddings_model],
                    multimodal_data=self.dataset.__class__.__name__
                    in MULTIMODAL_DATASETS,
                    dataset=self.dataset,
                    features_tsv=self.dataset.img_feature_tsv_file.format(split)
                    if hasattr(self.dataset, "img_feature_tsv_file")
                    else None,
                    idx2cap_id=id2cap_id_splits[split]
                    if id2cap_id_splits is not None
                    else id2cap_id_splits,
                    img_datafile=self.dataset.img_datafile,
                    sent_datafile=self.dataset.sent_datafile,
                    split=split,
                    only_text=self.only_text,
                )
            else:
                logger.debug(
                    f"embeddings file '{os.path.basename(outfile)}' already exist."
                )
        logger.info("All embedding h5 files ready!")

    @staticmethod
    def _convert_conllx_to_bert(
        input_conllx_pkl: str,
        output_file: str,
        model_name: str,
        bert_cased: bool,
        bert_model: str,
        multimodal_model: str,
        multimodal_config_file: str,
        multimodal_data: bool,
        dataset: DatasetBase,
        features_tsv: str = None,
        idx2cap_id: List[str] = None,
        img_datafile: str = None,
        sent_datafile: str = None,
        split: str = "train",
        only_text=False,
    ) -> None:
        """
        # Converts the raw file into a HDF5 file with embeddings create by
        BERT and BERT-Tokenizer. Crucially, do not do basic tokenization;
        PTB is tokenized. Just do wordpiece tokenization.
        Args:
            input_conllx_pkl: The filepath to the conllx file
            output_file: the filepath for where to store the hdf5 features
            model_name: the complete model_name as defined by 'pretrained_transformers'
            bert_cased: If the model should use cased (True) tokens or uncased
                        (False) tokens, default = False
            bert_model: whether to use BERT 'base' or 'large'
            multimodal_model: Which multimodal bert from VOLTA to use
            multimodal_config_file: path to the config for the chosen multimodal model.
            multimodal_data: Boolean if the dataset contains multimodal image data
            dataset: the dataset class
            features_tsv: path to the precomputed image features in a TSV
            idx2cap_id: mappings for index too caption ids
            img_datafile: path to pickle with a dict for all image data annotations
            sent_datafile: path to pickle with a dict for all caption data annotations
            split: which split we are currently processing
        Returns: None
        """
        dataset_name = dataset.__class__.__name__
        logger.info(
            f"from: {os.path.basename(input_conllx_pkl)}, "
            f"creating hdf5: {'/'.join(output_file.split('/')[-2:])}"
        )
        logger.info(f"making use of the bert model: {model_name.lower()}")
        tokenizer = BertTokenizer.from_pretrained(
            model_name.lower(),
            do_lower_case=not bert_cased,
            cache_dir="../pytorch_pretrained_bert/",
        )
        layer_count, feature_count = ProbingModuleMultimodal.get_bert_info(bert_model)
        config = BertConfig.from_json_file(multimodal_config_file)
        config.visual_target_weights = {}
        assert layer_count == config.num_attention_heads, (
            "Number of bert layers must match the heads in the volta config. "
            "Make sure you use bert-base, as this is the only version on volta."
        )
        assert feature_count == config.hidden_size, (
            "The feature count must match the hidden_size in the volta config. "
            "Make sure you use bert-base, as this is the only version on volta."
        )
        model = BertForVLPreTraining.from_pretrained(
            MULTIMODAL2PRETRAINEDFILE[multimodal_model],
            config=config,
            cache_dir="../pytorch_pretrained_bert/",
        )

        def forward_hook(module, input_, output):
            nonlocal baseline_embs
            baseline_embs = output

        def vis_forward_hook(module, input_, output):
            nonlocal vis_baseline_embs
            vis_baseline_embs = output

        _ = model.bert.embeddings.register_forward_hook(forward_hook)
        if multimodal_model != "UNITER":
            _ = model.bert.v_embeddings.register_forward_hook(vis_forward_hook)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            model = model.cuda()

        # find some statistics for creating the h5 datasets
        with open(input_conllx_pkl, "rb") as in_pkl:
            conllx_observations: dict = pickle.load(in_pkl)
        num_lines = len(conllx_observations)
        max_label_len = max(len(ob.index) for ob in conllx_observations.values())

        if dataset_name == "Flickr30k":
            (
                csv_rows,
                image_data,
                image_id2idx,
                max_num_boxes,
            ) = ProbingModuleMultimodal._open_image_data(
                dataset, features_tsv, idx2cap_id, img_datafile, multimodal_data, split
            )
        else:
            # the dummy full image region
            max_num_boxes = 1

        # start creating the hdf5
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w", libver="latest") as f_out:
            vision_layer_sets = {}
            text_layer_sets = {}
            img_id_data = f_out.create_dataset(
                name="image_id", shape=(num_lines,), fillvalue=0, dtype="i8"
            )
            box_data = f_out.create_dataset(
                name="boxes",
                shape=(num_lines, max_num_boxes, 4),
                fillvalue=0,
                dtype="i8",
                chunks=(1, max_num_boxes, 4),
                compression=9,
                shuffle=True,
            )
            num_box_data = f_out.create_dataset(
                name="num_boxes", shape=(num_lines,), fillvalue=0, dtype="i8"
            )
            phrase_id_data = f_out.create_dataset(
                name="box_phrase_id",
                shape=(num_lines, max_num_boxes),
                fillvalue=-1,
                dtype="i8",
            )

            for set_idx, layer in enumerate(config.t_ff_sublayers):
                text_layer_sets[set_idx] = f_out.create_dataset(
                    name="text" + str(layer),
                    shape=(num_lines, max_label_len, feature_count),
                    fillvalue=0,
                    dtype="f",
                    chunks=(1, max_label_len, feature_count),
                    compression=9,
                    shuffle=True,
                )
            for set_idx, layer in enumerate(config.v_ff_sublayers):
                vision_layer_sets[set_idx] = f_out.create_dataset(
                    name="vision" + str(layer),
                    shape=(num_lines, max_num_boxes, config.v_hidden_size),
                    fillvalue=0,
                    dtype="f",
                    chunks=(1, max_num_boxes, config.v_hidden_size),
                    compression=9,
                    shuffle=True,
                )
            text_baseline = f_out.create_dataset(
                name="text_baseline",
                shape=(num_lines, max_label_len, feature_count),
                fillvalue=0,
                dtype="f",
                chunks=(1, max_label_len, feature_count),
                compression=9,
                shuffle=True,
            )
            vision_baseline = f_out.create_dataset(
                name="vision_baseline",
                shape=(num_lines, max_num_boxes, config.v_feature_size),
                fillvalue=0,
                dtype="f",
                chunks=(1, max_num_boxes, config.v_hidden_size),
                compression=9,
                shuffle=True,
            )
            vision_mapped_baseline = f_out.create_dataset(
                name="vision_mapped_baseline",
                shape=(num_lines, max_num_boxes, config.v_hidden_size),
                fillvalue=0,
                dtype="f",
                chunks=(1, max_num_boxes, config.v_hidden_size),
                compression=9,
                shuffle=True,
            )
            # start filling the hdf5
            with torch.no_grad():
                dummy_img_feat = torch.zeros(1, 1, config.v_feature_size, device=device)
                dummy_img_loc = torch.zeros(
                    1, 1, config.num_locs, dtype=torch.float, device=device
                )
                dummy_img_mask = torch.zeros(1, 1, dtype=torch.long, device=device)
            for index, ob in track(
                conllx_observations.items(), description="raw2mmbert"
            ):
                baseline_embs: Union[torch.Tensor, None] = None
                vis_baseline_embs: Union[torch.Tensor, None] = None
                line = "[CLS] " + " ".join(ob.sentence) + " [SEP]"
                tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

                if multimodal_data and not only_text:
                    if dataset_name == "Flickr30k":
                        assert (
                            csv_rows and image_id2idx
                        ), "something went wrong in reading the csv/tsv"
                        assert (
                            idx2cap_id is not None
                        ), "idx2cap_id not should not be None here"
                        cap_id = idx2cap_id[index]
                        img_id = cap_id.split("_")[0]
                        if img_id not in image_id2idx.keys():
                            logger.warning(
                                "Somehow the image was not prepared into the csv/tsv.... {}".format(
                                    img_id
                                )
                            )
                            continue
                        item = csv_rows[image_id2idx[img_id]]
                        assert img_id == item["img_id"]
                        img_anns = image_data[cap_id]
                        num_boxes = np.frombuffer(
                            base64.b64decode(item["num_boxes"]), dtype=np.int  # type: ignore
                        )
                        tot_boxes = num_boxes.sum() + 1
                        # set mask
                        img_mask = torch.ones(
                            1, tot_boxes, dtype=torch.long, device=device
                        )
                        # create img features
                        try:
                            tmp = np.frombuffer(
                                base64.b64decode(item["image_feature"]),
                                dtype=np.float32,
                            ).reshape(-1, 2048)
                        except ValueError:
                            logger.warning(
                                f"Weird issue when reading image into np array. "
                                f"Buffer size not multiple of element size. "
                                f"Skipping image! {img_id}"
                            )
                            continue
                        full_img_feat = torch.tensor(
                            tmp.copy(), dtype=torch.float, device=device
                        )
                        # collect boxes
                        full_loc = torch.tensor(
                            [[0, 0, int(item["img_h"]), int(item["img_w"])]],
                            dtype=torch.float,
                            device=device,
                        )
                        if tot_boxes > 1:
                            tmp = np.frombuffer(
                                base64.b64decode(item["features"]), dtype=np.float32
                            ).reshape(-1, 2048)
                            img_features = torch.tensor(
                                tmp.copy(), dtype=torch.float, device=device
                            )
                            img_features = torch.cat(
                                (full_img_feat, img_features), dim=0
                            ).unsqueeze(0)
                            tmp = np.frombuffer(
                                base64.b64decode(item["boxes"]),
                                dtype=np.int,  # type:ignore
                            ).reshape(-1, 4)
                            img_loc = torch.tensor(
                                tmp.copy(), dtype=torch.float, device=device
                            )
                            img_loc = torch.cat((full_loc, img_loc), dim=0).unsqueeze(0)
                        else:
                            img_features = full_img_feat.unsqueeze(0)
                            img_loc = full_loc.unsqueeze(0)
                        # if the model expects 5 dimensional boxes,
                        # we create the 5th value as in Volta.
                        if config.num_locs == 5:
                            tmp = (
                                (img_loc[:, :, 3] - img_loc[:, :, 1])
                                * (img_loc[:, :, 2] - img_loc[:, :, 0])
                                / (float(item["img_w"]) * float(item["img_h"]))
                            ).unsqueeze(-1)
                            img_loc = torch.cat((img_loc, tmp), dim=-1)
                        # Fill ready data for visual datasets
                        num_box_data[index] = tot_boxes
                        box_data[index, :tot_boxes, :4] = (
                            img_loc[0, :, :4].detach().int().cpu().numpy()
                        )
                        phrase_id_data[index, 0] = 0
                        img_id_data[index] = int(img_id)
                        # collect the phrase_ids for each of the boxes.
                        # If no ID, we set to zero
                        for ix in range(1, tot_boxes):
                            phrase_id = 0
                            box = list(img_loc[0, ix, :4].detach().int().cpu().numpy())
                            for id_, boxlist in img_anns["boxes"].items():
                                if box in boxlist:
                                    phrase_id = int(id_)
                                    break
                            phrase_id_data[index, ix] = phrase_id
                    else:
                        logger.error(
                            f"Uknown dataset {dataset_name}. "
                            f"Known multimodal datasets are {MULTIMODAL_DATASETS}."
                        )
                        raise ValueError(f"Uknown dataset {dataset_name}.")
                else:
                    img_features = dummy_img_feat
                    img_loc = dummy_img_loc
                    img_mask = dummy_img_mask

                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([indexed_tokens], device=device)
                with torch.no_grad():
                    (
                        prediction_scores_t,
                        prediction_scores_v_dict,
                        seq_relationship_score,
                        all_attention_mask,
                        pooled_output,
                        encoded_layers_t,
                        encoded_layers_v,
                    ) = model(
                        input_ids=tokens_tensor,  # the ids for the sentence tokens
                        image_feat=img_features,  # features for the image.
                        image_loc=img_loc,  # the spatials, like position embs of text
                        token_type_ids=None,  # ids for the special tokens. default is set to zeros
                        attention_mask=None,  # default is set to ones
                        image_attention_mask=img_mask,  # default is set to ones
                        output_all_encoded_layers=True,
                        output_all_attention_masks=False,
                    )
                    if multimodal_model == "UNITER":
                        assert (
                            baseline_embs is not None
                        ), "baseline_embs should have been set, check forward_hook"
                        text_baseline_embs, mapped_vision_baseline_embs = baseline_embs
                    else:
                        assert (
                            baseline_embs is not None
                        ), "baseline_embs should have been set, check forward_hook"
                        assert (
                            vis_baseline_embs is not None
                        ), "vis_baseline_embs should have been set, check forward_hook"
                        text_baseline_embs = baseline_embs
                        mapped_vision_baseline_embs = vis_baseline_embs

                # place output into the datasets per layer
                # we start from one, since the encoded layers have the initial embedding prepended,
                # which we are not interested in
                untokenized_sent = ob.sentence
                untok_tok_mapping = (
                    ProbingModuleMultimodal._match_tokenized_to_untokenized(
                        tokenized_text, untokenized_sent
                    )
                )
                # FIRST THE TEXT LAYERS
                for set_idx, layer in enumerate(config.t_ff_sublayers):
                    single_layer_features = (
                        encoded_layers_t[layer].squeeze(0).detach().cpu().numpy()
                    )
                    # WE WANT TO PROBE FOR THE TREE FOR COMPLETE WORDS.
                    # THEREFORE WE FIND HOW MANY SUBTOKENS MAP TO COMPLETE WORDS.
                    # NOW WE TAKE THE AVERAGE OF THE EMBEDDINGS FOR THE SUBTOKENS,
                    # TO GET A SINGLE EMBEDDING FOR EVERY WORD.
                    for token_idx in range(len(untokenized_sent)):
                        text_layer_sets[set_idx][index, token_idx, :] = np.mean(
                            single_layer_features[
                                untok_tok_mapping[token_idx][0] : untok_tok_mapping[
                                    token_idx
                                ][-1]
                                + 1,
                                :,
                            ],
                            axis=0,
                        )
                text_baseline_embs = (
                    text_baseline_embs.squeeze(0).detach().cpu().numpy()
                )
                for token_idx in range(len(untokenized_sent)):
                    text_baseline[index, token_idx, :] = np.mean(
                        text_baseline_embs[
                            untok_tok_mapping[token_idx][0] : untok_tok_mapping[
                                token_idx
                            ][-1]
                            + 1,
                            :,
                        ],
                        axis=0,
                    )

                # NEXT THE VISION LAYERS
                for set_idx, layer in enumerate(config.v_ff_sublayers):
                    single_layer_features = (
                        encoded_layers_v[layer].squeeze(0).detach().cpu().numpy()
                    )
                    vision_layer_sets[set_idx][
                        index, : single_layer_features.shape[0], :
                    ] = single_layer_features
                # faster rcnn visual features
                vision_baseline[index, : img_features.shape[1], :] = (
                    img_features.squeeze(0).detach().cpu().numpy()
                )
                # save mapped vision embeddings
                mapped_vision_baseline_embs = (
                    mapped_vision_baseline_embs.squeeze(0).detach().cpu().numpy()
                )
                vision_mapped_baseline[
                    index, : mapped_vision_baseline_embs.shape[0], :
                ] = mapped_vision_baseline_embs

    @staticmethod
    def _open_image_data(
        dataset, features_tsv, idx2cap_id, img_datafile, multimodal_data, split
    ):
        dataset_name = dataset.__class__.__name__
        max_num_boxes = 1
        image_data = None
        csv_rows = []
        image_id2idx: Dict[str, int] = {}
        if not multimodal_data:
            return csv_rows, image_data, image_id2idx, max_num_boxes
        else:
            if dataset_name == "Flickr30k":
                if features_tsv is not None and os.path.exists(features_tsv):
                    with open(features_tsv, "r") as tsvfile:
                        reader = csv.DictReader(
                            tsvfile, delimiter="\t", fieldnames=dataset.tsv_fieldnames
                        )
                        for item in reader:
                            image_id2idx[item["img_id"]] = len(image_id2idx)
                            csv_rows.append(item)
                            nb = (
                                np.frombuffer(
                                    base64.b64decode(item["boxes"]), dtype=np.int
                                )
                                .reshape(-1, 4)
                                .shape[0]
                            )
                            max_num_boxes = max(
                                max_num_boxes, nb + 1
                            )  # we always at the full image as root box
                else:
                    logger.error(
                        "To use the Flickr30K dataset and to prepare its datafiles, "
                        "you must first extract the object features and give the path "
                        "to the tsv file. \n\t"
                        "- Follow instructions in 'data/README.md' to setup docker.\n\t"
                        "- Run the script 'extract_flickr30k_images.py'"
                    )
                    raise FileNotFoundError(features_tsv)
                if img_datafile is not None and os.path.exists(img_datafile):
                    with open(img_datafile, "rb") as imgfile:
                        image_data = pickle.load(imgfile)[split]
                else:
                    logger.error(
                        "To use the Flickr30K dataset and to prepare its embeddings. "
                        "First collect annotations and pass along path to the pickle."
                    )
                    raise FileNotFoundError
            else:
                logger.error(
                    f"Uknown image dataset {dataset_name}. "
                    f"Known datasets are 'Flickr30k'."
                )
                raise ValueError
            return csv_rows, image_data, image_id2idx, max_num_boxes

    # LABELS COMPUTE
    def _create_label_h5(
        self,
        task: TaskBase,
        input_conllx_pkl: str,
        output_file: str,
        unformatted_feature_file: str,
        scene_tree_file: str,
        *_,
        **__,
    ) -> None:
        super()._create_label_h5(task, input_conllx_pkl, output_file)
        task.add_task_label_dataset(
            input_conllx_pkl=input_conllx_pkl,
            unformatted_output_h5=output_file,
            unformatted_feature_file=unformatted_feature_file,
            scene_tree_file=scene_tree_file,
        )
