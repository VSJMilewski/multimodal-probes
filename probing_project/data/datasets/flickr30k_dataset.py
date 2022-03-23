import logging
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, Union

from probing_project.utils import track
from spacy import Language
from spacy_conll import init_parser  # type: ignore

from .base_dataset import MultimodalDatasetBase

logger = logging.getLogger(__name__)


class Flickr30k(MultimodalDatasetBase):
    def __init__(self, raw_dataset_path=None, intermediate_path=None):
        super(Flickr30k, self).__init__()
        self.filename = "flickr30k-entities"
        self.raw_dataset_path = raw_dataset_path
        self.intermediate_path = intermediate_path
        self.id_mappings_file = os.path.join(
            intermediate_path, self.filename + ".id_mappings.pkl"
        )
        self.img_feature_tsv_file = os.path.join(
            intermediate_path, self.filename + "-{}.obj_features.tsv"
        )
        self.img_datafile = os.path.join(
            self.intermediate_path, self.filename + ".img_data.pkl"
        )
        self.sent_datafile = os.path.join(
            self.intermediate_path, self.filename + ".sent_data.pkl"
        )

        self.tsv_fieldnames = [
            "img_id",
            "img_h",
            "img_w",
            "no_box",
            "scene",
            "box_identifiers",
            "num_boxes",
            "boxes",
            "features",
            "image_feature",
        ]

        assert all(
            os.path.isfile(self.img_feature_tsv_file.format(sp))
            for sp in ["train", "dev", "test"]
        ), (
            "To use the visual dataset and to prepare its datafiles, "
            "you must first extract the object features. \n\t"
            "- Follow the instructions in 'data/README.md' to setup docker.\n\t"
            "- Run the needed script. e.g. for flickr30k run 'extract_flickr30k_images.py'"
        )

    def prepare_intermediate_data(self, unformat_conllx_file):
        """
        When embeddings data must me prepared, we first need to precompute the raw data
        Returns:
        """
        assert (
            self.raw_dataset_path is not None
        ), "When needed to use the dataset, make sure to set the 'raw_dataset_path'"
        assert (
            self.intermediate_path is not None
        ), "When needed to use the dataset, make sure to set the 'intermediate_path'"
        test_files = [
            unformat_conllx_file.format(split) for split in ["train", "dev", "test"]
        ]
        if not (
            os.path.isfile(test_files[0])
            and os.path.isfile(test_files[1])
            and os.path.isfile(test_files[2])
        ):
            logger.info("creating intermediate conllx files")
            self._convert_flickr30k_splits_to_conllx(
                self.raw_dataset_path, unformat_conllx_file, self.id_mappings_file
            )

        if not (
            os.path.isfile(self.img_datafile) and os.path.isfile(self.sent_datafile)
        ):
            logger.info("creating intermediate flickr30k object and caption pickles")
            self._create_object_and_caption_annotations(
                id_mappings_file=self.id_mappings_file,
                dataset_path=self.raw_dataset_path,
                output_img_data_file=self.img_datafile,
                output_sent_data_file=self.sent_datafile,
            )
        logger.info("All intermediate files ready!")

    def _convert_flickr30k_splits_to_conllx(
        self, dataset_path: str, output_conllx_file: str, id_mappings_file: str
    ) -> None:
        """
        using splits from Flickr30k-Entities. Expectis all Flickr30 zips pre-extracted.
        Args:
            dataset_path: Path to the root dir of the Flickr30k dataset.
            output_conllx_file: Where to store the converted conllx formated tree files.
            id_mappings_file:
        Returns: None
        """
        #  create train/dev/test split
        img_dir = os.path.join(dataset_path, "flickr30k-images")
        ent_dir = os.path.join(dataset_path, "flickr30k_entities")
        assert os.path.isdir(img_dir), (
            "Make sure to download the flickr30k images, and extract the zip. "
            "Get the download link by filling out the form at the bottom of "
            "the following page:\n\thttp://shannon.cs.illinois.edu/DenotationGraph/"
        )
        assert os.path.isdir(ent_dir), (
            "Make sure to clone the flickr30k_entities repo from:"
            "\n\thttps://github.com/BryanPlummer/flickr30k_entities"
        )
        assert os.path.isdir(os.path.join(ent_dir, "Annotations")) and os.path.isdir(
            os.path.join(ent_dir, "Sentences")
        ), "Make sure to extract the flickr30k_entities annotations zip in the repo."

        os.makedirs(os.path.dirname(output_conllx_file), exist_ok=True)
        captions_in_splits = self._extract_spacy_conllx(ent_dir, output_conllx_file)
        with open(id_mappings_file, "wb") as f_map:
            pickle.dump(captions_in_splits, f_map)
        logger.info(
            "Images in Flickr30k train/dev/test splits: {}/{}/{}".format(
                len(captions_in_splits["train"]),
                len(captions_in_splits["dev"]),
                len(captions_in_splits["test"]),
            )
        )

    @staticmethod
    def _extract_spacy_conllx(
        ent_dir: Union[str, os.PathLike], output_conllx_file: str
    ):
        captions_in_splits = defaultdict(list)
        input_sentences_dir = os.path.join(ent_dir, "Sentences")
        all_conllx_files = os.path.join(
            os.path.dirname(output_conllx_file), "spacy_conllx"
        )
        if not os.path.isdir(all_conllx_files):
            logger.info(
                "Converting the flickr30k sentences to conll format using spacy..."
            )
            os.makedirs(all_conllx_files, exist_ok=True)
            Flickr30k._flickr30k_sentences2spacy_conllx(
                input_sentences_dir, all_conllx_files
            )
        else:
            logger.info(
                f"Folder for spacy generated conll files already exist. Assuming the "
                f"files exist as well. If this is not the case (and you get errors), "
                f"delete the folder {all_conllx_files}"
            )
        split2imgid = defaultdict(list)
        for split, file in zip(
            ["train", "dev", "test"], ["train.txt", "val.txt", "test.txt"]
        ):
            with open(os.path.join(ent_dir, file), "r") as f:
                for line in f:
                    split2imgid[split].append(int(line.strip()))
        for split in ["train", "dev", "test"]:
            with open(output_conllx_file.format(split), "w") as f_out:
                for im_id in track(
                    split2imgid[split],
                    description="extr. spacy conllx - {}".format(split),
                ):
                    # process all the needed captions for the image
                    for cap_file in os.listdir(
                        os.path.join(all_conllx_files, str(im_id))
                    ):
                        with open(
                            os.path.join(all_conllx_files, str(im_id), cap_file), "r"
                        ) as con_in:
                            for line in con_in:
                                f_out.write(line)
                            f_out.write("\n")
                        captions_in_splits[split].append(os.path.splitext(cap_file)[0])
        return captions_in_splits

    @staticmethod
    def _flickr30k_sentences2spacy_conllx(
        input_sentences_dir: str, output_conllx_dir: str
    ):
        @Language.component("force_single_sentence")
        def one_sentence_per_doc(doc):
            doc[0].sent_start = True
            for i in range(1, len(doc)):
                doc[i].sent_start = False
            return doc

        nlp = init_parser("en_core_web_trf", "spacy")
        nlp.add_pipe("force_single_sentence", before="parser")
        for file in track(
            os.listdir(input_sentences_dir), description="flickr sent2conll"
        ):
            img_id, file_ext = os.path.splitext(file)
            if file_ext == ".txt":
                with open(os.path.join(input_sentences_dir, file), "r") as f_in:
                    captions = f_in.readlines()
                save_dir = os.path.join(output_conllx_dir, img_id)
                os.makedirs(save_dir, exist_ok=True)
                for cap_idx, caption in enumerate(captions, start=1):
                    tokens = []
                    phrase_ids = []
                    phrase_id = "_"
                    # convert caption into list of tokens and list of phrase ids
                    for tok in caption.split():
                        if tok.startswith("["):
                            phrase_id = tok.split("/")[1].split("#")[1]
                        elif tok.endswith("]"):
                            # use spacy tokenizer on each token for
                            # consistency when dependency parsing
                            for t in nlp.tokenizer(tok[:-1]):
                                tokens.append(str(t))
                                phrase_ids.append(phrase_id)
                            phrase_id = "_"
                        else:
                            # use spacy tokenizer on each token for
                            # consistency when dependency parsing
                            for t in nlp.tokenizer(tok):
                                tokens.append(str(t))
                                phrase_ids.append(phrase_id)
                    caption = " ".join(tokens)
                    save_file = os.path.join(
                        save_dir, "{}_{}.conll".format(img_id, cap_idx)
                    )
                    doc = nlp(caption)
                    assert (
                        len(list(doc.sents)) == 1
                    ), "somehow multiple sentences created by spacy"
                    if len(doc) != len(phrase_ids):
                        logger.warning(
                            f"number of tokens should be equal to phrase_id sequence.\n"
                            f"It went wrong for image_id {img_id} with caption_id "
                            f"{cap_idx}.\n"
                            f"Lengths are {len(tokens)}/{len(phrase_ids)}/{len(doc)} "
                            f"for tokens/phrase_ids/doc.\n"
                            f"Skipping the caption!"
                        )
                        continue
                    with open(save_file, "w") as f_out:
                        for token, ph_id in zip(doc, phrase_ids):
                            f_out.write(token._.conll_str.strip() + "\t" + ph_id + "\n")

    @staticmethod
    def _create_object_and_caption_annotations(
        id_mappings_file: str,
        dataset_path: str,
        output_img_data_file: str,
        output_sent_data_file: str,
    ):
        img_dir = os.path.join(dataset_path, "flickr30k-images")
        ent_dir = os.path.join(dataset_path, "flickr30k_entities")
        ann_dir = os.path.join(ent_dir, "Annotations")
        sen_dir = os.path.join(ent_dir, "Sentences")
        sys.path.insert(0, ent_dir)
        from flickr30k_entities_utils import (  # type: ignore
            get_annotations,  # type: ignore
            get_sentence_data,  # type: ignore
        )

        assert os.path.isdir(img_dir), (
            "Make sure to download the flickr30k images, and extract the zip. "
            "Get the download link by through the form at the bottom of this page:"
            "\n\thttp://shannon.cs.illinois.edu/DenotationGraph/"
        )
        assert os.path.isdir(ent_dir), (
            "Make sure to clone the flickr30k_entities repo from:\n"
            "\thttps://github.com/BryanPlummer/flickr30k_entities"
        )
        assert os.path.isdir(ann_dir) and os.path.isdir(
            sen_dir
        ), "Make sure to extract the flickr30k_entities annotations zip in the repo."
        with open(id_mappings_file, "rb") as f_in:
            split_to_caption_ids = pickle.load(f_in)

        capid2img_data: Dict[str, dict] = {}
        capid2sent_data: Dict[str, dict] = {}
        for split in split_to_caption_ids.keys():
            capid2img_data[split] = {}
            capid2sent_data[split] = {}
            prev_img_id = -1
            img_sentences_data = None
            img_annotation_data = None
            for cap_id in track(
                split_to_caption_ids[split],
                description="obj+cap annotation - {}".format(split),
            ):
                img_id, cap_num = (int(i) for i in cap_id.split("_"))
                if img_id != prev_img_id:
                    img_sentences_data = get_sentence_data(
                        os.path.join(sen_dir, "{}.txt".format(img_id))
                    )
                    img_annotation_data = get_annotations(
                        os.path.join(ann_dir, "{}.xml".format(img_id))
                    )
                if img_sentences_data is not None and img_annotation_data is not None:
                    capid2img_data[split][cap_id] = img_annotation_data
                    capid2sent_data[split][cap_id] = img_sentences_data[cap_num - 1]
                else:
                    logger.warning(
                        "Shouldn't reach this point: Filling capid2img_data dict "
                        "and capid2sent_data dict, while the data to fill it with "
                        "is not loaded yet! Filling with 'None'!"
                    )
                    capid2img_data[split][cap_id] = None
                    capid2sent_data[split][cap_id] = None
        with open(output_sent_data_file, "wb") as f_out:
            pickle.dump(capid2sent_data, f_out)
        with open(output_img_data_file, "wb") as f_out:
            pickle.dump(capid2img_data, f_out)
        logger.info(
            "Saved the caption ids 2 sentence mapping and caption ids 2 image mapping."
        )
