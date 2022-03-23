# !/usr/bin/env python

# The root of bottom-up-attention repo. Do not need to change if using provided docker file.
BUTD_ROOT = '/opt/butd/'

# SPLIT to its folder name under IMG_ROOT
PP_FLICKR_SPLIT2NAME = {
    'train': 'train',
    'dev': 'valid',
    'test': 'test',
}

FLICKR_ENT_SPLIT2NAME = {
    'train': 'train',
    'dev': 'val',
    'test': 'test',
}

import os, sys

sys.path.insert(0, BUTD_ROOT + "/tools")
os.environ['GLOG_minloglevel'] = '2'

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms

import caffe
import argparse
import pprint
import base64
import numpy as np
import cv2
import csv
from tqdm import tqdm
import json
sys.path.insert(0, '/workspace/data/flickr30k_entities')
from flickr30k_entities_utils import get_sentence_data, get_annotations

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "no_box", "scene",
              "box_identifiers", "num_boxes", "boxes", "features", "image_feature"]


def load_pp_image_ids(img_root, split_name, splitfile):
    """images in the same directory are in the same split"""
    split_imgids = json.load(open(splitfile))[split_name]
    print('%s: %d' % (split_name, len(split_imgids)))
    path_and_id = []
    for id_ in split_imgids:
        filepath = os.path.join(img_root, id_+'.jpg')
        path_and_id.append((filepath, int(id_)))
    return path_and_id


def load_ent_image_ids(img_root, split_name, splitdir):
    """images in the same directory are in the same split"""
    path_and_id = []
    with open(os.path.join(splitdir, '{}.txt'.format(split_name)), 'r') as f:
        for id_ in f:
            id_ = id_.strip()
            filepath = os.path.join(img_root, id_ + '.jpg')
            path_and_id.append((filepath, int(id_)))
    print('%s: %d' % (split_name, len(path_and_id)))
    return path_and_id


def generate_tsv(prototxt, weights, image_ids, outfile, ann_dir):
    # First check if file exists, and if it is complete
    wanted_ids = set([image_id[1] for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile, "r") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                found_ids.add(int(item['img_id']))
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print('already completed {:d}'.format(len(image_ids)))
    else:
        print('missing {:d}/{:d}'.format(len(missing), len(image_ids)))
    if len(missing) > 0:
        print('check 1')
        # caffe.set_mode_cpu()
        # caffe.set_mode_gpu()
        # caffe.set_device(0)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        print('check 2')
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for im_file, image_id in tqdm(image_ids):
                if image_id in missing:
                    row = get_box_features_from_im(net, im_file, image_id, ann_dir)
                    if row is not None:
                        writer.writerow(row)


def get_box_features_from_im(net, im_file, image_id, ann_dir):
    """
    :param net:
    :param im_file: full path to an image
    :param image_id: the id of the image, also the filename
    :param ann_dir: the directory where annotations are stored
    :return: all information from detection and attr prediction
    """
    im = cv2.imread(im_file)
    img_annotation_data = get_annotations(os.path.join(ann_dir, '{}.xml'.format(image_id)))
    scene = np.array(img_annotation_data['scene'])
    no_box = np.array(img_annotation_data['nobox'])
    num_boxes = []
    idxes = []
    gt_boxes = []
    for boxes_idx, boxes in img_annotation_data['boxes'].items():
        num_boxes.append(len(boxes))
        idxes.append(boxes_idx)
        gt_boxes += boxes
    num_boxes = np.array(num_boxes)
    idxes = np.array(idxes)
    gt_boxes = np.array(gt_boxes)
    _, _, _, _ = im_detect(net, im, boxes=np.array([[0, 0, np.size(im, 0), np.size(im, 1)]]), force_boxes=True)
    full_im = net.blobs['pool5_flat'].data
    return_dict = {
            "img_id": image_id,
            "img_h": np.size(im, 0),
            "img_w": np.size(im, 1),
            "no_box": base64.b64encode(no_box),
            "scene": base64.b64encode(scene),  # int64
            "box_identifiers": base64.b64encode(idxes),  # int64
            "num_boxes": base64.b64encode(num_boxes),
            "boxes": base64.b64encode(gt_boxes),  # float32
            "features": base64.b64encode(np.array([])),  # float32
            "image_feature": base64.b64encode(full_im)  # float32
    }
    if gt_boxes.size == 0:
        tqdm.write("No boxes for current im:")
        return return_dict
    else:
        _, _, _, _ = im_detect(net, im, boxes=gt_boxes, force_boxes=True)
        pool5 = net.blobs['pool5_flat'].data
        return_dict["features"] = base64.b64encode(pool5)
        return return_dict


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', default='0', type=str, help='GPU id(s) to use')
    parser.add_argument('--def', dest='prototxt', default=None, type=str, help='prototxt file defining the network')
    parser.add_argument('--out', dest='outfile', default="/workspace/features/", type=str,
                        help='output filepath. file name is filled by script self')
    parser.add_argument('--cfg', dest='cfg_file', default=None, type=str, help='optional config file')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set config keys')
    parser.add_argument('--imgroot', type=str, default='/workspace/data/flickr30k-images/',
                        help='path to the images. If in docker, '
                             'leave default and add a link to the image dir, as explained in README')
    parser.add_argument('--caffemodel', type=str,
                        default='../snap/resnet101_faster_rcnn_final_iter_320000.caffemodel',
                        help='path to the caffemodels. If in docker, '
                             'leave default and add a link to the image dir, as explained in README')
    parser.add_argument('--ppsplitfile', type=str, default='/workspace/data/pp-flickr/pp-flickr_split.json',
                        help="path to jsonfile containing lists for 'train', 'valid', 'test' split. "
                             "assumes it to be from pp-flickr. If in docker,"
                             "leave default and add a link to the image dir, as explained in README'")
    parser.add_argument('--box_ann_dir', type=str, default='/workspace/data/flickr30k_entities/Annotations/',
                        help="path to dir with all xml box annotations per image. "
                             "assumes it to be from Flickr30k_entitities. If in docker,"
                             "leave default and add a link to the image dir, as explained in README'")
    parser.add_argument('--entititysplitdir', type=str, default='/workspace/data/flickr30k_entities/',
                        help="Directory with split files by flickr30k entities. Assumes directory to "
                             "contain txt files named 'train.txt', 'val.txt', and 'test.txt'. Files contain a list of "
                             "image ids.")
    parser.add_argument('--use_spacy', action='store_true', help="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()

    args.cfg_file = BUTD_ROOT + "experiments/cfgs/faster_rcnn_end2end_resnet.yml"
    args.prototxt = BUTD_ROOT + "models/vg/ResNet-101/faster_rcnn_end2end_final/test_gt.prototxt"
    args.outfile = os.path.join(args.outfile, 'flickr30k-entities-{}.obj_features.tsv')
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TEST.HAS_RPN = False
    cfg.DEDUP_BOXES = 0
    assert not cfg.TEST.HAS_RPN, "we want to use ground-truth boxes, make sure we don't use an RPN."

    for split in ['train', 'dev', 'test']:
        # Load image ids, need modification for new datasets.
        if args.use_spacy:
            image_ids = load_ent_image_ids(args.imgroot, FLICKR_ENT_SPLIT2NAME[split], args.entititysplitdir)
        else:
            image_ids = load_pp_image_ids(args.imgroot, PP_FLICKR_SPLIT2NAME[split], args.ppsplitfile)
        # Generate TSV files, normally do not need to modify
        generate_tsv(args.prototxt, args.caffemodel, image_ids, args.outfile.format(split), args.box_ann_dir)
