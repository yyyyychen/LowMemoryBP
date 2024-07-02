#!/usr/bin/env python3
# modified from https://github.com/KMnP/vpt/blob/main/src/data/datasets/json_dataset.py
"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import torch
import torch.utils.data
import torchvision
import numpy as np

import json

from collections import Counter


def read_json(filename: str):
    """read json files"""
    with open(filename, "rb") as f:
        data = json.load(f)
    return data


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, data_percentage, transform):

        self._split = split
        self.data_dir = data_dir
        self.data_percentage = data_percentage
        self._construct_imdb()
        self.transform = transform

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        return read_json(anno_path)

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()

        # Map class ids to contiguous ids
        self._class_ids = sorted(set(anno.values()))
        self._class_ids_counts = Counter(anno.values())
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self._class_num

    @property
    def _class_num(self):
        return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self._class_num
        if weight_type == "none":
            return [1.0] * cls_num

        cls_ids_counts = self._class_ids_counts

        num_per_cls = np.array([cls_ids_counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = torchvision.datasets.folder.default_loader(self._imdb[index]["im_path"])
        label = self._imdb[index]["class"]
        im = self.transform(im)

        return im, label

    def __len__(self):
        return len(self._imdb)