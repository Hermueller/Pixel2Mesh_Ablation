import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate

import torch.nn as nn

import config
from datasets.base_dataset import BaseDataset
from models.i2d import I2D

class ShapeNetDepth(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
        super().__init__()
        self.file_root = file_root
        with open(os.path.join(self.file_root, "meta", "shapenet.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}
        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.tensorflow = "_tf" in file_list_name # tensorflow version of data
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        
        self.device = torch.device("cpu")
        
        self.i2d = I2D().to(self.device)
        self.i2d.load_state_dict(torch.load("{}/i2d/fyn_model.pt".format(self.file_root), map_location='cpu'))
        print("i2d model loaded")


    def __getitem__(self, index):
        if self.tensorflow:
            filename = self.file_names[index][17:]
            label = filename.split("/", maxsplit=1)[0]
            pkl_path = os.path.join(self.file_root, "data_tf", filename)
            img_path = pkl_path[:-4] + ".png"
            with open(pkl_path) as f:
                data = pickle.load(open(pkl_path, 'rb'), encoding="latin1")
            pts, normals = data[:, :3], data[:, 3:]
            img = io.imread(img_path)
            img[np.where(img[:, :, 3] == 0)] = 255
            if self.resize_with_constant_border:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                       mode='constant', anti_aliasing=False)  # to match behavior of old versions
            else:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            img = img[:, :, :3].astype(np.float32)
        else:
            label, filename = self.file_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]


        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img
        img_normalized = img_normalized.reshape(1, img_normalized.shape[0], img_normalized.shape[1], img_normalized.shape[2])
        pred_depth = self.i2d(img_normalized)
        # pred_depth = torch.unsqueeze(pred_depth, 0)

        upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        pred_depth = upsample(pred_depth)

        img_normalized = torch.cat((img_normalized, pred_depth), 1)

        return {
            "images": img_normalized,
            "images_orig": img,
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": filename,
            "length": length
        }

    def __len__(self):
        return len(self.file_names)