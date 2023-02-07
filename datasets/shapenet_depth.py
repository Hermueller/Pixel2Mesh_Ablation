import json
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate
from collections import OrderedDict
from models.i2d import I2D

import matplotlib.pyplot as plt

import config
from datasets.base_dataset import BaseDataset


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
        
        self.model = I2D()
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
            img_org = img
            if self.resize_with_constant_border:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                       mode='constant', anti_aliasing=False)  # to match behavior of old versions
            else:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            img_org_size = img
            img = img[:, :, :3].astype(np.float32)
        else:
            label, filename = self.file_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img_org = data[0].astype(np.float32)
            img_org_size = img_org
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        # adjust brightness, noise, blur, etc. for Ablation
        """adj = T.ToPILImage()
        adjT = T.ToTensor()
        img = adjT(T.functional.adjust_brightness(adj(img), brightness_factor=1.6))"""

        # adjust saturation
        """adj = T.ToPILImage()
        adjT = T.ToTensor()
        img = adjT(T.functional.adjust_saturation(adj(img), 5))"""

        # add salt-and-pepper noise
        """rnd_tensor = torch.rand(img.shape[1], img.shape[2])
        img[0][rnd_tensor >= (1-0.005)] = 250
        img[0][rnd_tensor <= 0.005] = 5
        img[1][rnd_tensor >= (1 - 0.005)] = 250
        img[1][rnd_tensor <= 0.005] = 5
        img[2][rnd_tensor >= (1 - 0.005)] = 250
        img[2][rnd_tensor <= 0.005] = 5"""

        # add random noise (+/- the real value)
        """rnd_tensor = torch.rand(img.shape[0], img.shape[1], img.shape[2])
        rnd_noise = torch.rand(img.shape[0], img.shape[1], img.shape[2])
        img[rnd_tensor >= (1 - 0.1)] += (rnd_noise[rnd_tensor >= (1 - 0.1)] - 0.5)*10"""

        # depth
        img_org = img_org.astype(np.float32)
        img_cuda = torch.from_numpy(np.transpose(img_org[:, :, :3], (2, 0, 1)))
        img_cuda = img_cuda.unsqueeze(0).cuda()
        self.model = self.model.cuda()  # having this is __init__ resulted in all predictions being zero...
        self.model.load_state_dict(torch.load("datasets/preprocess/ext_models/fyn_model.pt", map_location='cpu'))
        img_depth = self.model(img_cuda)[0, :, :, :]

        # plt.imshow(img_depth[0,:,:].detach().cpu().numpy())
        # plt.show()

        img_depth = img_depth.permute(1, 2, 0)
        img_depth = img_depth.detach().cpu()
        if self.resize_with_constant_border:
            img_depth = transform.resize(img_depth, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)  # to match behavior of old versions
        else:
            img_depth = transform.resize(img_depth, (config.IMG_SIZE, config.IMG_SIZE))

        # TODO: same as img[np.where(img[:, :, 3] == 0)] = 255 before.
        #       discard the depth values that are not from the object itself.
        img_depth[np.where(img_org_size[:, :, 3] == 0)] = 0.000001

        img_depth = torch.from_numpy(img_depth).permute(2, 0, 1)
        img = torch.cat((img, img_depth), dim=0)

        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized.detach().numpy(),
            "images_orig": img.detach().numpy(),
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": filename,
            "length": length
        }

    def __len__(self):
        return len(self.file_names)


class ShapeNetDepthImageFolder(BaseDataset):

    def __init__(self, folder, normalization, shapenet_options):
        super().__init__()
        self.normalization = normalization
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.file_list = []
        for fl in os.listdir(folder):
            file_path = os.path.join(folder, fl)
            # check image before hand
            try:
                if file_path.endswith(".gif"):
                    raise ValueError("gif's are results. Not acceptable")
                Image.open(file_path)
                self.file_list.append(file_path)
            except (IOError, ValueError):
                print("=> Ignoring %s because it's not a valid image" % file_path)
        # depth model
        self.model = I2D()
        self.model.load_state_dict(torch.load("datasets/preprocess/ext_models/fyn_model.pt"))

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        img_org = img
        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img_org_size = img
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        # depth
        img_org = img_org.astype(np.float32)
        img_cuda = torch.from_numpy(np.transpose(img_org[:, :, :3], (2, 0, 1)))
        img_cuda = img_cuda.unsqueeze(0).cuda()
        self.model = self.model.cuda()  # having this is __init__ resulted in all predictions being zero...
        self.model.load_state_dict(torch.load("datasets/preprocess/ext_models/fyn_model.pt", map_location='cpu'))
        img_depth = self.model(img_cuda)[0, :, :, :]
        
        img_depth = img_depth.permute(1, 2, 0)
        img_depth = img_depth.detach().cpu()
        if self.resize_with_constant_border:
            img_depth = transform.resize(img_depth, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)  # to match behavior of old versions
        else:
            img_depth = transform.resize(img_depth, (config.IMG_SIZE, config.IMG_SIZE))

        img_depth[np.where(img_org_size[:, :, 3] == 0)] = 0.000001

        img_depth = torch.from_numpy(img_depth).permute(2, 0, 1)
        img = torch.cat((img, img_depth), dim=0)

        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "filepath": self.file_list[item]
        }

    def __len__(self):
        return len(self.file_list)


def shapenet_depth_collate(batch):
    if len(batch) > 1:
        all_equal = True
        
        for t in batch:
            if t["length"] != batch[0]["length"]:
                all_equal = False
                break
        
        points_orig, normals_orig = [], []

        if not all_equal:
            for t in batch:
                pts, normal = t["points"], t["normals"]
                length = pts.shape[0]
                choices = np.resize(np.random.permutation(length), 3000)   # num_points
                t["points"], t["normals"] = pts[choices], normal[choices]
                points_orig.append(torch.from_numpy(pts))
                normals_orig.append(torch.from_numpy(normal))

            ret = default_collate(batch)
            ret["points_orig"] = points_orig
            ret["normals_orig"] = normals_orig
            return ret

    ret = default_collate(batch)
    ret["points_orig"] = ret["points"] | []
    ret["normals_orig"] = ret["normals"] | []
    return ret


def get_shapenet_depth_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """

    return shapenet_depth_collate