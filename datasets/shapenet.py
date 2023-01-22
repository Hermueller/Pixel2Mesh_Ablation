import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import matplotlib.pyplot as plt

import config
from datasets.base_dataset import BaseDataset
from models.surface_normals.NNETUtils import *

class ShapeNet(BaseDataset):
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

        # model for surface normals
        self.surfaceNormalModel = load_checkpoint_on_device("datasets/scannet.pt", torch.device("cuda:0"))

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
            img_org = io.imread(img_path)
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
            img_org = data[0].astype(np.float32)
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        # **********************
        # surface normals
        # **********************

        # img_org = io.imread("D:\\_Entwicklung\\_Uni\\ml3d\\Pixel2Mesh_Ablation\\test-images\\01.png")
        img_resized = transform.resize(img_org, (480, 640), mode='constant', anti_aliasing=True) #img_org.convert("RGB").resize(size=(640, 480), resample=Image.BILINEAR)
        img_resized_rgb = img_resized[:,:,:3]
        
        # plt.imshow(img_resized_rgb)
        # plt.show()
        
        img_batch = torch.from_numpy(img_resized_rgb[None, :]).permute(0, 3, 1, 2).type(torch.FloatTensor)
        img_batch_gpu = img_batch.to(torch.device("cuda:0"))
        img_normals = img_to_surface_normals(self.surfaceNormalModel, img_batch_gpu)

        #remove data from objects surroundings
        img_normals[:,:,0][np.where(img_resized[:, :, 3] == 0)] = 0.000001
        img_normals[:,:,1][np.where(img_resized[:, :, 3] == 0)] = 0.000001
        img_normals[:,:,2][np.where(img_resized[:, :, 3] == 0)] = 0.000001

        # pred_norm_rgb = img_normals * 255
        # pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
        # pred_norm_rgb = pred_norm_rgb.astype(np.uint8)     
        
        # plt.title("output img")
        # plt.imshow(pred_norm_rgb)
        # plt.show()
        
        # plt.imsave("C:\\Users\\phste\\Desktop\\img-norm.png", pred_norm_rgb)
        # plt.imsave("C:\\Users\\phste\\Desktop\\img-rgb.png", img_resized)

        #resize surface normal output to fit other model's size 
        if self.resize_with_constant_border:
            img_normals = transform.resize(img_normals, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)  # to match behavior of old versions
        else:
            img_normals = transform.resize(img_normals, (config.IMG_SIZE, config.IMG_SIZE))
        
        #permute dimensions to fit other model
        img_surface_normals_tensor = torch.from_numpy(img_normals).type(torch.FloatTensor).permute(2, 0, 1)

        #combine rgb + surface normal data
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        img_cat_norm = self.normalize_img(img) if self.normalization else img
        img_normalized = torch.cat((img_cat_norm, img_surface_normals_tensor), dim=0)

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


class ShapeNetImageFolder(BaseDataset):

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

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "filepath": self.file_list[item]
        }

    def __len__(self):
        return len(self.file_list)


def shapenet_collate(batch):
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
    ret["points_orig"] = ret["points"]
    ret["normals_orig"] = ret["normals"]
    return ret


def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """

    return shapenet_collate