import torch
from models.surface_normals.NNET import NNET
import numpy as np
from PIL import Image

__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def img_to_surface_normals(model, rgbImage):
    norm_out_list, _, _ = model(rgbImage)
    norm_out = norm_out_list[-1]

    pred_norm = norm_out[:, :3, :, :]
    pred_kappa = norm_out[:, 3:, :, :]

    # to numpy arrays
    img = rgbImage.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
    pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
    pred_kappa = pred_kappa.detach().cpu().permute(0, 2, 3, 1).numpy()

    img = unnormalize(img[0, ...])

    # pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
    # pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
    # pred_norm_rgb = pred_norm_rgb.astype(np.uint8)   

    return ((pred_norm + 1) * 0.5) 

def load_checkpoint_on_device(fpath, device):
    model = NNET().to(device)
    ckpt = torch.load(fpath)['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    model.eval()
    return model

def unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out

def kappa_to_alpha(pred_kappa):
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    alpha = np.degrees(alpha)
    return alpha

def concat_image(image_path_list, concat_image_path):
    imgs = [Image.open(i).convert("RGB").resize((640, 480), resample=Image.BILINEAR) for i in image_path_list]
    imgs_list = []
    for i in range(len(imgs)):
        img = imgs[i]
        imgs_list.append(np.asarray(img))

        H, W, _ = np.asarray(img).shape
        imgs_list.append(255 * np.ones((H, 20, 3)).astype('uint8'))

    imgs_comb = np.hstack(imgs_list[:-1])
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(concat_image_path)