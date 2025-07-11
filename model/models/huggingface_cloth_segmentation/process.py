import os
import cv2
import gdown
import torch
import argparse
import numpy as np
from PIL import Image
from .options import opt
from .network import U2NET
import torch.nn.functional as F
from collections import OrderedDict
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings('ignore')


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean
        if isinstance(std, float):
            self.std = std
        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"


def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)


def generate_mask(input_image, net, palette, device = 'cpu'):
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    img = np.array(img)[:,:,:3]
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    alpha_out_dir = os.path.join(opt.output,'alpha')
    cloth_seg_out_dir = os.path.join(opt.output,'cloth_seg')
    os.makedirs(alpha_out_dir, exist_ok=True)
    os.makedirs(cloth_seg_out_dir, exist_ok=True)
    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()
    classes_to_save = []
    for cls in range(1, 4):
        if np.any(output_arr == cls):
            classes_to_save.append(cls)
    alpha_masks = []
    for cls in classes_to_save:
        alpha_mask = (output_arr == cls).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0] 
        alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
        alpha_masks.append(alpha_mask_img)
    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg.putpalette(palette)
    cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
    return alpha_masks, cloth_seg


def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY"
        gdown.download(url, file_path, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    return net


def get_model(checkpoint_path='./model/cloth_segm.pth', device='cpu'):
    model = load_seg_model(checkpoint_path, device=device)
    return model


def main(image, device, model):
    device = 'cuda:0' if device=='gpu' else 'cpu'
    palette = get_palette(4)
    img = Image.open(image).convert('RGB')
    masks, cloth_seg = generate_mask(img, net=model, palette=palette, device=device)
    return masks, cloth_seg, img


def make_model():
    model = load_seg_model("/home/deveshdatwani/closetx/celery/models/huggingface_cloth_segmentation/model/cloth_segm.pth")
    return model