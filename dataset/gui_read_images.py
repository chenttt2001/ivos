import os
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
from dataset.range_transform import im_normalization
import torch

def get_new_hw(h, w):  #
    if w <= 960:
        return h, w
    scale = w / h
    new_h = 480
    new_w = int(scale * new_h)
    return new_h, new_w

def pad_get_newhw_by(h, w):
    d = 16
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w

        return new_h, new_w

def get_tensor_images(imagespath):
    filepath = sorted(os.listdir(imagespath))
    flag = True
    inter_img_np = None
    for i in tqdm(filepath):
        imgpath = os.path.join(imagespath,i)
        img_np = cv2.imread(imgpath)
        new_h,new_w = get_new_hw(img_np.shape[0],img_np.shape[1])
        inter_img_np_resized = cv2.resize(img_np, (new_w,new_h), interpolation=cv2.INTER_CUBIC)
        trans1 = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        if flag:
            inter_img_np = trans1(inter_img_np_resized).unsqueeze(0)
            flag = False
        else:
            inter_img_np = torch.cat((inter_img_np,trans1(inter_img_np_resized).unsqueeze(0)),0)
    print('The sequence data is loaded')

    return inter_img_np