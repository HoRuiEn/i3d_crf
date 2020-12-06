import re
import json
import csv
import os
from glob import glob

import h5py
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Dataset


split_name_conversion = {
    'training': 'train_fns',
    'testing': 'test_fns'
}

pattern = r'P\d+S\d+G\d+B\d+H\d+UC\d+LC\d+A(\d+)R\d+_\d+'

def check_image_path(path):
    if not os.path.exists(path):
        raise ValueError("Image cannot be found: %s" %fn)

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start+num):
        fn = os.path.join(image_dir, vid, str(i).zfill(3)+'.jpg')
        check_image_path(fn)
        img = cv2.imread(fn)[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start+num):
        xfn = os.path.join(image_dir, vid, str(i).zfill(3)+'x.jpg')
        check_image_path(xfn)
        imgx = cv2.imread(xfn, cv2.IMREAD_GRAYSCALE)
        
        yfn = os.path.join(image_dir, vid, str(i).zfill(3)+'y.jpg')
        check_image_path(yfn)
        imgy = cv2.imread(yfn, cv2.IMREAD_GRAYSCALE)
    
        w,h = imgx.shape
        if w < 224 or h < 224:
            d = 224.-min(w,h)
            sc = 1+d/min(w,h)
            imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
            imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)

        imgx = (imgx/255.)*2 - 1
        imgy = (imgy/255.)*2 - 1
        img = np.asarray([imgx, imgy]).transpose([1,2,0])
        frames.append(img)
    
    return np.asarray(frames, dtype=np.float32)

def make_dataset(split_file, split, root, mode, snippets, num_classes=155):
    count_items = 0
    dataset = []
    with open(split_file, 'r') as f:
        train_test_split = json.load(f)

    vids = sorted(train_test_split[split_name_conversion[split]])    
    vids = [os.path.basename(fn)[:-8] for fn in vids]
    
    for vid in vids:
        fn = os.path.join(root, vid)
        if not os.path.exists(fn):
            continue

        num_frames = len(os.listdir(fn))
        
        for j in range(0, num_frames, snippets):
            if j + snippets > num_frames:
                continue
            label = np.zeros((num_classes, snippets), np.float32)
            ann = int(re.match(pattern, vid).groups()[0])
            
            for frame in range(j + 1, j + snippets + 1):
                label[ann, (frame-1)%snippets] = 1

            dataset.append((vid, j+1, label))
            count_items += 1
    
    print("Make dataset {}: {} examples".format(split, count_items*snippets))

    return dataset


class Uavhuman(Dataset):

    def __init__(self, split_file, split, root, mode, snippets, transforms=None, num_classes=155):
        self.data = make_dataset(split_file, split, root, mode, snippets, num_classes=num_classes)
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.snippets = snippets

    def __getitem__(self, index):
        vid, start, label = self.data[index]
        
        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start, self.snippets)
        else:
            imgs = load_flow_frames(self.root, vid, start, self.snippets)

        imgs = self.transforms(imgs)
        
        return video_to_tensor(imgs), torch.from_numpy(label)
    
    def __len__(self):
        return len(self.data)


class Uavhuman_eval(Dataset):

    def __init__(self, split_file, split, root, mode, snippets, transforms=None, num_classes=155):
        self.data = make_dataset(split_file, split, root, mode, snippets, num_classes=num_classes)
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.snippets = snippets

    def __getitem__(self, index):
        vid, start, label = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start, self.snippets)
        else:
            imgs = load_flow_frames(self.root, vid, start, self.snippets)

        imgs = self.transforms(imgs)

        return vid, start, video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
