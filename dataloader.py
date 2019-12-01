import logging
import os
from datetime import datetime
import pickle as pkl
import random
from multiprocessing import Pool

import PIL
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from torch.utils.data import Dataset
import torch


class NFNDataset(Dataset):
    def __init__(self, gt_root, noisy_root, landmark_root, transform=None):
        self.gt_root = gt_root
        self.noisy_root = noisy_root
        self.landmark_root = landmark_root

        self.transform = transform

        self.gt_files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(self.gt_root)
            for filename in files
            if filename.endswith(".png")
        ]
        self.gt_files.sort()

        self.noisy_files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(self.noisy_root)
            for filename in files
            if filename.endswith(".png")
        ]
        self.noisy_files.sort()

        self.landmark_files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(self.landmark_root)
            for filename in files
            if filename.endswith(".png")
        ]
        self.landmark_files.sort()


        if not len(self.gt_files) == len(self.noisy_files) or  (not len(self.noisy_files) == len(self.landmark_files)):
            assert("number of ground truth, noisy and landmark image files should be the same")

        self.length = len(self.gt_files)
        self.indexes = [idx for idx in range(self.length)]


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gt_path = self.gt_files[idx]
        noisy_path = self.noisy_files[idx]
        landmark_path = self.landmark_files[idx]

        gt_img = Image.open(gt_path)
        noisy_img = Image.open(noisy_path)
        landmark_img = Image.open(landmark_path)


        if self.transform:
            gt_img = self.transform(gt_img)
            noisy_img = self.transform(noisy_img)
            landmark_img = self.transform(landmark_img)

        x = torch.cat((noisy_img, landmark_img), 0)

        return gt_img, x


