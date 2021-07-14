import numpy as np
import cv2 as cv
from common.misc import *


class Data:
    def __init__(self, npz_path, train=True, action=None):
        self.img_path = []
        self.gt_pts2d = []
        self.gt_pts3d = []

        data = np.load(npz_path, allow_pickle=True)
        num_frame = 0

        print("INFO: Using Human3.6M dataset.")
        subject = {
            "subjects_train": ["S1/", "S5/", "S6/", "S7/", "S8/"],
            "subjects_test": ["S9/", "S11/"]
        }
        if action is None:
            if train:
                # training
                to_load = [item for item in data.files \
                    for S in subject["subjects_train"] if S in item]
            else:
                # validating
                to_load = [item for item in data.files \
                    for S in subject["subjects_test"] if S in item]
        else:
            # evaluating
            to_load = [item for item in data.files \
                for S in subject["subjects_test"] if S in item and action in item]

        import random
        random.seed(100)
        for act in to_load:
            frames = data[act].flatten()[0]
            reduced = random.sample(list(frames), int(len(frames))) \
                if action is None else random.sample(list(frames), int(len(frames)))
            num_frame += len(reduced)
            for f in reduced:
                gt_2d = self.zero_center(frames[f]["positions_2d"])
                gt_3d = self.zero_center(self.remove_joints( \
                        frames[f]["positions_3d"]))

                assert gt_2d.shape == (17,2) and gt_3d.shape == (17,3)
                self.gt_pts2d.append(gt_2d)
                self.gt_pts3d.append(gt_3d)
                self.img_path.append(frames[f]["directory"])

 
        print("INFO: Using ", num_frame, " frames")

    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            kpts_2d = self.gt_pts2d[index]
            kpts_3d = self.gt_pts3d[index]
        except:
            return None
        return img_path, kpts_2d, kpts_3d
        

    def __len__(self):
        return len(self.img_path)
    

    def remove_joints(self, kpts):
        """
        Get 17 joints from the original 32 (Human3.6M)
        """
        new_skel = np.zeros([17,3]) if kpts.shape[-1]==3 else np.zeros([17,2])

        keep = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
        for row in range(17):
            new_skel[row, :] = kpts[keep[row], :]
        return new_skel


    def zero_center(self, cam) -> np.array:
        """translate root joint to origin (0,0,0)"""
        return cam - cam[0,:]
