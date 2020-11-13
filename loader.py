import os
import cv2
import random
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset

from augmentations import Rotation, Scale, Translate, Flip
from augmentations import White_Noise, Gray, add_light, contrast_image, saturation_image, Equalization

# mean and std in my own data
mean = [0.3248, 0.3373, 0.3436, 0.2411, 0.2518, 0.1794]
std = [0.2365, 0.2438, 0.2502, 0.1624, 0.1484, 0.1807]

class Train_DataSet(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.proj_H = 64
        self.proj_W = 512
        self.proj_C = 6
        self.C_GT = 1

        self.calib_dir = glob(os.path.join(self.data_folder, "training", "calib", "*.txt"))
        self.image_dir = glob(os.path.join(self.data_folder, "training", "image_2", "*.png"))
        self.color_lidar_dir = glob(os.path.join(self.data_folder, "training", "color_txt", "*.txt"))
        self.velodyne_dir = glob(os.path.join(self.data_folder, "training", "velodyne", "*.bin"))
        self.gt_lidar_dir = glob(os.path.join(self.data_folder, "training", "gt_txt", "*.txt"))

    def __getitem__(self, item):
        # Build Data from Raw Camera & LiDAR data
        # Build a Camera
        img = cv2.imread(self.image_dir[item])
        height, width, channel = img.shape

        # Build a LiDAR
        velodyne = np.fromfile(self.velodyne_dir[item], dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

        coordinate = velodyne[:, :3]
        scan_x = coordinate[:, 0]
        scan_y = coordinate[:, 1]
        scan_z = coordinate[:, 2]

        add_ones = np.ones((coordinate.shape[0]))
        add_ones = np.expand_dims(add_ones, axis=1)
        coordinate_add_one = np.append(coordinate, add_ones, axis=1)

        intensity = velodyne[:, 3]

        # Build a Calibration
        calib_file = open(self.calib_dir[item], 'r')
        lines = calib_file.readlines()

        P2_rect = np.array(lines[2].split(" ")[1:]).astype(np.float32).reshape(3, 4)
        R0_rect = np.array(lines[4].split(" ")[1:]).astype(np.float32).reshape(3, 3)
        velo_to_cam = np.array(lines[5].split(" ")[1:]).astype(np.float32).reshape(3, 4)

        add_r0 = np.array([0., 0., 0.])
        add_r0_1 = np.array([0., 0., 0., 1.])
        R0_rect = np.append(R0_rect, add_r0.reshape(3, 1), axis=1)
        R0_rect = np.append(R0_rect, add_r0_1.reshape(1, 4), axis=0)

        add_velo = np.array([0., 0., 0., 1.])
        velo_to_cam = np.append(velo_to_cam, add_velo.reshape(1, 4), axis=0)

        homogeneous_matrix = np.matmul(np.matmul(P2_rect, R0_rect), velo_to_cam)

        result = np.matmul(homogeneous_matrix, coordinate_add_one.transpose(1, 0))

        u = result[0] / result[2]
        v = result[1] / result[2]

        condition = np.where(
            (result[0] >= 0) & (result[2] >= 0) & (scan_x >= 0) & (v < height) & (v >= 0) & (u >= 0) & (u < width))

        u = u[condition]
        v = v[condition]

        scan_x = scan_x[condition]
        scan_y = scan_y[condition]
        scan_z = scan_z[condition]
        intensity = intensity[condition]

        u = np.floor(u)
        u = np.minimum(width - 1, u)
        u = np.maximum(0, u).astype(np.int32)

        v = np.floor(v)
        v = np.minimum(height - 1, v)
        v = np.maximum(0, v).astype(np.int32)

        b = img[v, u][:, 0]
        g = img[v, u][:, 1]
        r = img[v, u][:, 2]

        dtheta = np.radians(0.4)
        dphi = np.radians(90. / 512.0)

        depth_ = np.sqrt(scan_x ** 2 + scan_y ** 2 + scan_z ** 2)
        range_ = np.sqrt(scan_x ** 2 + scan_y ** 2)

        depth_[depth_ == 0] = 0.000001
        range_[range_ == 0] = 0.000001

        phi = np.radians(45.) - np.arcsin(scan_y / range_)
        phi_ = (phi / dphi).astype(int)
        phi_[phi_ < 0] = 0
        phi_[phi_ >= 512] = 511

        theta = np.radians(2.) - np.arcsin(scan_z / depth_)
        theta_ = (theta / dtheta).astype(int)
        theta_[theta_ < 0] = 0
        theta_[theta_ >= 64] = 63

        fusion_map = np.zeros((self.proj_H, self.proj_W, self.proj_C))

        b = 1.0 * (b - b.min()) / (b.max() - b.min())
        g = 1.0 * (g - g.min()) / (g.max() - g.min())
        r = 1.0 * (r - r.min()) / (r.max() - r.min())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max() - scan_z.min())
        intensity = 1.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        depth_ = 1.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        fusion_map[theta_, phi_, 0] = b
        fusion_map[theta_, phi_, 1] = g
        fusion_map[theta_, phi_, 2] = r
        fusion_map[theta_, phi_, 3] = scan_z
        fusion_map[theta_, phi_, 4] = intensity
        fusion_map[theta_, phi_, 5] = depth_
        fusion_map = (fusion_map - mean) / std

        # Build a GT
        velodyne = self.gt_lidar_dir[item]
        with open(velodyne) as velo_object:
            contents = velo_object.readlines()

        points = []
        for content in contents:
            temp = []
            point = content.split(" ")

            temp.append(float(point[0]))
            temp.append(float(point[1]))
            temp.append(float(point[2]))
            temp.append(int(point[3]))
            temp.append(int(point[4]))
            temp.append(int(point[5]))
            points.append(temp)

        final = np.array(points)

        dtheta = np.radians(0.4)
        dphi = np.radians(90. / 512.0)

        point = final[:, :3]
        rgb = final[:, 3:]
        scan_x = point[:, 0]
        scan_y = point[:, 1]
        scan_z = point[:, 2]
        scan_r = rgb[:, 0]
        scan_g = rgb[:, 1]
        scan_b = rgb[:, 2]

        d = np.sqrt(pow(scan_x, 2) + pow(scan_y, 2) + pow(scan_z, 2))
        r = np.sqrt(pow(scan_x, 2) + pow(scan_y, 2))

        d[d == 0] = 0.000001
        r[r == 0] = 0.000001

        phi = np.radians(45.) - np.arcsin(scan_y / r)
        phi_ = (phi / dphi).astype(int)
        phi_[phi_ < 0] = 0
        phi_[phi >= 512] = 511

        theta = np.radians(2.) - np.arcsin(scan_z / d)
        theta_ = (theta / dtheta).astype(int)
        theta_[theta_ < 0] = 0
        theta_[theta_ >= 64] = 63

        depth_map_gt = np.zeros((self.proj_H, self.proj_W, self.C_GT))

        scan_b[scan_b == 255.0] = 1.0
        scan_g[scan_g == 255.0] = 1.0
        scan_r[scan_r == 255.0] = 1.0

        depth_map_gt[theta_, phi_, 0] = scan_b

        input_tensor = fusion_map
        label_tensor = depth_map_gt

        # Step 1
        # Rotation -10~10
        input_tensor, label_tensor = Rotation(input_tensor, label_tensor, 5)

        # Scale 0.9~1.3
        input_tensor, label_tensor = Scale(input_tensor, label_tensor)

        # Translate -50~50
        input_tensor, label_tensor = Translate(input_tensor, label_tensor, 5, 5)

        # Flip
        input_tensor, label_tensor = Flip(input_tensor, label_tensor, 0.5)

        # NHWC -> NCHW
        input_tensor = input_tensor.transpose(2, 0, 1)
        label_tensor = label_tensor.transpose(2, 0, 1)

        input_tensor = torch.tensor(input_tensor).float()
        label_tensor = torch.tensor(label_tensor).float()

        return input_tensor, label_tensor, self.image_dir[item]

    def __len__(self):
        return len(self.image_dir)

class Test_DataSet(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.proj_H = 64
        self.proj_W = 512
        self.proj_C = 6

        self.calib_dir = glob(os.path.join(self.data_folder, "testing", "calib", "*.txt"))
        self.image_dir = glob(os.path.join(self.data_folder, "testing", "image_2", "*.png"))
        self.color_lidar_dir = glob(os.path.join(self.data_folder, "testing", "color_txt", "*.txt"))
        self.velodyne_dir = glob(os.path.join(self.data_folder, "testing", "velodyne", "*.bin"))

    def __getitem__(self, item):
        # Build Data from Raw Camera & LiDAR data
        # Build a Camera
        img = cv2.imread(self.image_dir[item])
        height, width, channel = img.shape

        # Build a LiDAR
        velodyne = np.fromfile(self.velodyne_dir[item], dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

        coordinate = velodyne[:, :3]
        scan_x = coordinate[:, 0]
        scan_y = coordinate[:, 1]
        scan_z = coordinate[:, 2]

        add_ones = np.ones((coordinate.shape[0]))
        add_ones = np.expand_dims(add_ones, axis=1)
        coordinate_add_one = np.append(coordinate, add_ones, axis=1)

        intensity = velodyne[:, 3]

        # Build a Calibration
        calib_file = open(self.calib_dir[item], 'r')
        lines = calib_file.readlines()

        P2_rect = np.array(lines[2].split(" ")[1:]).astype(np.float32).reshape(3, 4)
        R0_rect = np.array(lines[4].split(" ")[1:]).astype(np.float32).reshape(3, 3)
        velo_to_cam = np.array(lines[5].split(" ")[1:]).astype(np.float32).reshape(3, 4)

        add_r0 = np.array([0., 0., 0.])
        add_r0_1 = np.array([0., 0., 0., 1.])
        R0_rect = np.append(R0_rect, add_r0.reshape(3, 1), axis=1)
        R0_rect = np.append(R0_rect, add_r0_1.reshape(1, 4), axis=0)

        add_velo = np.array([0., 0., 0., 1.])
        velo_to_cam = np.append(velo_to_cam, add_velo.reshape(1, 4), axis=0)

        homogeneous_matrix = np.matmul(np.matmul(P2_rect, R0_rect), velo_to_cam)

        result = np.matmul(homogeneous_matrix, coordinate_add_one.transpose(1, 0))

        u = result[0] / result[2]
        v = result[1] / result[2]

        condition = np.where(
            (result[0] >= 0) & (result[2] >= 0) & (scan_x >= 0) & (v < height) & (v >= 0) & (u >= 0) & (u < width))

        u = u[condition]
        v = v[condition]

        scan_x = scan_x[condition]
        scan_y = scan_y[condition]
        scan_z = scan_z[condition]
        intensity = intensity[condition]

        u = np.floor(u)
        u = np.minimum(width - 1, u)
        u = np.maximum(0, u).astype(np.int32)

        v = np.floor(v)
        v = np.minimum(height - 1, v)
        v = np.maximum(0, v).astype(np.int32)

        b = img[v, u][:, 0]
        g = img[v, u][:, 1]
        r = img[v, u][:, 2]

        dtheta = np.radians(0.4)
        dphi = np.radians(90. / 512.0)

        depth_ = np.sqrt(scan_x ** 2 + scan_y ** 2 + scan_z ** 2)
        range_ = np.sqrt(scan_x ** 2 + scan_y ** 2)

        depth_[depth_ == 0] = 0.000001
        range_[range_ == 0] = 0.000001

        phi = np.radians(45.) - np.arcsin(scan_y / range_)
        phi_ = (phi / dphi).astype(int)
        phi_[phi_ < 0] = 0
        phi_[phi_ >= 512] = 511

        theta = np.radians(2.) - np.arcsin(scan_z / depth_)
        theta_ = (theta / dtheta).astype(int)
        theta_[theta_ < 0] = 0
        theta_[theta_ >= 64] = 63

        fusion_map = np.zeros((self.proj_H, self.proj_W, self.proj_C))

        b = 1.0 * (b - b.min()) / (b.max() - b.min())
        g = 1.0 * (g - g.min()) / (g.max() - g.min())
        r = 1.0 * (r - r.min()) / (r.max() - r.min())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max() - scan_z.min())
        intensity = 1.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        depth_ = 1.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        fusion_map[theta_, phi_, 0] = b
        fusion_map[theta_, phi_, 1] = g
        fusion_map[theta_, phi_, 2] = r
        fusion_map[theta_, phi_, 3] = scan_z
        fusion_map[theta_, phi_, 4] = intensity
        fusion_map[theta_, phi_, 5] = depth_

        fusion_map = (fusion_map - mean) / std

        input_tensor = fusion_map

        # NHWC -> NCHW4
        input_tensor = input_tensor.transpose(2, 0, 1)

        input_tensor = torch.tensor(input_tensor).float()

        return input_tensor, input_tensor, self.image_dir[item]

    def __len__(self):
        return len(self.image_dir)