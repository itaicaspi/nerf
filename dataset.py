import itertools
import json
import math
import os
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from load_blender import load_blender_data
from nerf_core import get_camera_coords, get_rays
from replica_dataset import ReplicaDatasetCache


class NERFDataset(Dataset):
    def __init__(self, dataset, keep_data_on_cpu=False, device='cuda') -> None:
        self.device = device
        self.data = self.load_data(dataset)
        self.images = torch.tensor(self.data['images'], device=device, requires_grad=False)
        self.poses = torch.tensor(self.data['poses'], device=device, requires_grad=False)
        self.focal = torch.tensor(self.data['focal'], device=device, requires_grad=False)
        self.semantics = None
        if 'semantics' in self.data:
            self.semantics = torch.tensor(self.data['semantics'], device=device, requires_grad=False)
            self.semantics = self.semantics[:-1]
            self.test_semantics = self.semantics[-1]
        self.test_pose = self.poses[-1]
        self.test_image = self.images[-1]
        self.images = self.images[:-1]
        self.poses = self.poses[:-1]
        self.W = self.images.shape[2]
        self.H = self.images.shape[1]
        self.empty_tensor = torch.Tensor()

        self.white_background = self.data.get('white_background', False)

        # axes are defined with different signs for opencv vs opengl. This is dataset dependent
        # and is emboddied in the camera pose matrices. opencv is [1, 1, 1] and opengl is [1, -1, -1]
        self.axis_signs = torch.tensor(self.data.get('axis_signs', [1, -1, -1]), device=device, requires_grad=False)

        # init inputs and targets for training
        self.camera_coords = get_camera_coords(self.W, self.H, self.focal, self.axis_signs)
        all_rays_centers, all_rays_directions = [], []
        for camera_pose in self.poses:
            rays_center, rays_direction = get_rays(self.camera_coords, camera_pose)
            all_rays_centers.append(rays_center.reshape(-1, 3))
            all_rays_directions.append(rays_direction.reshape(-1, 3))
        self.all_rays_centers = torch.cat(all_rays_centers)
        self.all_rays_directions = torch.cat(all_rays_directions)
        self.all_target_colors = self.images.reshape(-1, 3)
        self.all_target_semantics = self.semantics.reshape(-1) if self.semantics is not None else None

        # keep the data on the cpu until it's needed so that we don't run out of gpu memory
        if keep_data_on_cpu:
            self.images.to('cpu')
            self.poses.to('cpu')
            self.camera_coords.to('cpu')
            if self.semantics is not None:
                self.semantics.to('cpu')

    def load_data(self, source='tiny_lego'):
        if source == 'tiny_lego':
            return np.load('data/tiny_nerf_data.npz')
        elif source == 'lego':
            images, poses, _, hwf, _ = load_blender_data(
                "../nerf/data/nerf_synthetic/lego", False, 8)
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:]) # replace background with white
            return {
                "images": images.astype(np.float32),
                "poses": poses.astype(np.float32),
                "focal": hwf[-1],
                "white_background": True
            }
        elif source == 'replica':
            dataset = ReplicaDatasetCache('/root/home/data/semantic_nerf/room_0/Sequence_1', range(550), range(1))#, 240, 320)
            hfov = 90
            focal = dataset.train_samples['image'].shape[2] / 2.0 / math.tan(math.radians(hfov / 2.0))
            return {
                "images": dataset.train_samples['image'].astype(np.float32),
                "poses": dataset.train_samples['T_wc'].astype(np.float32),
                "focal": focal,
                "semantics": dataset.train_samples['semantic_remap_clean'].astype(np.int64),
                "axis_signs": [1, 1, 1]
            }
        elif source == 'custom':
            root_dir = Path('/root/code/nerfstudio/data/itai5')
            with open(str(root_dir/"transforms.json")) as f:
                transforms = json.load(f)
            scale_down = 8
            focal = ((transforms['fl_x'] + transforms['fl_y']) / 2) / scale_down  # TODO: support different focal lengths for non ponhole cameras
            frames = transforms['frames']
            images = []
            poses = []
            for frame in frames:
                file_path = root_dir / frame['file_path'].replace('images/', f'images_{scale_down}/' if scale_down > 1 else 'images/')
                image = Image.open(str(file_path))
                images.append(np.array(image) / 255)
                pose = np.array(frame['transform_matrix'])
                poses.append(pose)
            return {
                "images": np.array(images).astype(np.float32),
                "poses": np.array(poses).astype(np.float32),
                "focal": focal,
                "axis_signs": [1, 1, 1]
            }

    def __len__(self):
        return len(self.all_target_colors)

    def __getitem__(self, index):
        return {
            'rays_center': self.all_rays_centers[index].to(self.device),
            'rays_direction': self.all_rays_directions[index].to(self.device),
            'target_color': self.all_target_colors[index].to(self.device),
            'target_semantics': self.all_target_semantics[index].to(self.device) if self.all_target_semantics is not None else self.empty_tensor
        }
