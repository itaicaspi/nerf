from dataclasses import dataclass
import itertools
import json
import math
import os
from symbol import parameters
from typing import List
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

from load_blender import load_blender_data
from replica_dataset import ReplicaDatasetCache

"""
References:
tiny-nerf colab
https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb#scrollTo=R1avtwVoAQTu

pytorch implementation
https://github.com/airalcorn2/pytorch-nerf/blob/master/run_nerf.py

nerf paper:
https://arxiv.org/pdf/2003.08934.pdf

camera extrinsics and intrinsics tutorial:
https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec

alpha compositing:
https://en.wikipedia.org/wiki/Alpha_compositing

nerf code:
https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L110
"""

@dataclass
class Config:
    near: float = 2  # t_n
    far: float = 6  # t_f
    num_coarse_samples: int = 64  # N_c
    num_fine_samples: int = 0  # N_f
    learning_rate: float = 1e-4
    L_position: int = 10
    L_direction: int = None
    num_layers_base: int = 8
    num_layers_head: int = None
    skip_connections: List[int] = None
    base_hidden_size: int = None
    base_output_size: int = None
    head_hidden_size: int = None
    use_separate_head_for_color: bool = True
    regularize_volume_density: bool = False # if true, adds noise to the base net output during training
    batch_size: int = 4096
    num_semantic_labels: int = 0
    semantic_loss_weight: float = 1


@dataclass
class Result:
    rgb: torch.Tensor = None
    depth: torch.Tensor = None
    acc: torch.Tensor = None
    disparity: torch.Tensor = None
    semantics: torch.Tensor = None

    def reshape(self, shape):
        self.rgb = self.rgb.reshape(*shape, 3)
        self.depth = self.depth.reshape(shape)
        self.acc = self.acc.reshape(shape)
        self.disparity = self.disparity.reshape(shape)
        self.semantics = self.semantics.reshape(*shape, self.semantics.shape[-1]) if self.semantics is not None else None
    
    @staticmethod
    def from_batches(batches):
        return Result(
            rgb=torch.cat([b.rgb for b in batches], dim=0),
            depth=torch.cat([b.depth for b in batches], dim=0),
            acc=torch.cat([b.acc for b in batches], dim=0),
            disparity=torch.cat([b.disparity for b in batches], dim=0),
            semantics=torch.cat([b.semantics for b in batches], dim=0) if batches[0].semantics is not None else None,
        )

    def save(self, root_dir: str):
        root_dir = Path(root_dir)
        root_dir.mkdir(exist_ok=True, parents=True)
        rgb_result = tensor_to_image(self.rgb)
        depth_result = tensor_to_image((1 - self.depth / self.depth.max())[..., None].expand(dataset.H, dataset.W, 3))
        white_bg_result = tensor_to_image(self.rgb + (1 - self.acc[..., None]))
        disparity_result = tensor_to_image(self.disparity, normalize=True)
        acc_result = tensor_to_image(self.acc, normalize=True)

        rgb_result.save(str(root_dir/f'rgb.png'))
        depth_result.save(str(root_dir/f'depth.png'))
        white_bg_result.save(str(root_dir/f'white.png'))
        disparity_result.save(str(root_dir/f'disparity.png'))
        acc_result.save(str(root_dir/f'acc.png'))

        if self.semantics is not None:
            semantics_result = tensor_to_image(self.semantics.argmax(dim=-1), normalize=True)
            semantics_result.save(str(root_dir/f'semantics.png'))


class PositionEncoding(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.output_size = 3 * (2 * L + 1)

    def forward(self, x):
        # positional encodings are [x, sin(2^0 * pi * x), cos(2^0 * pi * x), sin(2^1 * pi * x), cos(2^1 * pi * x), ...]
        embedding = [x]
        for i in range(self.L):
            for func in [torch.sin, torch.cos]:
                embedding.append(func(2.**i * x))
        return torch.cat(embedding, -1)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, skip_connections=None, output_activation=False):
        super().__init__()

        self.skip_connections = skip_connections or []

        # set layers
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        for i in range(1, num_layers):
            if i - 1 in self.skip_connections:
                self.layers.append(nn.Linear(hidden_size + input_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.activation = nn.ReLU()
        self.output_activation = output_activation

    def forward(self, input):
        x = input
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1 or self.output_activation:
                x = self.activation(x)
            if index in self.skip_connections:
                x = torch.cat([input, x], -1)
        return x


class NERFModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.position_encoder = PositionEncoding(config.L_position)
        self.direction_encoder = PositionEncoding(config.L_direction)

        # take the encoded position and output the volume density and 256 dimensional feature vector
        self.mlp_base = MLP(
            input_size=self.position_encoder.output_size, 
            hidden_size=config.base_hidden_size,
            output_size=config.base_output_size if config.use_separate_head_for_color else 4,
            num_layers=config.num_layers_base,
            skip_connections=config.skip_connections,
            output_activation=True
        )

        self.volume_density_linear = nn.Linear(config.base_output_size, 1)
        self.features_linear = nn.Linear(config.base_output_size, 256)
        self.semantics_linear = None
        if config.num_semantic_labels > 0:
            self.semantics_linear = nn.Linear(config.base_output_size, config.num_semantic_labels)

        # take the encoded direction and the 256 dimensional feature vector and output the RGB color
        if config.use_separate_head_for_color:
            self.mlp_head = MLP(
                input_size=self.direction_encoder.output_size + config.base_output_size,
                hidden_size=config.head_hidden_size, 
                output_size=3, 
                num_layers=config.num_layers_head)
        else:
            self.mlp_head = None

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, positions, directions, regularize_volume_density=False):
        # encode the 3D position using a positional encoding
        positions = self.position_encoder(positions)

        # encode the 3D viweing direction vector using a positional encoding
        directions = self.direction_encoder(directions)

        # pass the encoded position through the base MLP to get the volume density and 256 dimensional feature vector
        x = self.mlp_base(positions)
        volume_density = self.volume_density_linear(x)
        features = self.features_linear(x)
        semantics_logits = self.semantics_linear(x) if self.semantics_linear is not None else None

        # if we are training, add noise to the volume density to encourage regularization and prevent floater artifacts
        # See Appendix A of the paper
        if regularize_volume_density:
            volume_density = volume_density + torch.randn_like(volume_density)
        volume_density = self.relu(volume_density)

        if self.mlp_head is not None:
            # pass the encoded direction and feature vector through the head MLP to get the RGB color
            rgb_radiance = self.mlp_head(torch.cat([features, directions], dim=-1))
        else:
            rgb_radiance = features
        rgb_radiance = self.sigmoid(rgb_radiance)

        # return the opacity (volume density) and color (RGB radiance)
        return volume_density, rgb_radiance, semantics_logits


def get_camera_coords(W: int, H: int, focal: float, axis_signs: torch.Tensor):
    # generate two matrices representing the 2D coordinates at each point on the
    # pixel plane
    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')

    # now we need to convert from the pixel plane to the image plane.
    # first we translate the origin to the center of the image plane.
    # we ignore the scaling factor from pixel to meters for now.
    c_x, c_y = W/2, H/2
    x_i, y_i = u - c_x, v - c_y

    # convert the coordinates from the image plane to the camera plane 
    # 3D coordinates. we set z_c = 1, and use it to calculate x_c and y_c.
    # this is based on the similar triangles rule: x_c / z_c = x_i / f
    camera_coords = torch.stack([x_i/focal, y_i/focal, torch.ones_like(u)], dim=-1)
    camera_coords = camera_coords * axis_signs
    return camera_coords


def get_rays(camera_coords, camera_pose):
    # now that we have coordinates in the camera plane, we can convert them to
    # the world plane using the camera pose (camera extrinsics / camera to world)
    # rays are represented by r(t) = o + t*d.
    # this is done by R @ camera_coords
    # | R00 R01 R02 | | X_c |
    # | R10 R11 R12 | | Y_c |
    # | R20 R21 R22 | | Z_c |
    rays_direction = (camera_pose[:3, :3] @ camera_coords[..., None]).squeeze()
    rays_center = camera_pose[:3, -1].expand(rays_direction.shape)  # this is the camera center located in the world frame
    return rays_center, rays_direction


def get_coarse_t_vals(batch_size, near, far, num_samples, is_training=False):
    # select N_samples bin starts along the ray from t_n - near bound, t_f - far bound
    t_vals = torch.linspace(near, far, num_samples, device=device).expand([batch_size, num_samples])

    # for each bin, add noise for stratified sampling
    # in the paper this is represented by:
    # t_i ~ U[t_n + (i-1)/N * (t_f - t_n), t_n + i/N * (t_f - t_n)]
    if is_training:
        mid_t_vals = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        bin_starts = torch.cat([t_vals[..., :1], mid_t_vals], dim=-1)
        bin_ends = torch.cat([mid_t_vals, t_vals[..., -1:]], dim=-1)
        t_vals = bin_starts + torch.rand_like(t_vals) * (bin_ends - bin_starts)

    return t_vals


def get_fine_t_vals(batch_size, weights, t_vals, num_samples):
    with torch.no_grad():
        # convert weights to probabilities (pdf)
        weights = weights[..., 1:-1] + 1e-5 # prevent division by zero
        pdf = weights / weights.sum(dim=-1, keepdim=True)

        # get cdf from the pdf
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat((torch.zeros_like(cdf[..., 0, None]), cdf), dim=-1)

        t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        
        # inverse transform sampling. see https://en.wikipedia.org/wiki/Inverse_transform_sampling
        # get the largest x for which cdf(x) <= u
        # start by the generating uniform random numbers in the range [0, 1]
        u = torch.rand([batch_size, num_samples], device=device)

        # for each sampled value, find the value in the cdf that is closest to u. this amounts to finding x for which cdf(x) ~= u
        # note that we don't actually find x, but the index of x in the cdf
        indices = torch.searchsorted(cdf, u, right=True)
        start_indices = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
        end_indices = torch.clamp(indices, 0, cdf.shape[-1] - 1)

        # get start and end points of the interval in which the sampled value lies
        bin_starts = torch.gather(t_mids, -1, start_indices)
        bin_ends = torch.gather(t_mids, -1, end_indices)

        # get start and end cumulative probabilities of the interval in which the sampled value lies
        cdf_start = torch.gather(cdf, -1, start_indices)
        cdf_end = torch.gather(cdf, -1, end_indices)

        # get the position of each sampled value in the interval normalized to 0-1
        delta_probabilities = cdf_end - cdf_start
        # delta_probabilities = torch.clamp(delta_probabilities, 1e-5, 1.0) # prevent division by zero
        delta_probabilities = torch.where(delta_probabilities < 1e-5, torch.ones_like(delta_probabilities), delta_probabilities) # this seems like a mistake but I am not sure about it.
        relative_probabilities = (u - cdf_start) / delta_probabilities

        # get the fine t vals
        fine_t_vals = bin_starts + relative_probabilities * (bin_ends - bin_starts)
    return fine_t_vals


def convert_t_vals_to_sampled_points(t_vals, rays_center, rays_direction):
    # sampled points along the ray, calculated by point = center + t * direction
    sampled_points = rays_center[..., None, :] + t_vals[..., :, None] * rays_direction[..., None, :]
    return sampled_points

def volume_rendering(model, batch_size, num_samples, t_vals, rays_center, rays_direction, is_training=False):

    all_sampled_points = convert_t_vals_to_sampled_points(t_vals, rays_center, rays_direction)

    # concat the fine sampled points to the coarse sampled points and sort them so that we can later calculate
    # the distance between each sample in volume_rendering
    flattened_sampled_points = all_sampled_points.reshape(-1, 3)
    viewing_directions = rays_direction[..., None, :].expand([batch_size, num_samples, 3]).reshape(-1, 3)
    normalized_viewing_directions = viewing_directions / torch.norm(viewing_directions, dim=-1, keepdim=True)

    # run all the points through the fine model
    opacity, color, semantics_logits = model(flattened_sampled_points, normalized_viewing_directions, regularize_volume_density=is_training)

    # # reshape back to the image shape
    opacity = opacity.reshape([batch_size, num_samples])
    color = color.reshape([batch_size, num_samples, 3])
    semantics_logits = semantics_logits.reshape([batch_size, num_samples, -1]) if semantics_logits is not None else None

    # calculate the distances between the sampled points delta_i = t_(i+1) - t_i
    # doing this on the t values is the same as on sampled_points directly, since the ray direction is normalized
    # to a unit vector
    delta_ts = torch.cat([t_vals[..., 1:] - t_vals[..., :-1], inf.expand(t_vals[..., :1].shape)], dim=-1)

    # Multiply the distance between the sampled points by the norm of the rays direction
    # to account for non-unit direction vectors
    delta_ts = delta_ts * torch.linalg.norm(rays_direction[..., None, :], dim=-1)

    # calculate the density of each sample along the ray. Higher values imply higher likelihood of being absorbed
    # at this point
    alphas = 1 - torch.exp(-opacity * delta_ts)

    # In the paper (equation 3) they calculate T_i (the accumulated transmittance along the ray) as:
    # T_i = exp(-sum_{j=1}^{i-1} (sigma_j * delta_j))
    # this is the same as replacing the sum in the power as a product of the exponentials:
    # T_i = exp(-sum_{j=1}^{i-1} (sigma_j * delta_j)) = prod_{j=1}^{i-1} [ exp(-sigma_j * delta_j) ]
    # since a_i = 1 - exp(-sigma_i * delta_i), we can simplify this to:
    # T_i = prod_{j=1}^{i-1} [ 1 - a_j ]
    T_is = torch.cumprod(1 - alphas + eps, dim=-1)

    # we push a 1 to the beginning of the transmittance list to make sure
    # the ray makes it at least to the first step (TODO: better understand this)
    T_is = torch.roll(T_is, 1, -1)
    T_is[..., 0] = 1.0

    # calculate the weights for each sample along the ray (see equation 5 in the paper)
    weights = T_is * alphas

    # sum the weighted colors for all the samples
    rgb_map = torch.sum(weights[..., None] * color, dim=-2)

    # calculate the semantic classes map
    semantics = torch.sum(weights[..., None] * semantics_logits, dim=-2) if semantics_logits is not None else None
    
    with torch.no_grad():
        # calculate the depth of each pixel by calculating the expected distance
        depth_map = torch.sum(weights * t_vals, dim=-1)

        # calculate the sum of the weights along each ray
        acc_map = torch.sum(weights, dim=-1)

        # calculate the disparity map
        disparity_map = 1 / torch.maximum(eps, depth_map / torch.maximum(eps, acc_map))

    result = Result(
        rgb=rgb_map,
        depth=depth_map,
        acc=acc_map,
        disparity=disparity_map,
        semantics=semantics
    )
    return weights, result


def render_rays(config: Config, model: nn.Module, rays_center: torch.Tensor, rays_direction: torch.Tensor, is_training: bool=False, fine_model: nn.Module = None):
    num_coarse_samples = config.num_coarse_samples
    num_fine_samples = config.num_fine_samples
    batch_size = rays_center.shape[0]  # don't take it from the config, since it might be different when testing

    coarse_t_vals = get_coarse_t_vals(batch_size, config.near, config.far, num_coarse_samples, is_training)

    weights, coarse_result = volume_rendering(model, batch_size, num_coarse_samples, coarse_t_vals, rays_center, rays_direction, is_training=is_training)

    if config.num_fine_samples > 0:
        # get the t values for the fine sampling by sampling from the distribution defined 
        # by the coarse sampling done above
        fine_t_vals = get_fine_t_vals(batch_size, weights, coarse_t_vals, num_fine_samples)

        # concatenate the coarse and fine t values and convert them to sampled points
        # we need to sort them to make sure we can calculate the bin widths later in volume_rendering
        all_t_vals = torch.sort(torch.cat([coarse_t_vals, fine_t_vals], dim=-1)).values

        _, fine_result = volume_rendering(fine_model or model, batch_size, num_coarse_samples + num_fine_samples, all_t_vals, rays_center, rays_direction, is_training=is_training)
    else:
        fine_result = None

    return coarse_result, fine_result


class NERFDataset(Dataset):
    def __init__(self, dataset, keep_data_on_cpu=False) -> None:
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

        # axes are defined with different signs for opencv vs opengl. This is dataset dependent
        # and is emboddied in the camera pose matrices
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
            return np.load('tiny_nerf_data.npz')
        elif source == 'lego':
            images, poses, _, hwf, _ = load_blender_data(
                "../nerf/data/nerf_synthetic/lego", True, 8)
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:]) # replace background with white
            return {
                "images": images.astype(np.float32),
                "poses": poses.astype(np.float32),
                "focal": hwf[-1]
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

    def __len__(self):
        return len(self.all_target_colors)

    def __getitem__(self, index):
        return {
            'rays_center': self.all_rays_centers[index].to(device),
            'rays_direction': self.all_rays_directions[index].to(device),
            'target_color': self.all_target_colors[index].to(device),
            'target_semantics': self.all_target_semantics[index].to(device) if self.all_target_semantics is not None else self.empty_tensor
        }


class NERF:
    def __init__(self, config: Config, dataset: NERFDataset, device='cuda'):
        self.config = config
        # create models
        self.coarse_model = NERFModel(config)
        self.coarse_model.to(device)
        parameters = self.coarse_model.parameters()

        if config.num_fine_samples > 0:
            self.fine_model = NERFModel(config)
            self.fine_model.to(device)
            parameters = itertools.chain(parameters, self.fine_model.parameters())

        self.optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        self.image_criterion = nn.MSELoss()
        self.semantics_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.W = dataset.W
        self.H = dataset.H
        self.focal = dataset.focal
        self.axis_signs = dataset.axis_signs
        self.camera_coords = get_camera_coords(self.W, self.H, self.focal, self.axis_signs)
    
    def render_camera_pose(self, camera_pose, batch_size=1024):
        with torch.no_grad():
            # generate the rays
            rays_center, rays_direction = get_rays(self.camera_coords, camera_pose)
            rays_center = rays_center.reshape(-1, 3)
            rays_direction = rays_direction.reshape(-1, 3)

            # render the rays
            coarse_results, fine_results = [], []
            for i in range(0, rays_center.shape[0], batch_size):
                start = i
                end = min(i + batch_size, rays_center.shape[0])
                coarse_result, fine_result = self.render_rays(rays_center[start:end], rays_direction[start:end], rand=False)
                coarse_results.append(coarse_result)
                fine_results.append(fine_result)
            coarse_result = Result.from_batches(coarse_results)
            fine_result = Result.from_batches(fine_results)

            coarse_result.reshape((self.H, self.W))
            fine_result.reshape((self.H, self.W))

        return coarse_result, fine_result

    def render_rays(self, rays_center, rays_direction, rand=False):
        # render the rays
        coarse_result, fine_result = render_rays(self.config, self.coarse_model, rays_center, rays_direction, is_training=rand, fine_model=self.fine_model)

        return coarse_result, fine_result

    def step(self, rays_center, rays_direction, target_image, target_semantics=None):
        coarse_result, fine_result = self.render_rays(rays_center, rays_direction, rand=True)

        # calculate the image loss
        loss = self.image_criterion(coarse_result.rgb, target_image)
        if self.config.num_fine_samples > 0:
            loss += self.image_criterion(fine_result.rgb, target_image)

        # calculate the semantics loss
        if self.config.num_semantic_labels > 0:
            semantic_loss = self.semantics_criterion(coarse_result.semantics, target_semantics - 1)
            if self.config.num_fine_samples > 0:
                semantic_loss += self.semantics_criterion(fine_result.semantics, target_semantics - 1)
            loss += (self.config.semantic_loss_weight * semantic_loss)

        # train coarse model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


semantics_config = Config(
    num_layers_base = 8,
    num_layers_head = 1,
    skip_connections = [4],
    base_hidden_size = 256,
    base_output_size = 256,
    head_hidden_size = 128,
    near = .1,
    far = 10,
    num_coarse_samples = 64,
    num_fine_samples=128,
    L_position = 10,
    L_direction = 4,
    learning_rate = 5e-4,
    regularize_volume_density = True,
    batch_size = 1024,
    num_semantic_labels = 27,
    semantic_loss_weight = 1
)


original_config = Config(
    num_layers_base = 8,
    num_layers_head = 1,
    skip_connections = [4],
    base_hidden_size = 256,
    base_output_size = 256,
    head_hidden_size = 128,
    near = 2,
    far = 6,
    num_coarse_samples = 64,
    num_fine_samples=128,
    L_position = 10,
    L_direction = 4,
    learning_rate = 5e-4,
    regularize_volume_density = True,
    batch_size = 1024
)

tiny_config = Config(
    num_layers_base = 8,
    skip_connections = [4],
    use_separate_head_for_color = False,
    base_hidden_size = 256,
    near = 2,
    far = 6,
    num_coarse_samples = 64,
    L_position = 6,
    L_direction = 4,
    learning_rate = 5e-4,
    batch_size = 4096
)


# load data and setup models
device = 'cuda'
config = semantics_config
dataset = NERFDataset('replica')
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
nerf = NERF(config, dataset, device=device)

# training parameters
num_epochs = 100
plot_every_n_steps = 100
inf = torch.tensor([1e10], device=device)
eps = torch.tensor([1e-10], device=device)

def tensor_to_image(tensor, normalize=False):
    if normalize:
        tensor = tensor - torch.min(tensor)
        tensor = tensor / (torch.max(tensor) + eps)
    return Image.fromarray((tensor.detach().cpu().numpy() * 255).astype(np.uint8))

tensor_to_image(dataset.test_image).save(f'results/test_image.png')
if config.num_semantic_labels > 0:
    tensor_to_image(dataset.test_semantics, normalize=True).save(f'results/test_semantics.png')

total_steps = 0
for epoch in range(num_epochs):
    for iter, batch in enumerate(dataloader):
        loss = nerf.step(batch['rays_center'], batch['rays_direction'], batch['target_color'], batch['target_semantics'])
        print(f'epoch {epoch+1} / step {total_steps+1}: loss = {loss.item()}')
        total_steps += 1

        if total_steps % plot_every_n_steps == 0:
            with torch.no_grad():
                coarse_result, fine_result = nerf.render_camera_pose(dataset.test_pose, batch_size=config.batch_size)
                fine_result.save(f'results/{total_steps}')
    nerf.lr_scheduler.step()