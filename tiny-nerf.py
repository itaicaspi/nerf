from dataclasses import dataclass
import itertools
import math
from symbol import parameters
from typing import List
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

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


@dataclass
class Result:
    rgb: torch.Tensor
    depth: torch.Tensor
    acc: torch.Tensor

    def reshape(self, shape):
        self.rgb = self.rgb.reshape(*shape, 3)
        self.depth = self.depth.reshape(shape)
        self.acc = self.acc.reshape(shape)


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
                embedding.append(func(2.**i * torch.pi * x))
        return torch.cat(embedding, -1)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, skip_connections=None):
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

    def forward(self, input):
        x = input
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
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
            output_size=config.base_output_size + 1 if config.use_separate_head_for_color else 4,
            num_layers=config.num_layers_base,
            skip_connections=config.skip_connections
        )

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
        volume_density, features = x[..., 0], x[..., 1:]

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
        return volume_density, rgb_radiance


def get_camera_coords(W, H, focal):
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
    camera_coords = torch.stack([x_i/focal, -y_i/focal, -torch.ones_like(u)], dim=-1)
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
        bin_size = (far - near) / num_samples
        t_vals = t_vals + torch.rand_like(t_vals) * bin_size

    return t_vals


def get_fine_t_vals(batch_size, weights, bin_mid_points, num_samples):
    with torch.no_grad():
        # convert weights to probabilities (pdf)
        weights = weights + 1e-5 # prevent division by zero
        pdf = weights / weights.sum(dim=-1, keepdim=True)

        # get cdf from the pdf
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.roll(cdf, 1, -1)
        cdf[..., 0] = 0

        # inverse transform sampling. see https://en.wikipedia.org/wiki/Inverse_transform_sampling
        # get the largest x for which cdf(x) <= u
        # start by the generating uniform random numbers in the range [0, 1]
        u = torch.rand([batch_size, num_samples], device=device)

        # for each sampled value, find the value in the cdf that is closest to u. this amounts to finding x for which cdf(x) ~= u
        # note that we don't actually find x, but the index of x in the cdf
        indices = torch.searchsorted(cdf, u, right=True)
        indices = torch.clamp(indices, 1, cdf.shape[-1] - 1)

        # get start and end points of the interval in which the sampled value lies
        bin_starts = torch.gather(bin_mid_points, -1, indices - 1)
        bin_ends = torch.gather(bin_mid_points, -1, indices)

        # get start and end cumulative probabilities of the interval in which the sampled value lies
        cdf_start = torch.gather(cdf, -1, indices - 1)
        cdf_end = torch.gather(cdf, -1, indices)

        # get the position of each sampled value in the interval normalized to 0-1
        delta_probabilities = cdf_end - cdf_start
        delta_probabilities = torch.clamp(delta_probabilities, 1e-5, 1.0) # prevent division by zero
        relative_probabilities = (u - cdf_start) / delta_probabilities

        # get the fine t vals
        fine_t_vals = bin_starts + relative_probabilities * (bin_ends - bin_starts)
    return fine_t_vals


def convert_t_vals_to_sampled_points(t_vals, rays_center, rays_direction):
    # sampled points along the ray, calculated by point = center + t * direction
    sampled_points = rays_center[..., None, :] + t_vals[..., :, None] * rays_direction[..., None, :]
    return sampled_points

def volume_rendering(batch_size, opacity, color, num_samples, t_vals, rays_direction):
    # reshape back to the image shape
    opacity = opacity.reshape([batch_size, num_samples])
    color = color.reshape([batch_size, num_samples, 3])

    # calculate the distances between the sampled points delta_i = t_(i+1) - t_i
    # doing this on the t values is the same as on sampled_points directly, since the ray direction is normalized
    # to a unit vector
    inf = torch.tensor([1e10], device=device)
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
    T_is = torch.cumprod(1 - alphas + 1e-10, dim=-1)

    # we push a 1 to the beginning of the transmittance list to make sure
    # the ray makes it at least to the first step (TODO: better understand this)
    T_is = torch.roll(T_is, 1, -1)
    T_is[..., 0] = 1.0

    # calculate the weights for each sample along the ray (see equation 5 in the paper)
    weights = T_is * alphas

    # sum the weighted colors for all the samples
    rgb_map = torch.sum(weights[..., None] * color, dim=-2)

    with torch.no_grad():
        # calculate the depth of each pixel by calculating the expected distance
        depth_map = torch.sum(weights * t_vals, dim=-1)

        # calculate the sum of the weights along each ray
        acc_map = torch.sum(weights, dim=-1)

    return weights, rgb_map, depth_map, acc_map


def render_rays(config: Config, model: nn.Module, rays_center: torch.Tensor, rays_direction: torch.Tensor, is_training: bool=False, fine_model: nn.Module = None):
    near = config.near
    far = config.far
    num_coarse_samples = config.num_coarse_samples
    num_fine_samples = config.num_fine_samples
    batch_size = rays_center.shape[0]  # don't take it from the config, since it might be different when testing

    t_vals = get_coarse_t_vals(batch_size, near, far, num_coarse_samples, is_training)
    sampled_points = convert_t_vals_to_sampled_points(t_vals, rays_center, rays_direction)

    # reshape to a batch by converting to shape (W x H x N_samples, 3)
    flattened_sampled_points = sampled_points.reshape(-1, 3)
    viewing_directions = rays_direction[..., None, :].expand([batch_size, num_coarse_samples, 3]).reshape(-1, 3)

    # run the model to get the volume density and RGB radiance
    opacity, color = model(flattened_sampled_points, viewing_directions, regularize_volume_density=is_training)

    weights, coarse_rgb_map, coarse_depth_map, coarse_acc_map = volume_rendering(batch_size, opacity, color, num_coarse_samples, t_vals, rays_direction)

    coarse_result = Result(
        rgb=coarse_rgb_map,
        depth=coarse_depth_map,
        acc=coarse_acc_map
    )

    if config.num_fine_samples > 0:
        fine_model = fine_model or model

        # get bin mid points for the fine sampling
        mid_t_vals = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])

        # get the t values for the fine sampling by sampling from the distribution defined 
        # by the coarse sampling done above
        fine_t_vals = get_fine_t_vals(batch_size, weights[..., 1:-1], mid_t_vals, num_fine_samples)

        # concatenate the coarse and fine t values and convert them to sampled points
        # we need to sort them to make sure we can calculate the bin widths later in volume_rendering
        all_t_vals = torch.cat([t_vals, fine_t_vals], dim=-1)
        all_t_vals = torch.sort(all_t_vals, dim=-1).values
        all_sampled_points = convert_t_vals_to_sampled_points(all_t_vals, rays_center, rays_direction)

        # concat the fine sampled points to the coarse sampled points and sort them so that we can later calculate
        # the distance between each sample in volume_rendering
        flattened_sampled_points = all_sampled_points.reshape(-1, 3)
        viewing_directions = rays_direction[..., None, :].expand([batch_size, num_coarse_samples + num_fine_samples, 3]).reshape(-1, 3)

        # run all the points through the fine model
        fine_opacity, fine_color = fine_model(flattened_sampled_points, viewing_directions, regularize_volume_density=is_training)

        weights, fine_rgb_map, fine_depth_map, fine_acc_map = volume_rendering(batch_size, fine_opacity, fine_color, num_coarse_samples + num_fine_samples, all_t_vals, rays_direction)
        
        fine_result = Result(
            rgb=fine_rgb_map,
            depth=fine_depth_map,
            acc=fine_acc_map
        )
    else:
        fine_result = None

    return coarse_result, fine_result



class NERF:
    def __init__(self, config: Config, W, H, focal, device='cuda'):
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
        
        self.W = W
        self.H = H
        self.focal = focal
        self.camera_coords = get_camera_coords(W, H, focal)
    
    def render(self, camera_pose, rand=False):
        # generate the rays
        rays_center, rays_direction = get_rays(self.camera_coords, camera_pose)
        rays_center = rays_center.reshape(-1, 3)
        rays_direction = rays_direction.reshape(-1, 3)

        # render the rays
        coarse_result, fine_result = self.render_rays(rays_center, rays_direction, rand=rand)

        coarse_result.reshape((self.W, self.H))
        fine_result.reshape((self.W, self.H))

        return coarse_result, fine_result

    def render_rays(self, rays_center, rays_direction, rand=False):
        # render the rays
        coarse_result, fine_result = render_rays(self.config, self.coarse_model, rays_center, rays_direction, is_training=rand)

        return coarse_result, fine_result

    def step(self, rays_center, rays_direction, target):
        coarse_result, fine_result = self.render_rays(rays_center, rays_direction, rand=True)
        loss = torch.mean((coarse_result.rgb - target)**2)

        if self.config.num_fine_samples > 0:
            fine_loss = torch.mean((fine_result.rgb - target)**2)
            loss += fine_loss

        # train coarse model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


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
    batch_size = 4096
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


device = 'cuda'


class NERFDataset(Dataset):
    def __init__(self, keep_data_on_cpu=False) -> None:
        self.data = np.load('tiny_nerf_data.npz')
        self.images = torch.tensor(self.data['images'], device=device, requires_grad=False)
        self.poses = torch.tensor(self.data['poses'], device=device, requires_grad=False)
        self.focal = torch.tensor(self.data['focal'], device=device, requires_grad=False)
        self.test_pose = self.poses[101]
        self.images = self.images[:100]
        self.poses = self.poses[:100]
        self.W = self.images.shape[1]
        self.H = self.images.shape[2]

        # init inputs and targets for training
        self.camera_coords = get_camera_coords(self.W, self.H, self.focal)
        all_rays_centers, all_rays_directions = [], []
        for camera_pose in self.poses:
            rays_center, rays_direction = get_rays(self.camera_coords, camera_pose)
            all_rays_centers.append(rays_center.reshape(-1, 3))
            all_rays_directions.append(rays_direction.reshape(-1, 3))
        self.all_rays_centers = torch.cat(all_rays_centers)
        self.all_rays_directions = torch.cat(all_rays_directions)
        self.all_target_colors = self.images.reshape(-1, 3)

        # keep the data on the cpu until it's needed so that we don't run out of gpu memory
        if keep_data_on_cpu:
            self.images.to('cpu')
            self.poses.to('cpu')
            self.camera_coords.to('cpu')

    def __len__(self):
        return len(self.all_target_colors)

    def __getitem__(self, index):
        return {
            'rays_center': self.all_rays_centers[index].to(device),
            'rays_direction': self.all_rays_directions[index].to(device),
            'target_color': self.all_target_colors[index].to(device)
        }

config = original_config

# load data
dataset = NERFDataset()
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

nerf = NERF(config, dataset.W, dataset.H, dataset.focal, device=device)

# training parameters
num_epochs = 100
plot_every_n_steps = 100

def tensor_to_image(tensor):
    return Image.fromarray((tensor.detach().cpu().numpy() * 255).astype(np.uint8))

total_steps = 0
for epoch in range(num_epochs):
    for iter, batch in enumerate(dataloader):
        loss = nerf.step(batch['rays_center'], batch['rays_direction'], batch['target_color'])
        print(f'epoch {epoch} / step {total_steps}: loss = {loss.item()}')
        total_steps += 1

        if total_steps % plot_every_n_steps == 0:
            with torch.no_grad():
                coarse_result, fine_result = nerf.render(dataset.test_pose)
                Path('results').mkdir(exist_ok=True)
                rgb_result = tensor_to_image(fine_result.rgb)
                rgb_result.save(f'results/nerf_{total_steps}.png')
                depth_result = tensor_to_image((fine_result.depth / fine_result.depth.max())[..., None].expand(dataset.W, dataset.H, 3))
                depth_result.save(f'results/nerf_depth_{total_steps}.png')
                white_bg_result = tensor_to_image(fine_result.rgb + (1 - fine_result.acc[..., None]))
                white_bg_result.save(f'results/nerf_white_{total_steps}.png')
    nerf.lr_scheduler.step()