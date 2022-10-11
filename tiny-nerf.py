from dataclasses import dataclass
import math
from typing import List
import numpy as np
import torch
from torch import nn, optim
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
                # embedding.append(func(2.**i * torch.pi * x))  # TODO: why this does not work?
                embedding.append(func(2.**i * x))
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

    def forward(self, positions, directions):
        # encode the 3D position using a positional encoding
        positions = self.position_encoder(positions)

        # encode the 3D viweing direction vector using a positional encoding
        directions = self.direction_encoder(directions)

        # pass the encoded position through the base MLP to get the volume density and 256 dimensional feature vector
        x = self.mlp_base(positions)
        volume_density = self.relu(x[..., 0])
        feature_vector = x[..., 1:]

        if self.mlp_head is not None:
            # pass the encoded direction and feature vector through the head MLP to get the RGB color
            rgb_radiance = self.mlp_head(torch.cat([feature_vector, directions], dim=-1))
        else:
            rgb_radiance = feature_vector
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



def render_rays(config: Config, W, H, model, rays_center, rays_direction, rand=False):
    near = config.near
    far = config.far
    N_samples = config.num_coarse_samples

    # select N_samples bin starts along the ray from t_n - near bound, t_f - far bound
    t_vals = torch.linspace(near, far, N_samples, device=device).expand([W, H, N_samples])

    # for each bin, add noise for stratified sampling
    # in the paper this is represented by:
    # t_i ~ U[t_n + (i-1)/N * (t_f - t_n), t_n + i/N * (t_f - t_n)]
    if rand:
        bin_size = (far - near) / N_samples
        t_vals = t_vals + torch.rand_like(t_vals) * bin_size

    # sampled points along the ray, calculated by point = center + t * direction
    sampled_points = rays_center[..., None, :] + t_vals[..., :, None] * rays_direction[..., None, :]

    # reshape to a batch by converting to shape (W x H x N_samples, 3)
    flattened_sampled_points = sampled_points.reshape(-1, 3)
    viewing_directions = rays_direction[..., None, :].expand(sampled_points.shape).reshape(-1, 3)

    # run the model to get the volume density and RGB radiance
    opacity, color = model(flattened_sampled_points, viewing_directions)

    # reshape back to the image shape
    opacity = opacity.reshape([W, H, N_samples])
    color = color.reshape([W, H, N_samples, 3])

    # calculate the distances between the sampled points delta_i = t_(i+1) - t_i
    # doing this on the t values is the same as on sampled_points directly, since the ray direction is normalized
    # to a unit vector
    inf = torch.tensor([1e10], device=device)
    delta_ts = torch.cat([t_vals[..., 1:] - t_vals[..., :-1], inf.expand(t_vals[..., :1].shape)], dim=-1)

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

    # calculated the depth of each pixel by getting a weighted sums of the sampled points
    # and by that understanding the average point that contributed the most
    depth_map = torch.sum(weights * t_vals, dim=-1)

    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map



class NERF:
    def __init__(self, config: Config, W, H, focal, device='cuda'):
        self.config = config
        # create models
        self.coarse_model = NERFModel(config)
        self.coarse_model.to(device)
        self.coarse_optimizer = torch.optim.Adam(self.coarse_model.parameters(), lr=config.learning_rate)
        self.coarse_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.coarse_optimizer, gamma=0.99)

        if config.num_fine_samples > 0:
            self.fine_model = NERFModel(config)
            self.fine_model.to(device)
            self.fine_optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=config.learning_rate)
        
        self.W = W
        self.H = H
        self.focal = focal
        self.camera_coords = get_camera_coords(W, H, focal)
    
    def render(self, camera_pose, rand=False):
        # generate the rays
        rays_center, rays_direction = get_rays(self.camera_coords, camera_pose)

        # render the rays
        rgb_map, depth_map, acc_map = render_rays(self.config, self.W, self.H, self.coarse_model, rays_center, rays_direction, rand=rand)

        if self.config.num_fine_samples > 0:
            # TODO: render more samples
            pass

        return rgb_map, depth_map, acc_map

    def step(self, image, camera_pose):
        rgb, _, _ = nerf.render(camera_pose, rand=True)
        loss = torch.mean((rgb - image)**2)
        self.coarse_optimizer.zero_grad()
        loss.backward()
        self.coarse_optimizer.step()
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
)


device = 'cuda'

# load data
data = np.load('tiny_nerf_data.npz')
images = torch.tensor(data['images'], device=device, requires_grad=False)
poses = torch.tensor(data['poses'], device=device, requires_grad=False)
focal = torch.tensor(data['focal'], device=device, requires_grad=False)
test_pose = poses[101]
images = images[:100]
poses = poses[:100]
W, H = images.shape[1:3]

nerf = NERF(original_config, W, H, focal, device=device)

# training parameters
N_iters = 2000
plot_every = 100
epoch_size = len(images)

for i in range(N_iters):
    image_index = np.random.randint(images.shape[0])
    image, camera_pose = images[image_index], poses[image_index]
    loss = nerf.step(image, camera_pose)
    print(f'iter {i}: loss = {loss.item()}')
    if i % epoch_size == 0:
        nerf.coarse_lr_scheduler.step()
    if i % plot_every == 0:
        rgb, depth, acc = nerf.render(test_pose)
        Path('results').mkdir(exist_ok=True)
        rgb_result = Image.fromarray((rgb.detach().cpu().numpy() * 255).astype(np.uint8))
        rgb_result.save(f'results/nerf_{i}.png')
        depth_result = Image.fromarray((depth[..., None].expand(W, H, 3).detach().cpu().numpy() * 255).astype(np.uint8))
        depth_result.save(f'results/nerf_depth_{i}.png')