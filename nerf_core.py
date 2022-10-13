

import torch
from torch import nn
import numpy as np
from PIL import Image

from config import Config
from result import Result


device = 'cuda'

inf = torch.tensor([1e10], device=device, requires_grad=False)
eps = torch.tensor([1e-10], device=device, requires_grad=False)



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

def volume_rendering(model, batch_size, t_vals, rays_center, rays_direction, is_training=False):
    num_samples = t_vals.shape[-1]

    all_sampled_points = convert_t_vals_to_sampled_points(t_vals, rays_center, rays_direction)

    # concat the fine sampled points to the coarse sampled points and sort them so that we can later calculate
    # the distance between each sample in volume_rendering
    flattened_sampled_points = all_sampled_points.reshape(-1, 3)
    viewing_directions = rays_direction[..., None, :].expand([batch_size, num_samples, 3]).reshape(-1, 3)
    normalized_viewing_directions = viewing_directions / torch.norm(viewing_directions, dim=-1, keepdim=True)

    # run all the points through the fine model
    opacity, color, semantics_logits = model(flattened_sampled_points, normalized_viewing_directions, is_training=is_training)

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
    weights = alphas * T_is

    # sum the weighted colors for all the samples
    rgb_map = torch.sum(weights[..., None] * color, dim=-2)

    # calculate the semantic classes map
    semantics = torch.sum(weights[..., None] * semantics_logits, dim=-2) if semantics_logits is not None else None
    
    # calculate the sum of the weights along each ray
    acc_map = torch.sum(weights, dim=-1)

    # calculate an rgb map that has a white background
    white_rgb_map = rgb_map + (1 - acc_map[..., None])

    with torch.no_grad():
        # calculate the depth of each pixel by calculating the expected distance
        depth_map = torch.sum(weights * t_vals, dim=-1)

        # calculate the disparity map
        disparity_map = 1 / torch.maximum(eps, depth_map / torch.maximum(eps, acc_map))

    result = Result(
        rgb=rgb_map,
        white_rgb=white_rgb_map,
        depth=depth_map,
        acc=acc_map,
        disparity=disparity_map,
        semantics=semantics
    )
    return weights, result




def mse_to_psnr(loss): 
    return -10.*torch.log(loss)/np.log(10.)
