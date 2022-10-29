
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


import torch
from torch import nn, optim
from config import Config
from dataset import NERFDataset
from rendering_core import get_camera_coords, get_coarse_t_vals, get_fine_t_vals, get_rays, mse_to_psnr, volume_rendering
from result import Result


class NERF:
    def __init__(self, config: Config, dataset: NERFDataset, device='cuda'):
        self.config = config
        # create models
        self.coarse_model = config.network_class(config)
        self.coarse_model.to(device)
        parameters = list(self.coarse_model.parameters())

        self.fine_model = None
        if config.num_fine_samples > 0:
            self.fine_model = config.network_class(config)
            self.fine_model.to(device)
            parameters += list(self.fine_model.parameters())

        self.optimizer = optim.Adam(parameters, lr=config.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.learning_rate_decay)

        self.image_criterion = nn.MSELoss()
        self.semantics_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.W = dataset.W
        self.H = dataset.H
        self.focal = dataset.focal
        self.axis_signs = dataset.axis_signs
        self.white_background = dataset.white_background
        self.camera_coords = get_camera_coords(self.W, self.H, self.focal, self.axis_signs)
        self.camera_poses = dataset.poses
    
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
                coarse_result, fine_result = self.render_rays(rays_center[start:end], rays_direction[start:end], is_training=False)
                coarse_results.append(coarse_result)
                fine_results.append(fine_result)
            coarse_result = Result.from_batches(coarse_results)
            fine_result = Result.from_batches(fine_results)

            coarse_result.reshape((self.H, self.W))
            fine_result.reshape((self.H, self.W))

        return coarse_result, fine_result

    def render_rays(self, rays_center: torch.Tensor, rays_direction: torch.Tensor, is_training: bool=False):
        num_coarse_samples = self.config.num_coarse_samples
        num_fine_samples = self.config.num_fine_samples
        batch_size = rays_center.shape[0]  # don't take it from the config, since it might be different when testing

        coarse_t_vals = get_coarse_t_vals(batch_size, self.config.near, self.config.far, num_coarse_samples, is_training)
        weights, coarse_result = volume_rendering(self.coarse_model, batch_size, coarse_t_vals, rays_center, rays_direction, is_training=is_training)

        if self.config.num_fine_samples > 0:
            # get the t values for the fine sampling by sampling from the distribution defined 
            # by the coarse sampling done above
            fine_t_vals = get_fine_t_vals(batch_size, weights, coarse_t_vals, num_fine_samples)
            fine_t_vals = fine_t_vals.detach() # stop gradients

            # concatenate the coarse and fine t values and convert them to sampled points
            # we need to sort them to make sure we can calculate the bin widths later in volume_rendering
            all_t_vals = torch.sort(torch.cat([coarse_t_vals, fine_t_vals], dim=-1)).values

            _, fine_result = volume_rendering(self.fine_model, batch_size, all_t_vals, rays_center, rays_direction, is_training=is_training)
        else:
            fine_result = coarse_result

        return coarse_result, fine_result

    def step(self, batch):
        camera_indices, camera_coords, target_image, target_semantics = \
            batch['camera_indices'], batch['camera_coords'], batch['target_color'], batch['target_semantics']
        camera_poses = self.camera_poses[camera_indices.squeeze()]
        rays_center, rays_direction = get_rays(camera_coords, camera_poses)
        coarse_result, fine_result = self.render_rays(rays_center, rays_direction, is_training=True)

        # calculate the image loss
        coarse_rgb = coarse_result.white_rgb if self.white_background else coarse_result.rgb
        coarse_loss = self.image_criterion(coarse_rgb, target_image)
        psnr = mse_to_psnr(coarse_loss)
        loss = coarse_loss
        if self.config.num_fine_samples > 0:
            fine_rgb = fine_result.white_rgb if self.white_background else fine_result.rgb
            fine_loss = self.image_criterion(fine_rgb, target_image)
            psnr = mse_to_psnr(fine_loss)
            loss += fine_loss

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

        return loss.item(), psnr.item()

    def save(self, results_dir: str, epoch: int, total_steps: int, loss: float):
        torch.save({
            'epoch': epoch,
            'step': total_steps,
            'coarse_model_state_dict': self.coarse_model.state_dict(),
            'fine_model_state_dict': self.fine_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, f'{results_dir}/checkpoint_{total_steps}.pt')
        print(f'Saved checkpoint {total_steps}.pt to {results_dir}')

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.coarse_model.load_state_dict(checkpoint['coarse_model_state_dict'])
        self.fine_model.load_state_dict(checkpoint['fine_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loaded checkpoint {checkpoint_path}')
