import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from config import Config
from dataset import NERFDataset

from nerf_core import get_camera_coords, get_rays, mse_to_psnr, render_rays
from networks import NERFModel
from result import Result, tensor_to_image

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


class NERF:
    def __init__(self, config: Config, dataset: NERFDataset, device='cuda'):
        self.config = config
        # create models
        self.coarse_model = NERFModel(config)
        self.coarse_model.to(device)
        parameters = list(self.coarse_model.parameters())

        self.fine_model = None
        if config.num_fine_samples > 0:
            self.fine_model = NERFModel(config)
            self.fine_model.to(device)
            parameters += list(self.fine_model.parameters())

        self.optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        self.image_criterion = nn.MSELoss()
        self.semantics_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.W = dataset.W
        self.H = dataset.H
        self.focal = dataset.focal
        self.axis_signs = dataset.axis_signs
        self.white_background = dataset.white_background
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
            'coarse_model_state_dict': nerf.coarse_model.state_dict(),
            'fine_model_state_dict': nerf.fine_model.state_dict(),
            'optimizer_state_dict': nerf.optimizer.state_dict(),
            'loss': loss,
        }, f'{results_dir}/checkpoint_{total_steps}.pt')
        print(f'Saved checkpoint {total_steps}.pt to {results_dir}')

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.coarse_model.load_state_dict(checkpoint['coarse_model_state_dict'])
        self.fine_model.load_state_dict(checkpoint['fine_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loaded checkpoint {checkpoint_path}')


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
    volume_density_regularization = 0.1,
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
    num_fine_samples = 128,
    L_position = 10,
    L_direction = 4,
    learning_rate = 5e-4,
    volume_density_regularization = 1,
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
config = original_config
results_dir = "results/lego"
dataset = NERFDataset('lego')
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
nerf = NERF(config, dataset, device=device)

# training parameters
num_epochs = 100
plot_every_n_steps = 500
step_lr_every_n_steps = 1000
save_checkpoint_every_n_steps = 10000



# save reference target images
Path(results_dir).mkdir(exist_ok=True, parents=True)
tensor_to_image(dataset.test_image).save(f'{results_dir}/test_image.png')
if config.num_semantic_labels > 0:
    tensor_to_image(dataset.test_semantics, shift=True, scale=True).save(f'{results_dir}/test_semantics.png')

# train
total_steps = 0
for epoch in range(1, num_epochs+1):
    for iter, batch in enumerate(dataloader):
        loss, psnr = nerf.step(batch['rays_center'], batch['rays_direction'], batch['target_color'], batch['target_semantics'])

        total_steps += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        lr = nerf.lr_scheduler.get_last_lr()[0]
        print(f'{timestamp} epoch {epoch} / step {total_steps}: lr = {lr:.5f} loss = {loss:.5f} psnr = {psnr:.5f}')
        
        if total_steps % plot_every_n_steps == 0:
            with torch.no_grad():
                coarse_result, fine_result = nerf.render_camera_pose(dataset.test_pose, batch_size=config.batch_size)
                fine_result.save(f'{results_dir}/{total_steps}')
        
        if total_steps % step_lr_every_n_steps == 0:
            nerf.lr_scheduler.step()

        if total_steps % save_checkpoint_every_n_steps == 0:
            nerf.save(results_dir, epoch, total_steps, loss)