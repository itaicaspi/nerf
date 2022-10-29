import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from config import Config
from dataset import NERFDataset
from nerf import NERF
from networks import NERFNetwork, NERFTinyCudaNetwork
from result import tensor_to_image
from remote_plot import plt


instant_ngp_config = Config(
    num_layers_base = 1,
    num_layers_head = 2,
    skip_connections = [],
    base_hidden_size = 64,
    base_output_size = 15,
    head_hidden_size = 64,
    near = 2,
    far = 6,
    num_coarse_samples = 64,
    num_fine_samples = 128,
    learning_rate = 5e-4,
    learning_rate_decay = 0.99,
    volume_density_regularization = 0,
    batch_size = 1024,
    inference_batch_size=4096,
    network_class=NERFTinyCudaNetwork
)


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
    semantic_loss_weight = 1,
    network_class=NERFNetwork
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
    batch_size = 1024,
    inference_batch_size=4096,
    network_class=NERFNetwork
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
    batch_size = 4096,
    network_class=NERFNetwork
)


# load data and setup models
device = 'cuda'
config = instant_ngp_config
results_dir = "results/lego_instant_ngp"
dataset = NERFDataset('lego')
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=16)
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
start_time = datetime.now()
total_steps = 0
for epoch in range(1, num_epochs+1):
    for iter, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, psnr = nerf.step(batch)

        total_steps += 1
        timestamp = datetime.now() - start_time
        lr = nerf.lr_scheduler.get_last_lr()[0]
        print(f'{timestamp} epoch {epoch} / step {total_steps}: lr = {lr:.5f} loss = {loss:.5f} psnr = {psnr:.5f}')
        
        if total_steps % plot_every_n_steps == 0:
            with torch.no_grad():
                coarse_result, fine_result = nerf.render_camera_pose(dataset.test_pose, batch_size=config.inference_batch_size)
                fine_result.save(f'{results_dir}/{total_steps}')
                plt.imshow_native(tensor_to_image(fine_result.white_rgb))
        
        if total_steps % step_lr_every_n_steps == 0:
            nerf.lr_scheduler.step()

        if total_steps % save_checkpoint_every_n_steps == 0:
            nerf.save(results_dir, epoch, total_steps, loss)