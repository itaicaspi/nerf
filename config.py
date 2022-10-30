from dataclasses import dataclass
from typing import List
from torch import nn


@dataclass
class Config:
    network_class: nn.Module
    near: float = 2  # t_n
    far: float = 6  # t_f
    num_coarse_samples: int = 64  # N_c
    num_fine_samples: int = 0  # N_f
    learning_rate: float = 1e-4
    learning_rate_decay: float = 0.99
    pose_learning_rate: float = 1e-2
    pose_learning_rate_decay: float = 0.99
    L_position: int = 10
    L_direction: int = None
    num_layers_base: int = 8
    num_layers_head: int = None
    skip_connections: List[int] = None
    base_hidden_size: int = None
    base_output_size: int = None
    head_hidden_size: int = None
    use_separate_head_for_color: bool = True
    volume_density_regularization: float = 0 # if true, adds noise to the base net output during training
    batch_size: int = 4096
    inference_batch_size: int = 4096 * 4
    num_semantic_labels: int = 0
    semantic_loss_weight: float = 1
    train_camera_poses: bool = False
    position_encoding_coarse_to_fine_schedule: List[float] = None
