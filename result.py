
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
import numpy as np


def tensor_to_image(tensor, scale=False, shift=False):
    if shift:
        tensor = tensor - torch.min(tensor)
    if scale:
        tensor = tensor / (torch.max(tensor) + 1e-10)
    return Image.fromarray((tensor.detach().cpu().numpy() * 255).astype(np.uint8))


@dataclass
class Result:
    rgb: torch.Tensor = None
    white_rgb: torch.Tensor = None
    depth: torch.Tensor = None
    acc: torch.Tensor = None
    disparity: torch.Tensor = None
    semantics: torch.Tensor = None

    def reshape(self, shape):
        self.rgb = self.rgb.reshape(*shape, 3)
        self.white_rgb = self.white_rgb.reshape(*shape, 3)
        self.depth = self.depth.reshape(shape)
        self.acc = self.acc.reshape(shape)
        self.disparity = self.disparity.reshape(shape)
        self.semantics = self.semantics.reshape(*shape, self.semantics.shape[-1]) if self.semantics is not None else None
    
    @staticmethod
    def from_batches(batches):
        return Result(
            rgb=torch.cat([b.rgb for b in batches], dim=0),
            white_rgb=torch.cat([b.white_rgb for b in batches], dim=0),
            depth=torch.cat([b.depth for b in batches], dim=0),
            acc=torch.cat([b.acc for b in batches], dim=0),
            disparity=torch.cat([b.disparity for b in batches], dim=0),
            semantics=torch.cat([b.semantics for b in batches], dim=0) if batches[0].semantics is not None else None,
        )

    def save(self, root_dir: str):
        root_dir = Path(root_dir)
        root_dir.mkdir(exist_ok=True, parents=True)
        rgb_result = tensor_to_image(self.rgb)
        depth_result = tensor_to_image(self.depth, shift=True, scale=True)
        white_bg_result = tensor_to_image(self.white_rgb)
        disparity_result = tensor_to_image(self.disparity, scale=True)
        acc_result = tensor_to_image(self.acc, scale=True, shift=True)

        rgb_result.save(str(root_dir/f'rgb.png'))
        depth_result.save(str(root_dir/f'depth.png'))
        white_bg_result.save(str(root_dir/f'white.png'))
        disparity_result.save(str(root_dir/f'disparity.png'))
        acc_result.save(str(root_dir/f'acc.png'))

        if self.semantics is not None:
            semantics_result = tensor_to_image(self.semantics.argmax(dim=-1), shift=True, scale=True)
            semantics_result.save(str(root_dir/f'semantics.png'))

