

import torch
from torch import nn

from config import Config

class PositionEncoding(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.output_size = 3 * (2 * L + 1)
        self.scale_frequencies_by_pi = False
        self.freq_bands = 2.**torch.linspace(0., L - 1, L)
        if self.scale_frequencies_by_pi:
            self.freq_bands *= torch.pi

    def forward(self, x):
        # positional encodings are [x, sin(2^0 * pi * x), cos(2^0 * pi * x), sin(2^1 * pi * x), cos(2^1 * pi * x), ...]
        embedding = [x]
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                embedding.append(func(x * freq))
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
        self.features_linear = nn.Linear(config.base_output_size, config.base_output_size)
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
        self.volume_density_regularization = config.volume_density_regularization

    def forward(self, positions, directions, is_training=False):
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
        if is_training and self.volume_density_regularization > 0:
            volume_density = volume_density + torch.randn_like(volume_density) * self.volume_density_regularization
        volume_density = self.relu(volume_density)

        if self.mlp_head is not None:
            # pass the encoded direction and feature vector through the head MLP to get the RGB color
            rgb_radiance = self.mlp_head(torch.cat([features, directions], dim=-1))
        else:
            rgb_radiance = features
        rgb_radiance = self.sigmoid(rgb_radiance)

        # return the opacity (volume density) and color (RGB radiance)
        return volume_density, rgb_radiance, semantics_logits
