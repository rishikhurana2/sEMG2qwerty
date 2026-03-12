# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn

import torchvision.models as models
import math


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T) # (N, Channels, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width), # width * channels = num_features
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

class ResNet18Model(nn.Module):
    def __init__(self, channels_in: int, out_features: int):
        super().__init__()
        resnet_model = models.resnet18(weights=None)
		
        self.conv1 = nn.Conv2d(channels_in, out_channels=64, padding=1, stride=1, bias=False, kernel_size=3)
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
		
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, out_features)
	
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T,N,B,C,F = x.shape
        x = x.reshape(T*N,B,C,F)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #pooling
        x = self.pooling(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x.reshape(T,N,-1)
		
class LSTMEncoder(nn.Module):
    def __init__(
        self, 
        num_features: int,
        input_size: int, 
        hidden_size: int = 256, 
        num_layers: int = 3,    
        dropout: float = 0.3,   
        bidirectional: bool = True
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, 
            bidirectional=bidirectional,
            batch_first=False
        )
        print("Num LSTM layers: ", num_layers)
        print("Layer size: ", hidden_size)

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        x = self.proj(output)
        x = x + inputs
        x = self.layer_norm(x)
 
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pos_encoding)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)  # (T,1,C)

""" Single head attention layer -- implements the formulas for self-attention """
class SingleHeadAttentionLayer(nn.Module):
    """ An attention layer for processing after spectogram normalization and rotation invariant processing
    
    Args:
        num_features (int): dimension of a head for input of size
        (T, N, num_features)
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.Wq = nn.Linear(num_features, num_features)
        self.Wk = nn.Linear(num_features, num_features)
        self.Wv = nn.Linear(num_features, num_features)

    def forward(self, inputs):
        T_in, N, C = inputs.shape # C = num_features

        x = inputs.permute(1, 0, 2) # (N, T_in, num_features)
        
        Q = self.Wq(x) # (N, T_in, num_features)
        K = self.Wk(x)
        V = self.Wv(x)

        # a = softmax(QK^T/sqrt(d))V
        attention = torch.softmax(torch.matmul(Q, torch.transpose(K, 1, 2)) / math.sqrt(C), dim=2) # (N, T_in, T_in)
        delta     = torch.matmul(attention, V) # (N, T_in, num_features)

        return delta.permute(1, 0, 2) # (T_in, N, num_features)

""" Multi head attention layer -- uses PyTorch's nuilt in functionality to take full advantage of parallelism"""
class MultiHeadAttentionLayer(nn.Module):
    """ A multi-headed attention layer for processing
    
    Args: 
        num_features (int) : total dimension of multi-heading layer
        num_heads (int) : number of heads for this layer. We have that head_dim = num_features // num_heads
        dropout (float) : dropout used in training mode (default is 0.0, which is no dropout)
    """

    def __init__(self, num_features : int, num_heads : int = 12, dropout : float = 0.0):
        super().__init__()        
        # with batch_first=False, we can operate on (T, N, C)
        self.attn = nn.MultiheadAttention(num_features, num_heads, dropout, batch_first=False)
    
    def forward(self, inputs):
        attn_output, _ = self.attn(inputs, inputs, inputs)
        return attn_output # for self-attention, the q,k,v are all equal to each other



""" Transformer Block that uses single head attention """
class SingleHeadAttentionTransformerLayer(nn.Module):
    """
        Transformer layer that consists of Attention and fully connected layer
    """

    def __init__(self, num_features : int):
        super().__init__()

        self.ln1 = nn.LayerNorm(num_features)
        self.attn_layer = SingleHeadAttentionLayer(num_features)

        self.ln2 = nn.LayerNorm(num_features)
        self.fc_block = nn.Sequential(
            nn.Linear(num_features, 4 * num_features),
            nn.ReLU(),
            nn.Linear(4 * num_features, num_features),
        )
    
    def forward(self, x):
        # normalize + attention (with skip connection)
        x = x + self.attn_layer(self.ln1(x))
        
        # normalize + MLP (with skip connection)
        x = x + self.fc_block(self.ln2(x))

        return x

""" Transformer Block that uses multi head attention """
class MultiHeadAttentionTransformerLayer(nn.Module):
    def __init__(self, num_features : int, num_heads : int = 12, dropout : float = 0.0):
        super().__init__()

        self.ln1 = nn.LayerNorm(num_features)
        self.attn_layer = MultiHeadAttentionLayer(num_features, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(num_features)
        self.fc_block = nn.Sequential(
            nn.Linear(num_features, 4 * num_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * num_features, num_features)
        )
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # normalize + attention
        x = x + self.drop1(self.attn_layer(self.ln1(x)))
        
        # normalize + MLP
        x = x + self.drop2(self.fc_block(self.ln2(x)))

        return x

class SingleHeadTransformerNetwork(nn.Module):
    def __init__(self, num_layers, num_features):
        super().__init__()

        self.transformer_list = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_list.append(SingleHeadAttentionTransformerLayer(num_features))
    
    def forward(self, x):
        for transformer in self.transformer_list:
            x = transformer(x) 
        return x

class MultiHeadTransformerNetwork(nn.Module):
    def __init__(self, num_features : int, dropout : float = 0.0, heads : Sequence[int] = (12, 12, 12, 12)):
        super().__init__()
        
        num_layers = len(heads)
        self.transformer_list = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_list.append(MultiHeadAttentionTransformerLayer(num_features, heads[i], dropout))
        
    def forward(self, x):
        for transformer in self.transformer_list:
            x = transformer(x)
        return x
