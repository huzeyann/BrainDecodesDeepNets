from typing import Dict, List
from einops import rearrange, repeat
import torch
from torch import Tensor, nn

import numpy as np

from brainnet.position_encoding import PositionalEncoding


class FactorTopy(nn.Module):
    def __init__(
        self,
        n_vertices: int,
        layers: List[int] = list(range(12)),
        layer_widths: List[int] = [768] * 12,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        layers = [str(layer) for layer in layers]
        self.layers = layers
        self.local_token_bottleneck = nn.ModuleDict()
        self.global_token_bottleneck = nn.ModuleDict()
        for layer, width in zip(layers, layer_widths):
            self.local_token_bottleneck[layer] = nn.Conv2d(
                width, bottleneck_dim, 1, bias=False
            )
            self.global_token_bottleneck[layer] = nn.Linear(
                width, bottleneck_dim, bias=False
            )

        self.space_selector_mlp = nn.Sequential(
            PositionalEncoding(features=32),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 2, bias=False),
            nn.Tanh(),
        )
        nn.init.zeros_(self.space_selector_mlp[-2].weight)
        self.layer_selector_mlp = nn.Sequential(
            PositionalEncoding(features=32),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, len(layers), bias=False),
            nn.Softmax(dim=-1),
        )
        nn.init.zeros_(self.layer_selector_mlp[-2].weight)
        self.scale_selector_mlp = nn.Sequential(
            PositionalEncoding(features=32),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1, bias=False),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.scale_selector_mlp[-2].weight)
        

        dummy = nn.Linear(bottleneck_dim, n_vertices)
        self.weight = nn.Parameter(dummy.weight)  # (n_vertices, bottleneck_dim)
        self.bias = nn.Parameter(dummy.bias)  # (n_vertices,)

    def forward(
        self,
        local_tokens: Dict[int, Tensor],  # layer -> (batch, dim, h, w)
        global_tokens: Dict[int, Tensor],  # layer -> (batch, dim)
        coords: Tensor,  # (n_vertices, 3)
    ):
        assert list(local_tokens.keys()) == self.layers, "local tokens must have all layers"
        assert list(global_tokens.keys()) == self.layers, "global tokens must have all layers"
        
        ### channel align ###
        for layer in self.layers:
            local_tokens[layer] = self.local_token_bottleneck[layer](
                local_tokens[layer]
            )  # (batch, bottleneck_dim, h, w)
            global_tokens[layer] = self.global_token_bottleneck[layer](
                global_tokens[layer]
            )  # (batch, bottleneck_dim)

        ### multi-selectors ###
        sel_space = self.space_selector_mlp(coords)  # (n_vertices, 2)
        sel_layer = self.layer_selector_mlp(coords)  # (n_vertices, len(layers))
        sel_scale = self.scale_selector_mlp(coords)  # (n_vertices, 1)
        # - layer selector regularization, because of softmax
        _entropy = (torch.log(sel_layer + 1e-8) * sel_layer).sum(dim=-1).mean()
        normed_entropy = _entropy / np.log(len(self.layers))
        # - scale selector regularization, because of sigmoid
        _mse = (sel_scale - 0.5).pow(2).mean()
        
        reg = normed_entropy + _mse

        ### do the selection ###
        # - global tokens
        global_tokens = torch.stack(
            list(global_tokens.values()), dim=-1
        )  # (batch, bottleneck_dim, len(layers))
        global_tokens = repeat(global_tokens, "b d l -> b n d l", n=1)
        _sel_layer = repeat(sel_layer, "n l -> b n d l", b=1, d=1)
        v_global = (global_tokens * _sel_layer).sum(dim=-1)  # (batch, n_vertices, bottleneck_dim)
        # - local tokens
        bsz = local_tokens[self.layers[0]].shape[0]
        _sel_space = repeat(sel_space, "n d -> b n c d", b=bsz, c=1)
        _sel_layer = repeat(sel_layer, "n l -> b n l", b=1)
        v_local = None
        for i, layer in enumerate(self.layers):
            _local_tokens = local_tokens[layer] # (batch, bottleneck_dim, h, w)
            _v_local = nn.functional.grid_sample(
                _local_tokens,
                _sel_space,
                align_corners=False,
                mode="bilinear",
                padding_mode="zeros",
            )  # (batch, bottleneck_dim, n_vertices, 1)
            _v_local = _v_local.squeeze(-1)
            _v_local = rearrange(_v_local, "b d n -> b n d") 
            _v_local = _v_local * _sel_layer[:, :, i].unsqueeze(-1)
            if v_local is None:
                v_local = _v_local
            else:
                v_local = v_local + _v_local
        # - scale
        _sel_scale = repeat(sel_scale, "n d -> b n d", b=1)
        v = (1 - _sel_scale) * v_local + _sel_scale * v_global # (batch, n_vertices, bottleneck_dim)
        
        ### linear prediction ###
        w = self.weight.unsqueeze(0)  # (1, n_vertices, bottleneck_dim)
        b = self.bias.unsqueeze(0)  # (1, n_vertices)
        y = (v * w).mean(dim=-1) + b  # (batch, n_vertices)
        
        return y, reg
        