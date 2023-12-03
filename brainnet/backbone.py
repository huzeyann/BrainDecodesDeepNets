import open_clip
from open_clip.transformer import VisionTransformer

import torch
from torch import nn

import numpy as np

from einops import rearrange

from typing import List, Optional


class ModifiedCLIP(nn.Module):
    def __init__(self, ver="ViT-B-16", data="datacomp_l_s1b_b8k", **kwargs) -> None:
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(ver, pretrained=data)
        self.vision_model: VisionTransformer = model.visual
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()

    def get_tokens(
        self,
        x,
    ):
        #### original code #### begin

        ##############################
        ### patchify ###
        ##############################

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.vision_model.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.vision_model.grid_size[0],
                self.vision_model.patch_size[0],
                self.vision_model.grid_size[1],
                self.vision_model.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(
                x.shape[0],
                self.vision_model.grid_size[0] * self.vision_model.grid_size[1],
                -1,
            )
            x = self.vision_model.patchnorm_pre_ln(x)
            x = self.vision_model.conv1(x)
        else:
            x = self.vision_model.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.vision_model.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vision_model.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.vision_model.patch_dropout(x)
        x = self.vision_model.ln_pre(x)

        #### original code #### end

        #### modified code #### begin

        ##############################
        ### transformer ###
        ##############################

        x = x.permute(1, 0, 2)  # NLD -> LND

        local_tokens = {}
        global_tokens = {}
        for i, r in enumerate(self.vision_model.transformer.resblocks):
            x = r(x)  # [1+p**2, B, D]
            x_save = x.clone()
            x_save = x_save[1:, :, :]  # [p**2, B, D]
            p = int(np.sqrt(x_save.shape[0]))
            x_save = rearrange(x_save, "(p1 p2) b d -> b d p1 p2", p1=p, p2=p)
            local_tokens[str(i)] = x_save
            global_tokens[str(i)] = x[0, :, :]  # [B, D]

        return local_tokens, global_tokens


from dinov2.models.vision_transformer import DinoVisionTransformer


class ModifiedDiNOv2(nn.Module):
    def __init__(self, ver="dinov2_vitb14", **kwargs) -> None:
        super().__init__()
        vision_model = torch.hub.load("facebookresearch/dinov2", ver)
        self.vision_model: DinoVisionTransformer = vision_model
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()

    def get_tokens(
        self,
        x,
    ):
        #### original code #### begin
        x = self.vision_model.prepare_tokens_with_masks(x)
        #### original code #### end

        #### modified code #### begin
        local_tokens = {}
        global_tokens = {}
        for i, blk in enumerate(self.vision_model.blocks):
            x = blk(x)
            saved_x = x.clone()
            global_tokens[str(i)] = saved_x[:, 0, :]  # [B, C]
            saved_x = saved_x[:, 1:, :]  # remove cls token, [B, N, C]
            p = int(np.sqrt(saved_x.shape[1]))
            saved_x = rearrange(saved_x, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
            local_tokens[str(i)] = saved_x
        return local_tokens, global_tokens


from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling.sam import Sam


class ModifiedSAM(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sam: Sam = sam_model_registry["vit_b"](checkpoint=None)
        sd = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )
        sam.load_state_dict(sd)

        def new_forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            local_tokens, global_tokens = {}, {}
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                x_save = x.clone()
                x_save = x_save.permute(0, 3, 1, 2)
                local_tokens[f"{i}"] = x_save
                global_tokens[f"{i}"] = x_save.mean(dim=(2, 3))

            return local_tokens, global_tokens

        setattr(sam.image_encoder.__class__, "forward", new_forward)

        self.image_encoder = sam.image_encoder
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

    def get_tokens(
        self,
        x,
    ):
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear")
        local_tokens, global_tokens = self.image_encoder(x)
        return local_tokens, global_tokens


import timm


class ModifiedMAE(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(ModifiedMAE, self).__init__(**kwargs)

        sd = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
        )

        checkpoint_model = sd["model"]
        state_dict = self.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        self.requires_grad_(False)
        self.eval()

    def get_tokens(
        self,
        x,
    ):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        local_tokens = {}
        global_tokens = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            saved_x = x.clone()
            saved_x = saved_x[:, 1:, :]  # remove cls token, [B, N, C]
            p = int(np.sqrt(saved_x.shape[1]))
            saved_x = rearrange(saved_x, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
            local_tokens[str(i)] = saved_x
            global_tokens[str(i)] = x[:, 0, :]  # [B, C]
        return local_tokens, global_tokens


from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights, ViT_H_14_Weights
from torchvision.models import vit_b_16, vit_l_16, vit_h_14
from torchvision.models import list_models, get_model
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)


class ModifiedImgNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        model = get_model("vit_b_16", weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.requires_grad_(False)
        model.eval()
        layers = [f"encoder.layers.encoder_layer_{i}.add_1" for i in range(12)]
        model = create_feature_extractor(model, layers)

        self.model = model

    def get_tokens(
        self,
        x,
    ):
        em = self.model(x)
        out_list = list(em.values())

        local_tokens = {}
        global_tokens = {}
        for i, out in enumerate(out_list):
            saved_x = out.clone()
            saved_x = saved_x[:, 1:, :]  # remove cls token, [B, N, C]
            p = int(np.sqrt(saved_x.shape[1]))
            saved_x = rearrange(saved_x, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
            local_tokens[str(i)] = saved_x
            global_tokens[str(i)] = out[:, 0, :]  # [B, C]
        return local_tokens, global_tokens


import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

from timm.models.layers import PatchEmbed


class ModifiedMoCov3(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self,
        stop_grad_conv1=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        super().__init__(norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim
                )
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

        checkpoint = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        )

        linear_keyword = "head"
        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith("module.base_encoder") and not k.startswith(
                "module.base_encoder.%s" % linear_keyword
            ):
                # remove prefix
                state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = self.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {
            "%s.weight" % linear_keyword,
            "%s.bias" % linear_keyword,
        }

        # print("=> loaded pre-trained self '{}'".format(checkpoint))

        self.requires_grad_(False)
        self.eval()

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def get_tokens(
        self,
        x,
    ):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        local_tokens = {}
        global_tokens = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            saved_x = x.clone()
            saved_x = saved_x[:, 1:, :]  # remove cls token, [B, N, C]
            p = int(np.sqrt(saved_x.shape[1]))
            saved_x = rearrange(saved_x, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
            local_tokens[str(i)] = saved_x
            global_tokens[str(i)] = x[:, 0, :]  # [B, C]
        return local_tokens, global_tokens
