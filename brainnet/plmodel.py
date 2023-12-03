import logging
from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import R2Score
from brainnet.coords import load_coords
from brainnet.dataset import BrainDataset
from brainnet.plot_utils import make_training_plot
from brainnet.roi import nsdgeneral_indices
from brainnet.model import FactorTopy
import matplotlib.pyplot as plt

import cortex


class PLModel(pl.LightningModule):
    def __init__(self, config, backbone, cached=True, draw=True):
        super().__init__()
        self.config = config
        self.draw = draw

        # dataset
        self.dataset = BrainDataset(config.DATASET.DATA_DIR, config.DATASET.RESOLUTION)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [9000, 841]
        )

        # coordinates
        self.coords = load_coords()[nsdgeneral_indices]
        self.n_vertices = self.coords.shape[0]
        self.coords = nn.Parameter(torch.from_numpy(self.coords).float())
        self.coords.requires_grad = False  # freeze

        # model
        self.backbone = backbone
        self.cached = cached
        self.model = FactorTopy(
            n_vertices=self.n_vertices,
            layers=config.MODEL.LAYERS,
            layer_widths=config.MODEL.LAYER_WIDTHS,
            bottleneck_dim=config.MODEL.BOTTLENECK_DIM,
        )

        if self.cached:
            self.cached_local_tokens = {}
            self.cached_global_tokens = {}

        # metric
        self.train_r2 = R2Score(num_outputs=self.n_vertices, multioutput="raw_values")
        self.val_r2 = R2Score(num_outputs=self.n_vertices, multioutput="raw_values")

        self.logged_train_r2 = []
        self.logged_val_r2 = []

        cortex.download_subject(subject_id="fsaverage")

    def forward(self, x):
        with torch.no_grad():
            if self.cached:
                local_tokens, global_tokens = self.cached_forward(x)
            else:
                local_tokens, global_tokens = self.backbone.get_tokens(x)
            local_tokens = self.downsample_local_tokens(local_tokens)

        y, reg = self.model(local_tokens, global_tokens, self.coords)
        return y, reg

    def downsample_local_tokens(self, local_tokens):
        for layer in local_tokens:
            local_tokens[layer] = nn.functional.interpolate(
                local_tokens[layer], size=(8, 8), mode="bilinear", align_corners=False
            )
        return local_tokens

    def cached_forward(self, x):
        # trade memory for speed
        bsz = x.shape[0]
        device = x.device
        local_tokens, global_tokens = None, None
        for i in range(bsz):
            _x = x[i].unsqueeze(0)
            _hash = _x.sum().item()  # dirty hack
            if _hash not in self.cached_local_tokens:
                # compute cache
                _local_tokens, _global_tokens = self.backbone.get_tokens(_x)
                _local_tokens = self.downsample_local_tokens(_local_tokens)
                _local_tokens = {k: v.cpu() for k, v in _local_tokens.items()}
                _global_tokens = {k: v.cpu() for k, v in _global_tokens.items()}
                self.cached_local_tokens[_hash] = _local_tokens
                self.cached_global_tokens[_hash] = _global_tokens
            else:
                if self.current_epoch == 0 and self.training:
                    logging.warn("cache hit but in epoch 0, dirty hack is not working")
            # load cache
            _local_tokens = self.cached_local_tokens[_hash]
            _global_tokens = self.cached_global_tokens[_hash]
            _local_tokens = {k: v.to(device) for k, v in _local_tokens.items()}
            _global_tokens = {k: v.to(device) for k, v in _global_tokens.items()}
            if local_tokens is None:
                # initialize
                local_tokens = _local_tokens
                global_tokens = _global_tokens
            else:
                # concatenate
                for layer in local_tokens:
                    local_tokens[layer] = torch.cat(
                        [local_tokens[layer], _local_tokens[layer]], dim=0
                    )
                    global_tokens[layer] = torch.cat(
                        [global_tokens[layer], _global_tokens[layer]], dim=0
                    )
        return local_tokens, global_tokens

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, reg = self(x)
        self.train_r2.update(y_hat, y)
        loss = nn.functional.smooth_l1_loss(y_hat, y, beta=0.1)
        self.log("train_loss", loss)
        loss = loss + reg * self.config.REGULARIZATION.LAMBDA * self.get_decay()
        return loss

    def get_decay(self):
        current_step = self.global_step
        total_steps = self.config.REGULARIZATION.DECAY_TOTAL_STEPS
        decay = 1 - current_step / total_steps
        decay = max(decay, 0)
        return decay

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, reg = self(x)
        self.val_r2.update(y_hat, y)
        loss = nn.functional.smooth_l1_loss(y_hat, y, beta=0.1)
        self.log("val_loss", loss)
        # loss = loss + reg * self.config.REGULARIZATION.LAMBDA
        # return loss

    def on_validation_epoch_end(self):
        if self.global_step == 0:
            return

        self.logged_train_r2.append(self.train_r2.compute())
        self.train_r2.reset()
        self.log("train_r2", self.logged_train_r2[-1].mean())
        self.logged_val_r2.append(self.val_r2.compute())
        self.val_r2.reset()
        self.log("val_r2", self.logged_val_r2[-1].mean())

        if self.draw:
            from brainnet.plot_utils import make_training_plot

            sel_space, sel_layer, sel_scale = self.get_selectors()
            score = self.logged_val_r2[-1]
            fig, axs = make_training_plot(sel_space, sel_layer, sel_scale, score)
            fig.suptitle(
                f"Epoch={self.current_epoch:02d}        "
                + f"Step={self.global_step:05d}        "
                + f"R2={score.mean():.3f}",
                fontsize=24,
            )
            fig.tight_layout(pad=1)
            plt.show()
            plt.close()

    @torch.no_grad()
    def on_fit_end(self):
        print("on_fit_end: running clustering channels")
        sel_space, sel_layer, sel_scale = self.get_selectors()
        channel_clustering_dict, sel_channel = self.get_channel_clustering()
        self.channel_clustering_dict = channel_clustering_dict
        self.sel_channel = sel_channel
        self.sel_space = sel_space
        self.sel_layer = sel_layer
        self.sel_scale = sel_scale

        if self.draw:
            from brainnet.plot_utils import make_brainnet_plot

            fig, axs = make_brainnet_plot(sel_space, sel_layer, sel_scale, sel_channel)

            fig.tight_layout(pad=1)
            plt.show()
            plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.OPTIMIZER.LR)
        return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.DATASET.BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    ### for plotting ###
    @torch.no_grad()
    def get_selectors(self):
        sel_space = self.model.space_selector_mlp(self.coords)
        sel_layer = self.model.layer_selector_mlp(self.coords)
        sel_scale = self.model.scale_selector_mlp(self.coords)
        return sel_space, sel_layer, sel_scale

    ### channel clustering ###
    @torch.no_grad()
    def get_channel_clustering(self):
        from brainnet.clustering import cluster_channels

        channel_clustering_dict = cluster_channels(self.model.weight.detach())
        channel_indices = np.zeros(self.coords.shape[0])  # 37984
        for i, (k, v) in enumerate(channel_clustering_dict.items()):
            channel_indices[v] = k

        return channel_clustering_dict, channel_indices

    ### ROI top channel display ###
    @torch.no_grad()
    def get_top_channels(self, x):
        assert x.shape[1] == 3

        from brainnet.roi import rh_roi_dict, roi_names, nsdgeneral_indices

        local_tokens, global_tokens = self.backbone.get_tokens(x)
        # layer unique bottleneck
        for layer in local_tokens:
            local_tokens[layer] = self.model.local_token_bottleneck[layer](
                local_tokens[layer]
            )  # [B, D, H, W]
        stacked_local_tokens = torch.stack(
            [local_tokens[layer] for layer in local_tokens], dim=-1
        )  # [B, D, H, W, L]

        top_channels = {}  # roi -> [B, H, W, 3]
        sel_space, sel_layer, sel_scale = self.get_selectors()
        for roi in roi_names:
            # roi_indices
            roi_indices = rh_roi_dict[roi]
            fsaverage = np.zeros(327684)   # convert fsaverage and nsdgeneral space
            fsaverage[nsdgeneral_indices] = np.arange(nsdgeneral_indices.shape[0])
            roi_indices = fsaverage[roi_indices]
            # pca of weights
            w = self.model.weight[roi_indices]  # [n_roi_vertices, d]
            pca = torch.pca_lowrank(w.t(), q=3)
            pc_w = w.t() @ pca[-1]  # [d, 3]
            pc_w = -pc_w  # flip sign
            # roi-average of layer selection
            _sel_layer = sel_layer[roi_indices].mean(0)  # [N, L] -> [L]
            # roi-average selection of local tokens
            pc_ch = stacked_local_tokens @ _sel_layer.unsqueeze(-1)
            pc_ch = pc_ch.squeeze(-1)
            pc_ch = rearrange(pc_ch, "b c h w -> b h w c")
            # the rgb channels are the top 3 pca components (pc_w)
            pc_ch = pc_ch @ pc_w  # [B, H, W, 3]

            # normalize
            flat_pc_ch = pc_ch.flatten().detach().cpu().numpy()
            toppcen = np.percentile(flat_pc_ch, 95)
            botpcen = np.percentile(flat_pc_ch, 5)
            pc_ch = torch.clamp(pc_ch, botpcen, toppcen)
            pc_ch = (pc_ch - botpcen) / (toppcen - botpcen)
            pc_ch = torch.clamp(pc_ch, 0, 1)

            top_channels[roi] = pc_ch  # [B, H, W, 3]

        return top_channels

    ### ROI top channel display ###
    def draw_top_channels(self, x):
        # x is a batch of images [B, 3, H, W]
        # draw the top channels for ONLY the first image in the batch
        # return all images in the batch

        top_channels = self.get_top_channels(x)
        fig, axs = plt.subplots(3, 4, figsize=(8, 6))
        for i, roi in enumerate(top_channels):
            ax = axs.flatten()[i]
            ax.imshow(top_channels[roi][0].detach().cpu().numpy())
            # only show the first image in the batch
            ax.set_title(roi)
            ax.axis("off")
        fig.tight_layout()
        plt.show()
        plt.close()
        return top_channels  # roi -> [B, H, W, 3]


# if __name__ == "__main__":
#     from config import get_cfg_defaults
#     from backbone import ModifiedCLIP

#     cfg = get_cfg_defaults()
#     backbone = ModifiedCLIP()

#     plmodel = PLModel(cfg, backbone)

#     x = torch.rand(8, 3, 224, 224)
#     y, reg = plmodel(x)

#     trainer = pl.Trainer(max_epochs=3, accelerator="gpu", devices=[0], precision=16)

#     trainer.fit(plmodel)
