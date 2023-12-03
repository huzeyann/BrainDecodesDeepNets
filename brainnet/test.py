# %%
from brainnet.plmodel import PLModel
from brainnet.config import get_cfg_defaults
from brainnet.backbone import (
    ModifiedCLIP,
    ModifiedDiNOv2,
    ModifiedSAM,
    ModifiedImgNet,
    ModifiedMAE,
    ModifiedMoCov3,
)

import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch

# %%

cfg = get_cfg_defaults()

cfg.DATASET.DATA_DIR = '/data/huze/download/alg23/subj01'

cfg.MODEL.LAYERS = list(range(12))
cfg.MODEL.LAYER_WIDTHS = [768] * len(cfg.MODEL.LAYERS)

# cfg.DATASET.RESOLUTION = (448, 448)
# backbone = ModifiedDiNOv2()

cfg.DATASET.RESOLUTION = (224, 224)
backbone = ModifiedCLIP()

# cfg.DATASET.RESOLUTION = (1024, 1024)
# backbone = ModifiedSAM()

plmodel = PLModel(cfg, backbone, draw=True)
# plmodel.validation_epoch_end() is called on validation epoch to draw
plmodel = plmodel.cuda()
# %%

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="gpu",
    devices=[0],
    gradient_clip_val=0.5,
    precision=16,
    limit_train_batches=0.01,
    limit_val_batches=0.03,
    enable_checkpointing=False,
)

trainer.fit(plmodel)  # 30 min on 4090, 8GB of VRAM
# %%
# sd = plmodel.model.state_dict()
# torch.save(sd, "/workspace/assets/dino.pt")
# %%
# sd = torch.load("/workspace/assets/clip.pt")
# plmodel.model.load_state_dict(sd)
# %%
# plot raw layer selector weights
sel_space, sel_layer, sel_scale = plmodel.get_selectors()
sel_layer = sel_layer.detach().cpu().numpy()
from brainnet.plot_utils import make_single_ls_weight_plot

for i in range(12):
    png = f"/tmp/ls_weight_{i}.png"
    make_single_ls_weight_plot(sel_layer[:, i], png)
fig, axs = plt.subplots(2, 6, figsize=(15, 6))
for i in range(12):
    ax = axs.flatten()[i]
    ax.set_title(f"Layer {i}")
    png = f"/tmp/ls_weight_{i}.png"
    ax.imshow(plt.imread(png))
    ax.axis("off")
fig.tight_layout()
plt.show()
plt.close()
# %%
## plot top channels

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

image_path = "/workspace/assets/catchafish.jpg"
plt.imshow(Image.open(image_path))
plt.axis("off")
plt.title("Input image")
plt.show()
plt.close()

image = Image.open(image_path).convert("RGB")
transforms = transforms.Compose(
    [
        transforms.Resize(cfg.DATASET.RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
image = transforms(image).unsqueeze(0).cuda()

top_channels = plmodel.draw_top_channels(image)
# %%
