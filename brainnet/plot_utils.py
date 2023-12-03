import copy
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, ticker

from brainnet.roi import algo23_indices, nsdgeneral_indices

import cortex

# monkey patch
from cortex.quickflat.utils import _get_fig_and_ax


def center_crop(png_path, overlay=False):
    tmp_png = png_path
    from PIL import Image

    im = Image.open(tmp_png)
    width, height = im.size
    if not overlay:
        left = width * 0.358
        right = width * 0.64
        top = height * 0.3
    else:
        left = width * 0.33
        right = width * 0.67
        # left = width * 0.358
        # right = width * 0.64
        top = height * 0.25
    bottom = height * 0.9

    cropped_im = im.crop((left, top, right, bottom))

    cropped_im.save(tmp_png)


VMAX = 12
VMIN = 1
CMAP_IMG = "/tmp/cim.png"
SQUARE_CMAP = False
COLORFUL_PLOT = False
CBAR_TEXT = "Layer"
SUBJECT = "subj01"


def my_add_colorbar(
    fig,
    cimg,
    colorbar_ticks=None,
    colorbar_location=(0.4, 0.07, 0.2, 0.04),
    orientation="horizontal",
):
    """Add a colorbar to a flatmap plot

    Parameters
    ----------
    fig : matplotlib Figure object
        Figure into which to insert colormap
    cimg : matplotlib.image.AxesImage object
        Image for which to create colorbar. For reference, matplotlib.image.AxesImage
        is the output of imshow()
    colorbar_ticks : array-like
        values for colorbar ticks
    colorbar_location : array-like
        Four-long list, tuple, or array that specifies location for colorbar axes
        [left, top, width, height] (?)
    orientation : string
        'vertical' or 'horizontal'
    """
    from matplotlib import rc, rcParams

    rc("font", weight="bold")
    import matplotlib.pyplot as plt
    import os

    global VMAX, VMIN, SQUARE_CMAP
    global CMAP_IMG
    global CBAR_TEXT

    if not SQUARE_CMAP:
        colorbar_location = (0.445, 0.16, 0.1, 0.05)

        fig, _ = _get_fig_and_ax(fig)
        fig.add_axes(colorbar_location)
        fontsize = 28
        colorbar_ticks = [0, 4, 0, 1]
        if CBAR_TEXT != "time":
            colorbar_labels = [VMIN, VMAX, 0, 1]
        else:
            colorbar_labels = [0, 50, 0, 1]
        cbar = plt.imshow(
            plt.imread(CMAP_IMG), extent=colorbar_ticks, interpolation="bilinear"
        )
        # plt.axis('off')
        cbar.axes.set_xticks(colorbar_ticks[:2])
        cbar.axes.set_xticklabels(colorbar_labels[:2], fontdict=dict(size=fontsize))
        global COLORFUL_PLOT
        if COLORFUL_PLOT:
            cbar.axes.set_yticks(colorbar_ticks[2:])
            cbar.axes.set_yticklabels(colorbar_labels[2:], fontdict=dict(size=fontsize))
        else:
            cbar.axes.set_yticks([])
            cbar.axes.set_yticklabels([])
    if SQUARE_CMAP:
        colorbar_location = (0.445, 0.13, 0.1, 0.1)
        fig, _ = _get_fig_and_ax(fig)
        fig.add_axes(colorbar_location)
        fontsize = 28
        colorbar_ticks = [0, 4, 0, 4]
        colorbar_labels = ["L", "R", "", ""]
        cbar = plt.imshow(
            plt.imread(CMAP_IMG), extent=colorbar_ticks, interpolation="bilinear"
        )
        cbar.axes.set_xticks(colorbar_ticks[:2])
        cbar.axes.set_xticklabels(colorbar_labels[:2], fontdict=dict(size=fontsize))
        cbar.axes.set_yticks(colorbar_ticks[2:])
        cbar.axes.set_yticklabels(colorbar_labels[2:], fontdict=dict(size=fontsize))
        # y ticks right
        cbar.axes.yaxis.tick_right()

    # spines off
    cbar.axes.spines["top"].set_visible(False)
    cbar.axes.spines["right"].set_visible(False)
    cbar.axes.spines["bottom"].set_visible(False)
    cbar.axes.spines["left"].set_visible(False)
    # ticks off
    cbar.axes.tick_params(axis="both", which="both", length=0)

    # add text
    cbar.axes.text(
        0.5,
        1.2,
        CBAR_TEXT,
        horizontalalignment="center",
        verticalalignment="center",
        transform=cbar.axes.transAxes,
        fontsize=fontsize - 4,
    )

    return cbar


# monkey patch
cortex.quickflat.composite.add_colorbar = my_add_colorbar


def ent_transform(e, pow=3, add=0.2):
    e = 1 - e**pow + add
    return e


def ent(x):
    return (x * np.log(x)).mean(-1)


def normalized_ent(x):
    n = x.shape[-1]
    return ent(x) / ((1 / n) * np.log(1 / n))


def make_2d_layer_cmap(reverse=False):
    def rgb_from_values(values, vmax=1):
        import matplotlib.cm as cm

        values = values / vmax
        if reverse:
            rgb = cm.coolwarm(1 - values)
        else:
            rgb = cm.coolwarm(values)
        return rgb

    # 2d cmap
    grid = torch.linspace(0, 1, VMAX)
    # for i in range(VMAX):
    #     grid[i] = (i // VMAX) / VMAX
    rgb_grid = rgb_from_values(grid, vmax=1)
    rgb_grid_2d = np.stack([rgb_grid for _ in range(96)], axis=0)
    rgb_grid_2d = torch.tensor(rgb_grid_2d)
    y_grid = torch.linspace(0, 1, 96)
    rgb_grid_2d = rgb_grid_2d * ent_transform(y_grid)[:, None, None]
    rgb_grid_2d = rgb_grid_2d[:, :, :3]

    # show
    fig, ax = plt.subplots(figsize=(2, 12))
    plt.imshow(rgb_grid_2d)
    plt.axis("off")
    # plt.show()
    plt.savefig(CMAP_IMG, dpi=300, bbox_inches="tight")
    plt.close()


def make_1d_cmap(name="viridis"):
    def rgb_from_values(values, vmax=1):
        import matplotlib.cm as cm
        import matplotlib

        values = values / vmax
        cmap = matplotlib.colormaps[name]
        rgb = cmap(values)
        return rgb

    # 2d cmap
    grid = torch.linspace(0, 1, 128)
    rgb_grid = rgb_from_values(grid, vmax=1)
    rgb_grid_2d = np.stack([rgb_grid for _ in range(128)], axis=0)
    rgb_grid_2d = torch.tensor(rgb_grid_2d)
    rgb_grid_2d = rgb_grid_2d[:, :, :3]

    fig, ax = plt.subplots(figsize=(2, 10))
    plt.imshow(rgb_grid_2d)
    plt.axis("off")
    # plt.show()
    plt.savefig(CMAP_IMG, dpi=300, bbox_inches="tight")
    plt.close()


def make_2d_space_cmap():
    import colorstamps
    from PIL import Image

    cmap = colorstamps.stamps.get_const_J(r=50)  # (l,l,3) numpy array of rgb values
    cmap *= 255
    cmap = cmap.astype(np.uint8)
    img = Image.fromarray(cmap, "RGB")
    img.save(CMAP_IMG)


def rgb_from_mu(mu):
    color_gradient_img = plt.imread(CMAP_IMG)

    color_gradient_img = (
        torch.tensor(color_gradient_img).permute(2, 0, 1).float().unsqueeze(0)
    )
    from einops import rearrange, repeat

    mu = torch.tensor(mu).float()
    mu = repeat(mu, "n c -> b n d c", b=1, d=1)
    from torch.nn.functional import interpolate, grid_sample

    rgb = grid_sample(color_gradient_img, mu, align_corners=True)
    rgb = rgb.squeeze(0).squeeze(-1).t()
    rgb = rgb.numpy()
    return rgb


def make_space_plot(mu, path):
    global SQUARE_CMAP, CBAR_TEXT, COLORFUL_PLOT
    SQUARE_CMAP = True
    CBAR_TEXT = ""
    COLORFUL_PLOT = False

    make_2d_space_cmap()
    rgb = rgb_from_mu(mu)

    fsaverage = np.zeros((163842 * 2, 3))
    fsaverage /= 0  # nan to make other vertices transparent
    fsaverage[nsdgeneral_indices] = rgb

    r = cortex.Vertex(fsaverage[:, 0], "fsaverage", vmin=0, vmax=1)
    g = cortex.Vertex(fsaverage[:, 1], "fsaverage", vmin=0, vmax=1)
    b = cortex.Vertex(fsaverage[:, 2], "fsaverage", vmin=0, vmax=1)
    vertex_data = cortex.VertexRGB(r, g, b, "fsaverage")
    cortex.quickflat.make_png(
        path,
        vertex_data,
        with_curvature=False,
        with_rois=False,
        with_labels=False,
        with_sulci=False,
        with_colorbar=True,
    )

    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    plt.close()

    center_crop(path)


def make_layer_plot(ls, path, cbar_text="layer", reverse=False, overlay=False):
    global SQUARE_CMAP, CBAR_TEXT, COLORFUL_PLOT, VMAX, VMIN
    SQUARE_CMAP = False
    CBAR_TEXT = cbar_text
    COLORFUL_PLOT = True
    VMAX = ls.shape[-1]
    VMIN = 1
    make_2d_layer_cmap(reverse=reverse)

    def rgb_from_values(values, vmax=1):
        import matplotlib.cm as cm

        values = values / vmax
        if reverse:
            rgb = cm.coolwarm(1 - values)
        else:
            rgb = cm.coolwarm(values)
        return rgb

    ent_values = normalized_ent(ls)
    ls = ls.argmax(-1)
    rgb = rgb_from_values(ls, vmax=VMAX - 1)
    rgb[:, :3] *= ent_transform(ent_values)[:, None]

    fsaverage = np.zeros((163842 * 2, 4))
    fsaverage /= 0  # nan to make other vertices transparent
    fsaverage[nsdgeneral_indices] = rgb

    r = cortex.Vertex(fsaverage[:, 0], "fsaverage", vmin=0, vmax=1)
    g = cortex.Vertex(fsaverage[:, 1], "fsaverage", vmin=0, vmax=1)
    b = cortex.Vertex(fsaverage[:, 2], "fsaverage", vmin=0, vmax=1)
    a = cortex.Vertex(fsaverage[:, 3], "fsaverage", vmin=0, vmax=1)
    vertex_data = cortex.VertexRGB(r, g, b, "fsaverage", alpha=a)
    cortex.quickflat.make_png(
        path,
        vertex_data,
        with_curvature=False,
        with_rois=overlay,
        with_labels=overlay,
        with_sulci=False,
        with_colorbar=True,
    )
    COLORFUL_PLOT = False
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    plt.close()

    center_crop(path, overlay=overlay)


def make_1d_plot(
    score,
    path,
    cmap="inferno",
    cbartext="weight",
    vmin=0,
    vmax=1,
    cbar=True,
    overlay=False,
):
    make_1d_cmap(cmap)
    global SQUARE_CMAP, CBAR_TEXT, COLORFUL_PLOT, VMAX, VMIN
    SQUARE_CMAP = False
    CBAR_TEXT = cbartext
    COLORFUL_PLOT = False
    VMAX = vmax
    VMIN = vmin

    fsaverage = np.zeros(163842 * 2)
    fsaverage /= 0  # nan to make other vertices transparent
    fsaverage[nsdgeneral_indices] = score

    vertex_data = cortex.Vertex(fsaverage, "fsaverage", cmap=cmap, vmin=VMIN, vmax=VMAX)
    cortex.quickflat.make_png(
        path,
        vertex_data,
        with_curvature=False,
        with_rois=overlay,
        with_labels=overlay,
        with_sulci=False,
        with_colorbar=cbar,
    )
    dir_path = os.path.dirname(path)
    # plt.show()
    os.makedirs(dir_path, exist_ok=True)
    plt.close()

    center_crop(path, overlay=overlay)


def make_single_ls_weight_plot(weight, path, overlay=False):
    make_1d_plot(
        weight,
        path,
        cmap="inferno",
        cbartext="weight",
        vmin=0,
        vmax=1,
        overlay=overlay,
    )


def make_scale_plot(scale, path, overlay=False):
    make_1d_plot(
        scale,
        path,
        cmap="PuOr",
        cbartext="local       global",
        vmin=0,
        vmax=1,
        overlay=overlay,
    )


def make_score_plot(score, path, overlay=False):
    make_1d_plot(
        score,
        path,
        cmap="Reds",
        cbartext="brain r2",
        vmin=0,
        vmax=1.0,
        overlay=overlay,
    )


def make_channel_plot(channel, path, overlay=False):
    make_1d_plot(
        channel,
        path,
        cmap="tab20",
        cbartext="channel",
        vmin=1,
        vmax=20,
        overlay=overlay,
    )


def make_training_plot(space, layer, scale, score):
    space, layer, scale, score = (
        space.cpu().numpy(),
        layer.cpu().numpy(),
        scale.cpu().numpy(),
        score.cpu().numpy(),
    )
    scale = scale[:, 0]

    layer_png = "/tmp/layer.png"
    space_png = "/tmp/space.png"
    scale_png = "/tmp/scale.png"
    score_png = "/tmp/score.png"

    make_layer_plot(layer, layer_png)
    make_space_plot(space, space_png)
    make_scale_plot(scale, scale_png)
    make_score_plot(score, score_png)

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, png in zip(axs, [layer_png, space_png, scale_png, score_png]):
        ax.imshow(plt.imread(png))
        ax.axis("off")
    return fig, axs


def make_brainnet_plot(space, layer, scale, channel):
    space, layer, scale = (
        space.cpu().numpy(),
        layer.cpu().numpy(),
        scale.cpu().numpy(),
    )
    scale = scale[:, 0]

    layer_png = "/tmp/layer.png"
    space_png = "/tmp/space.png"
    scale_png = "/tmp/scale.png"
    channel_png = "/tmp/channel.png"

    make_layer_plot(layer, layer_png)
    make_space_plot(space, space_png)
    make_scale_plot(scale, scale_png)
    make_channel_plot(channel, channel_png)

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, png in zip(axs, [layer_png, space_png, scale_png, channel_png]):
        ax.imshow(plt.imread(png))
        ax.axis("off")
    return fig, axs
