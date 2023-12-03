import numpy as np

import nilearn
from nilearn import datasets, surface


def load_coords():
    fsaverage = nilearn.datasets.fetch_surf_fsaverage("fsaverage7")
    lh_coords, lh_faces = nilearn.surface.load_surf_mesh(fsaverage["sphere_left"])
    rh_coords, rh_faces = nilearn.surface.load_surf_mesh(fsaverage["sphere_right"])
    lh_xmin, lh_xmax = np.min(lh_coords[:, 0]), np.max(lh_coords[:, 0])
    lh_xmax = lh_xmin + (lh_xmax - lh_xmin) * 1.5  # offset to the right
    rh_xmin, rh_xmax = np.min(rh_coords[:, 0]), np.max(rh_coords[:, 0])
    if rh_xmin < lh_xmax:
        rh_coords[:, 0] += lh_xmax - rh_xmin
    coords = np.concatenate((lh_coords, rh_coords), axis=0)
    return coords
