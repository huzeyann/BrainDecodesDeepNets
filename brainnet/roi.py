import numpy as np

from brainnet.roi_data import ROI_DATA

roi_names = [
    "V1",
    "V2",
    "V3",
    "V4",
    "EBA",
    "FBA",
    "OFA",
    "FFA",
    "OPA",
    "PPA",
    "OWFA",
    "VWFA",
]
_inner_roi_names = [
    ["lh-V1v", "lh-V1d", "rh-V1v", "rh-V1d"],
    ["lh-V2v", "lh-V2d", "rh-V2v", "rh-V2d"],
    ["lh-V3v", "lh-V3d", "rh-V3v", "rh-V3d"],
    ["lh-hV4", "rh-hV4"],
    ["lh-EBA", "rh-EBA"],
    ["lh-FBA-1", "rh-FBA-1", "lh-FBA-2", "rh-FBA-2"],
    ["lh-OFA", "rh-OFA"],
    ["lh-FFA-1", "rh-FFA-1", "lh-FFA-2", "rh-FFA-2"],
    ["lh-OPA", "rh-OPA"],
    ["lh-PPA", "rh-PPA"],
    ["lh-OWFA", "rh-OWFA"],
    ["lh-VWFA-1", "rh-VWFA-1", "lh-VWFA-2", "rh-VWFA-2"],
]


def load_roi(roi):
    # path = str(files("brainnet").joinpath(f"roi_masks/{roi}.npy"))
    # print(f"loading {path}")
    # return np.load(path)
    return np.array(ROI_DATA[roi])

# right hemisphere only
rh_roi_dict = {}  # big roi name -> indices of fsaverage space vertices (327684)
for big, small in zip(roi_names, _inner_roi_names):
    roi_indices = []
    for roi in small:
        if "lh" in roi:
            continue  # skip lh
        indices = load_roi(roi)
        roi_indices.append(indices)
    roi_indices = np.concatenate(roi_indices)
    rh_roi_dict[big] = roi_indices

# left hemisphere only
lh_roi_dict = {}  # big roi name -> indices of fsaverage space vertices (327684)
for big, small in zip(roi_names, _inner_roi_names):
    roi_indices = []
    for roi in small:
        if "rh" in roi:
            continue  # skip rh
        indices = load_roi(roi)
        roi_indices.append(indices)
    roi_indices = np.concatenate(roi_indices)
    lh_roi_dict[big] = roi_indices

# both hemispheres
roi_dict = {}  # big roi name -> indices of fsaverage space vertices (327684)
for big, small in zip(roi_names, _inner_roi_names):
    roi_indices = []
    for roi in small:
        indices = load_roi(roi)
        roi_indices.append(indices)
    roi_indices = np.concatenate(roi_indices)
    roi_dict[big] = roi_indices


def load_algonauts23_indices():
    # _lh = np.load(
    #     str(files("brainnet").joinpath("roi_masks/lh.all-vertices_fsaverage_space.npy"))
    # )
    # _rh = np.load(
    #     str(files("brainnet").joinpath("roi_masks/rh.all-vertices_fsaverage_space.npy"))
    # )
    # binarized = np.concatenate([_lh, _rh])
    # indices = np.where(binarized == 1)[0]
    # return indices
    _lh = np.array(ROI_DATA["lh.all-vertices_fsaverage_space"])
    _rh = np.array(ROI_DATA["rh.all-vertices_fsaverage_space"])
    return np.concatenate([_lh, _rh])

algo23_indices = load_algonauts23_indices() # 39548

nsdgeneral_indices = load_roi("nsdgeneral") # 37984
