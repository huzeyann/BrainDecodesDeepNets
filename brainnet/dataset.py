import os

import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms

from brainnet.roi import algo23_indices, nsdgeneral_indices


class BrainDataset(Dataset):
    def __init__(self, data_dir, resolution=(224, 224)):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "training_split/training_images")
        self.fmri_dir = os.path.join(data_dir, "training_split/training_fmri")

        # image is loaded on the fly
        self.image_paths = sorted(os.listdir(self.image_dir))
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        ## fmri is loaded into memory
        lh_fmri = np.load(os.path.join(self.fmri_dir, "lh_training_fmri.npy"))
        rh_fmri = np.load(os.path.join(self.fmri_dir, "rh_training_fmri.npy"))
        _fmri = np.concatenate([lh_fmri, rh_fmri], axis=1)  # (9841, 39548)
        # 39548 is visual cortex vertices used by algonauts23, nsdgeneral (37984) + RSC (1564)
        # remove RSC, keep only nsdgeneral
        fmri = []
        for i in range(_fmri.shape[0]):
            _i_fsaverage = np.zeros(327684)
            _i_fsaverage[algo23_indices] = _fmri[i]
            _i_fmri = _i_fsaverage[nsdgeneral_indices]
            fmri.append(_i_fmri)
        fmri = np.stack(fmri, axis=0)
        self.fmri = fmri

        assert (
            len(self.image_paths) == self.fmri.shape[0]
        ), "image and fmri size mismatch"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        fmri = self.fmri[idx]
        fmri = torch.from_numpy(fmri).float()

        return image, fmri


if __name__ == "__main__":
    dataset = BrainDataset("/data/download/alg23/subj01")
    img, fmri = dataset[0]
    pass
