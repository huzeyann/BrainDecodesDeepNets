# %%
import os
import numpy as np

# %%
roi_dir = "./roi_masks"
rois = os.listdir(roi_dir)
rois = sorted(rois)

# %%
rois
# %%
roi_dict = {}
for roi_name in rois:
    if 'mapping' in roi_name:
        continue
    roi_path = os.path.join(roi_dir, roi_name)
    roi = np.load(roi_path)
    if 'lh.all-vertices_fsaverage_space' in roi_name:
        roi = np.where(roi == 1)[0]
    if 'rh.all-vertices_fsaverage_space' in roi_name:
        roi = np.where(roi == 1)[0] + 163842
    print(roi_name, roi.shape)
    roi_name = roi_name.replace('.npy', '')
    roi = roi.tolist()
    # roi = roi[:10]
    roi_dict[roi_name] = roi
# %%
# write to .py file
with open('roi_data.py', 'w') as f:
    f.write('ROI_DATA = ' + str(roi_dict))
# %%
