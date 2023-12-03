import torch
import numpy as np
from torchmetrics.functional import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
)
import scipy.cluster.hierarchy as shc


@torch.no_grad()
def gpu_kmeans_cluster(voxel_outs, n_clusters=100):
    from fast_pytorch_kmeans import KMeans

    kmeans = KMeans(
        n_clusters=n_clusters, verbose=True, mode="cosine", max_iter=1000, tol=1e-6
    )
    labels = kmeans.fit_predict(voxel_outs)
    return kmeans, labels


@torch.no_grad()
def cluster_channels(weights, target_num_rois=20):
    weights = weights.detach().cuda()

    kernel = weights @ weights.t()

    k = 1000
    km, labels = gpu_kmeans_cluster(kernel, n_clusters=k)
    c = km.centroids
    labels = labels.cpu().numpy()
    km_labels = labels

    # d = torch.cdist(c, c)
    d = pairwise_cosine_similarity(c)
    d[torch.isnan(d)] = 0
    # d = pairwise_euclidean_distance(c)
    d = d.cpu().numpy()

    Z = shc.linkage(d, method="ward", optimal_ordering=False)

    max_dist = 20
    num_rois = 0
    while num_rois != target_num_rois:
        dn_labels = shc.fcluster(Z, max_dist, criterion="distance")
        num_rois = len(np.unique(dn_labels))
        if num_rois > target_num_rois:
            max_dist *= 1.5
        elif num_rois < target_num_rois:
            max_dist *= 0.5
        else:
            pass

    vi_dict = {}  # i -> voxel indices
    kvi_dict = {}  # i -> km labels
    for i in np.unique(dn_labels):
        cluster_voxel_indices = []
        labels = (dn_labels == i).nonzero()[0]
        for l in labels:
            voxel_indices = (km_labels == l).nonzero()[0]
            cluster_voxel_indices.append(voxel_indices)
        cluster_voxel_indices = np.concatenate(cluster_voxel_indices)
        cluster_voxel_indices.sort()
        cluster_voxel_indices = cluster_voxel_indices
        vi_dict[i] = cluster_voxel_indices
        kvi_dict[i] = labels

    return vi_dict  #  i -> voxel indices
