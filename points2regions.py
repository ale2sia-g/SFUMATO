"""Points2Regions-style feature extraction for sparse spatial transcriptomics."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize


# ==========================
# CORE UTILS
# ==========================

def attribute_matrix(cat, unique_cat):
    X = np.array(cat).reshape((-1, 1))
    encoder = OneHotEncoder(
        categories=[np.array(unique_cat)],
        sparse_output=True,
        handle_unknown="ignore",
    )
    encoder.fit(X)
    y = encoder.transform(X)
    categories = list(encoder.categories_[0])
    return y, categories


def spatial_binning_matrix(xy, bin_width):
    mi, ma = xy.min(axis=0, keepdims=True), xy.max(axis=0, keepdims=True)
    xys = xy - mi
    grid = (ma - mi).flatten()
    bin_ids = (xys // bin_width).astype(int)
    bin_ids = tuple(x for x in bin_ids.T)
    size = tuple((grid // bin_width + 1).astype(int))
    linear_ind = np.ravel_multi_index(bin_ids, size)

    bin_matrix, linear_unique = attribute_matrix(linear_ind, np.unique(linear_ind))
    bin_matrix = bin_matrix.T

    sub_unique = np.unravel_index(linear_unique, size)
    grid_props = dict(
        grid_coords=sub_unique,
        grid_size=size,
        grid_offset=mi.flatten(),
        grid_scale=1.0 / bin_width,
        non_empty_bins=linear_unique,
    )

    return bin_matrix, grid_props


def xy_bin_from_grid(grid_props):
    x = (
        grid_props["grid_coords"][0] / grid_props["grid_scale"]
        + grid_props["grid_offset"][0]
    )
    y = (
        grid_props["grid_coords"][1] / grid_props["grid_scale"]
        + grid_props["grid_offset"][1]
    )
    return np.vstack((x, y)).T


def create_neighbors_matrix(grid_size, non_empty_indices):
    n_grid_pts = grid_size[0] * grid_size[1]

    non_empty_subindices = np.unravel_index(non_empty_indices, grid_size)

    neighbors_i = np.array([0, 0, -1, 1, 0])
    neighbors_j = np.array([-1, 1, 0, 0, 0])

    neighbor_candidates_i = non_empty_subindices[0][:, np.newaxis] + neighbors_i
    neighbor_candidates_j = non_empty_subindices[1][:, np.newaxis] + neighbors_j

    valid_neighbors = np.where(
        (0 <= neighbor_candidates_i)
        & (neighbor_candidates_i < grid_size[0])
        & (0 <= neighbor_candidates_j)
        & (neighbor_candidates_j < grid_size[1])
    )

    non_empty_indices = np.array(non_empty_indices)
    data = np.ones_like(valid_neighbors[0])
    rows_out = non_empty_indices[valid_neighbors[0]]
    cols_out = (
        neighbor_candidates_i[valid_neighbors] * grid_size[1]
        + neighbor_candidates_j[valid_neighbors]
    )

    neighbors = sp.csr_matrix(
        (data, (rows_out, cols_out)),
        shape=(n_grid_pts, n_grid_pts),
        dtype=bool,
    )
    neighbors = neighbors[non_empty_indices, :][:, non_empty_indices]
    return neighbors


def find_inverse_distance_weights(ij, A, B_pts, bin_width):
    cols, rows = ij
    distances = np.linalg.norm(A[rows, :] - B_pts[cols, :], axis=1)

    good_ind = distances <= bin_width * 1.000005
    vals = 1.0 / (distances + 1e-5)
    vals = vals[good_ind]
    rows = rows[good_ind]
    cols = cols[good_ind]

    sparse_matrix = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(A.shape[0], B_pts.shape[0]),
        dtype="float32",
    )
    sparse_matrix.eliminate_zeros()
    normalize(sparse_matrix, norm="l1", copy=False)

    return sparse_matrix


def smooth_features(
    features,
    xy_bin,
    grid_props,
    bin_width,
    smooth,
    min_markers_per_pixel,
    num_levels=4,
):
    """
    Multi-scale smoothing identical to inverse-distance interpolation,
    but takes pre-binned features as input instead of raw markers.

    Parameters
    ----------
    features : sparse matrix, shape (n_bins, n_genes)
        Raw binned counts.
    xy_bin : array, shape (n_bins, 2)
        Bin coordinates.
    grid_props : dict
        Grid metadata from spatial_binning_matrix.
    bin_width : float
        Base bin width.
    smooth : float
        Smoothing factor. The maximum smoothing width is bin_width * smooth.
    min_markers_per_pixel : int
        Minimum average marker density required for a bin to pass threshold.
    num_levels : int
        Number of smoothing levels.

    Returns
    -------
    features_smooth : sparse matrix
        Smoothed, log1p-transformed, row-normalized feature matrix.
    passed_threshold : array
        Boolean mask of bins passing the density threshold.
    norms : array
        Row normalization factors.
    """
    pixel_widths = np.linspace(bin_width, bin_width * smooth, num_levels)
    bin_center_xy = xy_bin + 0.5 * bin_width

    density = features.sum(axis=1)
    X = density.copy()

    Ws, Bs_list = [], []

    for level in range(1, num_levels):
        B_coarse, grid_props_coarse = spatial_binning_matrix(
            xy_bin,
            bin_width=pixel_widths[level],
        )
        xy_bin_coarse = xy_bin_from_grid(grid_props_coarse)

        N = create_neighbors_matrix(
            grid_props_coarse["grid_size"],
            grid_props_coarse["non_empty_bins"],
        )

        neighbors = N.dot(B_coarse).nonzero()
        low_res_center = xy_bin_coarse + 0.5 * pixel_widths[level]

        W = find_inverse_distance_weights(
            neighbors,
            bin_center_xy,
            low_res_center,
            pixel_widths[level],
        )

        Ws.append(W)
        Bs_list.append(B_coarse)

        density = density + W.dot(B_coarse.dot(X))

    passed_threshold = (density / num_levels >= min_markers_per_pixel).A.flatten()

    features = features.multiply(passed_threshold[:, None])
    features.eliminate_zeros()

    X_feat = features.copy()
    for Wi, Bi in zip(Ws, Bs_list):
        features = features + Wi.dot(Bi.dot(X_feat))

    features.data = np.log1p(features.data)

    s = features.sum(axis=1)
    norms = 1.0 / (s + 1e-5)
    norms = np.asarray(norms).ravel()
    norms[np.isinf(norms)] = 0.0

    features = features.multiply(norms[:, None]).tocsr()

    return features, passed_threshold, norms


def safe_row_normalize(X):
    X = X.tocsr().astype(np.float32)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    good = row_sums > 0
    if np.any(good):
        X[good] = normalize(X[good], norm="l1", axis=1)
    return X


# ==========================
# MAIN FUNCTION
# ==========================

def points2regions(
    xy,
    labels,
    bin_width,
    smooth,
    min_genes_per_bin=1,
    alpha=1.0,
    num_levels=4,
):
    """
    Feature extraction using multi-scale inverse-distance interpolation.

    Parameters
    ----------
    xy : array, shape (N, 2)
        Transcript coordinates.
    labels : array, shape (N,)
        Gene labels.
    bin_width : float
        Spatial bin width.
    smooth : float
        Smoothing factor. The maximum smoothing width is bin_width * smooth.
    min_genes_per_bin : int
        Minimum average gene density required for a bin to pass threshold.
    alpha : float
        Kept for backward compatibility. Not used.
    num_levels : int
        Number of smoothing levels.

    Returns
    -------
    dict with:
        X_all      : smoothed normalized all-gene feature matrix
        xy_bin     : bin coordinates
        good_bins  : boolean array marking bins passing threshold
        genes_all  : all gene names
        bin_size   : raw bin sizes
        bin_matrix : sparse transcript-to-bin matrix
    """
    del alpha

    xy = np.asarray(xy, dtype=np.float32)
    labels = np.asarray(labels).astype(str)

    B, grid_props = spatial_binning_matrix(xy, bin_width)
    xy_bin = xy_bin_from_grid(grid_props)
    B = B.astype("float32")

    unique_genes = np.unique(labels)

    attr_all, _ = attribute_matrix(labels, unique_genes)
    attr_all = attr_all.astype("bool")

    feat_all_raw = B @ attr_all

    feat_all, passed_threshold, _ = smooth_features(
        feat_all_raw,
        xy_bin,
        grid_props,
        bin_width,
        smooth,
        min_genes_per_bin,
        num_levels,
    )

    good_bins = passed_threshold
    bin_size = feat_all_raw.sum(axis=1).A.flatten()

    return {
        "X_all": feat_all,
        "xy_bin": xy_bin,
        "good_bins": good_bins,
        "genes_all": unique_genes,
        "bin_size": bin_size,
        "bin_matrix": B,
    }