import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import OneHotEncoder


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


def _xy_bin_from_grid(grid_props):
    x = (grid_props["grid_coords"][0] / grid_props["grid_scale"]
         + grid_props["grid_offset"][0])
    y = (grid_props["grid_coords"][1] / grid_props["grid_scale"]
         + grid_props["grid_offset"][1])
    return np.vstack((x, y)).T


def create_neighbors_matrix(grid_size, non_empty_indices):
    n_grid_pts = grid_size[0] * grid_size[1]
    rows, cols = np.indices(grid_size)
    row_indices = rows * grid_size[1] + cols

    non_empty_subindices = np.unravel_index(non_empty_indices, grid_size)

    neighbors_i = np.array([0, 0, -1, 1, 0])
    neighbors_j = np.array([-1, 1, 0, 0, 0])

    neighbor_candidates_i = non_empty_subindices[0][:, np.newaxis] + neighbors_i
    neighbor_candidates_j = non_empty_subindices[1][:, np.newaxis] + neighbors_j

    valid_neighbors = np.where(
        (0 <= neighbor_candidates_i) & (neighbor_candidates_i < grid_size[0]) &
        (0 <= neighbor_candidates_j) & (neighbor_candidates_j < grid_size[1])
    )

    non_empty_indices = np.array(non_empty_indices)
    data = np.ones_like(valid_neighbors[0])
    rows_out = non_empty_indices[valid_neighbors[0]]
    cols_out = (neighbor_candidates_i[valid_neighbors] * grid_size[1]
                + neighbor_candidates_j[valid_neighbors])

    neighbors = sp.csr_matrix(
        (data, (rows_out, cols_out)),
        shape=(n_grid_pts, n_grid_pts), dtype=bool
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
        shape=(A.shape[0], B_pts.shape[0]), dtype="float32"
    )
    sparse_matrix.eliminate_zeros()
    normalize(sparse_matrix, norm="l1", copy=False)
    return sparse_matrix


def _smooth_features(features, xy_bin, grid_props, bin_width, smooth,
                     min_markers_per_pixel, num_levels=4):
    """
    Multi-scale smoothing identical to inverse_distance_interpolation,
    but takes pre-binned features as input instead of raw markers.

    Parameters
    ----------
    features : sparse matrix (n_bins x n_genes)
        Raw binned counts — B @ attr
    xy_bin : (n_bins x 2) bin centers
    grid_props : dict from spatial_binning_matrix
    bin_width : float
    smooth : float  (analogous to factor — max_width = bin_width * smooth)
    min_markers_per_pixel : int
    num_levels : int

    Returns
    -------
    features_smooth : sparse matrix, normalized
    passed_threshold : bool array (n_bins,)
    norms : array (n_bins,)
    """
    pixel_widths = np.linspace(bin_width, bin_width * smooth, num_levels)

    bin_center_xy = xy_bin + 0.5 * bin_width

    # density for threshold — use raw sum across all genes
    density = features.sum(axis=1)
    X = density.copy()

    Ws, Bs_list = [], []

    for level in range(1, num_levels):
        # coarser binning
        B_coarse, grid_props_coarse = spatial_binning_matrix(
            xy_bin, bin_width=pixel_widths[level]
        )
        xy_bin_coarse = _xy_bin_from_grid(grid_props_coarse)

        N = create_neighbors_matrix(
            grid_props_coarse["grid_size"],
            grid_props_coarse["non_empty_bins"]
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

    # threshold
    passed_threshold = (density / num_levels >= min_markers_per_pixel).A.flatten()

    # smooth features
    features = features.multiply(passed_threshold[:, None])
    features.eliminate_zeros()

    X_feat = features.copy()
    for Wi, Bi in zip(Ws, Bs_list):
        features = features + Wi.dot(Bi.dot(X_feat))

    # log1p + normalize
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

def points2regions_withsplit(
    xy,
    labels,
    bin_width,
    smooth,
    rare_genes,
    min_genes_per_bin=1,
    alpha=1.0,
    num_levels=4,
):
    """
    Feature extraction with common/rare gene split,
    using multi-scale inverse distance interpolation
    (identical to Points2Regions class).

    Parameters
    ----------
    xy : (N, 2)
    labels : (N,)
    bin_width : float   (analogous to BIN)
    smooth : float      (analogous to factor — max radius = bin_width * smooth)
    rare_genes : list
    min_genes_per_bin : int
    alpha : float       weight of rare features in concatenation
    num_levels : int    number of resolution levels (default 4)

    Returns
    -------
    dict with:
        X_all, X_common, X_rare  — smoothed normalized feature matrices
        xy_bin                   — bin centers
        good_bins                — bool array
        genes_common, genes_rare — gene lists
        bin_size                 — raw bin sizes
        bin_matrix               — B (transcripts -> bins)
    """

    xy = np.asarray(xy, dtype=np.float32)
    labels = np.asarray(labels)

    # ==========================
    # BINNING — done once
    # ==========================
    B, grid_props = spatial_binning_matrix(xy, bin_width)
    xy_bin = _xy_bin_from_grid(grid_props)
    B = B.astype("float32")

    # ==========================
    # GENE SPLIT
    # ==========================
    unique_genes = np.unique(labels)
    rare_genes   = np.array(list(rare_genes))
    is_rare      = np.isin(unique_genes, rare_genes)
    genes_common = unique_genes[~is_rare]
    genes_rare   = unique_genes[is_rare]

    # ==========================
    # ATTRIBUTE MATRICES — binned counts
    # ==========================
    attr_all,    _ = attribute_matrix(labels, unique_genes)
    attr_common, _ = attribute_matrix(labels, genes_common)
    attr_rare,   _ = attribute_matrix(labels, genes_rare)

    attr_all    = attr_all.astype("bool")
    attr_common = attr_common.astype("bool")
    attr_rare   = attr_rare.astype("bool")

    feat_all_raw    = B @ attr_all
    feat_common_raw = B @ attr_common
    feat_rare_raw   = B @ attr_rare

    # ==========================
    # MULTI-SCALE SMOOTHING
    # good_bins defined from X_all (same as p2r)
    # ==========================
    feat_all, passed_threshold, _ = _smooth_features(
        feat_all_raw, xy_bin, grid_props,
        bin_width, smooth, min_genes_per_bin, num_levels
    )

    feat_common, _, _ = _smooth_features(
        feat_common_raw, xy_bin, grid_props,
        bin_width, smooth, min_genes_per_bin, num_levels
    )

    feat_rare, _, _ = _smooth_features(
        feat_rare_raw, xy_bin, grid_props,
        bin_width, smooth, min_genes_per_bin, num_levels
    )

    # good_bins from X_all threshold
    good_bins = passed_threshold

    # raw bin size for reference
    bin_size = feat_all_raw.sum(axis=1).A.flatten()

    # ==========================
    # CONCATENATION
    # ==========================
    X = sp.hstack([feat_common, alpha * feat_rare]).tocsr()

    return {
        "X_all":       feat_all,
        "X":           X,
        "X_common":    feat_common,
        "X_rare":      feat_rare,
        "xy_bin":      xy_bin,
        "good_bins":   good_bins,
        "genes_common": genes_common,
        "genes_rare":   genes_rare,
        "bin_size":    bin_size,
        "bin_matrix":  B,
    }