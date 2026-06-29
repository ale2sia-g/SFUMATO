import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
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


def _smooth_features(
    features,
    xy_bin,
    grid_props,
    bin_width,
    smooth,
    min_markers_per_pixel,
    num_levels=4,
):
    """
    Multi-scale smoothing identical to inverse_distance_interpolation,
    but takes pre-binned features as input instead of raw markers.
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
        xy_bin_coarse = _xy_bin_from_grid(grid_props_coarse)

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


def normalize_rare_gene_groups(rare_genes):
    if rare_genes is None:
        return {}

    if not isinstance(rare_genes, dict):
        raise TypeError(
            "rare_genes must be None, an empty dict, or a dictionary such as "
            "{'group_name': ['gene1', 'gene2']}."
        )

    out = {}
    for group_name, genes in rare_genes.items():
        group_name = str(group_name)

        if genes is None:
            gene_list = []
        elif isinstance(genes, str):
            gene_list = [genes]
        else:
            gene_list = [str(g) for g in genes]

        gene_list = sorted(set(gene_list))
        if gene_list:
            out[group_name] = gene_list

    return out


def build_rare_group_score_matrix(
    labels,
    B,
    xy_bin,
    grid_props,
    bin_width,
    smooth,
    min_genes_per_bin,
    num_levels,
    rare_gene_groups,
    unique_genes,
):
    n_bins = B.shape[0]

    if not rare_gene_groups:
        return (
            sp.csr_matrix((n_bins, 0), dtype=np.float32),
            [],
            {},
            {},
        )

    group_names = []
    group_scores = []
    group_genes_requested = {}
    group_genes_present = {}

    for group_name, requested_genes in rare_gene_groups.items():
        requested_genes = [str(g) for g in requested_genes]
        present_genes = sorted(set(requested_genes).intersection(set(unique_genes)))

        group_genes_requested[group_name] = requested_genes
        group_genes_present[group_name] = present_genes

        if len(present_genes) == 0:
            score = sp.csr_matrix((n_bins, 1), dtype=np.float32)
        else:
            attr_group, _ = attribute_matrix(labels, present_genes)
            attr_group = attr_group.astype("bool")
            feat_group_raw = B @ attr_group
            feat_group, _, _ = _smooth_features(
                feat_group_raw,
                xy_bin,
                grid_props,
                bin_width,
                smooth,
                min_genes_per_bin,
                num_levels,
            )

            # Collapse genes within the marker group into one group score per bin.
            score_values = np.asarray(feat_group.sum(axis=1)).ravel().astype(np.float32)
            score = sp.csr_matrix(score_values[:, None])

        group_names.append(group_name)
        group_scores.append(score)

    if len(group_scores) == 0:
        X_rare = sp.csr_matrix((n_bins, 0), dtype=np.float32)
    else:
        X_rare = sp.hstack(group_scores, format="csr").astype(np.float32)

    return X_rare, group_names, group_genes_requested, group_genes_present


# ==========================
# MAIN FUNCTION
# ==========================

def points2regions_withsplit(
    xy,
    labels,
    bin_width,
    smooth,
    rare_genes=None,
    min_genes_per_bin=1,
    alpha=1.0,
    num_levels=4,
):
    """
    Feature extraction using multi-scale inverse distance interpolation.

    Parameters
    ----------
    xy : (N, 2)
    labels : (N,)
    bin_width : float
    smooth : float
    rare_genes : dict or None
        None or {} means no rare-gene features.
        Dictionary format:
            {
                "group_A": ["gene1"],
                "group_B": ["gene2", "gene3"]
            }
        Each group becomes one score column in X_rare.
    min_genes_per_bin : int
    alpha : float
        Kept for API compatibility. It is not used because X_common/X stacked
        output is no longer returned.
    num_levels : int

    Returns
    -------
    dict with:
        X_all                    — smoothed normalized all-gene feature matrix
        X_rare                   — n_bins x n_rare_groups sparse score matrix
        rare_group_names         — group names matching X_rare columns
        rare_group_genes         — requested genes per group
        rare_group_genes_present — genes present in labels per group
        xy_bin                   — bin centers
        good_bins                — bool array
        genes_all                — all gene list
        bin_size                 — raw bin sizes
        bin_matrix               — B (transcripts -> bins)
    """

    xy = np.asarray(xy, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    rare_gene_groups = normalize_rare_gene_groups(rare_genes)

    B, grid_props = spatial_binning_matrix(xy, bin_width)
    xy_bin = _xy_bin_from_grid(grid_props)
    B = B.astype("float32")

    unique_genes = np.unique(labels)

    attr_all, _ = attribute_matrix(labels, unique_genes)
    attr_all = attr_all.astype("bool")
    feat_all_raw = B @ attr_all

    feat_all, passed_threshold, _ = _smooth_features(
        feat_all_raw,
        xy_bin,
        grid_props,
        bin_width,
        smooth,
        min_genes_per_bin,
        num_levels,
    )

    X_rare, rare_group_names, rare_group_genes, rare_group_genes_present = (
        build_rare_group_score_matrix(
            labels=labels,
            B=B,
            xy_bin=xy_bin,
            grid_props=grid_props,
            bin_width=bin_width,
            smooth=smooth,
            min_genes_per_bin=min_genes_per_bin,
            num_levels=num_levels,
            rare_gene_groups=rare_gene_groups,
            unique_genes=unique_genes,
        )
    )

    good_bins = passed_threshold
    bin_size = feat_all_raw.sum(axis=1).A.flatten()

    return {
        "X_all": feat_all,
        "X_rare": X_rare,
        "rare_group_names": np.array(rare_group_names, dtype=object),
        "rare_group_genes": rare_group_genes,
        "rare_group_genes_present": rare_group_genes_present,
        "xy_bin": xy_bin,
        "good_bins": good_bins,
        "genes_all": unique_genes,
        "bin_size": bin_size,
        "bin_matrix": B,
    }