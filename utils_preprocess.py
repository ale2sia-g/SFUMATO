"""Helper functions for preprocessing sparse spatial transcriptomics data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from skimage.color import lab2rgb
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from utils_config import resolve_path


STANDARD_X = "x"
STANDARD_Y = "y"
STANDARD_GENE = "target_name"
STANDARD_NUM_ID = "num_id"


@dataclass(frozen=True)
class PreprocessConfig:
    run_name: str
    input_csv: Path
    x_col: str
    y_col: str
    gene_col: str
    num_id_col: str
    k_list: list[int]
    bgm_oversample: float
    min_genes_per_bin: int
    bin_width: int
    factor: int
    seed: int
    use_svd: bool
    comp: int
    save_p2r: bool
    cache_dir: Path
    alpha: float = 1.0
    rare_genes: Any | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PreprocessConfig":
        columns = config.get("columns", {})
        preprocess = config.get("preprocess", {})
        return cls(
            run_name=config["run_name"],
            input_csv=resolve_path(config, config["input_csv"]),
            x_col=columns.get("x", "x"),
            y_col=columns.get("y", "y"),
            gene_col=columns.get("gene", "target_name"),
            num_id_col=columns.get("num_id", "num_id"),
            k_list=[int(k) for k in preprocess["k_list"]],
            bgm_oversample=float(preprocess.get("bgm_oversample", 1.5)),
            min_genes_per_bin=int(preprocess["min_genes_per_bin"]),
            bin_width=int(preprocess["bin_width"]),
            factor=int(preprocess["factor"]),
            seed=int(preprocess.get("seed", 8)),
            use_svd=bool(preprocess.get("use_svd", True)),
            comp=int(preprocess["comp"]),
            save_p2r=bool(preprocess.get("save_p2r", True)),
            cache_dir=resolve_path(config, preprocess["cache_dir"]),
            alpha=float(preprocess.get("alpha", 1.0)),
            rare_genes=preprocess.get("rare_genes"),
        )


def validate_and_standardize_dataframe(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    missing = [c for c in [cfg.x_col, cfg.y_col, cfg.gene_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            STANDARD_X: df[cfg.x_col].to_numpy(),
            STANDARD_Y: df[cfg.y_col].to_numpy(),
            STANDARD_GENE: df[cfg.gene_col].astype(str).to_numpy(),
        }
    )

    if cfg.num_id_col in df.columns:
        out[STANDARD_NUM_ID] = df[cfg.num_id_col].to_numpy()
    else:
        out[STANDARD_NUM_ID] = 1

    out[STANDARD_NUM_ID] = out[STANDARD_NUM_ID].astype("category")
    return out


def resolve_rare_genes(df: pd.DataFrame, rare_genes: Any | None) -> list[str]:
    # TODO: support user-provided rare-gene dictionaries.
    if isinstance(rare_genes, dict) and rare_genes:
        flattened: list[str] = []
        for value in rare_genes.values():
            if isinstance(value, str):
                flattened.append(value)
            else:
                flattened.extend([str(v) for v in value])
        if flattened:
            return sorted(set(flattened))

    if isinstance(rare_genes, (list, tuple, set)) and rare_genes:
        return sorted(set(str(g) for g in rare_genes))

    counts = df[STANDARD_GENE].value_counts()
    if counts.empty:
        raise ValueError("Cannot infer rare gene from an empty gene column.")
    return [str(counts.idxmin())]


def to_dense_float32(X: Any) -> np.ndarray:
    if sp.issparse(X):
        return X.astype(np.float32).toarray()
    return np.asarray(X, dtype=np.float32)


def rgb01_to_hex(rgb01: np.ndarray) -> str:
    rgb = np.clip(np.asarray(rgb01), 0.0, 1.0)
    rgb255 = (rgb * 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb255[0], rgb255[1], rgb255[2])


def tsne_colors_p2r(cluster_centers: np.ndarray, seed: int = 42) -> list[str]:
    n = len(cluster_centers)
    if n == 1:
        return ["#00ff00"]

    distances = pairwise_distances(cluster_centers)
    embedding = TSNE(
        n_components=2,
        perplexity=min(n - 1, 30),
        init="random",
        metric="precomputed",
        random_state=seed,
    ).fit_transform(distances)

    mu = 5
    out_ma, out_mi = 128, -128
    mi = np.percentile(embedding, q=mu, axis=0, keepdims=True)
    ma = np.percentile(embedding, q=100 - mu, axis=0, keepdims=True)
    colors = np.clip((embedding - mi) / (ma - mi + 1e-8), 0.0, 1.0)
    colors = (out_ma - out_mi) * colors + out_mi
    colors = np.hstack((np.ones((n, 1)) * 70, colors))
    colors_rgb = lab2rgb(colors)
    return [rgb01_to_hex(c) for c in colors_rgb]


def make_preprocess_stem(
    run_name: str,
    bin_width: int,
    factor: int,
    comp: int,
    use_svd: bool,
) -> str:
    dim_tag = f"_SVD{comp}" if use_svd else "_noSVD"
    return f"{run_name}_BIN{bin_width}_F{factor}{dim_tag}"


def make_bgm_stem(
    run_name: str,
    k_bgm: int,
    k: int,
    bin_width: int,
    factor: int,
    comp: int,
    use_svd: bool,
    bgm_oversample: float,
) -> str:
    dim_tag = f"_SVD{comp}" if use_svd else "_noSVD"
    if bgm_oversample > 1.0:
        return f"{run_name}_BGM{k_bgm}to{k}_BIN{bin_width}_F{factor}{dim_tag}"
    return f"{run_name}_BGM{k_bgm}_BIN{bin_width}_F{factor}{dim_tag}"


def p2r_cache_keys(k: int) -> tuple[str, str]:
    return f"cluster_p2r_bins_K{k}", f"p2r_color_bins_K{k}"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
