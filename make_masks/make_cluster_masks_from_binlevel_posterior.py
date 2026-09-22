#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


Image.MAX_IMAGE_PIXELS = None


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build semantic SFUMATO masks for Invasive and DCIS-like breast niches "
            "from selected posterior columns, compare them with pathology GeoJSON "
            "annotations, and compute Dice scores."
        )
    )
    p.add_argument("--bin-csv", required=True)
    p.add_argument("--transcript-proba-csv", required=True)
    p.add_argument("--annotation-geojson", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-size", nargs=2, type=int, required=True, metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--pixel-size-um", type=float, required=True)

    p.add_argument("--invasive-clusters", nargs="+", required=True)
    p.add_argument("--dcis-clusters", nargs="+", required=True)
    p.add_argument(
        "--cluster-prefix",
        default="p",
        help="Prefix used to convert numeric cluster ids to posterior columns, e.g. 4 -> p4.",
    )

    p.add_argument("--bin-id-col", default="bin_id")
    p.add_argument("--x-col", default="x")
    p.add_argument("--y-col", default="y")

    p.add_argument("--bin-size", type=int, default=7)
    p.add_argument("--sigma", type=float, default=7.0)
    p.add_argument("--low-threshold", type=float, default=0.10)
    p.add_argument("--high-threshold", type=float, default=0.30)
    p.add_argument("--threshold-scale", choices=("relative", "absolute"), default="relative")

    p.add_argument("--annotation-invasive-regex", default="invasive")
    p.add_argument("--annotation-dcis-regex", default=r"dcis\\s*1|dcis1|dcis\\s*2|dcis2")
    p.add_argument(
        "--annotation-label-fields",
        nargs="+",
        default=["classification.name", "class", "name", "label"],
        help="GeoJSON property fields searched for annotation labels. Use dots for nested fields.",
    )
    p.add_argument("--plot-max-dim", type=int, default=3000)
    return p.parse_args()


def cluster_to_prob_col(cluster, prefix="p"):
    cluster = str(cluster)
    if re.fullmatch(r"p\d+", cluster):
        return cluster
    if re.fullmatch(r"\d+", cluster):
        return f"{prefix}{cluster}"
    return cluster


def selected_prob_cols(args):
    invasive = [cluster_to_prob_col(c, args.cluster_prefix) for c in args.invasive_clusters]
    dcis = [cluster_to_prob_col(c, args.cluster_prefix) for c in args.dcis_clusters]
    return invasive, dcis


def validate_prob_cols(csv_path, prob_cols):
    header = pd.read_csv(csv_path, nrows=0)
    cols = set(header.columns)
    missing = [c for c in prob_cols if c not in cols]
    if missing:
        raise ValueError(f"Missing probability columns in {csv_path}: {missing}")


def load_bin_coordinates(bin_csv, bin_id_col, x_col, y_col):
    df = pd.read_csv(bin_csv, usecols=[bin_id_col, x_col, y_col])
    df = df.drop_duplicates(subset=[bin_id_col], keep="first")
    df[bin_id_col] = df[bin_id_col].astype(np.int64)
    df[x_col] = np.rint(df[x_col]).astype(np.int32)
    df[y_col] = np.rint(df[y_col]).astype(np.int32)
    return df.set_index(bin_id_col)


def load_group_posteriors(transcript_csv, bin_id_col, invasive_cols, dcis_cols):
    usecols = [bin_id_col] + sorted(set(invasive_cols + dcis_cols))

    print("Loading one posterior vector per bin...", flush=True)
    df = pd.read_csv(transcript_csv, usecols=usecols)

    # All transcripts in the same true bin_id should share the same posterior vector.
    df = df.drop_duplicates(subset=[bin_id_col], keep="first")
    df[bin_id_col] = df[bin_id_col].astype(np.int64)

    for col in usecols:
        if col != bin_id_col:
            df[col] = df[col].astype(np.float32)

    out = pd.DataFrame(index=df[bin_id_col].to_numpy(dtype=np.int64))
    out.index.name = bin_id_col
    out["Invasive"] = df[invasive_cols].sum(axis=1).to_numpy(dtype=np.float32)
    out["DCIS_like"] = df[dcis_cols].sum(axis=1).to_numpy(dtype=np.float32)

    # Group posteriors should be in [0, 1] when the selected clusters are disjoint
    # posterior classes. Clip defensively in case duplicated columns were passed.
    out["Invasive"] = out["Invasive"].clip(0.0, 1.0)
    out["DCIS_like"] = out["DCIS_like"].clip(0.0, 1.0)
    return out


def rasterize_bins_as_squares(x, y, p, width, height, bin_size):
    image = np.zeros((height, width), dtype=np.float32)
    half = bin_size // 2

    valid = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
        & np.isfinite(p)
        & (p > 0)
    )

    xv = x[valid]
    yv = y[valid]
    pv = np.clip(p[valid].astype(np.float32, copy=False), 0.0, 1.0)

    for cx, cy, val in zip(xv, yv, pv):
        x0 = max(0, int(cx) - half)
        x1 = min(width, int(cx) - half + bin_size)
        y0 = max(0, int(cy) - half)
        y1 = min(height, int(cy) - half + bin_size)
        patch = image[y0:y1, x0:x1]
        np.maximum(patch, val, out=patch)

    return image


def reconstruction_mask(smoothed, low_thr, high_thr, threshold_scale):
    max_val = float(np.nanmax(smoothed))

    if max_val <= 0:
        return np.zeros(smoothed.shape, dtype=bool), 0.0, 0.0, max_val

    if threshold_scale == "relative":
        low = low_thr * max_val
        high = high_thr * max_val
    else:
        low = low_thr
        high = high_thr

    if low > high:
        raise ValueError("low-threshold must be <= high-threshold")

    seed = smoothed >= high
    mask = smoothed >= low
    reconstructed = ndi.binary_propagation(seed, mask=mask)

    del seed, mask
    gc.collect()
    return reconstructed, float(low), float(high), max_val


def save_mask(mask, path):
    arr = np.empty(mask.shape, dtype=np.uint8)
    arr[mask] = 255
    arr[~mask] = 0
    Image.fromarray(arr, mode="L").save(path)
    del arr
    gc.collect()


def save_probability_png(image, path):
    max_val = float(np.nanmax(image))
    arr = np.empty(image.shape, dtype=np.uint8)

    if max_val > 0:
        scaled = image / max_val
        np.clip(scaled, 0.0, 1.0, out=scaled)
        np.multiply(scaled, 255, out=scaled)
        arr[:] = scaled.astype(np.uint8)
        del scaled
    else:
        arr.fill(0)

    Image.fromarray(arr, mode="L").save(path)
    del arr
    gc.collect()


def get_nested_property(properties, field):
    value = properties
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def feature_label(feature, label_fields):
    properties = feature.get("properties", {}) or {}
    values = []
    for field in label_fields:
        value = get_nested_property(properties, field)
        if value is not None:
            values.append(str(value))
    return " ".join(values)


def iter_polygon_rings(geometry):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        for ring in coords[:1]:
            yield ring
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon[:1]:
                yield ring


def rasterize_annotation_masks(geojson_path, width, height, label_fields, invasive_regex, dcis_regex):
    invasive_re = re.compile(invasive_regex, flags=re.IGNORECASE)
    dcis_re = re.compile(dcis_regex, flags=re.IGNORECASE)

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    invasive_img = Image.new("L", (width, height), 0)
    dcis_img = Image.new("L", (width, height), 0)
    invasive_draw = ImageDraw.Draw(invasive_img)
    dcis_draw = ImageDraw.Draw(dcis_img)

    counts = {"Invasive": 0, "DCIS_like": 0, "ignored": 0}

    for feature in data.get("features", []):
        label = feature_label(feature, label_fields)
        geometry = feature.get("geometry") or {}

        if invasive_re.search(label):
            draw = invasive_draw
            counts["Invasive"] += 1
        elif dcis_re.search(label):
            draw = dcis_draw
            counts["DCIS_like"] += 1
        else:
            counts["ignored"] += 1
            continue

        for ring in iter_polygon_rings(geometry):
            if len(ring) < 3:
                continue
            xy = [(float(pt[0]), float(pt[1])) for pt in ring]
            draw.polygon(xy, fill=255)

    invasive = np.array(invasive_img, dtype=np.uint8) > 0
    dcis = np.array(dcis_img, dtype=np.uint8) > 0
    return {"Invasive": invasive, "DCIS_like": dcis}, counts


def dice_score(pred, truth):
    pred = pred.astype(bool, copy=False)
    truth = truth.astype(bool, copy=False)
    denom = int(pred.sum()) + int(truth.sum())
    if denom == 0:
        return float("nan")
    inter = int(np.logical_and(pred, truth).sum())
    return float(2.0 * inter / denom)


def display_image(image, max_dim):
    h, w = image.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / max_dim)))
    return image[::step, ::step]


def plot_two_panel(images, title, output_path, cmap="magma", max_dim=3000, vmin=None, vmax=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, (name, image) in zip(axes, images.items()):
        shown = display_image(image, max_dim)
        ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_mask_comparison(pred_masks, annotation_masks, output_path, max_dim=3000):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    rows = [("Invasive", axes[0]), ("DCIS_like", axes[1])]

    for name, row_axes in rows:
        pred = display_image(pred_masks[name].astype(np.uint8), max_dim)
        ann = display_image(annotation_masks[name].astype(np.uint8), max_dim)

        row_axes[0].imshow(pred, cmap="gray", vmin=0, vmax=1)
        row_axes[0].set_title(f"SFUMATO {name} mask")
        row_axes[0].axis("off")

        row_axes[1].imshow(ann, cmap="gray", vmin=0, vmax=1)
        row_axes[1].set_title(f"Annotation {name} mask")
        row_axes[1].axis("off")

    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main():
    args = parse_args()

    width, height = args.image_size
    pixel_area_um2 = args.pixel_size_um ** 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    invasive_cols, dcis_cols = selected_prob_cols(args)
    all_cols = invasive_cols + dcis_cols
    validate_prob_cols(args.transcript_proba_csv, all_cols)

    print(f"Invasive posterior columns: {invasive_cols}", flush=True)
    print(f"DCIS-like posterior columns: {dcis_cols}", flush=True)

    print("Loading bin coordinates...", flush=True)
    bin_xy = load_bin_coordinates(args.bin_csv, args.bin_id_col, args.x_col, args.y_col)
    group_post = load_group_posteriors(
        args.transcript_proba_csv,
        args.bin_id_col,
        invasive_cols,
        dcis_cols,
    )

    common_bins = bin_xy.index.intersection(group_post.index)
    print(f"Bins with coordinates: {len(bin_xy)}", flush=True)
    print(f"Bins with posteriors:   {len(group_post)}", flush=True)
    print(f"Common bins:            {len(common_bins)}", flush=True)

    bin_xy = bin_xy.loc[common_bins]
    group_post = group_post.loc[common_bins]

    x = bin_xy[args.x_col].to_numpy(dtype=np.int32, copy=True)
    y = bin_xy[args.y_col].to_numpy(dtype=np.int32, copy=True)

    raw_images = {}
    smoothed_images = {}
    pred_masks = {}
    stats = {}

    for group_name in ["Invasive", "DCIS_like"]:
        print(f"Processing {group_name}", flush=True)
        p = group_post[group_name].to_numpy(dtype=np.float32, copy=True)
        raw = rasterize_bins_as_squares(
            x=x,
            y=y,
            p=p,
            width=width,
            height=height,
            bin_size=args.bin_size,
        )
        del p
        gc.collect()

        raw_images[group_name] = raw.copy()
        raw_path = out_dir / f"{group_name}_posterior_raw.png"
        save_probability_png(raw, raw_path)

        smoothed = raw
        ndi.gaussian_filter(smoothed, sigma=args.sigma, output=smoothed)
        smoothed_images[group_name] = smoothed.copy()
        smoothed_path = out_dir / f"{group_name}_posterior_smoothed.png"
        save_probability_png(smoothed, smoothed_path)

        mask, low_used, high_used, smoothed_max = reconstruction_mask(
            smoothed,
            args.low_threshold,
            args.high_threshold,
            args.threshold_scale,
        )
        pred_masks[group_name] = mask
        mask_path = out_dir / f"{group_name}_mask.png"
        save_mask(mask, mask_path)

        area_px = int(np.count_nonzero(mask))
        stats[group_name] = {
            "raw_path": str(raw_path),
            "smoothed_path": str(smoothed_path),
            "mask_path": str(mask_path),
            "area_px": area_px,
            "area_um2": float(area_px * pixel_area_um2),
            "smoothed_max": float(smoothed_max),
            "low_threshold_used": float(low_used),
            "high_threshold_used": float(high_used),
        }

        del raw, smoothed, mask
        gc.collect()

    plot_two_panel(
        raw_images,
        title="Summed SFUMATO posterior maps",
        output_path=out_dir / "semantic_posteriors_raw.png",
        cmap="magma",
        max_dim=args.plot_max_dim,
        vmin=0,
        vmax=1,
    )
    plot_two_panel(
        smoothed_images,
        title="Smoothed summed SFUMATO posterior maps",
        output_path=out_dir / "semantic_posteriors_smoothed.png",
        cmap="magma",
        max_dim=args.plot_max_dim,
        vmin=0,
        vmax=1,
    )

    print("Rasterizing pathology annotations...", flush=True)
    annotation_masks, annotation_counts = rasterize_annotation_masks(
        args.annotation_geojson,
        width=width,
        height=height,
        label_fields=args.annotation_label_fields,
        invasive_regex=args.annotation_invasive_regex,
        dcis_regex=args.annotation_dcis_regex,
    )

    for group_name, mask in annotation_masks.items():
        ann_path = out_dir / f"{group_name}_annotation_mask.png"
        save_mask(mask, ann_path)
        stats[group_name]["annotation_mask_path"] = str(ann_path)
        stats[group_name]["annotation_area_px"] = int(np.count_nonzero(mask))
        stats[group_name]["annotation_area_um2"] = float(np.count_nonzero(mask) * pixel_area_um2)
        stats[group_name]["dice"] = dice_score(pred_masks[group_name], mask)

    plot_mask_comparison(
        pred_masks,
        annotation_masks,
        output_path=out_dir / "semantic_masks_vs_annotations.png",
        max_dim=args.plot_max_dim,
    )

    summary = {
        "input_bin_csv": str(args.bin_csv),
        "input_transcript_proba_csv": str(args.transcript_proba_csv),
        "input_annotation_geojson": str(args.annotation_geojson),
        "image_width_px": int(width),
        "image_height_px": int(height),
        "pixel_size_um": float(args.pixel_size_um),
        "pixel_area_um2": float(pixel_area_um2),
        "bin_size_px": int(args.bin_size),
        "sigma_px": float(args.sigma),
        "threshold_mode": "binary_morphological_reconstruction_by_propagation",
        "threshold_scale": args.threshold_scale,
        "low_threshold_input": float(args.low_threshold),
        "high_threshold_input": float(args.high_threshold),
        "invasive_clusters": args.invasive_clusters,
        "invasive_posterior_columns": invasive_cols,
        "dcis_clusters": args.dcis_clusters,
        "dcis_posterior_columns": dcis_cols,
        "annotation_counts": annotation_counts,
        "n_bins_with_coordinates": int(len(bin_xy)),
        "n_bins_with_posteriors": int(len(group_post)),
        "n_common_bins": int(len(common_bins)),
        "outputs": {
            "raw_posterior_subplot": str(out_dir / "semantic_posteriors_raw.png"),
            "smoothed_posterior_subplot": str(out_dir / "semantic_posteriors_smoothed.png"),
            "mask_annotation_subplot": str(out_dir / "semantic_masks_vs_annotations.png"),
        },
        "semantic_classes": stats,
    }

    json_path = out_dir / "semantic_mask_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Wrote summary to {json_path}")
    for name, class_stats in stats.items():
        print(f"{name} Dice: {class_stats['dice']:.4f}")


if __name__ == "__main__":
    main()