#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


Image.MAX_IMAGE_PIXELS = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-csv", required=True)
    p.add_argument("--transcript-proba-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-size", nargs=2, type=int, required=True, metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--pixel-size-um", type=float, required=True)

    p.add_argument("--bin-id-col", default="bin_id")
    p.add_argument("--x-col", default="x")
    p.add_argument("--y-col", default="y")
    p.add_argument("--prob-cols", nargs="+", default=None)

    p.add_argument("--bin-size", type=int, default=7)
    p.add_argument("--sigma", type=float, default=7.0)
    p.add_argument("--low-threshold", type=float, default=0.10)
    p.add_argument("--high-threshold", type=float, default=0.30)
    p.add_argument("--threshold-scale", choices=("relative", "absolute"), default="relative")
    p.add_argument("--save-smoothed", action="store_true")
    return p.parse_args()


def infer_prob_cols(csv_path, requested):
    header = pd.read_csv(csv_path, nrows=0)
    cols = list(header.columns)

    if requested is not None:
        missing = [c for c in requested if c not in cols]
        if missing:
            raise ValueError(f"Missing probability columns: {missing}")
        return requested

    prob_cols = [c for c in cols if re.fullmatch(r"p\d+", str(c))]
    prob_cols = sorted(prob_cols, key=lambda c: int(c[1:]))

    if not prob_cols:
        raise ValueError("No posterior columns found. Expected p1, p2, ...")

    return prob_cols


def load_bin_coordinates(bin_csv, bin_id_col, x_col, y_col):
    df = pd.read_csv(bin_csv, usecols=[bin_id_col, x_col, y_col])
    df = df.drop_duplicates(subset=[bin_id_col], keep="first")

    df[bin_id_col] = df[bin_id_col].astype(np.int64)
    df[x_col] = np.rint(df[x_col]).astype(np.int32)
    df[y_col] = np.rint(df[y_col]).astype(np.int32)

    return df.set_index(bin_id_col)


def load_bin_posteriors(transcript_csv, bin_id_col, prob_cols):
    usecols = [bin_id_col] + prob_cols

    print("Loading one posterior vector per bin...", flush=True)
    df = pd.read_csv(transcript_csv, usecols=usecols)

    # All transcripts in the same true bin_id should share the same posterior vector.
    # We therefore keep only one row per bin, so each bin contributes once.
    df = df.drop_duplicates(subset=[bin_id_col], keep="first")

    df[bin_id_col] = df[bin_id_col].astype(np.int64)

    for col in prob_cols:
        df[col] = df[col].astype(np.float32)

    return df.set_index(bin_id_col)


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


def save_smoothed(smoothed, path):
    max_val = float(np.nanmax(smoothed))
    arr = np.empty(smoothed.shape, dtype=np.uint8)

    if max_val > 0:
        scaled = smoothed / max_val
        np.clip(scaled, 0.0, 1.0, out=scaled)
        np.multiply(scaled, 255, out=scaled)
        arr[:] = scaled.astype(np.uint8)
        del scaled
    else:
        arr.fill(0)

    Image.fromarray(arr, mode="L").save(path)
    del arr
    gc.collect()


def main():
    args = parse_args()

    width, height = args.image_size
    pixel_area_um2 = args.pixel_size_um ** 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_cols = infer_prob_cols(args.transcript_proba_csv, args.prob_cols)

    print("Loading bin coordinates...", flush=True)
    bin_xy = load_bin_coordinates(
        args.bin_csv,
        args.bin_id_col,
        args.x_col,
        args.y_col,
    )

    bin_post = load_bin_posteriors(
        args.transcript_proba_csv,
        args.bin_id_col,
        prob_cols,
    )

    common_bins = bin_xy.index.intersection(bin_post.index)
    print(f"Bins with coordinates: {len(bin_xy)}", flush=True)
    print(f"Bins with posteriors:   {len(bin_post)}", flush=True)
    print(f"Common bins:            {len(common_bins)}", flush=True)

    bin_xy = bin_xy.loc[common_bins]
    bin_post = bin_post.loc[common_bins]

    summary = {
        "input_bin_csv": str(args.bin_csv),
        "input_transcript_proba_csv": str(args.transcript_proba_csv),
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
        "mask_value_convention": "uint8 PNG, background=0, cluster=255",
        "rasterization": "one BINxBIN square per bin, one posterior vector per unique bin_id",
        "n_bins_with_coordinates": int(len(bin_xy)),
        "n_bins_with_posteriors": int(len(bin_post)),
        "n_common_bins": int(len(common_bins)),
        "clusters": {},
    }

    x = bin_xy[args.x_col].to_numpy(dtype=np.int32, copy=True)
    y = bin_xy[args.y_col].to_numpy(dtype=np.int32, copy=True)

    for i, prob_col in enumerate(prob_cols, start=1):
        print(f"[{i}/{len(prob_cols)}] Processing {prob_col}", flush=True)

        p = bin_post[prob_col].to_numpy(dtype=np.float32, copy=True)

        probability_image = rasterize_bins_as_squares(
            x=x,
            y=y,
            p=p,
            width=width,
            height=height,
            bin_size=args.bin_size,
        )

        del p
        gc.collect()

        ndi.gaussian_filter(
            probability_image,
            sigma=args.sigma,
            output=probability_image,
        )

        smoothed = probability_image

        mask, low_used, high_used, smoothed_max = reconstruction_mask(
            smoothed,
            args.low_threshold,
            args.high_threshold,
            args.threshold_scale,
        )

        area_px = int(np.count_nonzero(mask))
        area_um2 = float(area_px * pixel_area_um2)

        mask_path = out_dir / f"{prob_col}_mask.png"
        save_mask(mask, mask_path)

        smoothed_path = None
        if args.save_smoothed:
            smoothed_path = out_dir / f"{prob_col}_smoothed.png"
            save_smoothed(smoothed, smoothed_path)

        summary["clusters"][prob_col] = {
            "area_px": area_px,
            "area_um2": area_um2,
            "mask_path": str(mask_path),
            "smoothed_path": str(smoothed_path) if smoothed_path else None,
            "smoothed_max": float(smoothed_max),
            "low_threshold_used": float(low_used),
            "high_threshold_used": float(high_used),
        }

        del probability_image, smoothed, mask
        gc.collect()

    del x, y, bin_xy, bin_post
    gc.collect()

    json_path = out_dir / "cluster_areas_um2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Wrote {json_path}")
    print(f"Done. Wrote {len(prob_cols)} masks to {out_dir}")


if __name__ == "__main__":
    main()