#!/usr/bin/env python3
"""Reconstruct the circular hue equalization figure from a hues CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hues-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def hue_to_rgb(h: np.ndarray, sat: float = 0.8, val: float = 0.9) -> np.ndarray:
    h = np.asarray(h)
    hsv = np.stack([h, np.ones_like(h) * sat, np.ones_like(h) * val], axis=1)
    return mcolors.hsv_to_rgb(hsv)


def largest_gap_cut(hue: np.ndarray) -> tuple[float, int, float]:
    hue = np.asarray(hue, dtype=float)
    h_sorted = np.sort(hue)
    gaps = np.diff(np.concatenate([h_sorted, h_sorted[:1] + 1.0]))
    k_star = int(np.argmax(gaps))
    h_cut = (h_sorted[k_star] + 0.5 * gaps[k_star]) % 1.0
    return h_cut, k_star, gaps[k_star]


def plot_hue_equalization_from_csv(csv_path: Path, savepath: Path) -> None:
    df = pd.read_csv(csv_path)
    required = ["cluster", "hue_raw", "hue_rotated", "hue_equalized"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Hue CSV is missing required columns: {missing}")

    hue_raw = df["hue_raw"].to_numpy(float)
    hue_rot = df["hue_rotated"].to_numpy(float)
    hue_eq = df["hue_equalized"].to_numpy(float)
    h_cut, _, _ = largest_gap_cut(hue_raw)
    labels = [str(int(c)) for c in df["cluster"]]

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.25])
    axes = [
        fig.add_subplot(gs[0, 0], projection="polar"),
        fig.add_subplot(gs[0, 1], projection="polar"),
        fig.add_subplot(gs[0, 2], projection="polar"),
    ]
    hist_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    panels = [
        ("A", "Original hue", hue_raw),
        ("B", "After circular rotation", hue_rot),
        ("C", "After equalization", hue_eq),
    ]

    for ax, (letter, title, h) in zip(axes, panels):
        theta = 2 * np.pi * h
        colors = hue_to_rgb(h)
        ax.scatter(theta, np.ones_like(theta), c=colors, s=150, edgecolor="black", linewidth=0.4)
        for t, lab in zip(theta, labels):
            ax.text(t, 1.13, lab, fontsize=7, ha="center", va="center")
        ax.set_title(f"{letter}. {title}", fontsize=13, fontweight="bold", pad=18)
        ax.set_yticks([])
        ax.set_ylim(0, 1.22)
        ax.grid(alpha=0.35)

    theta_cut_raw = 2 * np.pi * h_cut
    axes[0].plot([theta_cut_raw, theta_cut_raw], [0, 1.5], linewidth=1.5, color="black")
    axes[1].plot([0, 0], [0, 1.5], linewidth=1.5, color="black")

    hist_data = [
        ("Original hue distribution", hue_raw),
        ("Rotated hue distribution", hue_rot),
        ("Equalized hue distribution", hue_eq),
    ]
    for ax, (title, h) in zip(hist_axes, hist_data):
        colors = hue_to_rgb(h)
        ax.hist(h, bins=np.linspace(0, 1, 21), color="lightgray", edgecolor="black")
        ax.scatter(h, np.full_like(h, -0.15), c=colors, s=150, edgecolor="black", linewidth=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.35, None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Hue")
        ax.set_ylabel("Count")

    fig.suptitle(
        "Circular hue rotation and equalization for cluster colour assignment",
        fontsize=15,
        fontweight="bold",
    )
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_hue_equalization_from_csv(args.hues_csv, args.output)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
