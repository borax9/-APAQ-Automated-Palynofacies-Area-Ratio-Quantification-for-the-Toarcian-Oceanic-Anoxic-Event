# -*- coding: utf-8 -*-
"""
palynofacies_full.py
====================
Full Palynofacies + TAI analysis using the YOLO26 segmentation model.

Outputs:
  1. tyson_ternary_plot.png     — Tyson (1995) 9-field ternary plot
  2. relative_abundance_log.png — Relative abundance in geological log format
  3. tai_diagram.png            — TAI maturity diagram
  4. summary_charts.png         — Bar + Pie + Boxplot summary
  5. palynofacies_report.xlsx   — Per Image + Summary + TAI sheets
  6. palynofacies_results.csv

Usage:
  python palynofacies_full.py --weights best.pt --images test_images/ --mag 10

Installation:
  pip install ultralytics matplotlib pandas openpyxl opencv-python
"""

import os
import argparse, sys, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES   = ["AOM", "Background", "Palynomorph", "Phytoclast"]
PALYNO_GROUPS = ["AOM", "Palynomorph", "Phytoclast"]
BG_CLASS      = "Background"
CONF_THRESH   = 0.25
IOU_THRESH    = 0.45

COLORS = {
    "AOM"        : "#8B0000",    # Dark red — geological standard
    "Palynomorph": "#1A5E20",    # Dark green
    "Phytoclast" : "#1A3A6B",    # Dark blue
    "Background" : "#AAAAAA",
}

# Calibration: measured with micrometric ruler
CALIBRATION = {
    10: {"um_per_pixel": 1.2403},
    20: {"um_per_pixel": 0.6202},
    40: {"um_per_pixel": 0.3101},
}

# ══════════════════════════════════════════════════════════════════════════════
# TYSON (1995) TERNARY — EXACT COORDINATES
# Corners: AOM=bottom left, Palynomorphs=bottom right, Phytoclasts=top
# Reference: Tyson R.V. (1995) Sedimentary Organic Matter, Chapman & Hall
# ══════════════════════════════════════════════════════════════════════════════
TYSON_FIELDS = [
    {   # I: Highly proximal
        "num": "I", "color": "#FFFFFF",
        "name": "Highly proximal\nshelf or basin",
        "pts": [(0,0,100),(20,0,80),(15,15,70),(0,15,85)],
        "label_pos": (8, 5, 87),
    },
    {   # II: Marginal dysoxic-oxic
        "num": "II", "color": "#FFFFFF",
        "name": "Marginal dysoxic-\noxic shelf",
        "pts": [(0,15,85),(15,15,70),(20,20,60),(20,40,40),(0,40,60),(0,20,80)],
        "label_pos": (8, 25, 67),
    },
    {   # III: Heterolithic oxic
        "num": "III", "color": "#FFFFFF",
        "name": "Heterolithic oxic\nshelf (proximal)",
        "pts": [(0,40,60),(20,40,40),(20,60,20),(0,60,40)],
        "label_pos": (8, 52, 40),
    },
    {   # IVa: Shelf to basin transition
        "num": "IVa", "color": "#FFFFFF",
        "name": "Shelf to basin\ntransition",
        "pts": [(15,15,70),(20,0,80),(40,0,60),(35,20,45),(20,20,60)],
        "label_pos": (26, 12, 62),
    },
    {   # IVb: Shelf to basin transition
        "num": "IVb", "color": "#FFFFFF",
        "name": "Shelf to basin\ntransition",
        "pts": [(20,20,60),(35,20,45),(40,20,40),(40,40,20),(20,40,40)],
        "label_pos": (30, 30, 40),
    },
    {   # V: Mud-dominated oxic
        "num": "V", "color": "#FFFFFF",
        "name": "Mud-dominated oxic\nshelf (distal)",
        "pts": [(0,60,40),(20,60,20),(20,80,0),(0,80,20),(0,100,0)],
        "label_pos": (8, 72, 20),
    },
    {   # VI: Proximal suboxic-anoxic
        "num": "VI", "color": "#FFFFFF",
        "name": "Proximal suboxic-\nanoxic shelf",
        "pts": [(20,0,80),(60,0,40),(55,15,30),(35,20,45),(20,20,60),(15,15,70)],
        "label_pos": (35, 8, 57),
    },
    {   # VII: Distal dysoxic-anoxic
        "num": "VII", "color": "#FFFFFF",
        "name": "Distal dysoxic-\nanoxic shelf",
        "pts": [(40,20,40),(55,15,30),(60,0,40),(80,0,20),(60,20,20),(40,40,20),(40,20,40)],
        "label_pos": (58, 18, 24),
    },
    {   # VIII: Distal dysoxic-anoxic basin
        "num": "VIII", "color": "#FFFFFF",
        "name": "Distal dysoxic-\nanoxic basin",
        "pts": [(20,40,40),(40,40,20),(40,60,0),(20,80,0),(20,60,20)],
        "label_pos": (30, 55, 15),
    },
    {   # IX: Distal suboxic-anoxic basin
        "num": "IX", "color": "#FFFFFF",
        "name": "Distal suboxic-\nanoxic basin",
        "pts": [(60,0,40),(100,0,0),(80,20,0),(80,0,20)],
        "label_pos": (72, 8, 20),
    },
]

# Field boundaries
FIELD_BOUNDARIES = [
    [(0,20,80),  (20,20,60), "solid"],   # I/II lower bound
    [(20,0,80),  (20,20,60), "solid"],   # I/IVa right bound
    [(20,20,60), (20,40,40), "solid"],   # II/IVb left bound
    [(20,40,40), (40,40,20), "solid"],   # III-IVb/VIII bound
    [(0,40,60),  (20,40,40), "solid"],   # II/III bound
    [(0,60,40),  (20,60,20), "solid"],   # III/V bound
    [(20,60,20), (40,60,0),  "solid"],   # V/VIII bound
    [(20,0,80),  (60,0,40),  "solid"],   # IVa-VI middle bound
    [(35,20,45), (40,20,40), "solid"],   # IVa/IVb-VII transition
    [(40,20,40), (40,40,20), "solid"],   # VII/IVb+VIII bound
    [(40,40,20), (80,20,0),  "solid"],   # VII/VIII+IX bound
    [(60,0,40),  (80,0,20),  "solid"],   # VI/VII+IX bound
    [(20,20,60), (35,20,45), "dashed"],  # IVa/IVb internal (dashed)
    [(35,20,45), (55,15,30), "dashed"],  # internal transition dashed
]

# ══════════════════════════════════════════════════════════════════════════════
# TAI SCALE
# ══════════════════════════════════════════════════════════════════════════════
TAI_SCALE = [
    (1.0, 1.5,  "#FFFF99", "Immature",            "Pre-oil generation"),
    (1.5, 2.0,  "#FFD700", "Immature–Early Mature","Early oil generation"),
    (2.0, 2.5,  "#FFA500", "Early Mature",         "Peak oil generation"),
    (2.5, 3.0,  "#FF6600", "Mature",               "Late oil / condensate"),
    (3.0, 3.7,  "#CC3300", "Late Mature",          "Wet gas window"),
    (3.7, 4.5,  "#660000", "Overmature",           "Dry gas window"),
    (4.5, 5.0,  "#1A0000", "Overmature+",          "Post-gas (dead carbon)"),
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def tc(aom, pal, phy):
    """Ternary (AOM, Pal, Phy) → Cartesian (x, y)."""
    total = aom + pal + phy
    if total == 0:
        return 0.5, 0.0
    p = pal / total
    h = phy / total
    x = p + h * 0.5
    y = h * (3 ** 0.5) / 2
    return x, y

def get_magnification(images_path: Path, mag_arg):
    if mag_arg:
        return int(mag_arg)
    folder = images_path.name
    for kw, m in [("40x",40),("40X",40),("20x",20),("20X",20),("10x",10),("10X",10)]:
        if kw in folder:
            return m
    return 10

def pixels_to_um2(px: float, mag: int) -> float:
    cal = CALIBRATION.get(mag, CALIBRATION[10])
    return px * (cal["um_per_pixel"] ** 2)

def classify_facies(aom, pal, phy):
    centers = {
        "I":    (5,  5,  90), "II":  (8,  28, 64), "III": (8,  52, 40),
        "IVa":  (27, 10, 63), "IVb": (30, 30, 40), "V":   (8,  73, 19),
        "VI":   (35, 8,  57), "VII": (58, 18, 24), "VIII":(30, 55, 15),
        "IX":   (72, 8,  20),
    }
    best, dmin = "IVb", float("inf")
    for fn, (ca, cp, ch) in centers.items():
        d = ((aom-ca)**2 + (pal-cp)**2 + (phy-ch)**2)**0.5
        if d < dmin:
            dmin, best = d, fn
    name_map = {
        "I":"Highly proximal shelf or basin",
        "II":"Marginal dysoxic-oxic shelf",
        "III":"Heterolithic oxic shelf (proximal)",
        "IVa":"Shelf to basin transition (proximal)",
        "IVb":"Shelf to basin transition (distal)",
        "V":"Mud-dominated oxic shelf (distal)",
        "VI":"Proximal suboxic-anoxic shelf",
        "VII":"Distal dysoxic-anoxic shelf",
        "VIII":"Distal dysoxic-anoxic basin",
        "IX":"Distal suboxic-anoxic basin",
    }
    return best, name_map.get(best, "")

def estimate_tai(image_rgb, mask_np):
    """Estimate TAI via color analysis of pixels under the palynomorph mask."""
    if mask_np.shape != image_rgb.shape[:2]:
        mask_np = cv2.resize(mask_np.astype(np.uint8),
                             (image_rgb.shape[1], image_rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    pixels = image_rgb[mask_np == 1]
    if len(pixels) < 50:
        return None, None
    hsv = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_RGB2HSV)[0]
    hue = np.mean(hsv[:, 0])
    val = np.mean(hsv[:, 2])
    sat = np.mean(hsv[:, 1])

    if val < 50:
        tai, status = 4.2, "Overmature (Dry Gas Window)"
    elif val < 90:
        tai, status = 3.2, "Late Mature (Wet Gas)"
    elif val < 140 and sat > 60:
        tai, status = 2.7, "Mature (Oil Window)"
    elif hue < 25:
        tai, status = 2.2, "Early Mature"
    elif hue < 40:
        tai, status = 1.8, "Immature–Early Mature"
    else:
        tai, status = 1.3, "Immature"
    return tai, status

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def analyze_image(model, img_path: Path, mag: int) -> dict:
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None

    results = model(str(img_path), conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
    counts, areas_px = defaultdict(int), defaultdict(float)
    total_pixels = 0
    tai_scores = []

    for r in results:
        h, w = r.orig_shape
        total_pixels = h * w
        if r.masks is None:
            continue
        for mask_t, cls_id in zip(r.masks.data, r.boxes.cls):
            cls_name = CLASS_NAMES[int(cls_id)]
            mask_np  = mask_t.cpu().numpy()
            counts[cls_name]   += 1
            areas_px[cls_name] += float(mask_np.sum())
            if cls_name == "Palynomorph" and img_rgb is not None:
                tai, status = estimate_tai(img_rgb, mask_np)
                if tai is not None:
                    tai_scores.append(tai)

    total_pal_px    = sum(areas_px.get(g, 0) for g in PALYNO_GROUPS)
    total_pal_count = sum(counts.get(g, 0)   for g in PALYNO_GROUPS)

    row = {
        "image"        : img_path.name,
        "magnification": f"{mag}x",
        "total_pixels" : total_pixels,
        "total_um2"    : pixels_to_um2(total_pixels, mag),
    }
    for g in PALYNO_GROUPS:
        px  = areas_px.get(g, 0.0)
        cnt = counts.get(g, 0)
        row[f"{g}_count"]     = cnt
        row[f"{g}_area_px"]   = px
        row[f"{g}_area_um2"]  = pixels_to_um2(px, mag)
        row[f"{g}_area_pct"]  = (px  / total_pal_px    * 100) if total_pal_px    > 0 else 0.0
        row[f"{g}_count_pct"] = (cnt / total_pal_count * 100) if total_pal_count > 0 else 0.0

    row[f"{BG_CLASS}_count"]    = counts.get(BG_CLASS, 0)
    row[f"{BG_CLASS}_area_px"]  = areas_px.get(BG_CLASS, 0.0)
    row["total_palyno_px"]      = total_pal_px
    row["total_palyno_um2"]     = pixels_to_um2(total_pal_px, mag)
    row["total_palyno_count"]   = total_pal_count

    if tai_scores:
        mean_tai = np.mean(tai_scores)
        row["Estimated_TAI"] = mean_tai
        if   mean_tai >= 3.7: row["Maturity_Status"] = "Overmature (Dry Gas)"
        elif mean_tai >= 3.0: row["Maturity_Status"] = "Late Mature (Wet Gas)"
        elif mean_tai >= 2.5: row["Maturity_Status"] = "Mature (Oil Window)"
        elif mean_tai >= 2.0: row["Maturity_Status"] = "Early Mature"
        elif mean_tai >= 1.5: row["Maturity_Status"] = "Immature–Early Mature"
        else:                 row["Maturity_Status"] = "Immature"
    else:
        row["Estimated_TAI"]   = np.nan
        row["Maturity_Status"] = "No Palynomorph Detected"

    return row

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: TYSON TERNARY
# ══════════════════════════════════════════════════════════════════════════════
def draw_tyson_ternary(ax):
    """Draws the Tyson (1995) ternary diagram."""
    for field in TYSON_FIELDS:
        cart = [tc(a, p, h) for (a, p, h) in field["pts"]]
        poly = Polygon(cart, closed=True, facecolor="#FAFAFA",
                       edgecolor="none", linewidth=0, zorder=1)
        ax.add_patch(poly)

    for (p1, p2, style) in FIELD_BOUNDARIES:
        x1, y1 = tc(*p1)
        x2, y2 = tc(*p2)
        ls = "--" if style == "dashed" else "-"
        ax.plot([x1, x2], [y1, y2], color="black",
                lw=0.9, ls=ls, zorder=3)

    corners = [tc(100,0,0), tc(0,100,0), tc(0,0,100), tc(100,0,0)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    ax.plot(xs, ys, color="black", lw=2.5, zorder=5)

    for frac in [0.2, 0.4, 0.6, 0.8]:
        p = frac * 100; q = 100 - p
        tick_len = 0.015
        x1, y1 = tc(q, 0, p)
        x2, y2 = tc(q-tick_len*100, tick_len*50, p)
        ax.plot([x1, x2], [y1, y2], "k-", lw=1, zorder=5)
        x1, y1 = tc(0, p, q)
        x2, y2 = tc(tick_len*100, p-tick_len*50, q)
        ax.plot([x1, x2], [y1, y2], "k-", lw=1, zorder=5)
        x1, y1 = tc(p, q, 0)
        ax.plot([x1, x1], [y1, y1+tick_len*0.866], "k-", lw=1, zorder=5)

    for field in TYSON_FIELDS:
        la, lp, lh = field["label_pos"]
        lx, ly = tc(la, lp, lh)
        ax.text(lx, ly+0.015, field["num"],
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="black", zorder=6,
                fontfamily="serif")
        short = field["name"].split("\n")[0]
        ax.text(lx, ly-0.025, short,
                ha="center", va="center",
                fontsize=5, color="#444444", zorder=6)

    ax.text(0.5, 0.866 + 0.055, "Phytoclasts %",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="black", zorder=7)
    ax.text(-0.01, -0.055, "AOM %",
            ha="right", va="top",
            fontsize=13, fontweight="bold", color="black", zorder=7)
    ax.text(1.01, -0.055, "Palynomorphs %",
            ha="left", va="top",
            fontsize=13, fontweight="bold", color="black", zorder=7)

    for frac in [0.2, 0.4, 0.6, 0.8]:
        p = int(frac * 100); q = 100 - p
        x, y = tc(q, 0, p)
        ax.text(x - 0.03, y, f"{p}", ha="right", va="center",
                fontsize=8, color="black")
        x, y = tc(0, p, q)
        ax.text(x + 0.03, y, f"{p}", ha="left", va="center",
                fontsize=8, color="black")
        x, y = tc(p, q, 0)
        ax.text(x, y - 0.04, f"{p}", ha="center", va="top",
                fontsize=8, color="black")

    ax.set_xlim(-0.20, 1.30)
    ax.set_ylim(-0.12, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")

def plot_tyson_ternary(df: pd.DataFrame, out_path: Path, mag: int):
    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor("white")
    ax.set_title(
        "Tyson (1995) Ternary Diagram — Palynofacies Classification\n"
        f"n = {len(df)} samples  |  {mag}x magnification  |  "
        f"{datetime.now().strftime('%Y-%m-%d')}",
        fontsize=13, fontweight="bold", pad=18, fontfamily="serif"
    )
    draw_tyson_ternary(ax)
    xs, ys = [], []
    for _, row in df.iterrows():
        x, y = tc(row["AOM_area_pct"], row["Palynomorph_area_pct"], row["Phytoclast_area_pct"])
        xs.append(x); ys.append(y)
    ax.scatter(xs, ys, c="black", s=50, zorder=8, alpha=0.75,
               edgecolors="white", linewidth=0.5, label="Sample", marker="o")
    ma = df["AOM_area_pct"].mean()
    mp = df["Palynomorph_area_pct"].mean()
    mh = df["Phytoclast_area_pct"].mean()
    mx, my = tc(ma, mp, mh)
    fnum, fname = classify_facies(ma, mp, mh)
    ax.scatter([mx], [my], c="red", s=180, zorder=9,
               edgecolors="white", linewidth=1.5, marker="*", label="Mean")
    ax.annotate(
        f"Mean — Field {fnum}\n"
        f"AOM: {ma:.1f}%\nPal: {mp:.1f}%\nPhy: {mh:.1f}%",
        xy=(mx, my), xytext=(mx + 0.13, my + 0.07),
        fontsize=8, color="red", fontweight="bold", fontfamily="serif",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9),
        zorder=10
    )
    handles = [
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="black", markersize=8, label="Sample"),
        plt.Line2D([0],[0], marker="*", color="w",
                   markerfacecolor="red", markersize=13, label=f"Mean (Field {fnum})"),
    ]
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.28, 1.0), fontsize=9,
              framealpha=0.9, edgecolor="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Tyson ternary plot → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: GEOLOGICAL LOG
# ══════════════════════════════════════════════════════════════════════════════
def plot_relative_abundance_log(df: pd.DataFrame, out_path: Path):
    n = len(df)
    depths = np.arange(1, n + 1)
    fig = plt.figure(figsize=(16, max(10, n * 0.35 + 3)))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Palynofacies Relative Abundance — Stratigraphic Log\n"
        "Tyson (1995) Components",
        fontsize=13, fontweight="bold", fontfamily="serif", y=0.98
    )
    gs = gridspec.GridSpec(1, 7, figure=fig, wspace=0.08,
                           left=0.12, right=0.97, top=0.92, bottom=0.06)
    col_configs = [
        ("AOM_area_pct",        "AOM\nArea %",      COLORS["AOM"],         0, 100),
        ("Palynomorph_area_pct","Palynomorph\nArea %", COLORS["Palynomorph"],0, 100),
        ("Phytoclast_area_pct", "Phytoclast\nArea %", COLORS["Phytoclast"], 0, 100),
        ("AOM_count_pct",       "AOM\nCount %",     COLORS["AOM"],         0, 100),
        ("Palynomorph_count_pct","Pal\nCount %",    COLORS["Palynomorph"], 0, 100),
        ("Phytoclast_count_pct","Phy\nCount %",     COLORS["Phytoclast"],  0, 100),
    ]
    axes = []
    for col_i, (col, title, color, xmin, xmax) in enumerate(col_configs):
        ax = fig.add_subplot(gs[0, col_i])
        axes.append(ax)
        vals = df[col].values
        ax.fill_betweenx(depths, 0, vals, color=color, alpha=0.55, linewidth=0)
        ax.plot(vals, depths, color=color, lw=1.2, alpha=0.9)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(n + 0.5, 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_minor_locator(MultipleLocator(12.5))
        ax.grid(True, axis="x", color="lightgray", lw=0.5, alpha=0.7)
        ax.grid(True, axis="x", which="minor", color="#eeeeee", lw=0.3)
        ax.set_title(title, fontsize=9, fontweight="bold",
                     color=color, fontfamily="serif", pad=4)
        ax.tick_params(axis="x", labelsize=7)
        if col_i == 0:
            ax.set_yticks(depths)
            ax.set_yticklabels([f"{i}" for i in depths], fontsize=6.5)
            ax.set_ylabel("Sample No.", fontsize=8, fontfamily="serif")
        else:
            ax.set_yticks(depths)
            ax.set_yticklabels([])
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        mean_val = vals.mean()
        ax.axvline(mean_val, color=color, lw=1.0, ls="--", alpha=0.7)
        ax.text(mean_val + 1, 0.3, f"μ={mean_val:.1f}", fontsize=6, color=color, va="top")

    ax_f = fig.add_subplot(gs[0, 6])
    axes.append(ax_f)
    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(n + 0.5, 0.5)
    ax_f.set_yticks(depths)
    ax_f.set_yticklabels([])
    ax_f.set_title("Facies\nField", fontsize=9, fontweight="bold",
                   fontfamily="serif", pad=4)
    ax_f.xaxis.set_visible(False)
    facies_colors = {
        "I":"#E8F5E9","II":"#E3F2FD","III":"#F1F8E9",
        "IVa":"#FFFDE7","IVb":"#FFF8E1","V":"#E3F2FD",
        "VI":"#FFF3E0","VII":"#FFEBEE","VIII":"#F3E5F5","IX":"#FFCDD2"
    }
    for i, (_, row) in enumerate(df.iterrows(), 1):
        fn, _ = classify_facies(row["AOM_area_pct"],
                                row["Palynomorph_area_pct"],
                                row["Phytoclast_area_pct"])
        fc = facies_colors.get(fn, "#FFFFFF")
        ax_f.barh(i, 1, height=0.85, color=fc, edgecolor="gray", lw=0.3)
        ax_f.text(0.5, i, fn, ha="center", va="center",
                  fontsize=7, fontweight="bold", fontfamily="serif")
    for ax in axes:
        for d in depths:
            ax.axhline(d + 0.5, color="lightgray", lw=0.3, zorder=0)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Relative abundance log → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3: TAI DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def plot_tai_diagram(df: pd.DataFrame, out_path: Path):
    valid = df[df["Estimated_TAI"].notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [2.5, 3, 2]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Thermal Alteration Index (TAI) — Organic Maturity Assessment",
                 fontsize=13, fontweight="bold", fontfamily="serif")
    ax0 = axes[0]
    ax0.set_xlim(0, 1)
    ax0.set_ylim(1.0, 5.0)
    ax0.invert_yaxis()
    ax0.set_title("TAI Reference Scale", fontsize=10, fontweight="bold",
                  fontfamily="serif")
    ax0.set_ylabel("TAI Value", fontsize=9, fontfamily="serif")
    ax0.set_xticks([])
    for (tai_min, tai_max, color, label, desc) in TAI_SCALE:
        ax0.barh((tai_min + tai_max) / 2, 0.8,
                 height=(tai_max - tai_min) * 0.92,
                 color=color, edgecolor="black", lw=0.5, left=0.1)
        ax0.text(0.5, (tai_min + tai_max) / 2,
                 f"{label}\n({desc})",
                 ha="center", va="center", fontsize=7.5,
                 fontfamily="serif", color="black" if tai_max < 3.5 else "white")
        ax0.text(0.08, (tai_min + tai_max) / 2,
                 f"{tai_min:.1f}–{tai_max:.1f}",
                 ha="right", va="center", fontsize=7, color="black")
    ax0.set_yticks([1, 1.5, 2, 2.5, 3, 3.7, 4.5, 5])
    ax1 = axes[1]
    if len(valid) > 0:
        n = len(valid)
        depths = np.arange(1, n + 1)
        tai_vals = valid["Estimated_TAI"].values
        sample_colors = []
        for tv in tai_vals:
            c = "#FFD700"
            for (tmin, tmax, col, _, __) in TAI_SCALE:
                if tmin <= tv < tmax:
                    c = col; break
            sample_colors.append(c)
        ax1.barh(depths, tai_vals, color=sample_colors,
                 edgecolor="black", lw=0.4, height=0.7)
        ax1.set_xlim(1.0, 5.0)
        ax1.set_ylim(n + 0.5, 0.5)
        ax1.set_yticks(depths)
        ax1.set_yticklabels(valid["image"].str[:20].values, fontsize=6.5)
        ax1.set_xlabel("Estimated TAI", fontsize=9, fontfamily="serif")
        ax1.set_title("Sample TAI Values", fontsize=10,
                      fontweight="bold", fontfamily="serif")
        ax1.axvline(tai_vals.mean(), color="red", lw=1.5, ls="--",
                    label=f"Mean TAI = {tai_vals.mean():.2f}")
        ax1.legend(fontsize=8)
        for (tmin, tmax, color, _, __) in TAI_SCALE:
            ax1.axvspan(tmin, tmax, alpha=0.12, color=color)
        ax1.grid(True, axis="x", color="lightgray", lw=0.5)
    else:
        ax1.text(0.5, 0.5, "No TAI data\n(Palynomorphs not detected)",
                 ha="center", va="center", fontsize=11, color="gray",
                 transform=ax1.transAxes)
    ax2 = axes[2]
    if len(valid) > 0:
        tai_vals = valid["Estimated_TAI"].values
        bins = [1.0, 1.5, 2.0, 2.5, 3.0, 3.7, 4.5, 5.0]
        bin_colors = [c for (_, _, c, __, ___) in TAI_SCALE]
        n_hist, _, patches = ax2.hist(tai_vals, bins=bins, edgecolor="black",
                                      lw=0.7, orientation="horizontal")
        for patch, color in zip(patches, bin_colors):
            patch.set_facecolor(color)
        ax2.set_ylim(1.0, 5.0)
        ax2.invert_yaxis()
        ax2.set_xlabel("Count", fontsize=9, fontfamily="serif")
        ax2.set_ylabel("TAI", fontsize=9, fontfamily="serif")
        ax2.set_title("TAI Distribution", fontsize=10,
                      fontweight="bold", fontfamily="serif")
        ax2.grid(True, axis="x", color="lightgray", lw=0.5)
        mean_t = tai_vals.mean()
        mstatus = "Immature"
        for (tmin, tmax, _, status, __) in TAI_SCALE:
            if tmin <= mean_t < tmax:
                mstatus = status; break
        textstr = (f"n = {len(tai_vals)}\n"
                   f"Mean TAI = {mean_t:.2f}\n"
                   f"Min = {tai_vals.min():.2f}\n"
                   f"Max = {tai_vals.max():.2f}\n"
                   f"Status: {mstatus}")
        ax2.text(0.97, 0.03, textstr, transform=ax2.transAxes,
                 fontsize=8, va="bottom", ha="right",
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
                 fontfamily="serif")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                 fontsize=11, color="gray", transform=ax2.transAxes)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  TAI diagram → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4: SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary_charts(df: pd.DataFrame, out_path: Path, mag: int):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Palynofacies Summary Statistics — {mag}x Magnification",
                 fontsize=13, fontweight="bold", fontfamily="serif")
    colors = [COLORS[g] for g in PALYNO_GROUPS]
    means_a = [df[f"{g}_area_pct"].mean() for g in PALYNO_GROUPS]
    stds_a  = [df[f"{g}_area_pct"].std()  for g in PALYNO_GROUPS]
    bars = axes[0,0].bar(PALYNO_GROUPS, means_a, color=colors,
                         edgecolor="white", lw=1.2,
                         yerr=stds_a, capsize=5)
    axes[0,0].set_ylabel("Mean Area %"); axes[0,0].set_ylim(0, 110)
    axes[0,0].set_title("Mean Area % (±SD)", fontfamily="serif")
    for bar, v, s in zip(bars, means_a, stds_a):
        axes[0,0].text(bar.get_x()+bar.get_width()/2, v+s+1,
                       f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    means_u = [df[f"{g}_area_um2"].mean() for g in PALYNO_GROUPS]
    stds_u  = [df[f"{g}_area_um2"].std()  for g in PALYNO_GROUPS]
    bars2 = axes[0,1].bar(PALYNO_GROUPS, means_u, color=colors,
                          edgecolor="white", lw=1.2,
                          yerr=stds_u, capsize=5)
    axes[0,1].set_ylabel("Mean Area (µm²)")
    axes[0,1].set_title("Mean Area µm² (±SD)", fontfamily="serif")
    for bar, v in zip(bars2, means_u):
        axes[0,1].text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()*1.01,
                       f"{v:.0f}", ha="center", fontsize=9)
    axes[0,2].pie(means_a, labels=PALYNO_GROUPS, colors=colors,
                  autopct="%1.1f%%", startangle=90,
                  wedgeprops=dict(edgecolor="white", lw=1.5))
    axes[0,2].set_title("Area Distribution", fontfamily="serif")
    means_c = [df[f"{g}_count"].mean() for g in PALYNO_GROUPS]
    stds_c  = [df[f"{g}_count"].std()  for g in PALYNO_GROUPS]
    bars3 = axes[1,0].bar(PALYNO_GROUPS, means_c, color=colors,
                          edgecolor="white", lw=1.2,
                          yerr=stds_c, capsize=5)
    axes[1,0].set_ylabel("Mean Object Count")
    axes[1,0].set_title("Mean Object Count (±SD)", fontfamily="serif")
    for bar, v in zip(bars3, means_c):
        axes[1,0].text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()*1.01,
                       f"{v:.1f}", ha="center", fontsize=9)
    data_box = [df[f"{g}_area_pct"].values for g in PALYNO_GROUPS]
    bp = axes[1,1].boxplot(data_box, labels=PALYNO_GROUPS,
                           patch_artist=True,
                           medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    axes[1,1].set_ylabel("Area %")
    axes[1,1].set_title("Area % Distribution", fontfamily="serif")
    for g, c in zip(PALYNO_GROUPS, colors):
        axes[1,2].scatter(df[f"{g}_count"], df[f"{g}_area_um2"],
                         color=c, label=g, alpha=0.65, s=45,
                         edgecolors="white", lw=0.5)
    axes[1,2].set_xlabel("Object Count")
    axes[1,2].set_ylabel("Area (µm²)")
    axes[1,2].set_title("Count vs Area µm²", fontfamily="serif")
    axes[1,2].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Summary charts → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVING REPORTS
# ══════════════════════════════════════════════════════════════════════════════
def save_reports(df: pd.DataFrame, out_dir: Path, mag: int):
    cal = CALIBRATION.get(mag, CALIBRATION[10])
    csv_path = out_dir / "palynofacies_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV → {csv_path}")
    xlsx_path = out_dir / "palynofacies_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Per Image", index=False)
        summary = pd.DataFrame({
            "Class"         : PALYNO_GROUPS,
            "Mean Area %"   : [df[f"{g}_area_pct"].mean()  for g in PALYNO_GROUPS],
            "Std Area %"    : [df[f"{g}_area_pct"].std()   for g in PALYNO_GROUPS],
            "Min Area %"    : [df[f"{g}_area_pct"].min()   for g in PALYNO_GROUPS],
            "Max Area %"    : [df[f"{g}_area_pct"].max()   for g in PALYNO_GROUPS],
            "Mean Area µm²" : [df[f"{g}_area_um2"].mean()  for g in PALYNO_GROUPS],
            "Mean Count"    : [df[f"{g}_count"].mean()     for g in PALYNO_GROUPS],
            "Total Count"   : [int(df[f"{g}_count"].sum()) for g in PALYNO_GROUPS],
            "Mean Count %"  : [df[f"{g}_count_pct"].mean() for g in PALYNO_GROUPS],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
        facies_rows = []
        for _, row in df.iterrows():
            fn, fname = classify_facies(row["AOM_area_pct"],
                                        row["Palynomorph_area_pct"],
                                        row["Phytoclast_area_pct"])
            facies_rows.append({"Image": row["image"],
                                 "Facies Field": fn, "Environment": fname})
        pd.DataFrame(facies_rows).to_excel(
            writer, sheet_name="Facies Classification", index=False)
        if "Estimated_TAI" in df.columns and df["Estimated_TAI"].notna().any():
            vt = df["Estimated_TAI"].dropna()
            tai_sum = pd.DataFrame({
                "Metric": ["Mean TAI","Min TAI","Max TAI","Std TAI","n samples"],
                "Value" : [vt.mean(), vt.min(), vt.max(), vt.std(), int(len(vt))]
            })
            tai_sum.to_excel(writer, sheet_name="TAI Summary", index=False)
            df[["image","Estimated_TAI","Maturity_Status"]].to_excel(
                writer, sheet_name="TAI Per Image", index=False)
        pd.DataFrame({
            "Magnification"   : [f"{mag}x"],
            "µm per pixel"    : [cal["um_per_pixel"]],
            "µm² per pixel"   : [cal["um_per_pixel"]**2],
        }).to_excel(writer, sheet_name="Calibration", index=False)
    print(f"  Excel → {xlsx_path}")

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(df: pd.DataFrame, mag: int):
    ma = df["AOM_area_pct"].mean()
    mp = df["Palynomorph_area_pct"].mean()
    mh = df["Phytoclast_area_pct"].mean()
    fnum, fname = classify_facies(ma, mp, mh)
    cal = CALIBRATION.get(mag, CALIBRATION[10])
    print("\n" + "="*70)
    print(f"PALYNOFACIES & TAI ANALYSIS RESULTS — Tyson (1995)")
    print(f"Magnification: {mag}x  |  Calibration: {cal['um_per_pixel']} µm/px")
    print("="*70)
    print(f"Total images: {len(df)}\n")
    print(f"{'Class':15s} {'Area%':>8s} {'±std':>6s} "
          f"{'µm²(avg)':>12s} {'Count(avg)':>11s} {'Count%':>8s}")
    print("-"*70)
    for g in PALYNO_GROUPS:
        print(f"{g:15s} "
              f"{df[f'{g}_area_pct'].mean():8.2f}% "
              f"{df[f'{g}_area_pct'].std():6.2f}  "
              f"{df[f'{g}_area_um2'].mean():12.1f}  "
              f"{df[f'{g}_count'].mean():11.1f}  "
              f"{df[f'{g}_count_pct'].mean():8.2f}%")
    print("="*70)
    print(f"\n🗺️  Dominant Facies: Field {fnum} — {fname}")
    print(f"   AOM={ma:.1f}%  Palynomorph={mp:.1f}%  Phytoclast={mh:.1f}%")
    if "Estimated_TAI" in df.columns and df["Estimated_TAI"].notna().any():
        vt = df["Estimated_TAI"].dropna()
        mt = vt.mean()
        mstatus = "Immature"
        for (tmin, tmax, _, status, __) in TAI_SCALE:
            if tmin <= mt < tmax:
                mstatus = status; break
        print(f"\n🔥 Thermal Maturity (TAI):")
        print(f"   Mean TAI = {mt:.2f}  →  {mstatus}")
        print(f"   Range: {vt.min():.2f} – {vt.max():.2f}  (n={len(vt)})")
    print(f"\nFacies distribution per image:")
    fc = Counter()
    for _, row in df.iterrows():
        fn, _ = classify_facies(row["AOM_area_pct"],
                                row["Palynomorph_area_pct"],
                                row["Phytoclast_area_pct"])
        fc[fn] += 1
    for fn in ["I","II","III","IVa","IVb","V","VI","VII","VIII","IX"]:
        n = fc.get(fn, 0)
        if n > 0:
            print(f"  Field {fn:4s}: {n:3d} images ({n/len(df)*100:.1f}%)")
    print("="*70)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Palynofacies + TAI Full Analyzer — Tyson (1995)")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--images",  required=True, help="Test images folder")
    parser.add_argument("--mag",     type=int, default=None,
                        help="Magnification (10/20/40). Can be auto-detected.")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    images_path  = Path(args.images)
    mag          = get_magnification(images_path, args.mag)
    out_dir      = Path(args.out_dir) if args.out_dir else \
                   weights_path.parent.parent / f"palynofacies_{mag}x"
    out_dir.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg",".jpeg",".png",".bmp",".tiff",".JPG",".JPEG",".PNG"}
    images = sorted(p for p in images_path.iterdir() if p.suffix in exts)
    if not images:
        print(f"❌ No images found: {images_path}"); sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  PalyTOAE — Palynofacies Full Analyzer")
    print(f"  Model     : {weights_path.name}")
    print(f"  Images    : {len(images)}  |  {mag}x")
    print(f"  Output    : {out_dir}")
    print(f"{'='*55}\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics"); sys.exit(1)

    model = YOLO(str(weights_path))
    rows = []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i:3d}/{len(images)}] {img_path.name}", end="\r")
        rows.append(analyze_image(model, img_path, mag))
    print(f"\n✅ Analysis completed ({len(rows)} images)\n")

    df = pd.DataFrame(rows)
    print_summary(df, mag)

    print("\nSaving outputs:")
    save_reports(df, out_dir, mag)
    plot_tyson_ternary(df, out_dir / "tyson_ternary_plot.png", mag)
    plot_relative_abundance_log(df, out_dir / "relative_abundance_log.png")
    plot_tai_diagram(df, out_dir / "tai_diagram.png")
    plot_summary_charts(df, out_dir / "summary_charts.png", mag)

    print(f"\n🎯 All outputs saved to: {out_dir}")
    print("   ├── tyson_ternary_plot.png    (Tyson 1995, 9 fields)")
    print("   ├── relative_abundance_log.png (Geological log)")
    print("   ├── tai_diagram.png           (TAI maturity)")
    print("   ├── summary_charts.png        (Statistics)")
    print("   └── palynofacies_report.xlsx  (6 sheets)")

if __name__ == "__main__":
    main()