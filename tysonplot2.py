import argparse, sys, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # <-- Hata buradaydı, düzeltildi.
from matplotlib.patches import Polygon

warnings.filterwarnings("ignore")

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["AOM", "Background", "Palynomorph", "Phytoclast"]
PALYNO_GROUPS = ["AOM", "Palynomorph", "Phytoclast"]
BG_CLASS      = "Background"
CONF_THRESH   = 0.25
IOU_THRESH    = 0.45

COLORS = {
    "AOM"        : "#C0392B",
    "Palynomorph": "#27AE60",
    "Phytoclast" : "#2980B9",
}
MEAN_COLOR = "#E91E63"

# ─── TYSON (1995) FASİES ALANLARI ─────────────────────────────────────────────
TYSON_FIELDS = [
    {"num": "I",    "name": "Highly proximal shelf or basin",       "color": "#C8E6C9", "pts": [(0,0,100),(10,0,90),(0,10,90)]},
    {"num": "II",   "name": "Marginal dysoxic-oxic shelf",          "color": "#BBDEFB", "pts": [(10,0,90),(40,0,60),(40,10,50),(0,50,50),(0,10,90)]},
    {"num": "III",  "name": "Heterolithic oxic shelf (proximal)",   "color": "#C8E6C9", "pts": [(0,50,50),(30,50,20),(0,80,20)]},
    {"num": "IVa",  "name": "Shelf to basin transition (a)",        "color": "#FFF9C4", "pts": [(40,10,50),(0,50,50),(10,50,40),(40,20,40)]},
    {"num": "IVb",  "name": "Shelf to basin transition (b)",        "color": "#FFE0B2", "pts": [(40,20,40),(10,50,40),(30,50,20),(40,40,20)]},
    {"num": "V",    "name": "Mud-dominated oxic shelf (distal)",    "color": "#BBDEFB", "pts": [(30,50,20),(50,50,0),(0,100,0),(0,80,20)]},
    {"num": "VI",   "name": "Proximal suboxic-anoxic shelf",        "color": "#FFE0B2", "pts": [(40,0,60),(40,10,50),(40,20,40),(60,0,40)]},
    {"num": "VII",  "name": "Distal dysoxic-anoxic shelf",          "color": "#FFCDD2", "pts": [(40,40,20),(60,40,0),(50,50,0),(30,50,20)]},
    {"num": "VIII", "name": "Distal dysoxic-anoxic basin",          "color": "#E1BEE7", "pts": [(60,0,40),(80,0,20),(80,20,0),(60,40,0),(40,40,20),(40,20,40)]},
    {"num": "IX",   "name": "Distal suboxic-anoxic basin",          "color": "#FFCDD2", "pts": [(80,0,20),(80,20,0),(100,0,0)]},
]

# ─── KÖŞELERİN GÖRSEL PİKSEL KOORDİNATLARI ───────────────────────────────────
_PX = {"aom": (76, 530), "pal": (660, 530), "phy": (368, 105)}

def tc(aom, pal, phy):
    """(AOM%, PAL%, PHY%) → görsel piksel koordinatı (px, py)."""
    total = aom + pal + phy
    if total == 0:
        return 368.0, 317.0
    a, p, h = aom / total, pal / total, phy / total
    px = a * _PX["aom"][0] + p * _PX["pal"][0] + h * _PX["phy"][0]
    py = a * _PX["aom"][1] + p * _PX["pal"][1] + h * _PX["phy"][1]
    return px, py


# ─── MODEL FONKSİYONLARI ──────────────────────────────────────────────────────
def analyze_image(model, img_path, class_names):
    results = model(str(img_path), conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
    counts, areas = defaultdict(int), defaultdict(float)
    total_pixels = 0
    for r in results:
        h, w = r.orig_shape
        total_pixels = h * w
        if r.masks is None:
            continue
        for mask, cls_id in zip(r.masks.data, r.boxes.cls):
            cls_name = class_names[int(cls_id)]
            counts[cls_name] += 1
            areas[cls_name]  += float(mask.cpu().numpy().sum())
    row = {"image": img_path.name, "total_pixels": total_pixels}
    total_palyno_area  = sum(areas.get(g, 0) for g in PALYNO_GROUPS)
    total_palyno_count = sum(counts.get(g, 0) for g in PALYNO_GROUPS)
    for g in PALYNO_GROUPS:
        row[f"{g}_count"] = counts.get(g, 0)
        row[f"{g}_area"]  = areas.get(g, 0.0)
        a = areas.get(g, 0.0)
        c = counts.get(g, 0)
        row[f"{g}_area_pct"]  = (a / total_palyno_area  * 100) if total_palyno_area  > 0 else 0.0
        row[f"{g}_count_pct"] = (c / total_palyno_count * 100) if total_palyno_count > 0 else 0.0
    row[f"{BG_CLASS}_count"] = counts.get(BG_CLASS, 0)
    row[f"{BG_CLASS}_area"]  = areas.get(BG_CLASS, 0.0)
    row["total_palyno_area"]  = total_palyno_area
    row["total_palyno_count"] = total_palyno_count
    return row


def classify_facies(aom_pct, pal_pct, phy_pct):
    best_field, best_name, min_dist = "IVa", "Shelf to basin transition (a)", float("inf")
    for field in TYSON_FIELDS:
        pts = field["pts"]
        c_aom = sum(p[0] for p in pts) / len(pts)
        c_pal = sum(p[1] for p in pts) / len(pts)
        c_phy = sum(p[2] for p in pts) / len(pts)
        dist  = ((aom_pct - c_aom)**2 + (pal_pct - c_pal)**2 + (phy_pct - c_phy)**2)**0.5
        if dist < min_dist:
            min_dist, best_field, best_name = dist, field["num"], field["name"]
    return best_field, best_name


# ─── ANA TYSON TERNARY PLOT ───────────────────────────────────────────────────
def plot_tyson_ternary(df, out_path, ref_img_str):
    IMG_W, IMG_H = 736, 610

    fig = plt.figure(figsize=(20, 11))

    # ── Sol: görsel + üçgen  ──────────────────────────────────────────────────
    ax = fig.add_axes([0.02, 0.05, 0.56, 0.88])
    ax.set_xlim(-60, IMG_W + 60)
    ax.set_ylim(IMG_H + 60, -80)   # y ters
    ax.set_aspect("equal")
    ax.axis("off")

    # Referans görsel
    ref_img_path = Path(ref_img_str)
    if ref_img_path.exists():
        try:
            img = plt.imread(str(ref_img_path))
            ax.imshow(img, extent=[0, IMG_W - 1, IMG_H - 1, 0],
                      aspect="auto", zorder=1, alpha=1.0)
        except Exception as e:
            print(f"  ⚠️ Referans görsel yüklenemedi: {e}")
    else:
        print(f"  ⚠️ '{ref_img_str}' bulunamadı.")

    # Numune noktaları
    xs = [tc(row["AOM_area_pct"], row["Palynomorph_area_pct"], row["Phytoclast_area_pct"])[0]
          for _, row in df.iterrows()]
    ys = [tc(row["AOM_area_pct"], row["Palynomorph_area_pct"], row["Phytoclast_area_pct"])[1]
          for _, row in df.iterrows()]
    ax.scatter(xs, ys, c="#1A1A2E", s=80, zorder=6, alpha=0.92,
               edgecolors="white", linewidth=1.0)

    # Ortalama noktası
    ma = df["AOM_area_pct"].mean()
    mp = df["Palynomorph_area_pct"].mean()
    mh = df["Phytoclast_area_pct"].mean()
    mx, my = tc(ma, mp, mh)
    ax.scatter([mx], [my], c=MEAN_COLOR, s=300, zorder=7,
               edgecolors="white", linewidth=1.5, marker="*")

    # Köşe etiketleri
    pad = 22
    ax.text(_PX["aom"][0] - pad, _PX["aom"][1] + pad,
            "AOM\n(Amorphous\nOrganic Matter)",
            ha="right", va="top", fontsize=10, fontweight="bold",
            color=COLORS["AOM"], zorder=8)
    ax.text(_PX["pal"][0] + pad, _PX["pal"][1] + pad,
            "Palynomorphs\n(Spores/Pollen/\nMicroplankton)",
            ha="left", va="top", fontsize=10, fontweight="bold",
            color=COLORS["Palynomorph"], zorder=8)
    ax.text(_PX["phy"][0], _PX["phy"][1] - pad,
            "Phytoclasts\n(Wood/Plant Debris)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=COLORS["Phytoclast"], zorder=8)

    # Başlık
    fig.text(0.30, 0.995,
             "Tyson (1995) Ternary Plot — Palynofacies Analysis",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.30, 0.958,
             f"n = {len(df)} samples  |  {datetime.now().strftime('%Y-%m-%d')}",
             ha="center", va="top", fontsize=11, color="#555555")

    # ── Sağ: Lejant  ─────────────────────────────────────────────────────────
    ax_leg = fig.add_axes([0.60, 0.05, 0.38, 0.88])
    ax_leg.axis("off")

    y_cursor = 0.98

    def leg_title(text, y):
        ax_leg.text(0.0, y, text,
                    transform=ax_leg.transAxes,
                    fontsize=11, fontweight="bold", color="#1A1A2E",
                    va="top")
        return y - 0.045

    def leg_item(color, label, y, marker=None):
        if marker == "*":
            ax_leg.scatter([0.018], [y - 0.012], s=220, c=color, marker="*",
                           transform=ax_leg.transAxes, zorder=5,
                           edgecolors="white", linewidth=0.8, clip_on=False)
        elif marker == "o":
            ax_leg.scatter([0.018], [y - 0.012], s=80, c=color, marker="o",
                           transform=ax_leg.transAxes, zorder=5,
                           edgecolors="white", linewidth=0.8, clip_on=False)
        else:
            rect = mpatches.FancyBboxPatch(
                (0.0, y - 0.026), 0.038, 0.028,
                boxstyle="round,pad=0.002",
                facecolor=color, edgecolor="#999", linewidth=0.7,
                transform=ax_leg.transAxes, clip_on=False)
            ax_leg.add_patch(rect)
        ax_leg.text(0.052, y - 0.009, label,
                    transform=ax_leg.transAxes,
                    fontsize=9.2, va="center", color="#222222")
        return y - 0.042

    # — Tyson Alanları —
    y_cursor = leg_title("Tyson (1995) Facies Fields", y_cursor)
    ax_leg.plot([0, 1], [y_cursor + 0.008, y_cursor + 0.008],
               color="#CCCCCC", linewidth=0.8,
               transform=ax_leg.transAxes)
    y_cursor -= 0.008
    for field in TYSON_FIELDS:
        label = f"  {field['num']:>4s}  {field['name']}"
        y_cursor = leg_item(field["color"], label, y_cursor)

    y_cursor -= 0.018
    ax_leg.plot([0, 1], [y_cursor + 0.008, y_cursor + 0.008],
               color="#CCCCCC", linewidth=0.8,
               transform=ax_leg.transAxes)

    # — İstatistik & Semboller —
    y_cursor = leg_title("Statistics & Symbols", y_cursor)
    y_cursor = leg_item(COLORS["AOM"],          f"AOM (mean {ma:.1f}%)",          y_cursor)
    y_cursor = leg_item(COLORS["Palynomorph"],  f"Palynomorph (mean {mp:.1f}%)",  y_cursor)
    y_cursor = leg_item(COLORS["Phytoclast"],   f"Phytoclast (mean {mh:.1f}%)",   y_cursor)
    y_cursor = leg_item("#1A1A2E", "Sample Point", y_cursor, marker="o")
    y_cursor = leg_item(MEAN_COLOR, "Mean Data Point", y_cursor, marker="*")

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Tyson ternary plot kaydedildi → {out_path}")


def plot_summary_charts(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Palynofacies Summary", fontsize=14, fontweight="bold")
    means_area  = [df[f"{g}_area_pct"].mean()  for g in PALYNO_GROUPS]
    means_count = [df[f"{g}_count_pct"].mean() for g in PALYNO_GROUPS]
    colors      = [COLORS[g] for g in PALYNO_GROUPS]
    for ax, vals, title, ylabel in [
        (axes[0], means_area,  "Mean Area %",  "Mean Area %"),
        (axes[1], means_count, "Mean Count %", "Mean Count %"),
    ]:
        bars = ax.bar(PALYNO_GROUPS, vals, color=colors, edgecolor="white", lw=1.2)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, 105)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[2].pie(means_area, labels=PALYNO_GROUPS, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[2].set_title("Overall Area Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_charts.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_reports(df, out_dir):
    df.to_csv(out_dir / "palynofacies_results.csv", index=False, float_format="%.3f")
    with pd.ExcelWriter(out_dir / "palynofacies_report.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Per Image", index=False)
        summary = pd.DataFrame({
            "Class"       : PALYNO_GROUPS,
            "Mean Area %" : [df[f"{g}_area_pct"].mean()  for g in PALYNO_GROUPS],
            "Std Area %"  : [df[f"{g}_area_pct"].std()   for g in PALYNO_GROUPS],
            "Mean Count %": [df[f"{g}_count_pct"].mean() for g in PALYNO_GROUPS],
            "Total Count" : [int(df[f"{g}_count"].sum()) for g in PALYNO_GROUPS],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)


def print_summary(df):
    ma = df["AOM_area_pct"].mean()
    mp = df["Palynomorph_area_pct"].mean()
    mh = df["Phytoclast_area_pct"].mean()
    fnum, fname = classify_facies(ma, mp, mh)
    print("\n" + "=" * 65)
    print(f"Baskın Tyson Fasiyes Alanı : Field {fnum} ── {fname}")
    print(f"Ortalama Oranlar: AOM={ma:.1f}% | Palynomorph={mp:.1f}% | Phytoclast={mh:.1f}%")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Palynofacies Analyzer")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images",  required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--ref_img", default="Screenshot 2026-06-06 222542.jpg",
                        help="Arka plana bindirilecek Tyson diyagram görseli")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    images_path  = Path(args.images)
    out_dir = (Path(args.out_dir) if args.out_dir
               else weights_path.parent.parent / "palynofacies")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_path.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        print(f"❌ Görüntü bulunamadı: {images_path}")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    rows = []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}")
        rows.append(analyze_image(model, img_path, CLASS_NAMES))

    df = pd.DataFrame(rows)
    print_summary(df)
    save_reports(df, out_dir)
    plot_summary_charts(df, out_dir)
    plot_tyson_ternary(df, out_dir / "tyson_ternary_plot.png", args.ref_img)


if __name__ == "__main__":
    main()