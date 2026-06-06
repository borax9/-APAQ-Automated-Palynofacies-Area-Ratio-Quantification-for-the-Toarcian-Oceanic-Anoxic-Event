"""
5_palynofacies.py
=================
YOLO26 segmentation modeliyle otomatik Palynofacies analizi.
Tyson (1995) Ternary Plot — 9 fasies alanı (I-IX).

Kullanım:
  python 5_palynofacies.py --weights best.pt --images klasor/

Kurulum:
  pip install ultralytics matplotlib pandas openpyxl
"""

import argparse, sys, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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

# ─── TYSON (1995) 9 FASİES ALANI ─────────────────────────────────────────────
# Köşe sırası: AOM (sol), Palynomorph (sağ), Phytoclast (üst)
# Koordinatlar (AOM%, PAL%, PHY%) — toplam 100
# Kaynak: Tyson R.V. 1995, Sedimentary Organic Matter

TYSON_FIELDS = [
    {
        "num"  : "I",
        "name" : "Highly proximal shelf or basin",
        "color": "#D5E8D4",   # açık yeşil
        # Phytoclast dominant >80%, AOM ve Pal az
        "pts"  : [(0,0,100),(20,0,80),(20,20,60),(0,20,80)],
    },
    {
        "num"  : "II",
        "name" : "Marginal dysoxic-oxic basin",
        "color": "#DAE8FC",   # açık mavi
        # Phytoclast >60%, Palynomorph orta
        "pts"  : [(0,20,80),(20,20,60),(20,40,40),(0,40,60)],
    },
    {
        "num"  : "III",
        "name" : "Heterolithic oxic shelf (proximal)",
        "color": "#D5E8D4",
        # Phytoclast 40-60%, Pal orta
        "pts"  : [(0,40,60),(20,40,40),(20,60,20),(0,60,40)],
    },
    {
        "num"  : "IV",
        "name" : "Shelf to basin transition",
        "color": "#FFF2CC",   # sarı
        # Orta alan — karma
        "pts"  : [(20,0,80),(40,0,60),(40,20,40),(20,20,60)],
    },
    {
        "num"  : "V",
        "name" : "Mud-dominated oxic shelf (distal)",
        "color": "#DAE8FC",
        # Palynomorph dominant
        "pts"  : [(0,60,40),(20,60,20),(20,80,0),(0,80,20)],
    },
    {
        "num"  : "VI",
        "name" : "Proximal suboxic-anoxic shelf",
        "color": "#FFE6CC",   # turuncu
        # AOM orta, Phytoclast orta
        "pts"  : [(20,0,80),(60,0,40),(60,20,20),(40,20,40),(40,0,60)],
    },
    {
        "num"  : "VII",
        "name" : "Distal dysoxic-anoxic shelf",
        "color": "#F8CECC",   # açık kırmızı
        # AOM yüksek >60%
        "pts"  : [(40,20,40),(60,20,20),(60,40,0),(40,40,20)],
    },
    {
        "num"  : "VIII",
        "name" : "Distal dysoxic-anoxic shelf",
        "color": "#E1D5E7",   # açık mor
        # AOM >60%, Pal orta
        "pts"  : [(20,60,20),(40,60,0),(40,40,20),(20,40,40)],  # düzeltildi
    },
    {
        "num"  : "IX",
        "name" : "Distal suboxic-anoxic basin",
        "color": "#F8CECC",
        # AOM >80%
        "pts"  : [(60,0,40),(100,0,0),(60,40,0),(60,20,20)],
    },
]
# ──────────────────────────────────────────────────────────────────────────────


def tc(aom, pal, phy):
    """Ternary → Kartezyen (AOM=sol alt, Pal=sağ alt, Phy=üst)"""
    t = aom + pal + phy
    if t == 0: return 0.5, 0.333
    a, p, h = aom/t, pal/t, phy/t
    x = p + h * 0.5
    y = h * (3**0.5) / 2
    return x, y


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
        row[f"{g}_area_pct"]  = (a/total_palyno_area*100)  if total_palyno_area  > 0 else 0.0
        row[f"{g}_count_pct"] = (c/total_palyno_count*100) if total_palyno_count > 0 else 0.0

    row[f"{BG_CLASS}_count"] = counts.get(BG_CLASS, 0)
    row[f"{BG_CLASS}_area"]  = areas.get(BG_CLASS, 0.0)
    row["total_palyno_area"]  = total_palyno_area
    row["total_palyno_count"] = total_palyno_count
    return row


def classify_facies(aom_pct, pal_pct, phy_pct):
    """Hangi Tyson fasies alanına düştüğünü tahmin et."""
    if phy_pct >= 80:
        return "I",   "Highly proximal shelf or basin"
    elif phy_pct >= 60 and pal_pct <= 20:
        return "II",  "Marginal dysoxic-oxic basin"
    elif phy_pct >= 40 and pal_pct >= 20:
        return "III", "Heterolithic oxic shelf (proximal)"
    elif phy_pct >= 60 and aom_pct >= 20:
        return "IV",  "Shelf to basin transition"
    elif pal_pct >= 60:
        return "V",   "Mud-dominated oxic shelf (distal)"
    elif aom_pct >= 40 and phy_pct >= 20:
        return "VI",  "Proximal suboxic-anoxic shelf"
    elif aom_pct >= 60 and phy_pct >= 20:
        return "VII", "Distal dysoxic-anoxic shelf"
    elif aom_pct >= 40 and pal_pct >= 40:
        return "VIII","Distal dysoxic-anoxic shelf"
    elif aom_pct >= 60:
        return "IX",  "Distal suboxic-anoxic basin"
    else:
        return "IV",  "Shelf to basin transition (mixed)"


def draw_tyson_ternary(ax):
    """Tyson 9 fasies alanlı ternary üçgen çiz."""

    # ── Fasies alanlarını doldur ──────────────────────────────────
    for field in TYSON_FIELDS:
        cart_pts = [tc(a,p,h) for (a,p,h) in field["pts"]]
        poly = Polygon(cart_pts, closed=True,
                       facecolor=field["color"], edgecolor="#888888",
                       linewidth=0.6, alpha=0.85, zorder=1)
        ax.add_patch(poly)

        # Alan numarası + kısa çevresel not
        cx = np.mean([pt[0] for pt in cart_pts])
        cy = np.mean([pt[1] for pt in cart_pts])
        ax.text(cx, cy + 0.012, field["num"],
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="#333333", zorder=4)
        ax.text(cx, cy - 0.025,
                field["name"],
                ha="center", va="center",
                fontsize=5.5, color="#555555",
                wrap=True, zorder=4,
                multialignment="center")

    # ── Üçgen çerçeve ────────────────────────────────────────────
    corners = [tc(100,0,0), tc(0,100,0), tc(0,0,100)]
    tri = Polygon(corners, closed=True,
                  fill=False, edgecolor="black", linewidth=2, zorder=5)
    ax.add_patch(tri)

    # ── Izgara (%20 aralıklı) ────────────────────────────────────
    for frac in [0.2, 0.4, 0.6, 0.8]:
        p = frac * 100
        q = 100 - p
        # AOM ekseni
        ax.plot(*zip(tc(p,q,0), tc(p,0,q)),
                color="gray", lw=0.4, ls="--", alpha=0.5, zorder=2)
        # Pal ekseni
        ax.plot(*zip(tc(0,p,q), tc(q,p,0)),
                color="gray", lw=0.4, ls="--", alpha=0.5, zorder=2)
        # Phy ekseni
        ax.plot(*zip(tc(q,0,p), tc(0,q,p)),
                color="gray", lw=0.4, ls="--", alpha=0.5, zorder=2)

    # ── Köşe etiketleri ──────────────────────────────────────────
    ax.text(tc(100,0,0)[0]-0.04, tc(100,0,0)[1]-0.04,
            "AOM", ha="right", va="top",
            fontsize=14, fontweight="bold", color=COLORS["AOM"], zorder=6)
    ax.text(tc(0,100,0)[0]+0.04, tc(0,100,0)[1]-0.04,
            "Palynomorphs", ha="left", va="top",
            fontsize=14, fontweight="bold", color=COLORS["Palynomorph"], zorder=6)
    ax.text(tc(0,0,100)[0], tc(0,0,100)[1]+0.04,
            "Phytoclasts", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=COLORS["Phytoclast"], zorder=6)

    # ── Eksen % etiketleri ───────────────────────────────────────
    for frac in [0.2, 0.4, 0.6, 0.8]:
        p = int(frac * 100)
        q = 100 - p
        # AOM sol kenar
        x, y = tc(p, q, 0)
        ax.text(x-0.025, y, f"{p}%", ha="right", va="center",
                fontsize=7, color="#666666")
        # Pal sağ kenar
        x, y = tc(0, p, q)
        ax.text(x+0.025, y, f"{p}%", ha="left", va="center",
                fontsize=7, color="#666666")
        # Phy alt kenar
        x, y = tc(q, p, 0)
        ax.text(x, y-0.03, f"{p}%", ha="center", va="top",
                fontsize=7, color="#666666")

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.12, 1.02)
    ax.axis("off")


def plot_tyson_ternary(df, out_path):
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_aspect("equal")

    ax.set_title(
        "Tyson (1995) Ternary Plot — Palynofacies Analysis\n"
        f"n = {len(df)} samples  |  {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=14, fontweight="bold", pad=16
    )

    draw_tyson_ternary(ax)

    # ── Örnekleri çiz ────────────────────────────────────────────
    xs, ys = [], []
    for _, row in df.iterrows():
        aom = row["AOM_area_pct"] / 100
        pal = row["Palynomorph_area_pct"] / 100
        phy = row["Phytoclast_area_pct"] / 100
        x, y = tc(aom*100, pal*100, phy*100)
        xs.append(x)
        ys.append(y)

    ax.scatter(xs, ys, c="#2C3E50", s=55, zorder=7, alpha=0.8,
               edgecolors="white", linewidth=0.6, label="Sample (area %)")

    # ── Ortalama noktası ─────────────────────────────────────────
    ma = df["AOM_area_pct"].mean()
    mp = df["Palynomorph_area_pct"].mean()
    mh = df["Phytoclast_area_pct"].mean()
    mx, my = tc(ma, mp, mh)
    ax.scatter([mx], [my], c="red", s=200, zorder=8,
               edgecolors="white", linewidth=1.5,
               marker="*", label="Mean")

    fasies_num, fasies_name = classify_facies(ma, mp, mh)
    ax.annotate(
        f"Mean  Field {fasies_num}\n"
        f"AOM:{ma:.1f}%  Pal:{mp:.1f}%  Phy:{mh:.1f}%",
        xy=(mx, my), xytext=(mx+0.10, my+0.06),
        fontsize=8, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.85),
        zorder=9
    )

    # ── Fasies alanı legend ───────────────────────────────────────
    field_patches = [
        mpatches.Patch(facecolor=f["color"], edgecolor="#888",
                       label=f"Field {f['num']}: {f['name']}")
        for f in TYSON_FIELDS
    ]
    corner_patches = [
        mpatches.Patch(color=COLORS["AOM"],         label=f"AOM (mean {ma:.1f}%)"),
        mpatches.Patch(color=COLORS["Palynomorph"], label=f"Palynomorph (mean {mp:.1f}%)"),
        mpatches.Patch(color=COLORS["Phytoclast"],  label=f"Phytoclast (mean {mh:.1f}%)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#2C3E50",
                   markersize=8, label="Sample"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="red",
                   markersize=12, label="Mean"),
    ]

    leg1 = ax.legend(handles=field_patches, loc="upper left",
                     bbox_to_anchor=(-0.02, 1.0),
                     fontsize=7, framealpha=0.9,
                     title="Tyson (1995) Palynofacies Fields",
                     title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=corner_patches, loc="lower left",
              bbox_to_anchor=(-0.02, 0.0),
              fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Tyson ternary plot → {out_path}")


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
            ax.text(bar.get_x()+bar.get_width()/2, v+1,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    wedges, texts, autotexts = axes[2].pie(
        means_area, labels=PALYNO_GROUPS, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    axes[2].set_title("Overall Area Distribution")

    plt.tight_layout()
    p = out_dir / "summary_charts.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Summary charts → {p}")


def save_reports(df, out_dir):
    csv_path = out_dir / "palynofacies_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"  CSV → {csv_path}")

    xlsx_path = out_dir / "palynofacies_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Per Image", index=False)

        summary = pd.DataFrame({
            "Class":         PALYNO_GROUPS,
            "Mean Area %":   [df[f"{g}_area_pct"].mean()  for g in PALYNO_GROUPS],
            "Std Area %":    [df[f"{g}_area_pct"].std()   for g in PALYNO_GROUPS],
            "Min Area %":    [df[f"{g}_area_pct"].min()   for g in PALYNO_GROUPS],
            "Max Area %":    [df[f"{g}_area_pct"].max()   for g in PALYNO_GROUPS],
            "Mean Count %":  [df[f"{g}_count_pct"].mean() for g in PALYNO_GROUPS],
            "Total Count":   [int(df[f"{g}_count"].sum()) for g in PALYNO_GROUPS],
            "Total Area px": [df[f"{g}_area"].sum()       for g in PALYNO_GROUPS],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"  Excel → {xlsx_path}")


def print_summary(df):
    ma = df["AOM_area_pct"].mean()
    mp = df["Palynomorph_area_pct"].mean()
    mh = df["Phytoclast_area_pct"].mean()
    fnum, fname = classify_facies(ma, mp, mh)

    print("\n" + "="*65)
    print("PALYNOFACIES ANALİZ SONUÇLARI — Tyson (1995)")
    print("="*65)
    print(f"Toplam görüntü : {len(df)}")
    print(f"\n{'Sınıf':15s} {'Ort.Alan%':>10s} {'Std':>7s} "
          f"{'Ort.Count%':>11s} {'Toplam Obje':>12s}")
    print("-"*65)
    for g in PALYNO_GROUPS:
        print(f"{g:15s} "
              f"{df[f'{g}_area_pct'].mean():10.2f}% "
              f"{df[f'{g}_area_pct'].std():7.2f} "
              f"{df[f'{g}_count_pct'].mean():11.2f}% "
              f"{int(df[f'{g}_count'].sum()):12d}")
    print("="*65)
    print(f"\n🗺️  Baskın Tyson Fasies Alanı : Field {fnum}")
    print(f"   {fname}")
    print(f"   AOM={ma:.1f}%  Palynomorph={mp:.1f}%  Phytoclast={mh:.1f}%")

    # Tüm fasies dağılımı
    print(f"\nGörüntü başına fasies dağılımı:")
    facies_counts = defaultdict(int)
    for _, row in df.iterrows():
        fn, _ = classify_facies(row["AOM_area_pct"],
                                row["Palynomorph_area_pct"],
                                row["Phytoclast_area_pct"])
        facies_counts[fn] += 1
    for fn in ["I","II","III","IV","V","VI","VII","VIII","IX"]:
        n = facies_counts.get(fn, 0)
        if n > 0:
            name = next(f["name"] for f in TYSON_FIELDS if f["num"] == fn)
            print(f"  Field {fn:4s}: {n:3d} görüntü  ({n/len(df)*100:.1f}%)  — {name}")


def main():
    parser = argparse.ArgumentParser(description="Palynofacies Analyzer — Tyson 1995")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images",  required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--conf",    type=float, default=CONF_THRESH)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    images_path  = Path(args.images)
    out_dir = Path(args.out_dir) if args.out_dir else \
              weights_path.parent.parent / "palynofacies"
    out_dir.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg",".jpeg",".png",".bmp",".tiff"}
    images = sorted(p for p in images_path.iterdir()
                    if p.suffix.lower() in exts)
    if not images:
        print(f"❌ Görüntü bulunamadı: {images_path}"); sys.exit(1)

    print(f"✅ {len(images)} görüntü  |  Model: {weights_path.name}")
    print(f"   Çıktı: {out_dir}\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics"); sys.exit(1)

    model = YOLO(str(weights_path))

    rows = []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i:3d}/{len(images)}] {img_path.name}", end="\r")
        rows.append(analyze_image(model, img_path, CLASS_NAMES))
    print(f"\n✅ Analiz tamamlandı\n")

    df = pd.DataFrame(rows)
    print_summary(df)

    print("\nÇıktılar kaydediliyor:")
    save_reports(df, out_dir)
    plot_summary_charts(df, out_dir)
    plot_tyson_ternary(df, out_dir / "tyson_ternary_plot.png")

    print(f"\n🎯 Tüm çıktılar: {out_dir}")
    print("   ├── palynofacies_results.csv")
    print("   ├── palynofacies_report.xlsx")
    print("   ├── summary_charts.png")
    print("   └── tyson_ternary_plot.png")


if __name__ == "__main__":
    main()
