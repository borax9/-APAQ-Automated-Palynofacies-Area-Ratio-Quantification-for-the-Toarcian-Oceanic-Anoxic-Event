import argparse, sys, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 
import matplotlib.patches as mpatches
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

# ─── TYSON (1995) 9 FASİES ALANI (DÜZELTİLMİŞ) ────────────────────────────────
TYSON_FIELDS = [
    {
        "num"  : "I",
        "name" : "Highly proximal shelf or basin",
        "color": "#D5E8D4",
        "pts"  : [(0,0,100),(20,0,80),(20,20,60),(0,20,80)],
    },
    {
        "num"  : "II",
        "name" : "Marginal dysoxic-oxic shelf",  # Basin ifadesi Shelf olarak düzeltildi
        "color": "#DAE8FC",
        "pts"  : [(0,20,80),(20,20,60),(20,40,40),(0,40,60)],
    },
    {
        "num"  : "III",
        "name" : "Heterolithic oxic shelf (proximal)",
        "color": "#D5E8D4",
        "pts"  : [(0,40,60),(20,40,40),(20,60,20),(0,60,40)],
    },
    {
        "num"  : "IV",
        "name" : "Shelf to basin transition",
        "color": "#FFF2CC",
        "pts"  : [(20,0,80),(40,0,60),(40,20,40),(20,20,60)],
    },
    {
        "num"  : "V",
        "name" : "Mud-dominated oxic shelf (distal)",
        "color": "#DAE8FC",
        "pts"  : [(0,60,40),(20,60,20),(20,80,0),(0,80,20)],
    },
    {
        "num"  : "VI",
        "name" : "Proximal suboxic-anoxic shelf",
        "color": "#FFE6CC",
        "pts"  : [(20,0,80),(60,0,40),(60,20,20),(40,20,40),(40,0,60)],
    },
    {
        "num"  : "VII",
        "name" : "Distal dysoxic-anoxic shelf",
        "color": "#F8CECC",
        "pts"  : [(40,20,40),(60,20,20),(60,40,0),(40,40,20)],
    },
    {
        "num"  : "VIII",
        "name" : "Distal dysoxic-anoxic basin",  # VII ile karışmaması için Basin olarak düzeltildi
        "color": "#E1D5E7",
        "pts"  : [(20,60,20),(40,60,0),(40,40,20),(20,40,40)],  
    },
    {
        "num"  : "IX",
        "name" : "Distal suboxic-anoxic basin",
        "color": "#F8CECC",
        "pts"  : [(60,0,40),(100,0,0),(60,40,0),(60,20,20)],
    },
]

def tc(aom, pal, phy):
    """Ternary → Kartezyen Dönüşümü (AOM=Sol Alt, Pal=Sağ Alt, Phy=Üst Apex)"""
    total = aom + pal + phy
    if total == 0: 
        return 0.5, 0.333
    a = aom / total
    p = pal / total
    h = phy / total
    
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
    field_centers = {
        "I":    (10, 10, 80), "II":   (10, 30, 60), "III":  (10, 50, 40),
        "IV":   (30, 10, 60), "V":    (10, 70, 20), "VI":   (40, 10, 50),
        "VII":  (50, 30, 20), "VIII": (30, 50, 20), "IX":   (70, 10, 20)
    }
    best_field = "IV"
    min_dist = float("inf")
    for f_num, center in field_centers.items():
        dist = ((aom_pct - center[0])**2 + (pal_pct - center[1])**2 + (phy_pct - center[2])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            best_field = f_num
            
    for f in TYSON_FIELDS:
        if f["num"] == best_field:
            return f["num"], f["name"]
    return "IV", "Shelf to basin transition"

def draw_tyson_ternary(ax):
    """Tyson Alanlarını ve Dinamik Ok Göstergelerini Çiz"""
    # Alan Poligonlarını Doldur
    for field in TYSON_FIELDS:
        cart_pts = [tc(a,p,h) for (a,p,h) in field["pts"]]
        poly = Polygon(cart_pts, closed=True, facecolor=field["color"], 
                       edgecolor="#B0B0B0", linewidth=0.5, alpha=0.8, zorder=1)
        ax.add_patch(poly)

        # Alan Numaraları ve Yazıları
        cx = np.mean([pt[0] for pt in cart_pts])
        cy = np.mean([pt[1] for pt in cart_pts])
        ax.text(cx, cy + 0.015, field["num"], ha="center", va="center",
                fontsize=11, fontweight="bold", color="#2C3E50", zorder=4)
        ax.text(cx, cy - 0.02, field["name"], ha="center", va="center",
                fontsize=5.5, color="#555555", wrap=True, zorder=4, multialignment="center")

    # Ana Üçgen Çerçevesi
    corners = [tc(100,0,0), tc(0,100,0), tc(0,0,100)]
    tri = Polygon(corners, closed=True, fill=False, edgecolor="#1A1A1A", linewidth=2, zorder=5)
    ax.add_patch(tri)

    # İnce Kılavuz Çizgileri (%20 Aralıklı İç Izgara)
    for frac in [0.2, 0.4, 0.6, 0.8]:
        p = frac * 100; q = 100 - p
        ax.plot(*zip(tc(p,q,0), tc(p,0,q)), color="gray", lw=0.4, ls=":", alpha=0.4, zorder=2)
        ax.plot(*zip(tc(0,p,q), tc(q,p,0)), color="gray", lw=0.4, ls=":", alpha=0.4, zorder=2)
        ax.plot(*zip(tc(q,0,p), tc(0,q,p)), color="gray", lw=0.4, ls=":", alpha=0.4, zorder=2)

    # Köşe Ana Başlıkları
    ax.text(-0.02, -0.02, "AOM\n(Amorphous Organic Matter)", ha="right", va="top", fontsize=12, fontweight="bold", color=COLORS["AOM"])
    ax.text(1.02, -0.02, "Palynomorphs\n(Spores/Pollen/Microplankton)", ha="left", va="top", fontsize=12, fontweight="bold", color=COLORS["Palynomorph"])
    ax.text(0.5, 0.866 + 0.03, "Phytoclasts\n(Wood/Plant Debris)", ha="center", va="bottom", fontsize=12, fontweight="bold", color=COLORS["Phytoclast"])

    # ── YÜZDELER YERİNE DİNAMİK OK KILAVUZLARI (PetroStrat Stili) ──
    # 1. AOM Artış Yönü (Sağdan Sola)
    ax.annotate("", xy=(0.1, -0.06), xytext=(0.9, -0.06), arrowprops=dict(arrowstyle="->", color=COLORS["AOM"], lw=1.5))
    ax.text(0.5, -0.09, "Increasing AOM ──", ha="center", va="top", color=COLORS["AOM"], fontsize=9, fontweight="bold")

    # 2. Phytoclast Artış Yönü (Aşağıdan Yukarıya - Sol Kenar)
    ax.annotate("", xy=(0.5 - 0.05, 0.866), xytext=(0.0 - 0.05, 0.0), arrowprops=dict(arrowstyle="->", color=COLORS["Phytoclast"], lw=1.5))
    ax.text(0.25 - 0.06, 0.433, "── Increasing Phytoclasts", ha="center", va="bottom", color=COLORS["Phytoclast"], fontsize=9, fontweight="bold", rotation=60)

    # 3. Palynomorph Artış Yönü (Yukarıdan Aşağı Sağ Köşeye - Sağ Kenar)
    ax.annotate("", xy=(1.0 + 0.05, 0.0), xytext=(0.5 + 0.05, 0.866), arrowprops=dict(arrowstyle="->", color=COLORS["Palynomorph"], lw=1.5))
    ax.text(0.75 + 0.06, 0.433, "Increasing Palynomorphs ──", ha="center", va="bottom", color=COLORS["Palynomorph"], fontsize=9, fontweight="bold", rotation=-60)

    ax.set_xlim(-0.25, 1.45)
    ax.set_ylim(-0.15, 0.98)
    ax.axis("off")

def plot_tyson_ternary(df, out_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")

    ax.set_title(
        "Tyson (1995) Ternary Plot — Palynofacies Analysis\n"
        f"n = {len(df)} samples  |  {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=14, fontweight="bold", pad=20
    )

    draw_tyson_ternary(ax)

    # Örnek Noktalarını Çiz
    xs, ys = [], []
    for _, row in df.iterrows():
        x, y = tc(row["AOM_area_pct"], row["Palynomorph_area_pct"], row["Phytoclast_area_pct"])
        xs.append(x); ys.append(y)

    ax.scatter(xs, ys, c="#2C3E50", s=60, zorder=7, alpha=0.85, edgecolors="white", linewidth=0.7, label="Sample")

    # Ortalama Noktası (Mean) Hesaplama ve Projeksiyonu
    ma, mp, mh = df["AOM_area_pct"].mean(), df["Palynomorph_area_pct"].mean(), df["Phytoclast_area_pct"].mean()
    mx, my = tc(ma, mp, mh)
    
    ax.scatter([mx], [my], c="red", s=220, zorder=8, edgecolors="white", linewidth=1.5, marker="*", label="Mean")

    fasies_num, _ = classify_facies(ma, mp, mh)
    ax.annotate(
        f"Mean: Field {fasies_num}\n"
        f"AOM: {ma:.1f}%\nPal: {mp:.1f}%\nPhy: {mh:.1f}%",
        xy=(mx, my), xytext=(mx + 0.12, my + 0.08),
        fontsize=9, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9),
        zorder=10
    )

    # Sağ Kenara Hizalanmış Temiz Lejant Yapısı (Çakışmalar Önlendi)
    field_patches = [
        mpatches.Patch(facecolor=f["color"], edgecolor="#888", label=f"Field {f['num']}: {f['name']}")
        for f in TYSON_FIELDS
    ]
    corner_patches = [
        mpatches.Patch(color=COLORS["AOM"],         label=f"AOM (mean {ma:.1f}%)"),
        mpatches.Patch(color=COLORS["Palynomorph"], label=f"Palynomorph (mean {mp:.1f}%)"),
        mpatches.Patch(color=COLORS["Phytoclast"],  label=f"Phytoclast (mean {mh:.1f}%)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#2C3E50", markersize=8, label="Sample"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="red", markersize=12, label="Mean Data Point"),
    ]

    leg1 = ax.legend(handles=field_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, framealpha=0.9, title="Tyson (1995) Fields", title_fontsize=9)
    ax.add_artist(leg1)
    
    ax.legend(handles=corner_patches, loc="upper left", bbox_to_anchor=(1.02, 0.45),
              fontsize=8, framealpha=0.9, title="Statistics & Legend", title_fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Tyson ternary plot başarıyla düzeltildi → {out_path}")

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
            ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    axes[2].pie(means_area, labels=PALYNO_GROUPS, colors=colors, autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[2].set_title("Overall Area Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_charts.png", dpi=150, bbox_inches="tight")
    plt.close()

def save_reports(df, out_dir):
    df.to_csv(out_dir / "palynofacies_results.csv", index=False, float_format="%.3f")
    with pd.ExcelWriter(out_dir / "palynofacies_report.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Per Image", index=False)
        summary = pd.DataFrame({
            "Class": PALYNO_GROUPS,
            "Mean Area %":   [df[f"{g}_area_pct"].mean()  for g in PALYNO_GROUPS],
            "Std Area %":    [df[f"{g}_area_pct"].std()   for g in PALYNO_GROUPS],
            "Mean Count %":  [df[f"{g}_count_pct"].mean() for g in PALYNO_GROUPS],
            "Total Count":   [int(df[f"{g}_count"].sum()) for g in PALYNO_GROUPS]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

def print_summary(df):
    ma, mp, mh = df["AOM_area_pct"].mean(), df["Palynomorph_area_pct"].mean(), df["Phytoclast_area_pct"].mean()
    fnum, fname = classify_facies(ma, mp, mh)
    print("\n" + "="*65)
    print(f"Baskın Tyson Fasies Alanı : Field {fnum} ── {fname}")
    print(f"Ortalama Oranlar: AOM={ma:.1f}% | Palynomorph={mp:.1f}% | Phytoclast={mh:.1f}%")
    print("="*65)

def main():
    parser = argparse.ArgumentParser(description="Palynofacies Analyzer")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images",  required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    images_path  = Path(args.images)
    out_dir = Path(args.out_dir) if args.out_dir else weights_path.parent.parent / "palynofacies"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_path.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
    if not images:
        print(f"❌ Görüntü bulunamadı: {images_path}"); sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    rows = []
    for i, img_path in enumerate(images, 1):
        rows.append(analyze_image(model, img_path, CLASS_NAMES))

    df = pd.DataFrame(rows)
    print_summary(df)
    save_reports(df, out_dir)
    plot_summary_charts(df, out_dir)
    plot_tyson_ternary(df, out_dir / "tyson_ternary_plot.png")

if __name__ == "__main__":
    main()