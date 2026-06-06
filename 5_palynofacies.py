"""
5_palynofacies.py
=================
YOLO26 segmentation modeliyle otomatik Palynofacies analizi.

- Her görüntü için AOM / Palynomorph / Phytoclast mask alanı ve obje sayısı
- Background hariç tutulur
- Genel toplam istatistikleri
- Tyson Ternary Plot (matplotlib)
- CSV + Excel rapor çıktısı

Kullanım:
  python 5_palynofacies.py --weights best.pt --images klasor/

Kurulum:
  pip install ultralytics matplotlib pandas openpyxl
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore")

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["AOM", "Background", "Palynomorph", "Phytoclast"]
PALYNO_GROUPS = ["AOM", "Palynomorph", "Phytoclast"]   # Ternary için 3 köşe
BG_CLASS      = "Background"                            # Hesaplamadan hariç

CONF_THRESH   = 0.25
IOU_THRESH    = 0.45

# Renk paleti
COLORS = {
    "AOM"         : "#C0392B",   # kırmızı
    "Palynomorph" : "#27AE60",   # yeşil
    "Phytoclast"  : "#2980B9",   # mavi
    "Background"  : "#95A5A6",   # gri
}

# Tyson fasies bölgeleri (AOM-Palynomorph-Phytoclast köşeleri)
TYSON_FACIES = [
    # (isim, renk, köşe koordinatları [(aom,pal,phy), ...])
    ("I – Anoxic marine",        "#FADBD8", [(80,10,10),(100,0,0),(80,20,0),(60,20,20)]),
    ("II – Oxic marine",         "#D5F5E3", [(10,80,10),(0,100,0),(0,80,20),(20,60,20)]),
    ("III – Marine/terrestrial", "#D6EAF8", [(10,10,80),(0,0,100),(20,0,80),(20,20,60)]),
    ("IV – Mixed",               "#FEF9E7", [(33,33,34),(60,20,20),(20,60,20),(20,20,60)]),
]
# ──────────────────────────────────────────────────────────────────────────────


def ternary_to_cartesian(aom, pal, phy):
    """Ternary koordinatları (0-1) Kartezyen'e çevir."""
    total = aom + pal + phy
    if total == 0:
        return 0.5, 0.5
    a, b, c = aom/total, pal/total, phy/total
    x = 0.5 * (2*b + c) / (a + b + c)
    y = (np.sqrt(3)/2) * c / (a + b + c)
    return x, y


def analyze_image(model, img_path: Path, class_names: list) -> dict:
    """Tek görüntüyü analiz et, mask alan ve obje sayısı döndür."""
    results = model(
        str(img_path),
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False
    )

    counts = defaultdict(int)
    areas  = defaultdict(float)
    total_pixels = 0

    for r in results:
        h, w = r.orig_shape
        total_pixels = h * w

        if r.masks is None:
            continue

        for mask, cls_id in zip(r.masks.data, r.boxes.cls):
            cls_name = class_names[int(cls_id)]
            mask_np  = mask.cpu().numpy()
            area     = float(mask_np.sum())
            counts[cls_name] += 1
            areas[cls_name]  += area

    # Sadece palynofacies grupları
    row = {"image": img_path.name, "total_pixels": total_pixels}
    total_palyno_area  = 0
    total_palyno_count = 0

    for grp in PALYNO_GROUPS:
        row[f"{grp}_count"] = counts.get(grp, 0)
        row[f"{grp}_area"]  = areas.get(grp, 0.0)
        total_palyno_area  += areas.get(grp, 0.0)
        total_palyno_count += counts.get(grp, 0)

    row[f"{BG_CLASS}_count"] = counts.get(BG_CLASS, 0)
    row[f"{BG_CLASS}_area"]  = areas.get(BG_CLASS, 0.0)
    row["total_palyno_area"]  = total_palyno_area
    row["total_palyno_count"] = total_palyno_count

    # Yüzde hesapla (background hariç)
    for grp in PALYNO_GROUPS:
        a = areas.get(grp, 0.0)
        c = counts.get(grp, 0)
        row[f"{grp}_area_pct"]  = (a / total_palyno_area  * 100) if total_palyno_area  > 0 else 0.0
        row[f"{grp}_count_pct"] = (c / total_palyno_count * 100) if total_palyno_count > 0 else 0.0

    return row


def draw_ternary_background(ax):
    """Ternary üçgen çiz."""
    # Köşeler: AOM=sol alt, Palynomorph=sağ alt, Phytoclast=üst
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(triangle)

    # Izgara çizgileri (%20 aralıklı)
    for frac in [0.2, 0.4, 0.6, 0.8]:
        # AOM ekseni
        p1 = np.array(ternary_to_cartesian(frac, 1-frac, 0))
        p2 = np.array(ternary_to_cartesian(frac, 0, 1-frac))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "gray", lw=0.4, alpha=0.5)
        # Palynomorph ekseni
        p1 = np.array(ternary_to_cartesian(0, frac, 1-frac))
        p2 = np.array(ternary_to_cartesian(1-frac, frac, 0))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "gray", lw=0.4, alpha=0.5)
        # Phytoclast ekseni
        p1 = np.array(ternary_to_cartesian(0, 1-frac, frac))
        p2 = np.array(ternary_to_cartesian(1-frac, 0, frac))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "gray", lw=0.4, alpha=0.5)

    # Köşe etiketleri
    offset = 0.06
    ax.text(0 - offset,      0 - offset*0.8, "AOM",         ha="center", va="top",    fontsize=13, fontweight="bold", color=COLORS["AOM"])
    ax.text(1 + offset,      0 - offset*0.8, "Palynomorph", ha="center", va="top",    fontsize=13, fontweight="bold", color=COLORS["Palynomorph"])
    ax.text(0.5,  np.sqrt(3)/2 + offset,     "Phytoclast",  ha="center", va="bottom", fontsize=13, fontweight="bold", color=COLORS["Phytoclast"])

    # Eksen % etiketleri
    for frac in [0.2, 0.4, 0.6, 0.8]:
        pct = int(frac * 100)
        # AOM ekseni (sol kenar)
        x, y = ternary_to_cartesian(frac, 1-frac, 0)
        ax.text(x - 0.03, y, f"{pct}%", ha="right", va="center", fontsize=7, color="gray")
        # Phytoclast ekseni (sağ kenar)
        x, y = ternary_to_cartesian(0, 1-frac, frac)
        ax.text(x + 0.03, y, f"{pct}%", ha="left",  va="center", fontsize=7, color="gray")


def plot_tyson_ternary(df: pd.DataFrame, out_path: Path):
    """Tyson Ternary Plot çiz ve kaydet."""
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    # Başlık
    ax.set_title(
        "Tyson Ternary Plot — Palynofacies Analysis\n"
        f"(n={len(df)} images | {datetime.now().strftime('%Y-%m-%d')})",
        fontsize=14, fontweight="bold", pad=20
    )

    draw_ternary_background(ax)

    # Her görüntüyü nokta olarak çiz (alan bazlı)
    xs, ys = [], []
    for _, row in df.iterrows():
        aom = row["AOM_area_pct"] / 100
        pal = row["Palynomorph_area_pct"] / 100
        phy = row["Phytoclast_area_pct"] / 100
        x, y = ternary_to_cartesian(aom, pal, phy)
        xs.append(x)
        ys.append(y)

    ax.scatter(xs, ys, c="#2C3E50", s=60, zorder=5, alpha=0.75,
               edgecolors="white", linewidth=0.5, label="Sample (area %)")

    # Genel ortalama noktası
    mean_aom = df["AOM_area_pct"].mean() / 100
    mean_pal = df["Palynomorph_area_pct"].mean() / 100
    mean_phy = df["Phytoclast_area_pct"].mean() / 100
    mx, my = ternary_to_cartesian(mean_aom, mean_pal, mean_phy)
    ax.scatter([mx], [my], c="red", s=180, zorder=6,
               edgecolors="white", linewidth=1.5, marker="*", label="Mean")
    ax.annotate(
        f"Mean\nAOM:{mean_aom*100:.1f}%\nPal:{mean_pal*100:.1f}%\nPhy:{mean_phy*100:.1f}%",
        xy=(mx, my), xytext=(mx+0.08, my+0.05),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8)
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS["AOM"],          label=f"AOM (mean: {mean_aom*100:.1f}%)"),
        mpatches.Patch(color=COLORS["Palynomorph"],  label=f"Palynomorph (mean: {mean_pal*100:.1f}%)"),
        mpatches.Patch(color=COLORS["Phytoclast"],   label=f"Phytoclast (mean: {mean_phy*100:.1f}%)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#2C3E50",
                   markersize=8, label="Sample"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="red",
                   markersize=12, label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.18, 1.0), fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Tyson plot → {out_path}")


def plot_summary_charts(df: pd.DataFrame, out_dir: Path):
    """Bar chart + pie chart özet."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Palynofacies Summary", fontsize=14, fontweight="bold")

    # 1. Ortalama alan %
    means_area = [df[f"{g}_area_pct"].mean() for g in PALYNO_GROUPS]
    bars = axes[0].bar(PALYNO_GROUPS, means_area,
                       color=[COLORS[g] for g in PALYNO_GROUPS], edgecolor="white", linewidth=1.2)
    axes[0].set_ylabel("Mean Area %")
    axes[0].set_title("Mean Area % per Class")
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, means_area):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # 2. Ortalama obje sayısı
    means_count = [df[f"{g}_count_pct"].mean() for g in PALYNO_GROUPS]
    bars2 = axes[1].bar(PALYNO_GROUPS, means_count,
                        color=[COLORS[g] for g in PALYNO_GROUPS], edgecolor="white", linewidth=1.2)
    axes[1].set_ylabel("Mean Count %")
    axes[1].set_title("Mean Count % per Class")
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars2, means_count):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # 3. Pie chart (alan bazlı)
    total_areas = [df[f"{g}_area_pct"].mean() for g in PALYNO_GROUPS]
    wedges, texts, autotexts = axes[2].pie(
        total_areas,
        labels=PALYNO_GROUPS,
        colors=[COLORS[g] for g in PALYNO_GROUPS],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    axes[2].set_title("Overall Area Distribution")

    plt.tight_layout()
    out_path = out_dir / "summary_charts.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Summary charts → {out_path}")


def save_reports(df: pd.DataFrame, out_dir: Path):
    """CSV ve Excel rapor kaydet."""
    # CSV
    csv_path = out_dir / "palynofacies_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"  CSV → {csv_path}")

    # Excel — per-image + summary sheet
    xlsx_path = out_dir / "palynofacies_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Per-image sheet
        df.to_excel(writer, sheet_name="Per Image", index=False)

        # Summary sheet
        summary = pd.DataFrame({
            "Class": PALYNO_GROUPS,
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


def print_summary(df: pd.DataFrame):
    """Terminale özet tablo yaz."""
    print("\n" + "="*65)
    print("PALYNOFACIES ANALİZ SONUÇLARI")
    print("="*65)
    print(f"Toplam görüntü : {len(df)}")
    print(f"{'Sınıf':15s} {'Ort.Alan%':>10s} {'Std':>7s} {'Ort.Count%':>11s} {'Toplam Obje':>12s}")
    print("-"*65)
    for grp in PALYNO_GROUPS:
        print(f"{grp:15s} "
              f"{df[f'{grp}_area_pct'].mean():10.2f}% "
              f"{df[f'{grp}_area_pct'].std():7.2f} "
              f"{df[f'{grp}_count_pct'].mean():11.2f}% "
              f"{int(df[f'{grp}_count'].sum()):12d}")
    print("="*65)

    # Dominant fasies tahmini
    mean_aom = df["AOM_area_pct"].mean()
    mean_pal = df["Palynomorph_area_pct"].mean()
    mean_phy = df["Phytoclast_area_pct"].mean()
    dominant = max(
        [("AOM", mean_aom), ("Palynomorph", mean_pal), ("Phytoclast", mean_phy)],
        key=lambda x: x[1]
    )
    print(f"\nDominant bileşen : {dominant[0]} ({dominant[1]:.1f}%)")
    if mean_aom > 50:
        print("Fasies yorumu   : Anoxic marine (Tip I) — AOM dominant")
    elif mean_pal > 50:
        print("Fasies yorumu   : Oxic marine (Tip II) — Palynomorph dominant")
    elif mean_phy > 50:
        print("Fasies yorumu   : Terrestrial/deltaic (Tip III) — Phytoclast dominant")
    else:
        print("Fasies yorumu   : Mixed (Tip IV) — karma bileşim")


def main():
    parser = argparse.ArgumentParser(description="Palynofacies Analyzer")
    parser.add_argument("--weights", required=True, help="best.pt yolu")
    parser.add_argument("--images",  required=True, help="Test görüntüleri klasörü")
    parser.add_argument("--out_dir", default=None,  help="Çıktı klasörü")
    parser.add_argument("--conf",    type=float, default=CONF_THRESH)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    images_path  = Path(args.images)
    out_dir      = Path(args.out_dir) if args.out_dir else weights_path.parent.parent / "palynofacies"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Görüntüleri topla
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    images = sorted([p for p in images_path.iterdir()
                     if p.suffix.lower() in img_extensions])

    if not images:
        print(f"❌ Görüntü bulunamadı: {images_path}")
        sys.exit(1)

    print(f"✅ {len(images)} görüntü bulundu")
    print(f"   Model   : {weights_path}")
    print(f"   Çıktı   : {out_dir}\n")

    # Model yükle
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(weights_path))

    # Her görüntüyü analiz et
    rows = []
    for i, img_path in enumerate(images, 1):
        print(f"[{i:3d}/{len(images)}] {img_path.name}", end="\r")
        row = analyze_image(model, img_path, CLASS_NAMES)
        rows.append(row)

    print(f"\n✅ Analiz tamamlandı — {len(rows)} görüntü")

    # DataFrame oluştur
    df = pd.DataFrame(rows)

    # Terminale özet yaz
    print_summary(df)

    # Raporlar
    print("\nÇıktılar kaydediliyor:")
    save_reports(df, out_dir)
    plot_summary_charts(df, out_dir)
    plot_tyson_ternary(df, out_dir / "tyson_ternary_plot.png")

    print(f"\n🎯 Tüm çıktılar: {out_dir}")
    print("   ├── palynofacies_results.csv")
    print("   ├── palynofacies_report.xlsx  (per-image + summary sheet)")
    print("   ├── summary_charts.png")
    print("   └── tyson_ternary_plot.png")


if __name__ == "__main__":
    main()
