"""
5_palynofacies.py
=================
YOLO26 segmentation modeliyle otomatik Palynofacies analizi.
- Area ratio quantification (piksel + µm²)
- Tyson (1995) 9 fasies alanlı ternary plot
- 10x / 20x / 40x büyütme desteği

Kullanım:
  python 5_palynofacies.py --weights best.pt --images klasor/
  python 5_palynofacies.py --weights best.pt --images klasor/ --mag 20

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

# ─── KALİBRASYON — MİKROMETRE CETVELİNDEN ÖLÇÜLEN DEĞERLER ──────────────────
# Buraya kendi mikroskobunun değerlerini gir
# µm_per_pixel: 1 pikselin kaç mikrometre olduğu
# Hesaplama: cetvel görüntüsünde bilinen mesafe (µm) / o mesafenin piksel sayısı
CALIBRATION = {
    10: {"um_per_pixel": 1.2403},   # 10x  → ölçtükten sonra güncelle
    20: {"um_per_pixel": 0.6202},   # 20x  → ölçtükten sonra güncelle
    40: {"um_per_pixel": 0.3101},   # 40x  → ölçtükten sonra güncelle
}
# Klasör adından büyütmeyi otomatik algıla
MAGNIFICATION_KEYWORDS = {
    "10x": 10, "10X": 10,
    "20x": 20, "20X": 20,
    "40x": 40, "40X": 40,
}
# ──────────────────────────────────────────────────────────────────────────────

# ─── TYSON (1995) 9 FASİES ALANI ─────────────────────────────────────────────
TYSON_FIELDS = [
    {"num":"I",    "name":"Highly proximal shelf or basin",        "color":"#D5E8D4",
     "pts":[(0,0,100),(20,0,80),(20,20,60),(0,20,80)]},
    {"num":"II",   "name":"Marginal dysoxic-oxic basin",           "color":"#DAE8FC",
     "pts":[(0,20,80),(20,20,60),(20,40,40),(0,40,60)]},
    {"num":"III",  "name":"Heterolithic oxic shelf (proximal)",    "color":"#D5F5E3",
     "pts":[(0,40,60),(20,40,40),(20,60,20),(0,60,40)]},
    {"num":"IV",   "name":"Shelf to basin transition",             "color":"#FFF2CC",
     "pts":[(20,0,80),(40,0,60),(40,20,40),(20,20,60)]},
    {"num":"V",    "name":"Mud-dominated oxic shelf (distal)",     "color":"#DAE8FC",
     "pts":[(0,60,40),(20,60,20),(20,80,0),(0,80,20)]},
    {"num":"VI",   "name":"Proximal suboxic-anoxic shelf",         "color":"#FFE6CC",
     "pts":[(20,0,80),(60,0,40),(60,20,20),(40,20,40),(40,0,60)]},
    {"num":"VII",  "name":"Distal dysoxic-anoxic shelf",           "color":"#F8CECC",
     "pts":[(40,20,40),(60,20,20),(60,40,0),(40,40,20)]},
    {"num":"VIII", "name":"Distal dysoxic-anoxic shelf",           "color":"#E1D5E7",
     "pts":[(20,60,20),(40,60,0),(40,40,20),(20,40,40)]},
    {"num":"IX",   "name":"Distal suboxic-anoxic basin",           "color":"#F8CECC",
     "pts":[(60,0,40),(100,0,0),(60,40,0),(60,20,20)]},
]
# ──────────────────────────────────────────────────────────────────────────────


def get_magnification(images_path: Path, mag_arg: int) -> int:
    """Klasör adından veya argümandan büyütmeyi belirle."""
    if mag_arg:
        return mag_arg
    folder = images_path.name
    for kw, mag in MAGNIFICATION_KEYWORDS.items():
        if kw in folder:
            return mag
    print(f"⚠️  Büyütme belirlenemedi (klasör: {folder}). --mag ile belirt.")
    print(f"   Varsayılan: 10x kullanılıyor.")
    return 10


def pixels_to_um2(pixels: float, mag: int) -> float:
    """Piksel alanını µm²'ye çevir."""
    cal = CALIBRATION.get(mag, CALIBRATION[10])
    um_per_px = cal["um_per_pixel"]
    return pixels * (um_per_px ** 2)


def tc(aom, pal, phy):
    """Ternary → Kartezyen (AOM=sol, Pal=sağ, Phy=üst)."""
    t = aom + pal + phy
    if t == 0: return 0.5, 0.333
    a, p, h = aom/t, pal/t, phy/t
    x = p + h * 0.5
    y = h * (3**0.5) / 2
    return x, y


def analyze_image(model, img_path: Path, class_names: list, mag: int) -> dict:
    """Tek görüntüyü analiz et."""
    results = model(str(img_path), conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
    counts, areas_px = defaultdict(int), defaultdict(float)
    total_pixels = 0

    for r in results:
        h, w = r.orig_shape
        total_pixels = h * w
        if r.masks is None:
            continue
        for mask, cls_id in zip(r.masks.data, r.boxes.cls):
            cls_name = class_names[int(cls_id)]
            counts[cls_name]    += 1
            areas_px[cls_name]  += float(mask.cpu().numpy().sum())

    row = {
        "image"       : img_path.name,
        "magnification": f"{mag}x",
        "total_pixels": total_pixels,
        "total_um2"   : pixels_to_um2(total_pixels, mag),
    }

    total_palyno_px    = sum(areas_px.get(g, 0) for g in PALYNO_GROUPS)
    total_palyno_count = sum(counts.get(g, 0)   for g in PALYNO_GROUPS)

    for g in PALYNO_GROUPS:
        px  = areas_px.get(g, 0.0)
        cnt = counts.get(g, 0)
        um2 = pixels_to_um2(px, mag)

        row[f"{g}_count"]      = cnt
        row[f"{g}_area_px"]    = px
        row[f"{g}_area_um2"]   = um2
        row[f"{g}_area_pct"]   = (px  / total_palyno_px    * 100) if total_palyno_px    > 0 else 0.0
        row[f"{g}_count_pct"]  = (cnt / total_palyno_count * 100) if total_palyno_count > 0 else 0.0

    row[f"{BG_CLASS}_count"]   = counts.get(BG_CLASS, 0)
    row[f"{BG_CLASS}_area_px"] = areas_px.get(BG_CLASS, 0.0)
    row[f"{BG_CLASS}_area_um2"]= pixels_to_um2(areas_px.get(BG_CLASS, 0.0), mag)
    row["total_palyno_px"]     = total_palyno_px
    row["total_palyno_um2"]    = pixels_to_um2(total_palyno_px, mag)
    row["total_palyno_count"]  = total_palyno_count
    return row


def classify_facies(aom_pct, pal_pct, phy_pct):
    """Tyson fasies alanını belirle."""
    if phy_pct >= 80: return "I",    "Highly proximal shelf or basin"
    elif phy_pct >= 60 and pal_pct <= 20: return "II",  "Marginal dysoxic-oxic basin"
    elif phy_pct >= 40 and pal_pct >= 20: return "III", "Heterolithic oxic shelf (proximal)"
    elif phy_pct >= 60 and aom_pct >= 20: return "IV",  "Shelf to basin transition"
    elif pal_pct >= 60:                   return "V",   "Mud-dominated oxic shelf (distal)"
    elif aom_pct >= 40 and phy_pct >= 20: return "VI",  "Proximal suboxic-anoxic shelf"
    elif aom_pct >= 60 and phy_pct >= 20: return "VII", "Distal dysoxic-anoxic shelf"
    elif aom_pct >= 40 and pal_pct >= 40: return "VIII","Distal dysoxic-anoxic shelf"
    elif aom_pct >= 60:                   return "IX",  "Distal suboxic-anoxic basin"
    else:                                 return "IV",  "Shelf to basin transition (mixed)"


def draw_tyson_ternary(ax):
    """Tyson 9 fasies alanlı ternary üçgen."""
    for field in TYSON_FIELDS:
        cart_pts = [tc(a, p, h) for (a, p, h) in field["pts"]]
        poly = Polygon(cart_pts, closed=True,
                       facecolor=field["color"], edgecolor="#888",
                       linewidth=0.6, alpha=0.85, zorder=1)
        ax.add_patch(poly)
        cx = np.mean([pt[0] for pt in cart_pts])
        cy = np.mean([pt[1] for pt in cart_pts])
        ax.text(cx, cy+0.012, field["num"],
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="#333", zorder=4)
        ax.text(cx, cy-0.025, field["name"],
                ha="center", va="center", fontsize=5.5,
                color="#555", zorder=4, multialignment="center")

    corners = [tc(100,0,0), tc(0,100,0), tc(0,0,100)]
    tri = Polygon(corners, closed=True, fill=False,
                  edgecolor="black", linewidth=2, zorder=5)
    ax.add_patch(tri)

    for frac in [0.2, 0.4, 0.6, 0.8]:
        p = frac * 100; q = 100 - p
        ax.plot(*zip(tc(p,q,0), tc(p,0,q)), color="gray", lw=0.4, ls="--", alpha=0.5, zorder=2)
        ax.plot(*zip(tc(0,p,q), tc(q,p,0)), color="gray", lw=0.4, ls="--", alpha=0.5, zorder=2)
        ax.plot(*zip(tc(q,0,p), tc(0,q,p)), color="gray", lw=0.4, ls="--", alpha=0.5, zorder=2)
        ax.text(tc(p,q,0)[0]-0.025, tc(p,q,0)[1], f"{int(p)}%",
                ha="right", va="center", fontsize=7, color="#666")
        ax.text(tc(0,p,q)[0]+0.025, tc(0,p,q)[1], f"{int(p)}%",
                ha="left",  va="center", fontsize=7, color="#666")
        ax.text(tc(q,p,0)[0], tc(q,p,0)[1]-0.03, f"{int(p)}%",
                ha="center", va="top",   fontsize=7, color="#666")

    ax.text(tc(100,0,0)[0]-0.04, tc(100,0,0)[1]-0.04,
            "AOM", ha="right", va="top", fontsize=14,
            fontweight="bold", color=COLORS["AOM"], zorder=6)
    ax.text(tc(0,100,0)[0]+0.04, tc(0,100,0)[1]-0.04,
            "Palynomorphs", ha="left", va="top", fontsize=14,
            fontweight="bold", color=COLORS["Palynomorph"], zorder=6)
    ax.text(tc(0,0,100)[0], tc(0,0,100)[1]+0.04,
            "Phytoclasts", ha="center", va="bottom", fontsize=14,
            fontweight="bold", color=COLORS["Phytoclast"], zorder=6)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.12, 1.02)
    ax.axis("off")


def plot_tyson_ternary(df: pd.DataFrame, out_path: Path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_aspect("equal")
    ax.set_title(
        f"Tyson (1995) Ternary Plot — Palynofacies Analysis {title_suffix}\n"
        f"n = {len(df)} samples  |  {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=14, fontweight="bold", pad=16
    )
    draw_tyson_ternary(ax)

    xs, ys = [], []
    for _, row in df.iterrows():
        x, y = tc(row["AOM_area_pct"], row["Palynomorph_area_pct"], row["Phytoclast_area_pct"])
        xs.append(x); ys.append(y)
    ax.scatter(xs, ys, c="#2C3E50", s=55, zorder=7, alpha=0.8,
               edgecolors="white", linewidth=0.6, label="Sample")

    ma = df["AOM_area_pct"].mean()
    mp = df["Palynomorph_area_pct"].mean()
    mh = df["Phytoclast_area_pct"].mean()
    mx, my = tc(ma, mp, mh)
    ax.scatter([mx], [my], c="red", s=200, zorder=8,
               edgecolors="white", linewidth=1.5, marker="*", label="Mean")
    fnum, fname = classify_facies(ma, mp, mh)
    ax.annotate(
        f"Mean — Field {fnum}\nAOM:{ma:.1f}%  Pal:{mp:.1f}%  Phy:{mh:.1f}%",
        xy=(mx, my), xytext=(mx+0.10, my+0.06),
        fontsize=8, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.85),
        zorder=9
    )

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
                     bbox_to_anchor=(-0.02, 1.0), fontsize=7, framealpha=0.9,
                     title="Tyson (1995) Palynofacies Fields", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=corner_patches, loc="lower left",
              bbox_to_anchor=(-0.02, 0.0), fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Tyson plot → {out_path}")


def plot_area_quantification(df: pd.DataFrame, out_dir: Path, mag: int):
    """Area ratio quantification grafikleri."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Area Ratio Quantification — {mag}x Magnification",
                 fontsize=15, fontweight="bold")

    colors = [COLORS[g] for g in PALYNO_GROUPS]

    # 1. Ortalama Alan % (bar)
    means_pct = [df[f"{g}_area_pct"].mean() for g in PALYNO_GROUPS]
    stds_pct  = [df[f"{g}_area_pct"].std()  for g in PALYNO_GROUPS]
    bars = axes[0,0].bar(PALYNO_GROUPS, means_pct, color=colors,
                         edgecolor="white", lw=1.2,
                         yerr=stds_pct, capsize=5, error_kw={"elinewidth":1.5})
    axes[0,0].set_ylabel("Mean Area %"); axes[0,0].set_title("Mean Area % (±std)")
    axes[0,0].set_ylim(0, 110)
    for bar, v, s in zip(bars, means_pct, stds_pct):
        axes[0,0].text(bar.get_x()+bar.get_width()/2, v+s+2,
                       f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # 2. Ortalama Alan µm²
    means_um2 = [df[f"{g}_area_um2"].mean() for g in PALYNO_GROUPS]
    stds_um2  = [df[f"{g}_area_um2"].std()  for g in PALYNO_GROUPS]
    bars2 = axes[0,1].bar(PALYNO_GROUPS, means_um2, color=colors,
                          edgecolor="white", lw=1.2,
                          yerr=stds_um2, capsize=5, error_kw={"elinewidth":1.5})
    axes[0,1].set_ylabel("Mean Area (µm²)")
    axes[0,1].set_title("Mean Area µm² (±std)")
    for bar, v in zip(bars2, means_um2):
        axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                       f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")

    # 3. Pie chart (alan bazlı)
    axes[0,2].pie(means_pct, labels=PALYNO_GROUPS, colors=colors,
                  autopct="%1.1f%%", startangle=90,
                  wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[0,2].set_title("Area Distribution (%)")

    # 4. Ortalama obje sayısı
    means_cnt = [df[f"{g}_count"].mean() for g in PALYNO_GROUPS]
    stds_cnt  = [df[f"{g}_count"].std()  for g in PALYNO_GROUPS]
    bars3 = axes[1,0].bar(PALYNO_GROUPS, means_cnt, color=colors,
                          edgecolor="white", lw=1.2,
                          yerr=stds_cnt, capsize=5, error_kw={"elinewidth":1.5})
    axes[1,0].set_ylabel("Mean Count"); axes[1,0].set_title("Mean Object Count (±std)")
    for bar, v in zip(bars3, means_cnt):
        axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                       f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

    # 5. Box plot — alan % dağılımı
    data_box = [df[f"{g}_area_pct"].values for g in PALYNO_GROUPS]
    bp = axes[1,1].boxplot(data_box, labels=PALYNO_GROUPS, patch_artist=True,
                           medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[1,1].set_ylabel("Area %"); axes[1,1].set_title("Area % Distribution (boxplot)")

    # 6. Scatter — alan vs obje sayısı
    for g, c in zip(PALYNO_GROUPS, colors):
        axes[1,2].scatter(df[f"{g}_count"], df[f"{g}_area_um2"],
                         color=c, label=g, alpha=0.6, s=40, edgecolors="white")
    axes[1,2].set_xlabel("Object Count"); axes[1,2].set_ylabel("Area (µm²)")
    axes[1,2].set_title("Count vs Area µm²")
    axes[1,2].legend(fontsize=9)

    plt.tight_layout()
    out_path = out_dir / f"area_quantification_{mag}x.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Area quantification → {out_path}")


def save_reports(df: pd.DataFrame, out_dir: Path, mag: int):
    """CSV ve Excel kaydet."""
    csv_path = out_dir / f"palynofacies_{mag}x.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV → {csv_path}")

    xlsx_path = out_dir / f"palynofacies_{mag}x.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Per Image", index=False)
        cal = CALIBRATION.get(mag, CALIBRATION[10])

        summary_data = {
            "Class"          : PALYNO_GROUPS,
            "Mean Area %"    : [df[f"{g}_area_pct"].mean()  for g in PALYNO_GROUPS],
            "Std Area %"     : [df[f"{g}_area_pct"].std()   for g in PALYNO_GROUPS],
            "Min Area %"     : [df[f"{g}_area_pct"].min()   for g in PALYNO_GROUPS],
            "Max Area %"     : [df[f"{g}_area_pct"].max()   for g in PALYNO_GROUPS],
            "Mean Area µm²"  : [df[f"{g}_area_um2"].mean()  for g in PALYNO_GROUPS],
            "Std Area µm²"   : [df[f"{g}_area_um2"].std()   for g in PALYNO_GROUPS],
            "Mean Count"     : [df[f"{g}_count"].mean()     for g in PALYNO_GROUPS],
            "Total Count"    : [int(df[f"{g}_count"].sum()) for g in PALYNO_GROUPS],
            "Mean Count %"   : [df[f"{g}_count_pct"].mean() for g in PALYNO_GROUPS],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Kalibrasyon bilgisi
        cal_df = pd.DataFrame({
            "Magnification": [f"{mag}x"],
            "µm per pixel" : [cal["um_per_pixel"]],
            "µm² per pixel": [cal["um_per_pixel"]**2],
        })
        cal_df.to_excel(writer, sheet_name="Calibration", index=False)

    print(f"  Excel → {xlsx_path}")


def print_summary(df: pd.DataFrame, mag: int):
    ma = df["AOM_area_pct"].mean()
    mp = df["Palynomorph_area_pct"].mean()
    mh = df["Phytoclast_area_pct"].mean()
    fnum, fname = classify_facies(ma, mp, mh)
    cal = CALIBRATION.get(mag, CALIBRATION[10])

    print("\n" + "="*70)
    print(f"PALYNOFACİES ANALİZ SONUÇLARI — Tyson (1995) | {mag}x büyütme")
    print(f"Kalibrasyon: {cal['um_per_pixel']} µm/piksel")
    print("="*70)
    print(f"{'Sınıf':15s} {'Alan%':>8s} {'±std':>6s} {'µm²(ort)':>12s} {'Count(ort)':>11s} {'Count%':>8s}")
    print("-"*70)
    for g in PALYNO_GROUPS:
        print(f"{g:15s} "
              f"{df[f'{g}_area_pct'].mean():8.2f}% "
              f"{df[f'{g}_area_pct'].std():6.2f} "
              f"{df[f'{g}_area_um2'].mean():12.1f} "
              f"{df[f'{g}_count'].mean():11.1f} "
              f"{df[f'{g}_count_pct'].mean():8.2f}%")
    print("="*70)
    print(f"\n🗺️  Baskın Fasies Alanı: Field {fnum} — {fname}")
    print(f"   AOM={ma:.1f}%  Palynomorph={mp:.1f}%  Phytoclast={mh:.1f}%")

    print(f"\nGörüntü başına fasies dağılımı:")
    from collections import Counter as Ctr
    fc = Ctr()
    for _, row in df.iterrows():
        fn, _ = classify_facies(row["AOM_area_pct"],
                                row["Palynomorph_area_pct"],
                                row["Phytoclast_area_pct"])
        fc[fn] += 1
    for fn in ["I","II","III","IV","V","VI","VII","VIII","IX"]:
        n = fc.get(fn, 0)
        if n > 0:
            name = next(f["name"] for f in TYSON_FIELDS if f["num"] == fn)
            print(f"  Field {fn:4s}: {n:3d} görüntü ({n/len(df)*100:.1f}%) — {name}")


def main():
    parser = argparse.ArgumentParser(description="Palynofacies Analyzer — Tyson 1995")
    parser.add_argument("--weights", required=True, help="best.pt yolu")
    parser.add_argument("--images",  required=True, help="Test görüntüleri klasörü")
    parser.add_argument("--mag",     type=int, default=None,
                        help="Büyütme (10, 20, 40). Belirtilmezse klasör adından algılanır.")
    parser.add_argument("--out_dir", default=None, help="Çıktı klasörü")
    parser.add_argument("--conf",    type=float, default=CONF_THRESH)
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
        print(f"❌ Görüntü bulunamadı: {images_path}"); sys.exit(1)

    print(f"\n✅ {len(images)} görüntü | {mag}x büyütme | {CALIBRATION[mag]['um_per_pixel']} µm/px")
    print(f"   Model : {weights_path.name}")
    print(f"   Çıktı : {out_dir}\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics"); sys.exit(1)

    model = YOLO(str(weights_path))

    rows = []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i:3d}/{len(images)}] {img_path.name}", end="\r")
        rows.append(analyze_image(model, img_path, CLASS_NAMES, mag))
    print(f"\n✅ Analiz tamamlandı\n")

    df = pd.DataFrame(rows)
    print_summary(df, mag)

    print("\nÇıktılar kaydediliyor:")
    save_reports(df, out_dir, mag)
    plot_area_quantification(df, out_dir, mag)
    plot_tyson_ternary(df, out_dir / "tyson_ternary_plot.png",
                       title_suffix=f"({mag}x)")

    print(f"\n🎯 Tüm çıktılar: {out_dir}")
    print("   ├── palynofacies_{mag}x.csv")
    print("   ├── palynofacies_{mag}x.xlsx  (Per Image + Summary + Calibration)")
    print("   ├── area_quantification_{mag}x.png  (6 grafik)")
    print("   └── tyson_ternary_plot.png")


if __name__ == "__main__":
    main()
