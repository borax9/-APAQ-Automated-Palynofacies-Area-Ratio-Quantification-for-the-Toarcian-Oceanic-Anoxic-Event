# -*- coding: utf-8 -*-
"""
palynofacies_full.py
====================
Full Palynofacies + TAI analysis using YOLO segmentation model.

Outputs:
  1. tyson_ternary_plot.png     — Clean Tyson Ternary Plot (Data points only)
  2. relative_abundance_log.png — Geological log format relative abundance
  3. tai_diagram.png            — TAI maturity bar chart + Color Evolution Scale
  4. summary_charts.png         — Bar + Pie + Boxplot
  5. palynofacies_report.xlsx   — Per Image + Summary + TAI sheets
  6. palynofacies_results.csv

Usage:
  python palynofacies_full.py --weights best.pt --images test_images/ --mag 10

Installation:
  pip install ultralytics matplotlib pandas openpyxl opencv-python
"""

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

# ==============================================================================
# SETTINGS
# ==============================================================================
CLASS_NAMES    = ["AOM", "Background", "Palynomorph", "Phytoclast"]
COLORS_HEX     = ["#DDA0DD", "#D3D3D3", "#FFD700", "#8B4513"] # Plum, LightGray, Gold, SaddleBrown
# TAI Standard Colors: Immature (Pale Yellow) -> Post-Mature (Black)
TAI_COLORS_HEX = ["#FFF8DC", "#FFE4B5", "#FFA500", "#FF4500", "#8B0000", "#000000"] 

# Micron per pixel mappings based on magnification (10x, 20x, 40x, 50x)
MICRON_PER_PIXEL = {10: 1.20, 20: 0.60, 40: 0.30, 50: 0.24}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_tyson_environment(aom, palyno, phyto):
    """
    Determines the Tyson (1995) 9-field Palynofacies Environment based on percentages.
    Used for tabular data classification.
    """
    total = aom + palyno + phyto
    if total == 0: return "Unknown", 0
    p_aom   = (aom / total) * 100
    p_palyno = (palyno / total) * 100
    p_phyto  = (phyto / total) * 100

    if p_phyto <= 50 and p_palyno <= 10 and p_aom >= 40: return "Field I: Highly Marginal Dysoxic-Anoxic", 1
    if p_phyto <= 50 and p_palyno > 10 and p_palyno <= 40 and p_aom >= 40: return "Field II: Marginal Dysoxic-Anoxic", 2
    if p_phyto <= 30 and p_palyno <= 25 and p_aom >= 55: return "Field III: Heterotrophic Offshore", 3
    if p_phyto <= 50 and p_palyno <= 30 and p_aom > 20 and p_aom < 55: return "Field IV: Shelf to Basin Transition", 4
    if p_phyto > 50 and p_palyno <= 30 and p_aom <= 30: return "Field V: Mud-dominated Oxic Shelf (Proximal)", 5
    if p_phyto > 40 and p_phyto <= 70 and p_palyno <= 20 and p_aom >= 20: return "Field VI: Proximal Suboxic-Anoxic Shelf", 6
    if p_phyto <= 35 and p_palyno > 25 and p_aom >= 40: return "Field VII: Distal Dysoxic-Anoxic Shelf", 7
    if p_phyto <= 45 and p_palyno > 30 and p_aom <= 40: return "Field VIII: Distal Dysoxic-Oxic Shelf", 8
    if p_phyto <= 20 and p_palyno <= 10 and p_aom >= 75: return "Field IX: Proximal Suboxic-Anoxic Basin", 9
    
    if p_aom >= 50: return "Field III: Heterotrophic Offshore", 3
    if p_phyto >= 50: return "Field V: Mud-dominated Oxic Shelf (Proximal)", 5
    return "Field IV: Shelf to Basin Transition", 4

def get_tai_info(mean_intensity):
    """
    Estimates Thermal Alteration Index (TAI) and Vitrinite Reflectance (Ro%) based on particle brightness.
    """
    if mean_intensity > 180:   return "1.5 to 2.0", "Immature", "0.20 - 0.40%", TAI_COLORS_HEX[0]
    elif mean_intensity > 130: return "2.0 to 2.5", "Early Mature", "0.40 - 0.60%", TAI_COLORS_HEX[1]
    elif mean_intensity > 90:  return "2.5 to 2.8", "Peak Mature", "0.60 - 0.85%", TAI_COLORS_HEX[2]
    elif mean_intensity > 50:  return "2.8 to 3.2", "Late Mature", "0.85 - 1.35%", TAI_COLORS_HEX[3]
    elif mean_intensity > 20:  return "3.2 to 3.7", "Overmature", "1.35 - 2.00%", TAI_COLORS_HEX[4]
    else:                      return "3.7 to 4.0+", "Post Mature", "> 2.00%", TAI_COLORS_HEX[5]

# ==============================================================================
# MAIN ANALYSIS ENGINE
# ==============================================================================
def analyze_image(model, img_path, mag):
    scale = MICRON_PER_PIXEL.get(mag, 1.0)
    img = cv2.imread(str(img_path))
    if img is None:
        return {"Image_Name": img_path.name, "Status": "Error Loading File"}
        
    h, w, _ = img.shape
    total_pixels = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = model.predict(img, verbose=False, conf=0.25, iou=0.45)
    result = results[0]

    counts = Counter()
    area_pixels = defaultdict(float)
    intensities = defaultdict(list)

    if result.masks is not None:
        for mask, cls_idx in zip(result.masks.xy, result.boxes.cls):
            cls_name = CLASS_NAMES[int(cls_idx)]
            counts[cls_name] += 1
            
            if len(mask) < 3: continue
            poly = np.array(mask, dtype=np.int32)
            
            p_area = cv2.contourArea(poly)
            area_pixels[cls_name] += p_area
            
            mask_img = np.zeros_like(gray)
            cv2.fillPoly(mask_img, [poly], 255)
            pts = gray[mask_img == 255]
            if len(pts) > 0:
                intensities[cls_name].extend(pts.tolist())

    aom_px   = area_pixels["AOM"]
    palyno_px = area_pixels["Palynomorph"]
    phyto_px  = area_pixels["Phytoclast"]
    total_om_px = aom_px + palyno_px + phyto_px

    if total_om_px > 0:
        p_aom   = (aom_px / total_om_px) * 100
        p_palyno = (palyno_px / total_om_px) * 100
        p_phyto  = (phyto_px / total_om_px) * 100
    else:
        p_aom, p_palyno, p_phyto = 0.0, 0.0, 0.0

    env_name, env_id = get_tyson_environment(p_aom, p_palyno, p_phyto)
    
    om_intensities = intensities["Phytoclast"] + intensities["Palynomorph"]
    mean_intensity = np.mean(om_intensities) if len(om_intensities) > 0 else 255.0
    tai_val, stage, ro_val, _ = get_tai_info(mean_intensity)

    res_dict = {
        "Image_Name": img_path.name,
        "Total_Pixels": total_pixels,
        "Total_OM_Pixels": total_om_px,
        "AOM_Count": counts["AOM"],
        "Palynomorph_Count": counts["Palynomorph"],
        "Phytoclast_Count": counts["Phytoclast"],
        "AOM_Area_SqMicron": aom_px * (scale**2),
        "Palynomorph_Area_SqMicron": palyno_px * (scale**2),
        "Phytoclast_Area_SqMicron": phyto_px * (scale**2),
        "Total_OM_Area_SqMicron": total_om_px * (scale**2),
        "SOM_Percentage": (total_om_px / total_pixels) * 100,
        "AOM_Relative_Pct": p_aom,
        "Palynomorph_Relative_Pct": p_palyno,
        "Phytoclast_Relative_Pct": p_phyto,
        "Tyson_Environment": env_name,
        "Tyson_Field_ID": env_id,
        "Mean_OM_Intensity": mean_intensity,
        "Estimated_TAI": tai_val,
        "Thermal_Maturity_Stage": stage,
        "Estimated_Ro": ro_val
    }
    return res_dict

# ==============================================================================
# VISUALIZATION MODULES (ENGLISH)
# ==============================================================================
def plot_tyson_ternary(df, out_path, mag):
    """Draws a clean, empty ternary plot with correct data projections."""
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.set_aspect('equal')

    # Barycentric to Cartesian coordinate conversion
    def get_xy(aom, palyno, phyto):
        total = aom + palyno + phyto
        if total == 0: return 0.5, 0.0
        a = aom / total
        p = palyno / total
        ph = phyto / total
        # AOM (Top), Palynomorphs (Left), Phytoclasts (Right)
        x = ph + 0.5 * a
        y = (np.sqrt(3) / 2) * a
        return x, y

    # Draw the main clean outer triangle boundary
    triangle = Polygon([(0,0), (1,0), (0.5, np.sqrt(3)/2)], facecolor='none', edgecolor='#333333', linewidth=2, zorder=3)
    ax.add_patch(triangle)

    # Plot the specific data points correctly calculated inside the triangle
    for _, row in df.iterrows():
        if row.get("Total_OM_Pixels", 0) > 0:
            x, y = get_xy(row["AOM_Relative_Pct"], row["Palynomorph_Relative_Pct"], row["Phytoclast_Relative_Pct"])
            ax.scatter(x, y, color='#FF4500', edgecolor='white', s=65, zorder=5, alpha=0.9)

    # Axis Labels
    ax.text(0.5, (np.sqrt(3)/2) + 0.03, "AOM (100%)", ha='center', va='bottom', fontsize=12, weight='bold')
    ax.text(-0.02, -0.02, "Palynomorphs (100%)", ha='right', va='top', fontsize=12, weight='bold')
    ax.text(1.02, -0.02, "Phytoclasts (100%)", ha='left', va='top', fontsize=12, weight='bold')

    # Visual boundaries adjustments
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, 1.0)
    ax.axis('off')
    
    plt.title("Tyson (1995) Palynofacies Ternary Plot", fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_relative_abundance_log(df, out_path):
    if len(df) == 0: return
    fig, ax = plt.subplots(figsize=(6, max(4, len(df)*0.4)))
    
    y_pos = np.arange(len(df))
    aom = df["AOM_Relative_Pct"].values
    palyno = df["Palynomorph_Relative_Pct"].values
    phyto = df["Phytoclast_Relative_Pct"].values
    names = df["Image_Name"].values

    ax.barh(y_pos, aom, label="AOM", color=COLORS_HEX[0], edgecolor='w', height=0.6)
    ax.barh(y_pos, palyno, left=aom, label="Palynomorphs", color=COLORS_HEX[2], edgecolor='w', height=0.6)
    ax.barh(y_pos, phyto, left=aom+palyno, label="Phytoclasts", color=COLORS_HEX[3], edgecolor='w', height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Relative Abundance (%)", fontsize=11, weight='bold')
    ax.set_xlim(0, 100)
    ax.invert_yaxis() 
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)
    plt.title("Palynofacies Relative Abundance Log", fontsize=13, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_tai_diagram(df, out_path):
    """Plots TAI Distribution Bar Chart + Visual Color Evolution Map at the bottom."""
    # Create two stacked subplots: Main Bar chart (top), Color Legend (bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), gridspec_kw={'height_ratios': [4, 1]})
    
    categories = ["1.5 - 2.0\n(Immature)", "2.0 - 2.5\n(Early)", "2.5 - 2.8\n(Peak)", "2.8 - 3.2\n(Late)", "3.2 - 3.7\n(Overmature)", "3.7 - 4.0+\n(Post-Mat.)"]
    counts = [0]*6
    
    for val in df["Estimated_TAI"].values:
        if "1.5" in str(val): counts[0]+=1
        elif "2.0" in str(val): counts[1]+=1
        elif "2.5" in str(val): counts[2]+=1
        elif "2.8" in str(val): counts[3]+=1
        elif "3.2" in str(val): counts[4]+=1
        elif "3.7" in str(val): counts[5]+=1

    # TOP CHART: Histogram
    bars = ax1.bar(categories, counts, color=TAI_COLORS_HEX, edgecolor='#444444', width=0.6, linewidth=1.2)
    ax1.set_ylabel("Number of Sub-samples / Fields", fontsize=11, weight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_title("Thermal Alteration Index (TAI) Distribution Map", fontsize=13, weight='bold', pad=15)
    
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, str(int(yval)), ha='center', va='bottom', weight='bold')

    # BOTTOM CHART: Palynomorph Color Evolution Scale
    ax2.axis('off')
    ax2.set_title("Palynomorph / Kerogen Color Evolution Reference", fontsize=10, weight='bold', color='#555555')
    
    for i, color in enumerate(TAI_COLORS_HEX):
        # Draw color squares representing organic matter under microscope
        rect = mpatches.Rectangle((i, 0), 1, 0.8, facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        # Clean up labels for a single line text
        clean_label = categories[i].replace('\n', ' ')
        ax2.text(i + 0.5, -0.2, clean_label, ha='center', va='top', fontsize=9)

    ax2.set_xlim(0, 6)
    ax2.set_ylim(-0.5, 1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_summary_charts(df, out_path):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    mean_areas = [df["AOM_Area_SqMicron"].mean(), df["Palynomorph_Area_SqMicron"].mean(), df["Phytoclast_Area_SqMicron"].mean()]
    ax1.bar(["AOM", "Palynomorphs", "Phytoclasts"], mean_areas, color=[COLORS_HEX[0], COLORS_HEX[2], COLORS_HEX[3]], edgecolor='k', width=0.4)
    ax1.set_ylabel("Mean Area ($\mu m^2$)", weight='bold')
    ax1.set_title("Absolute Abundance (Mean Area per Field)", weight='bold')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    total_areas = [df["AOM_Area_SqMicron"].sum(), df["Palynomorph_Area_SqMicron"].sum(), df["Phytoclast_Area_SqMicron"].sum()]
    if sum(total_areas) > 0:
        ax2.pie(total_areas, labels=["AOM", "Palynomorphs", "Phytoclasts"], colors=[COLORS_HEX[0], COLORS_HEX[2], COLORS_HEX[3]], autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'w'})
    ax2.set_title("Total Volumetric Organic Matter Budget", weight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.boxplot([df["AOM_Relative_Pct"], df["Palynomorph_Relative_Pct"], df["Phytoclast_Relative_Pct"]], labels=["AOM", "Palynomorphs", "Phytoclasts"], patch_artist=True,
                boxprops=dict(facecolor='#F5F5F5', color='k'), medianprops=dict(color='red', linewidth=1.5))
    ax3.set_ylabel("Relative Proportion (%)", weight='bold')
    ax3.set_title("Kerogen Variance Profile Boxplot", weight='bold')
    ax3.grid(axis='y', linestyle=':', alpha=0.6)

    ax4 = fig.add_subplot(gs[1, 1])
    env_counts = df["Tyson_Environment"].value_counts()
    env_counts.plot(kind='barh', color='#4682B4', ax=ax4, edgecolor='k', height=0.5)
    ax4.set_xlabel("Field Count Frequency", weight='bold')
    ax4.set_title("Facies Classification Hit Count Summary", weight='bold')
    ax4.grid(axis='x', linestyle=':', alpha=0.6)

    plt.suptitle("PalyTOAE Core Statistical Summary Panels", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ==============================================================================
# EXCEL GENERATOR
# ==============================================================================
def save_reports(df, out_dir, mag):
    out_dir = Path(out_dir)
    csv_path = out_dir / "palynofacies_results.csv"
    xlsx_path = out_dir / "palynofacies_report.xlsx"

    df.to_csv(csv_path, index=False, encoding='utf-8')

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Per Image Data", index=False)
        
        summary_data = {
            "Parameter": [
                "Total Images Analyzed", "Mean SOM Area per Field (Sq Microns)", "Mean SOM Area (%)",
                "Global Mean AOM Pct", "Global Mean Palynomorph Pct", "Global Mean Phytoclast Pct",
                "Dominant Tyson Environment Field Code"
            ],
            "Value": [
                len(df), df["Total_OM_Area_SqMicron"].mean(), df["SOM_Percentage"].mean(),
                df["AOM_Relative_Pct"].mean(), df["Palynomorph_Relative_Pct"].mean(), df["Phytoclast_Relative_Pct"].mean(),
                str(df["Tyson_Environment"].mode()[0] if len(df) > 0 else "N/A")
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary Metrics", index=False)

        tai_counts = df["Estimated_TAI"].value_counts().to_frame().reset_index()
        tai_counts.columns = ["TAI Range Interval", "Field Hit Count Frequency"]
        tai_counts.to_excel(writer, sheet_name="Thermal Maturity TAI", index=False)

def print_summary(df, mag):
    print(f"\n" + "="*60)
    print(f"               PALYTOAE PROJECT SUMMARY REPORT")
    print(f"")
    print(f"  Total Fields Evaluated   : {len(df)}")
    print(f"  Optical Magnification   : {mag}x (Scale Factor: {MICRON_PER_PIXEL.get(mag, 1.0)} um/px)")
    print(f"  Mean Kerogen Cover (SOM) : {df['SOM_Percentage'].mean():.2f} %")
    print(f"-"*60)
    print(f"  GLOBAL MEAN RELATIVE ABUNDANCE PROFILE:")
    print(f"    - Amorphous Organic Matter (AOM) : {df['AOM_Relative_Pct'].mean():.2f} %")
    print(f"    - Palynomorph Population         : {df['Palynomorph_Relative_Pct'].mean():.2f} %")
    print(f"    - Phytoclast Fragments           : {df['Phytoclast_Relative_Pct'].mean():.2f} %")
    print(f"-"*60)
    print(f"  DOMINANT SEDIMENTARY FACIES SIGNATURE:")
    mode_env = df['Tyson_Environment'].mode()[0] if len(df) > 0 else 'N/A'
    print(f"    => {mode_env}")
    print(f"  DOMINANT THERMAL ALTERATION SCALE (TAI):")
    mode_tai = df['Estimated_TAI'].mode()[0] if len(df) > 0 else 'N/A'
    mode_stage = df['Thermal_Maturity_Stage'].mode()[0] if len(df) > 0 else 'N/A'
    print(f"    => TAI {mode_tai} ({mode_stage})")
    print(f"=====" + "="*55)

# ==============================================================================
# MAIN TERMINAL ARGUMENT ROUTER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PalyTOAE Full Automatic Palynofacies System Pipeline.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model.pt file.")
    parser.add_argument("--images", type=str, required=True, help="Directory containing target images.")
    parser.add_argument("--out_dir", type=str, default="palynofacies_output", help="Directory where results are exported.")
    parser.add_argument("--mag", type=int, default=10, choices=[10, 20, 40, 50], help="Microscope lens magnification factor.")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    images_path  = Path(args.images)
    out_dir      = Path(args.out_dir)
    mag          = args.mag

    out_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in images_path.iterdir() if p.suffix.lower() in valid_exts] if images_path.exists() else []

    if not weights_path.exists():
        print(f"❌ Weights file not found at location: {weights_path}"); sys.exit(1)
    if len(images) == 0:
        print(f"❌ No valid image formats found in target path: {images_path}"); sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  PalyTOAE — Palynofacies Full Analyzer")
    print(f"  Model Weight File : {weights_path.name}")
    print(f"  Total Target Images: {len(images)}  |  Lens Magnification: {mag}x")
    print(f"  Destination Path   : {out_dir}")
    print(f"{'='*55}\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Required module missing! Execute: pip install ultralytics"); sys.exit(1)

    model = YOLO(str(weights_path))

    rows = []
    for i, img_path in enumerate(images, 1):
        print(f"  Processing image [{i:3d}/{len(images)}] -> {img_path.name}", end="\r")
        rows.append(analyze_image(model, img_path, mag))
    print(f"\n✅ Processing array and inferences complete! ({len(rows)} images processed)\n")

    df = pd.DataFrame(rows)
    print_summary(df, mag)

    print("\nWriting systematic files and data logs to disk:")
    save_reports(df, out_dir, mag)
    print("  -> Exported spreadsheet metrics database (.xlsx & .csv)")
    plot_tyson_ternary(df, out_dir / "tyson_ternary_plot.png", mag)
    print("  -> Generated clean Tyson (1995) Ternary Data plot")
    plot_relative_abundance_log(df, out_dir / "relative_abundance_log.png")
    print("  -> Generated Stratigraphic Organic Logging Profile chart")
    plot_tai_diagram(df, out_dir / "tai_diagram.png")
    print("  -> Rendered Thermal Alteration Maturity index histogram with Color Reference")
    plot_summary_charts(df, out_dir / "summary_charts.png")
    print("  -> Compiled cross-abundance bar/pie data panels")
    
    print(f"\n🚀 Pipeline operation successfully finished! Results ready inside: '{out_dir}'\n")

if __name__ == "__main__":
    main()