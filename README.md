# -APAQ-Automated-Palynofacies-Area-Ratio-Quantification-for-the-Toarcian-Oceanic-Anoxic-Event
A YOLOv26 Instance Segmentation Methodology

Special augmentation script for T-OAE datasets: bora.py

Dataset Preparation:

C:\paly_toae\

    dataset\          ← Label in Roboflow&CVAT&Ultralytics&LabelImg. Extract the zip file of downloaded labeled images
    
    dataset_split\    ← Script will create automaticaly , leave blank folder
    
    scripts\          ← bora_toae.ipynb


Step 1 — Create Environment


    conda create -n palytoae python=3.10 -y


    conda activate palytoae


Step 2 — Setup Libraries


    pip install ultralytics tensorboard pyyaml rich


Step 3 — Go to Script Folder


    cd "C:\paly_toae\scripts"


Step 4 - Run the Script 


    python bora_toae.py


Next Steps - Follow the instructions in the script (  python bora_toae.py )


AFTER TRAINING - AUTOMATED PALYNOFACIES ANALYSIS

📊 How Script works?
Input: 100 test image + best.pt
Her görüntü için hesaplar:

Her sınıftan kaç obje detect edildi
Her sınıfın mask alanı (piksel) ve yüzdesi (Background hariç)

Çıktılar:
Dosyaİçerikpalynofacies_results.csvHer görüntü için tüm sayılarpalynofacies_report.xlsxPer-image + Summary sheetsummary_charts.pngBar + Pie charttyson_ternary_plot.pngTyson Ternary Plot
Tyson plot'ta:

Her görüntü ayrı nokta olarak çizilir
Kırmızı yıldız = tüm dataset'in ortalaması
Fasies yorumu otomatik yazılır (Anoxic / Oxic / Terrestrial / Mixed)


Kullanım (training bittikten sonra):

      pip install ultralytics matplotlib pandas openpyxl

      python 5_palynofacies.py \

      weights C:\palytoae\runs\exp_XXX\weights\best.pt \
      
      images  C:\palytoae\test_images\
    
