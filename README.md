# -APAQ-Automated-Palynofacies-Area-Ratio-Quantification-for-the-Toarcian-Oceanic-Anoxic-Event
A YOLOv26 Instance Segmentation Methodology

Special augmentation script for T-OAE datasets: bora.py

Veri Hazirlama:

C:\paly_toae\

    dataset\          ← Roboflow/CVAT/Ultralytics/LabelImg'dan indirdiğin etiketlenen verilere ait zip'i buraya çıkart
    
    dataset_split\    ← Script otomatik oluşturacak, boş bırak 
    
    scripts\          ← bora_toae.ipynb


Adım 1 — Ortam oluştur


    conda create -n palytoae python=3.10 -y


    conda activate palytoae


Adım 2 — Kütüphaneleri kur


    pip install ultralytics tensorboard pyyaml rich


Adım 3 — Script klasörüne git


    cd "C:\paly_toae\scripts"


Adim 4 - Script Calistir


    python bora_toae.py
