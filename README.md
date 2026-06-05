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
    
