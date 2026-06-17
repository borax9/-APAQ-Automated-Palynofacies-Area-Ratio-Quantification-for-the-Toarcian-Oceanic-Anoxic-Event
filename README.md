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


    python bora_toae_paly.py


Next Steps - Follow the instructions in the script (  python bora_toae.py )


AFTER TRAINING - AUTOMATED PALYNOFACIES ANALYSIS - Follow the Full Script

📊 How Script works?
Input: test images + model.pt


How to use model after training to apply palynofacies:

Open python:

    pip install ultralytics matplotlib pandas openpyxl

    cd C:/bora_toae
    
    jupyter notebook
    
Through Jupyter Notebook interface just follow the script: 'local_palynofacies_analysis_.ipynb'


To able to test palynofacies with different magnifications create folder repository in your environment:

    test_images/
    10x/   ← 10x görüntüler
    20x/   ← 20x görüntüler  
    40x/   ← 40x görüntüler

    python 5_palynofacies.py --weights best.pt --images test_images/10x/
    python 5_palynofacies.py --weights best.pt --images test_images/20x/
    python 5_palynofacies.py --weights best.pt --images test_images/40x/


    ## Hardware Calibration & Spatial Quantification

The post-processing analytical engine (**Phase 2 - `palynofacies_full.py`**) automatically converts pixel-level segmentation masks predicted by the YOLOv26 model into absolute real-world spatial areas ($\mu m^2$).

### Default Hardware Profile
The built-in calibration factors are optimized for the baseline hardware setup used in this study (**Olympus BX51 microscope equipped with a Zeiss Axiocam 105 camera at $2560 \times 1920$ pixels resolution**):
* **10x Objective:** $1.2403\ \mu m/\text{pixel}$
* **20x Objective:** $0.6202\ \mu m/\text{pixel}$
* **40x Objective:** $0.3101\ \mu m/\text{pixel}$

### Image Scaling Workflow
To ensure scale-invariant feature extraction, input images are resized to $640 \times 640$ pixels for the YOLOv26 model. Once instances are segmented, the script dynamically maps the masks back onto the original high-resolution ($2560 \times 1920$) matrix. Absolute area quantification is performed using these original pixels multiplied by the hardware-specific constants.

### How to Adapt for Other Microscope Systems
If you wish to deploy this pipeline on a different optical platform (e.g., Nikon, Leica, Zeiss setups), you must update the calibration constants in `palynofacies_full.py` to maintain quantitative accuracy for absolute measurements:
1. Capture an image of a standard **stage micrometer** using your custom setup.
2. Measure the pixel length of a known physical distance (e.g., $100\ \mu m$) using an image analyzer.
3. Calculate your custom factor: `Calibration Factor = Distance (µm) / Pixel Count`.
4. Modify the calibration variables in the `palynofacies_full.py` script.
5. All necessary code are in the 'microscope.py' script just add your values and update the 'palynofacies_full.py' script.

*Note: For paleoenvironmental proxy analysis via the Tyson Ternary Plot, the pipeline calculates relative area percentages (% Area). Since relative ratios within a single frame are strictly scale-invariant, the distribution output remains mathematically accurate even without modifying the default calibration factors.*
