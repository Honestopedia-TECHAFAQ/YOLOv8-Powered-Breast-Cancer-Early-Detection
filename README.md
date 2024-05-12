# YOLOv8 for Breast Cancer Detection on Mammography Images

This repository contains code for building and training a YOLOv8 model for early detection of breast cancer on mammography images. The goal is to automate the identification of cancerous regions in mammograms, ultimately improving the accuracy of breast cancer diagnoses.

## Requirements

- Python 3
- PyTorch
- torchvision
- scikit-learn
- PIL (Python Imaging Library)

You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
data_dir/
├── train/
│   ├── cancerous/
│   └── non_cancerous/
└── val/
    ├── cancerous/
    └── non_cancerous/
yolov8-breast-cancer-detection/
├── data/
│   ├── train/
│   │   ├── cancerous/
│   │   └── non_cancerous/
│   └── val/
│       ├── cancerous/
│       └── non_cancerous/
├── model.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── LICENSE
