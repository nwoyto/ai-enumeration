# SpaceNet Building Detection

This project is organized for building detection using the SpaceNet dataset. It includes code for model training, data preprocessing, and evaluation, as well as scripts and notebooks for end-to-end workflow.

## Directory Structure

```
spacenet-building-detection/
├── code/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── requirements.txt
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_sagemaker_training_launcher.ipynb
│   └── (Optional) 03_model_evaluation.ipynb
├── data/
│   ├── raw/
│   │   ├── SN2_buildings_train_AOI_2_Vegas.tar.gz
│   │   └── AOI_2_Vegas_test_public.tar.gz
│   └── processed/
│       ├── AOI_2_Vegas_Train/
│       │   ├── RGB-PanSharpen/
│       │   ├── MUL-PanSharpen/
│       │   └── geojson/
│       │       └── buildings/
│       ├── AOI_2_Vegas_Test_Public/
│       └── processed_masks/
│           ├── train/
│           │   └── masks/
│           └── val/
│               └── masks/
│           └── test/ (Optional)
│               └── masks/
├── scripts/
│   ├── preprocess_data.py
│   └── deploy_endpoint.py
└── README.md
```


