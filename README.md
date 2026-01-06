# 🛰️ Satellite Imagery–Based Property Valuation  
### CDC X Yhills Open Projects 2025–2026  
**Data Science Problem Statement**

**Author:** Kumar Manas, IIT Roorkee  
**Initiative:** Career Development Cell (CDC), IIT Roorkee  

---

## 📌 Project Overview

Traditional property valuation models rely primarily on structured housing attributes such as size, number of rooms, and location coordinates. While effective, these features often fail to capture **neighborhood-level visual context**, including surrounding infrastructure, green cover, and urban layout.

This project builds a **multimodal machine learning pipeline** that integrates:

- 📊 **Tabular housing data**
- 🛰️ **Satellite imagery acquired using the Mapbox Static Images API**

to predict residential property prices more holistically.

A pretrained **Convolutional Neural Network (CNN)** is used to extract visual features from satellite images, which are then fused with tabular features and passed to a **gradient-boosted regression model**.

---

## 🎯 Objectives

- Predict residential property prices using a **multimodal approach**
- Programmatically acquire satellite images using **latitude–longitude coordinates**
- Extract meaningful visual representations using **deep learning**
- Combine tabular and image-based features for **regression**
- Interpret visual features using **Grad-CAM explainability**
- Build a **clean, reproducible, end-to-end pipeline**

---

## 📊 Dataset

### Tabular Data

- **Source:** King County House Sales Dataset (Kaggle)
- **Size:** 16,209 properties

**Key Features**
- Living area, lot size  
- Bedrooms, bathrooms  
- Construction grade and condition  
- Latitude & longitude  
- Neighborhood statistics (`sqft_living15`, `sqft_lot15`)  
- Waterfront indicator  

- **Target Variable:** Property price

---

### Satellite Imagery

- Acquired using the **Mapbox Static Images API**
- Centered at each property’s geographic coordinates
- Captures:
  - Neighborhood layout  
  - Road networks  
  - Green cover  
  - Land-use patterns  

---

### Sampling Strategy

- A **stratified sample (~5,000 properties)** was created from the training set using **price quantiles** to ensure representation across all price ranges
- The **full test dataset** was retained for final inference

---

## 🧠 Methodology

### 1. Exploratory Data Analysis (EDA)

- Analyzed price distribution and skewness
- Studied relationships between price and:
  - Structural features  
  - Neighborhood attributes  
  - Geographic coordinates  
- Identified strong **spatial clustering effects**, motivating the use of satellite imagery

---

### 2. Tabular Baseline Models

- **Linear Regression** (baseline)
- **XGBoost Regression** (non-linear baseline)

---

### 3. Image Feature Extraction

- Satellite images processed using **ResNet-18** (pretrained on ImageNet)
- Final classification layer removed
- Extracted **512-dimensional image embeddings**
- CNN weights frozen to prevent overfitting

---

### 4. Multimodal Fusion

- Image embeddings concatenated with tabular features
- Final regression performed using **XGBoost**

---

### 5. Explainability

- **Grad-CAM** applied to the CNN encoder
- Visualized attention patterns for:
  - Low-priced properties  
  - High-priced properties  
- Validated that the model focuses on **semantically meaningful regions**

---

## 📈 Results

| Model                          | RMSE ($) | R²   |
|--------------------------------|----------|------|
| Tabular XGBoost                |  117,558 | 0.8898 |
| Multimodal (Tabular + Satellite) |  139,433 |  0.8593 |

**Key Observations**
- The multimodal model demonstrates **competitive performance**
- Incorporates **visual neighborhood context** not present in tabular data
- Grad-CAM confirms **meaningful spatial attention patterns**
- Final model chosen for its **representational richness and interpretability**

---

## 📂 Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/
│   │   ├── train_sampled.csv
│   │   ├── image_embeddings.npy
│   │   └── image_ids.npy
│   └── images/
│       ├── train/
│       └── test/
│
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb
│   ├── 02_tabular_baseline.ipynb
│   ├── 02a_fetch_test_images.ipynb
│   ├── 03_image_eda.ipynb
│   ├── 04_image_encoder.ipynb
│   ├── 05_multimodal_.ipynb
│   ├── 06_gradcam.ipynb
│   └── 07_inference.ipynb
│
├── models/
│   └── xgb_multimodal.pkl
│
├── submissions/
│   └── 23119016_final.csv
│   └── 23119016_report.pdf
├── .env.example
├── requirements.txt
└── README.md

🔐 API Key Handling

Satellite images are downloaded using the Mapbox Static Images API.
To keep credentials secure, API keys are stored in a .env file.
