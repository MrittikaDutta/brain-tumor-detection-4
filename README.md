# Brain Tumor Detection from MRI Scans using Deep Learning  

**Author:** Mrittika Dutta  
**Affiliation:** Computer Science Department  
**Project Context:** Research Project on Biomedical Image Analysis  
**Status:** *In Progress* 🚀  

---

## 🩺 1. Purpose of the Project  

This project explores the application of **Artificial Neural Networks (ANNs)** and **deep learning** methods in the field of **medical image analysis**, focusing on **MRI-based brain tumor classification**.  

The goal is to simulate a research-grade workflow that integrates both **technical proficiency** (machine learning, data engineering, and image processing) and **scientific reasoning** (model interpretability, medical relevance, and ethical evaluation).  

I aim to design a robust pipeline capable of:  
- Preprocessing MRI brain images efficiently,  
- Extracting both **visual and tabular features**,  
- Training state-of-the-art deep learning architectures,  
- Comparing traditional ML approaches (**LightGBM**, **CatBoost**, **XGBoost**) with CNN-based models, and  
- Interpreting model predictions using **explainable AI** tools such as **Grad-CAM** and **SHAP**.  

This project aligns with the spirit of independent academic research—encouraging **autonomy**, **experimentation**, and **critical analysis** of AI methods in healthcare.  

---

## 🧭 2. Strategy: How I Plan to Approach This Problem  

To achieve these objectives, the project follows a structured roadmap:

### **Step 1 — Dataset Exploration**
- Conduct an initial **overview of training and testing MRI scans**.  
- Visualize differences between classes (tumorous vs. non-tumorous regions).  
- Analyze dataset quality, class balance, and MRI slice variability.  

### **Step 2 — Feature Extraction**
- Extract **low-level image descriptors** (texture, entropy, energy, dissimilarity).  
- Build tabular datasets to compare classical ML models against CNNs.  
- Evaluate correlation between handcrafted features and model performance.  

### **Step 3 — Deep Learning Pipeline**
- Implement a **Convolutional Neural Network (CNN)** baseline using **TensorFlow/Keras**.  
- Explore transfer learning using pretrained models (**ResNet**, **EfficientNet**).  
- Integrate **LightGBM**, **CatBoost**, and **XGBoost** as comparative models on feature vectors.  

### **Step 4 — Model Evaluation**
- Apply cross-validation and visualize **ROC-AUC**, **accuracy**, and **confusion matrices**.  
- Introduce **explainability tools** (Grad-CAM visualizations to highlight tumor regions).  

### **Step 5 — Optimization & Deployment (Future Work)**
- Experiment with hybrid **CNN–GBM architectures**.  
- Consider lightweight models for **real-time medical inference**.  
- Package the pipeline as a **reproducible research framework**.  

---

## 🧰 3. Tools & Libraries  

| Category | Libraries |
|-----------|------------|
| **Data Handling** | `pandas`, `numpy`, `os`, `tqdm` |
| **Image Processing** | `OpenCV`, `scikit-image`, `Pillow` |
| **Machine Learning** | `scikit-learn`, `lightgbm`, `xgboost`, `catboost` |
| **Deep Learning** | `TensorFlow`, `Keras`, *(PyTorch planned)* |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Explainability** | `SHAP`, `Grad-CAM` |

---

## 🧪 4. Preliminary Experiments  

The first experiments will focus on analyzing **texture-based features** (contrast, dissimilarity, energy) using the **Grey-Level Co-occurrence Matrix (GLCM)**.  
These handcrafted descriptors can already provide early diagnostic value, serving as a **baseline before deep learning integration**.  

Subsequently, CNN-based models will be trained to compare representational power and robustness to noise or rotation.  

---

## 🧬 5. Expected Outcomes  

- A **reproducible pipeline** for MRI image preprocessing and classification.  
- Quantitative comparison between **traditional ML** and **deep learning** approaches.  
- Insights into **which visual features correlate most strongly** with tumor presence.  
- Visualization of **activation maps** to interpret model attention on medical regions of interest.  

---

## 🧩 6. Future Directions  

- Integration of **3D CNNs** to analyze volumetric MRI data.  
- Experimentation with **self-supervised learning** for limited labeled data.  
- Exploration of **federated learning** for privacy-preserving training.  
- Development of a **web-based demo interface** for interactive inference visualization.  
- Preparation of a **publication-ready notebook** detailing methodology and results.  


---

## 🧾 License  

This project is released under the **MIT License** for open scientific collaboration and educational purposes.  

---
