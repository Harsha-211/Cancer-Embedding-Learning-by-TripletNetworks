# Cancer-Embedding-Learning-by-TripletNetworks
# 🧠🫁🩺 Multi-Organ Medical Image Embedding using Triplet Loss and KNN

This project uses deep learning to generate discriminative embeddings from medical images of **brain**, **breast**, and **lung**, and applies **K-Nearest Neighbors (KNN)** for classification. It aims to improve understanding of class similarities in medical scans — e.g., benign vs malignant tumors — using a metric learning approach.

---

## 💡 Project Idea

Instead of training a standard classifier, we train a neural network to **learn visual similarity** between images using **Triplet Loss**. This forms a meaningful embedding space where:

- Similar images (same class) are close
- Dissimilar images (different classes) are far

Then we use **KNN** in this space to classify new images.

---

## 📂 Dataset

The dataset consists of ~500 images per class from 10 medical categories:

| Class Label            | Index |
|------------------------|--------|
| Brain-No-tumor         | 0      |
| Brain-Glioma           | 1      |
| Brain-Meningioma       | 2      |
| Brain-Pituitary        | 3      |
| Breast-Benign          | 4      |
| Breast-Malignant       | 5      |
| Breast-Normal          | 6      |
| Lung-Benign            | 7      |
| Lung-Malignant         | 8      |
| Lung-Normal            | 9      |

*Note: Data was preprocessed and resized. No augmentation applied in v1.*

---

## 🔧 Methodology

1. **Triplet Network** trained with `TripletMarginLoss`:
   - Anchor, Positive (same class), Negative (different class)
   - Feature extractor: CNN with intermediate FC layer

2. **Embedding Visualization**:
   - 2D plots with PCA/t-SNE to verify clustering quality

3. **Classification**:
   - Embeddings passed to `KNeighborsClassifier`
   - Evaluated using Accuracy, F1, Precision, Recall

---

## 📊 Results (Initial Run)

- **Accuracy**: ~41%
- **Challenges**:
  - Some classes (e.g., Benign vs Malignant) are visually similar
  - Dataset size small for deep metric learning

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Brain-Pituitary (3) | 0.50 | 0.68 | 0.57 |
| Brain-No-Tumor (0)  | 0.27 | 0.54 | 0.36 |
| Others              | ~0.30 | ~0.20–0.40 | ~0.25–0.38 |

---

## 📌 TODO (Next Steps)

- ✅ Label encoding and embedding setup
- 🔄 Data Augmentation (rotation, noise, zoom)
- 📈 Train with Hard/Semi-hard Triplet Mining
- 🔬 Try ResNet18/50 as backbone
- 🧪 Explore contrastive loss, Siamese nets

---

## 🧠 Inspiration

This project shows how **embedding-based learning** can help in situations where:

- Classes are visually similar (e.g., tumors)
- You want to **generalize to unseen classes** later
- Interpretability of feature space matters

---
