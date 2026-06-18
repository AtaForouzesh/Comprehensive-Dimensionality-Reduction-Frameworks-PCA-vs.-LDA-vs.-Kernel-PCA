# Comprehensive-Dimensionality-Reduction-Frameworks-PCA-vs.-LDA-vs.-Kernel-PCA
A comprehensive machine learning benchmark comparing Unsupervised Linear (PCA), Supervised Linear (LDA), and Non-Linear (Kernel PCA) dimensionality reduction techniques.

## 📌 Project Overview
This repository features an advanced data science benchmark evaluation comparing three mathematical approaches to dimensionality reduction and feature space projection:
1. **Principal Component Analysis (PCA)** – Unsupervised Linear Variance Optimization
2. **Linear Discriminant Analysis (LDA)** – Supervised Linear Class Maximization  
3. **Kernel PCA (KPCA)** – Unsupervised Non-Linear High-Dimensional Inversion

---

## 🧮 Theoretical Framework Foundations

### 1. Principal Component Analysis (PCA)
* **Paradigm:** Unsupervised Linear.
* **Mechanism:** Computes the eigenvectors and eigenvalues of the data covariance matrix. It maps variables orthogonally into a lower-dimensional hyperplane to track the directions of maximum descriptive variance without observing target outputs.

### 2. Linear Discriminant Analysis (LDA)
* **Paradigm:** Supervised Linear.
* **Mechanism:** Maximizes the explicit distance boundaries among targets. It optimizes the operational ratio of **Between-Class Variance** to **Within-Class Variance**.

### 3. Kernel PCA (KPCA)
* **Paradigm:** Unsupervised Non-Linear.
* **Mechanism:** Uses the **"Kernel Trick"** (Radial Basis Function Kernel) to project non-linearly separable raw attributes into an arbitrary, higher-dimensional Hilbert feature space where the points become linearly distinct. It then extracts orthogonal principal configurations without executing expensive inner-product scaling computations.

---

## 📊 Evaluation Matrix

| Pipeline Workspace | Reduction Paradigm | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| **Nutritional Profiling (Pizza)** | PCA (Linear, Unsupervised) | *76%* | *75%* | *74%* |
| **Morphological Taxonomy (Iris)** | LDA (Linear, Supervised) | *97%* | *96%* | *96%* |
| **Diagnostic Analytics (Diabetes)**| Kernel PCA (Non-Linear) | *69%* | *67%* | *68%* |
