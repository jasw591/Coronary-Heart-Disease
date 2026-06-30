# Coronary Heart Disease Risk-Stratification Pipeline

This repository contains an end-to-end unsupervised machine learning pipeline designed to risk-stratify patients for Coronary Heart Disease (CHD) using the **Framingham Heart Study** dataset. By dropping clinical labels and clustering profiles, the model uncovers hidden patterns in patient health metrics and aligns them against real-world medical outcomes.

---

# Pipeline Overview & Methodology

1. **Data Preprocessing & Imputation:** Missing values are imputed using median values to handle anomalies without skewing distributions, followed by `StandardScaler` normalization.
2. **Cluster Optimization:** The model evaluates optimal groups ($K$) dynamically using the **Elbow Method** and **Silhouette Scoring**.
3. **Unsupervised Clustering:** Executes final profile groupings using `K-Means` ($K=2$).
4. **Supervised Realignment:** Realigns machine-generated profiles with true clinical outcomes (`TenYearCHD`) to measure predictive baseline mapping.

---

#  Model Visualizations

The script automatically generates and saves the following performance analytics plots:

# 1. Optimal Cluster Evaluation (Elbow Method)
Used to find the optimal inflection point for patient clustering based on Within-Cluster Sum of Squares (Inertia).
![Elbow Method Curve](Elbow%20Method%20Curve.png)
# 2. Silhouette Score Optimization
Measures how cohesive and well-separated the patient profile clusters are across different $K$ values.
![Silhouette Scores Plot](Silhouette%20Scores%20Plot.png)

# 3. Machine Groupings vs. Ground-Truth Realities
A side-by-side comparative analysis validating unsupervised cluster distribution against real clinical medical outcomes.
![Model vs Reality Comparison](Model%20vs%20Reality%20Comparison.png)

---

# Performance Metrics & Results

After mapping the unsupervised clusters back to the real ground-truth labels, the realigned pipeline achieved the following validation score:

* **Corrected Model Accuracy:** `68.56%`

# Classification Report:
```text
              precision    recall  f1-score   support

 Healthy (0)       0.87      0.73      0.79      3594
     CHD (1)       0.20      0.37      0.26       644

    accuracy                           0.68      4238
   macro avg       0.54      0.55      0.53      4238
weighted avg       0.77      0.68      0.71      4238
