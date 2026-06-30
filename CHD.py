# -*- coding: utf-8 -*-
"""
Coronary Heart Disease Risk-Stratification Pipeline
Converted for standard local execution in VS Code.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

# ==========================================
# 1. DATA LOADING & INITIAL EXPLORATION
# ==========================================
print("--- Loading Dataset ---")
# Ensure 'framingham.csv' is in the same directory as this script in VS Code
df = pd.read_csv(r'C:\Users\Jayita Diwan\Downloads\CHD\framingham.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:\n", df.isnull().sum())

# ==========================================
# 2. PREPROCESSING & CLEANING
# ==========================================
print("\n--- Preprocessing Data ---")
# Drop the target variable for True Unsupervised Learning
df_unsupervised = df.drop('TenYearCHD', axis=1)

# Handle Missing Data (Imputation using Median)
imputer = SimpleImputer(strategy='median')
df_clean = pd.DataFrame(imputer.fit_transform(df_unsupervised), columns=df_unsupervised.columns)

print("Missing values after imputation:\n", df_clean.isnull().sum())

# Scale the data (Mean = 0, Std Dev = 1)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

print("\nScaled data sample:")
print(df_scaled.head())

# ==========================================
# 3. ELBOW METHOD FOR OPTIMAL K
# ==========================================
print("\n--- Running Elbow Method ---")
inertia_values = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_values, marker='o', linestyle='--')
plt.title('The Elbow Method: Finding Optimal Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show(block=False)  # block=False allows the script to keep running while showing the plot

# ==========================================
# 4. K-MEANS CLUSTERING (K=2) & INSIGHTS
# ==========================================
print("\n--- Fitting Final K-Means (K=2) ---")
optimal_k = 2
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(df_scaled)

# Add cluster labels back to the original dataframe
df['Patient_Profile'] = cluster_labels

# Analyze profiles
profile_analysis = df.groupby('Patient_Profile').mean()
print("\nPatient Profile Analysis (Transposed):")
print(profile_analysis.T)

# Unsupervised CHD Rates
chd_rates = df.groupby('Patient_Profile')['TenYearCHD'].mean() * 100
print("\nPercentage of patients who developed CHD per profile:")
print(chd_rates)

# ==========================================
# 5. SILHOUETTE SCORE EVALUATION
# ==========================================
print("\n--- Calculating Silhouette Scores ---")
k_range_sil = range(2, 11)
silhouette_scores = []

for k in k_range_sil:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels_test = kmeans_test.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, cluster_labels_test)
    silhouette_scores.append(score)
    print(f"For K={k}, the Silhouette Score is: {score:.4f}")

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_range_sil, silhouette_scores, marker='o', color='teal', linestyle='-', linewidth=2)
plt.title('Finding the Optimal K using Silhouette Scores')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score (Higher is Better)')
plt.xticks(k_range_sil)
plt.grid(True)
plt.show(block=False)

# ==========================================
# 6. VISUAL COMPARISON: PREDICTED VS REALITY
# ==========================================
machine_counts = df['Patient_Profile'].value_counts().sort_index()
real_counts = df['TenYearCHD'].value_counts().sort_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Chart 1: Unsupervised Grouping
colors_machine = plt.cm.viridis(np.linspace(0, 1, len(machine_counts)))
bars1 = ax1.bar([f'Cluster {i}' for i in machine_counts.index], machine_counts, color=colors_machine, alpha=0.85)
ax1.set_title("What the Machine Predicted\n(Unsupervised Grouping)", fontsize=13, pad=10, weight='bold')
ax1.set_ylabel("Total Number of Patients", fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Chart 2: Ground Truth
colors_real = ['#2b7bba', '#e74c3c']
bars2 = ax2.bar(['0\n(Remained Healthy)', '1\n(Developed CHD)'], real_counts, color=colors_real, alpha=0.85)
ax2.set_title("What was in the Dropped Column\n(The Ground-Truth Reality)", fontsize=13, pad=10, weight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# Add text labels on bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 50, f'{int(yval):,}', ha='center', va='bottom', weight='bold')

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 50, f'{int(yval):,}', ha='center', va='bottom', weight='bold')

plt.suptitle("Side-by-Side Comparison: Machine Grouping vs. Real Medical Outcomes", fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.show(block=False)

# ==========================================
# 7. ARTIFICIAL HIGH-K ACCURACY TEST (K=10)
# ==========================================
print("\n--- Testing Artificial Accuracy (K=10) ---")
df_test = pd.read_csv(r'C:\Users\Jayita Diwan\Downloads\CHD\framingham.csv')
X_raw = df_test.drop('TenYearCHD', axis=1)
y_true = df_test['TenYearCHD']

X_clean = SimpleImputer(strategy='median').fit_transform(X_raw)
X_scaled_test = StandardScaler().fit_transform(X_clean)

high_k = 10
kmeans_high = KMeans(n_clusters=high_k, random_state=42, n_init=10)
high_labels = kmeans_high.fit_predict(X_scaled_test)

df_mapping = pd.DataFrame({'Cluster': high_labels, 'Truth': y_true})
cluster_predictions = df_mapping.groupby('Cluster')['Truth'].apply(lambda x: x.mode()[0])
y_pred_high = [cluster_predictions[cluster] for cluster in high_labels]

new_acc = accuracy_score(y_true, y_pred_high) * 100
print(f"Artificial Accuracy with K=10: {new_acc:.2f}%")

# ==========================================
# 8. CORRECTED MODEL ALIGNMENT & REPORT
# ==========================================
print("\n--- Running Corrected Model Realignment ---")
kmeans_final_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
raw_labels = kmeans_final_2.fit_predict(X_scaled_test)

mapping_df = pd.DataFrame({'Cluster': raw_labels, 'Truth': y_true})
high_risk_cluster = mapping_df.groupby('Cluster')['Truth'].mean().idxmax()
low_risk_cluster = 0 if high_risk_cluster == 1 else 1

y_pred_corrected = pd.Series(raw_labels).map({low_risk_cluster: 0, high_risk_cluster: 1})
final_accuracy = accuracy_score(y_true, y_pred_corrected) * 100

print(f"   CORRECTED MODEL ACCURACY SCORE: {final_accuracy:.2f}%")
print("\nComplete Unsupervised Classification Report:")
print(classification_report(y_true, y_pred_corrected, target_names=['Healthy (0)', 'CHD (1)']))

# Keep plots open at the end of the script execution
plt.show()
