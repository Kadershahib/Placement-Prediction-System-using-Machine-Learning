# eda.py - Run this first to understand your data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/Placement_Data_Full_Class.csv')

# Basic info
print('Shape:', df.shape)
print('\nColumn Names:', df.columns.tolist())
print('\nFirst 5 rows:')
print(df.head())
print('\nMissing Values:')
print(df.isnull().sum())
print('\nData Types:')
print(df.dtypes)
print('\nStatistics:')
print(df.describe())

# Placement count
print('\nPlacement Count:')
print(df['status'].value_counts())

# Plot 1: Placement Distribution
plt.figure(figsize=(6,4))
df['status'].value_counts().plot(kind='bar', color=['#2E75B6','#C00000'])
plt.title('Placement Distribution')
plt.xlabel('Status')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/placement_dist.png')
plt.show()

# Plot 2: CGPA vs Placement
plt.figure(figsize=(8,5))
sns.boxplot(x='status', y='mba_p', data=df)
plt.title('MBA % by Placement Status')
plt.savefig('plots/cgpa_placement.png')
plt.show()

# Plot 3: Correlation Heatmap
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.title('Feature Correlation')
plt.tight_layout()
plt.savefig('plots/correlation.png')
plt.show()

print('EDA Complete! Check plots folder.')
