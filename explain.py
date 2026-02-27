# explain.py - SHAP Explainability
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load model and data
model = joblib.load('models/placement_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

df = pd.read_csv('data/Placement_Data_Full_Class.csv')
df['target'] = (df['status'] == 'Placed').astype(int)

le = LabelEncoder()
cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

drop_cols = ['status', 'salary', 'sl_no', 'target']
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols)
X_scaled = scaler.transform(X)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# Summary plot - shows most important features
plt.figure()
shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('plots/shap_summary.png', bbox_inches='tight')
plt.show()

# Bar plot
shap.summary_plot(shap_values, X, plot_type='bar',
                  feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('plots/shap_bar.png', bbox_inches='tight')
plt.show()

print('SHAP plots saved!')
print('These plots tell you WHY each student gets placed or not.')

