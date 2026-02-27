# train.py - Complete Model Training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP A: Load and Clean Data
# ============================================
df = pd.read_csv('data/Placement_Data_Full_Class.csv')
print('Original shape:', df.shape)

# Drop salary column (only for placed students)
# We train classification separately
df_class = df.copy()

# Encode target variable
df_class['target'] = (df_class['status'] == 'Placed').astype(int)

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']

for col in cat_cols:
    if col in df_class.columns:
        df_class[col] = le.fit_transform(df_class[col].astype(str))

# Drop non-numeric or unnecessary columns
drop_cols = ['status', 'salary', 'sl_no']
drop_cols = [c for c in drop_cols if c in df_class.columns]
df_class.drop(columns=drop_cols, inplace=True)

print('Cleaned shape:', df_class.shape)
print('Class balance:\n', df_class['target'].value_counts())

# ============================================
# STEP B: Split Data
# ============================================
X = df_class.drop('target', axis=1)
y = df_class['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP C: Train Multiple Models
# ============================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
}

results = {}
print('\n--- Model Comparison ---')
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    results[name] = {'accuracy': acc, 'cv_score': cv_score}
    print(f'{name}: Test Acc={acc:.4f}, CV Score={cv_score:.4f}')

# ============================================
# STEP D: Best Model - XGBoost
# ============================================
best_model = models['XGBoost']
y_pred = best_model.predict(X_test_scaled)

print('\n--- Best Model: XGBoost ---')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Placed','Placed'],
            yticklabels=['Not Placed','Placed'])
plt.title('Confusion Matrix - XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.show()

# Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
feat_imp.sort_values(ascending=True).plot(kind='barh', figsize=(8,6), color='#2E75B6')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.show()

# Model Comparison Bar Chart
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
plt.figure(figsize=(8,4))
plt.bar(names, accs, color=['#2E75B6','#70AD47','#ED7D31','#C00000'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
for i, v in enumerate(accs):
    plt.text(i, v+0.005, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/model_comparison.png')
plt.show()

# ============================================
# STEP E: Save Model
# ============================================
joblib.dump(best_model, 'models/placement_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

print('\nModel saved to models/placement_model.pkl')
print('Training Complete!')