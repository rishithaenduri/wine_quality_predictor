import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Generate realistic wine quality dataset
np.random.seed(42)
n_samples = 2000

def generate_wine_data(n):
    data = []
    for _ in range(n):
        quality = np.random.choice([3,4,5,6,7,8,9], p=[0.01,0.04,0.25,0.38,0.22,0.08,0.02])
        
        base = (quality - 3) / 6  # normalized quality 0-1
        
        # Features correlated with quality
        alcohol = np.random.normal(9 + base*3.5, 0.8)
        volatile_acidity = np.random.normal(0.7 - base*0.4, 0.12)
        sulphates = np.random.normal(0.45 + base*0.35, 0.1)
        citric_acid = np.random.normal(0.2 + base*0.2, 0.1)
        fixed_acidity = np.random.normal(7.5 + base*1.5, 0.9)
        residual_sugar = np.random.normal(2.5 + np.random.exponential(1.5), 1.2)
        chlorides = np.random.normal(0.08 - base*0.03, 0.02)
        free_so2 = np.random.normal(25 + base*10, 8)
        total_so2 = np.random.normal(100 + base*20, 20)
        density = np.random.normal(0.997 - base*0.003, 0.001)
        pH = np.random.normal(3.3 - base*0.1, 0.15)
        
        row = {
            'fixed_acidity': round(max(4, min(15, fixed_acidity)), 2),
            'volatile_acidity': round(max(0.1, min(1.5, volatile_acidity)), 3),
            'citric_acid': round(max(0, min(1, citric_acid)), 2),
            'residual_sugar': round(max(0.5, min(20, residual_sugar)), 2),
            'chlorides': round(max(0.01, min(0.6, chlorides)), 3),
            'free_sulfur_dioxide': round(max(1, min(70, free_so2)), 1),
            'total_sulfur_dioxide': round(max(10, min(280, total_so2)), 1),
            'density': round(max(0.99, min(1.004, density)), 4),
            'pH': round(max(2.8, min(4.0, pH)), 2),
            'sulphates': round(max(0.2, min(2.0, sulphates)), 2),
            'alcohol': round(max(7, min(15, alcohol)), 1),
            'quality': quality
        }
        data.append(row)
    return pd.DataFrame(data)

df = generate_wine_data(n_samples)

# Bin quality into 3 classes: Low (3-4), Medium (5-6), High (7-9)
def categorize(q):
    if q <= 4: return 0  # Low
    elif q <= 6: return 1  # Medium
    else: return 2  # High

df['quality_class'] = df['quality'].apply(categorize)

X = df.drop(['quality', 'quality_class'], axis=1)
y_class = df['quality_class']
y_score = df['quality']

feature_names = X.columns.tolist()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class)

# Train GBC for classification
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)

print("=== Wine Quality Classifier ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"CV Score: {cross_val_score(gbc, X_scaled, y_class, cv=5).mean():.4f}")
print()
print(classification_report(y_test, y_pred, target_names=['Low (3-4)', 'Medium (5-6)', 'High (7-9)']))

# Feature importances
importances = dict(zip(feature_names, gbc.feature_importances_.tolist()))
print("\nFeature Importances:")
for k, v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")

# Save model, scaler, feature info
joblib.dump(gbc, '/home/claude/wine_model.pkl')
joblib.dump(scaler, '/home/claude/wine_scaler.pkl')

model_info = {
    'feature_names': feature_names,
    'feature_importances': importances,
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'classes': ['Low (3-4)', 'Medium (5-6)', 'High (7-9)'],
    'class_labels': [0, 1, 2],
    'feature_ranges': {
        col: {'min': round(float(df[col].min()), 3), 
              'max': round(float(df[col].max()), 3),
              'mean': round(float(df[col].mean()), 3)}
        for col in feature_names
    }
}

with open('/home/claude/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\nModel saved successfully!")
print(f"Files: wine_model.pkl, wine_scaler.pkl, model_info.json")
