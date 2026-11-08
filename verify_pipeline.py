import sys
import os
import warnings

# Suppress library warnings
warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Ensure UTF-8 output
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 65)
print(" PIPELINE & SMOTE VERIFICATION RUN")
print("=" * 65)

# 1. Load dataset
df = pd.read_csv('spotify_churn_dataset.csv')
print(f"[1] Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

total_count = len(df)
churn_counts = df['Churn'].value_counts().to_dict()
print(f"    - Overall class distribution (0 = Stayed, 1 = Churned): {churn_counts}")
print(f"    - Overall churn rate: {df['Churn'].mean():.2%}")

# 2. Preprocessing & Feature Engineering
for col in ['subscription_type', 'device_type', 'gender', 'country', 'favorite_genre']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df = df.dropna()
if 'user_id' in df.columns:
    df = df.drop('user_id', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn']

if 'listening_hours' in X.columns and 'subscription_length' in X.columns:
    X['engagement_ratio'] = X['listening_hours'] / (X['subscription_length'] + 1)

if 'monthly_charges' in X.columns and 'listening_hours' in X.columns:
    X['value_per_hour'] = X['monthly_charges'] / (X['listening_hours'] + 1)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

if categorical_features:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(categorical_features),
        index=X.index
    )
    X = pd.concat([X[numerical_features], encoded_df], axis=1)

print(f"\n[2] Feature Engineering & Encoding Complete:")
print(f"    - Cleaned feature matrix X shape: {X.shape}")

# 3. Train-Test Split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Stratified Train-Test Split (80/20):")
print(f"    - Training set: {X_train.shape[0]} samples")
print(f"    - Test set:     {X_test.shape[0]} samples")
print(f"    - X_train class counts: {y_train.value_counts().to_dict()}")
print(f"    - X_test class counts:  {y_test.value_counts().to_dict()} (untouched for testing)")

# 4. Apply SMOTE
print(f"\n[4] Applying SMOTE to Training Set:")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

balanced_counts = pd.Series(y_train_balanced).value_counts().to_dict()
print(f"    - Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"    - After SMOTE:  {balanced_counts}")
print(f"    - Synthetic samples generated: {len(y_train_balanced) - len(y_train)}")

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
print(f"\n[5] Feature Scaling (StandardScaler):")
print(f"    - Final X_train_scaled shape: {X_train_scaled.shape}")
print(f"    - Final X_test_scaled shape:  {X_test_scaled.shape}")

# Verification assertions
assert balanced_counts[0] == balanced_counts[1], "Classes are not balanced after SMOTE!"
assert len(X_test) == len(y_test), "Test set size mismatch!"
assert X_train_scaled.shape[1] == X_test_scaled.shape[1], "Feature count mismatch!"

print("\n" + "=" * 65)
print(" VERIFICATION RESULT: PASSED (SMOTE successfully validated)")
print("=" * 65)
