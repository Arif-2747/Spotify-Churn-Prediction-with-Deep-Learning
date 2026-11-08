import sys
import os
import warnings

# Configure environment before imports
os.environ["KERAS_HOME"] = os.path.abspath(".keras")
os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore')

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import tensorflow as tf

print("=" * 65)
print(" SPOTIFY CHURN PREDICTION - SAVED MODEL EVALUATION")
print("=" * 65)

# 1. Load data
df = pd.read_csv('spotify_churn_dataset.csv')
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

cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

if cat_cols:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
    X = pd.concat([X[num_cols], encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Load model and scaler
scaler = joblib.load('models/feature_scaler.pkl')
model = tf.keras.models.load_model('models/mlp_churn_model.h5')

X_test_scaled = scaler.transform(X_test)
y_pred_proba = model.predict(X_test_scaled, verbose=0).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

# 3. Calculate Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 CLASSIFICATION PERFORMANCE (Held-Out Test Set: n={len(y_test)}):")
print(f"  • Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"  • Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"  • Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"  • F1-Score:  {f1:.4f}")
print(f"  • AUC-ROC:   {auc:.4f}")

print(f"\n🔍 CONFUSION MATRIX:")
print(f"                 Predicted: Stayed (0)    Predicted: Churned (1)")
print(f"  Actual Stayed:        {tn:6d} (TN)              {fp:6d} (FP)")
print(f"  Actual Churned:       {fn:6d} (FN)              {tp:6d} (TP)")

print(f"\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)'], digits=4))

# 4. Business Impact Analysis
cost_per_campaign = 10
revenue_per_retention = 50
revenue_lost_per_churn = 240

total_targeted = tp + fp
campaign_cost = total_targeted * cost_per_campaign
revenue_saved = tp * revenue_per_retention
revenue_lost = fn * revenue_lost_per_churn
net_roi = revenue_saved - campaign_cost - revenue_lost
roi_pct = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0

print("💼 BUSINESS & FINANCIAL IMPACT ANALYSIS:")
print(f"  • Total test set users:                {len(y_test):,}")
print(f"  • Actual churners:                     {sum(y_test):,}")
print(f"  • Users targeted for retention:        {total_targeted:,}")
print(f"  • True Positives (saved opportunities): {tp:,}")
print(f"  • False Alarms (budget wasted on FP):  {fp:,}")
print(f"  • Missed Churners (lost to competitors): {fn:,}")
print(f"  --------------------------------------------------")
print(f"  • Campaign Cost ($10 / target):        ${campaign_cost:,.2f}")
print(f"  • Revenue Saved ($50 / retention):     ${revenue_saved:,.2f}")
print(f"  • Revenue Lost ($240 / missed churn):  ${revenue_lost:,.2f}")
print(f"  • Net Financial Impact / ROI:          ${net_roi:,.2f}")
print(f"  • ROI Percentage:                      {roi_pct:.1f}%")
print(f"  • Decision Recommendation:             {'✅ PROCEED WITH DEPLOYMENT' if net_roi > 0 else '⚠️ CAUTION / PILOT ONLY'}")
print("=" * 65)

