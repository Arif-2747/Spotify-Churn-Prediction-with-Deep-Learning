# Enhanced Spotify Churn Prediction Pipeline with a Deep Learning Focus
# ====================================================================================

import pandas as pd
import numpy as np
import subprocess
import sys
import warnings

# Configure output encoding to support emoji/unicode characters on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import os

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ====================================================================================
# UTILITY FUNCTIONS
# ====================================================================================

def install_and_import(package):
    """Install package if not available"""
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}.")

def print_section(title):
    """Print formatted section headers"""
    print("\n" + "=" * 60)
    print(f" {title.upper()} ")
    print("=" * 60)

def calculate_business_metrics(cm, cost_per_campaign=10, revenue_per_retention=50, revenue_lost_per_churn=240):
    """Calculate comprehensive business metrics from a confusion matrix"""
    tn, fp, fn, tp = cm.ravel()
    
    # Campaign costs and benefits
    total_targeted = tp + fp
    campaign_cost = total_targeted * cost_per_campaign
    revenue_saved = tp * revenue_per_retention
    revenue_lost = fn * revenue_lost_per_churn
    
    # Net ROI calculation
    net_roi = revenue_saved - campaign_cost - revenue_lost
    roi_percentage = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0
    
    return {
        'total_targeted': total_targeted,
        'campaign_cost': campaign_cost,
        'revenue_saved': revenue_saved,
        'revenue_lost': revenue_lost,
        'net_roi': net_roi,
        'roi_percentage': roi_percentage,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
    }

# Check for required libraries
required_packages = ['tensorflow', 'scikit-learn', 'pandas', 'numpy', 'imbalanced-learn']
for package in required_packages:
    install_and_import(package)

# ====================================================================================
# STEP 1: ENHANCED DATA PREPROCESSING
# ====================================================================================

print_section("STEP 1: ENHANCED DATA PREPROCESSING")

try:
    df = pd.read_csv('spotify_churn_dataset.csv')
    print(f"✓ Dataset loaded successfully: {df.shape}")
except FileNotFoundError:
    print("❌ Error: spotify_churn_dataset.csv not found.")
    print("Creating synthetic dataset for demonstration...")
    
    # Create synthetic Spotify churn data
    np.random.seed(42)
    n_samples = 10000
    
    df = pd.DataFrame({
        'user_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['M', 'F', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'Other'], n_samples, p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.15]),
        'subscription_type': np.random.choice(['Free', 'Premium', 'Family'], n_samples, p=[0.6, 0.3, 0.1]),
        'monthly_charges': np.random.normal(12, 5, n_samples).clip(0, 30),
        'total_charges': np.random.normal(150, 100, n_samples).clip(0, 1000),
        'subscription_length': np.random.randint(1, 60, n_samples),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet', 'Smart_TV'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'listening_hours': np.random.normal(25, 15, n_samples).clip(0, 100),
        'playlist_count': np.random.randint(0, 200, n_samples),
        'favorite_genre': np.random.choice(['Pop', 'Rock', 'Hip-Hop', 'Jazz', 'Classical', 'Other'], n_samples),
        'skips_per_session': np.random.normal(8, 5, n_samples).clip(0, 50),
        'premium_features_used': np.random.randint(0, 10, n_samples)
    })
    
    # Create target variable with realistic patterns
    churn_prob = (
        0.1 +  # Base churn rate
        (df['subscription_type'] == 'Free') * 0.3 +  # Free users more likely to churn
        (df['listening_hours'] < 10) * 0.4 +  # Low engagement increases churn
        (df['subscription_length'] < 6) * 0.2 +  # New users more likely to churn
        np.random.normal(0, 0.1, n_samples)  # Random noise
    ).clip(0, 1)
    
    df['Churn'] = np.random.binomial(1, churn_prob, n_samples)
    print(f"✓ Synthetic dataset created: {df.shape}")

# Data cleaning and preprocessing
print("\n📊 Data Overview:")
print(f"- Dataset shape: {df.shape}")
print(f"- Missing values: {df.isnull().sum().sum()}")
print(f"- Churn rate: {df['Churn'].mean():.2%}")

# Handle missing values
for col in ['subscription_type', 'device_type', 'gender', 'country', 'favorite_genre']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Remove rows with missing values in critical columns
df = df.dropna()

# Drop user_id as it's not predictive
if 'user_id' in df.columns:
    df = df.drop('user_id', axis=1)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"\n✓ Data cleaned - Final shape: {X.shape}")

# Advanced feature engineering
print("\n🔧 Feature Engineering:")

# Create interaction features
if 'listening_hours' in X.columns and 'subscription_length' in X.columns:
    X['engagement_ratio'] = X['listening_hours'] / (X['subscription_length'] + 1)
    print("- Created engagement ratio feature")

if 'monthly_charges' in X.columns and 'listening_hours' in X.columns:
    X['value_per_hour'] = X['monthly_charges'] / (X['listening_hours'] + 1)
    print("- Created value per hour feature")

# Encode categorical variables
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

if categorical_features:
    print(f"- Encoding {len(categorical_features)} categorical features")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(categorical_features),
        index=X.index
    )
    X = pd.concat([X[numerical_features], encoded_df], axis=1)

print(f"✓ Final feature count: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance with SMOTE
print("\n⚖️ Handling Class Imbalance:")
print(f"- Original class distribution: {y_train.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"- After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# Scale features for the neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# ====================================================================================
# STEP 2: MULTI-LAYER PERCEPTRON (MLP) MODEL
# ====================================================================================

print_section("STEP 2: DEEP LEARNING MODEL")

def build_enhanced_mlp(input_dim, dropout_rate=0.3, l2_reg=0.001):
    """Build an enhanced MLP with regularization and batch normalization"""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    return model

print("🧠 Building Enhanced MLP model...")
mlp_model = build_enhanced_mlp(X_train_scaled.shape[1])

# Enhanced callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
]

print("🔄 Training MLP model...")
history = mlp_model.fit(
    X_train_scaled, y_train_balanced,
    epochs=150,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=0
)

# Evaluate MLP
y_pred_mlp_proba = mlp_model.predict(X_test_scaled, verbose=0)
y_pred_mlp = (y_pred_mlp_proba > 0.5).astype(int).flatten()

mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_precision = precision_score(y_test, y_pred_mlp)
mlp_recall = recall_score(y_test, y_pred_mlp)
mlp_f1 = f1_score(y_test, y_pred_mlp)
mlp_auc = roc_auc_score(y_test, y_pred_mlp_proba)

print(f"✓ MLP Training Complete!")
print(f"   Accuracy: {mlp_accuracy:.4f} | Precision: {mlp_precision:.4f} | Recall: {mlp_recall:.4f} | F1: {mlp_f1:.4f} | AUC: {mlp_auc:.4f}")

# ====================================================================================
# STEP 3: COMPREHENSIVE EVALUATION & BUSINESS INSIGHTS
# ====================================================================================

print_section("STEP 3: COMPREHENSIVE EVALUATION & BUSINESS INSIGHTS")

# Detailed analysis of the MLP model
print(f"\n🔍 DETAILED ANALYSIS - ENHANCED MLP:")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_mlp)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"Actual    No Churn  Churn")
print(f"No Churn    {tn:6d}  {fp:5d}")
print(f"Churn       {fn:6d}  {tp:5d}")

# Business metrics calculation
business_metrics = calculate_business_metrics(cm)

print(f"\n💼 BUSINESS IMPACT ANALYSIS:")
print(f"- Total customers in test set: {len(y_test):,}")
print(f"- Actual churners: {sum(y_test):,}")
print(f"- Customers targeted for retention: {business_metrics['total_targeted']:,}")
print(f"- Successfully identified churners: {tp:,}")
print(f"- False alarms (non-churners targeted): {fp:,}")
print(f"- Missed churners: {fn:,}")

print(f"\n💰 FINANCIAL IMPACT:")
print(f"- Campaign cost: ${business_metrics['campaign_cost']:,.2f}")
print(f"- Revenue saved (retained customers): ${business_metrics['revenue_saved']:,.2f}")
print(f"- Revenue lost (missed churners): ${business_metrics['revenue_lost']:,.2f}")
print(f"- Net ROI: ${business_metrics['net_roi']:,.2f}")
print(f"- ROI percentage: {business_metrics['roi_percentage']:.1f}%")

roi_status = "✅ POSITIVE" if business_metrics['net_roi'] > 0 else "❌ NEGATIVE"
print(f"- ROI Status: {roi_status}")

# Risk assessment
print(f"\n⚠️ RISK ASSESSMENT:")
precision = business_metrics['precision']
recall = business_metrics['recall']

if precision < 0.5:
    print("- HIGH RISK: Low precision - many false positives may waste campaign budget")
elif precision < 0.7:
    print("- MEDIUM RISK: Moderate precision - some campaign waste expected")
else:
    print("- LOW RISK: High precision - efficient campaign targeting")

if recall < 0.5:
    print("- HIGH RISK: Low recall - missing many potential churners")
elif recall < 0.7:
    print("- MEDIUM RISK: Moderate recall - some churners will be missed")
else:
    print("- LOW RISK: High recall - catching most potential churners")

# Implementation recommendations
print(f"\n🎯 IMPLEMENTATION RECOMMENDATIONS:")

if business_metrics['net_roi'] > 0:
    print("\n✅ PROCEED WITH IMPLEMENTATION:")
    print("1. Start with a pilot program (10-15% of identified churners)")
    print("2. A/B test different retention strategies")
    print("3. Monitor key metrics weekly")
    print("4. Retrain model monthly with new data")
else:
    print("\n⛔ IMPLEMENTATION NOT RECOMMENDED:")
    print("1. Improve model performance before deployment")
    print("2. Consider different retention strategies")
    print("3. Reassess cost-benefit assumptions")

print(f"\n📈 MONITORING FRAMEWORK:")
print("- Model Performance Metrics:")
print("  • Prediction accuracy vs actual churn (weekly)")
print("  • Feature drift detection (monthly)")
print("  • Model recalibration needs (quarterly)")

print("- Business Performance Metrics:")
print("  • Campaign conversion rate")
print("  • Customer retention rate improvement")
print("  • Net ROI per campaign")
print("  • Customer lifetime value (CLV) impact")

print(f"\n🔄 MODEL MAINTENANCE:")
print("- Retrain model with new data monthly")
print("- Monitor for concept drift")
print("- Update feature engineering as needed")
print("- Regular performance audits")

# Save the MLP model
if not os.path.exists('models'):
    os.makedirs('models')
mlp_model.save('models/mlp_churn_model.h5')
print("\n💾 Enhanced MLP model saved as 'models/mlp_churn_model.h5'")
joblib.dump(scaler, 'models/feature_scaler.pkl')
print("💾 Feature scaler saved as 'models/feature_scaler.pkl'")

print(f"\n🎉 ANALYSIS COMPLETE!")
print(f"Model Performance:")
print(f"  - Accuracy: {mlp_accuracy:.4f}")
print(f"  - Precision: {mlp_precision:.4f}")
print(f"  - Recall: {mlp_recall:.4f}")
print(f"  - F1-Score: {mlp_f1:.4f}")
print(f"  - AUC: {mlp_auc:.4f}")
print(f"Business Impact:")
print(f"  - Net ROI: ${business_metrics['net_roi']:,.2f}")
print(f"  - ROI percentage: {business_metrics['roi_percentage']:.1f}%")
