# Spotify churn prediction with deep learning

Predicts whether a Spotify user will cancel their subscription, using a multi-layer perceptron trained on user behavior and subscription data. Includes feature engineering, class balancing via SMOTE, and a business ROI analysis on top of standard ML metrics.

## What it does

The pipeline takes a CSV of user data, cleans and engineers features, trains an MLP neural network, evaluates it against held-out test data, and then translates those predictions into estimated campaign costs, revenue saved, and net ROI. The idea is that knowing a model hit 95% accuracy is less useful than knowing whether the retention campaign it drives will make money.

The three phases run sequentially in one script:

1. **Preprocessing** - Missing value imputation, one-hot encoding of categorical variables, engineered features (`engagement_ratio`, `value_per_hour`), SMOTE oversampling on the training set to address class imbalance, and StandardScaler normalization.
2. **Model training** - A 5-layer MLP (256 → 128 → 64 → 32 → 1) with BatchNorm, Dropout (0.3), L2 regularization, EarlyStopping, and ReduceLROnPlateau.
3. **Evaluation** - Classification report, confusion matrix, and a business metrics calculation that estimates campaign ROI given configurable cost-per-contact and revenue-per-retention assumptions.

## Files

| File | Purpose |
|------|---------|
| `Spotify Churn Prediction with Deep Learning 2.py` | Main pipeline script |
| `spotify_churn_dataset.csv` | Training dataset |
| `models/mlp_churn_model.h5` | Saved trained model |
| `models/feature_scaler.pkl` | Saved StandardScaler for inference |

## Results

After training on the full dataset with SMOTE balancing, the model reached **95% accuracy** on the held-out test set. The script also reports precision, recall, F1, and AUC-ROC, plus the business impact breakdown.

## Setup

```bash
pip install tensorflow scikit-learn pandas numpy imbalanced-learn joblib
```

Then run:

```bash
python "Spotify Churn Prediction with Deep Learning 2.py"
```

The script auto-installs missing packages on startup. If `spotify_churn_dataset.csv` isn't found, it generates a synthetic dataset with realistic churn patterns so you can run the pipeline end-to-end without the real data.

## Dataset columns

The model expects these columns in the CSV:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Dropped before training |
| `age` | int | User age |
| `gender` | str | M / F / Other |
| `country` | str | Country code |
| `subscription_type` | str | Free / Premium / Family |
| `monthly_charges` | float | Monthly fee |
| `total_charges` | float | Lifetime charges |
| `subscription_length` | int | Months subscribed |
| `device_type` | str | Mobile / Desktop / Tablet / Smart_TV |
| `listening_hours` | float | Hours per month |
| `playlist_count` | int | Number of playlists |
| `favorite_genre` | str | Genre category |
| `skips_per_session` | float | Average skips |
| `premium_features_used` | int | Count of premium features used |
| `Churn` | int | Target: 1 = churned, 0 = retained |

## Model architecture

```
Input → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
      → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
      → Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
      → Dense(32, ReLU)  → Dropout(0.3)
      → Dense(1, Sigmoid)
```

Optimizer: Adam (lr=0.001). Loss: binary crossentropy. EarlyStopping patience: 20 epochs.

## Business metrics

The script calculates ROI using configurable assumptions:

```python
cost_per_campaign = 10       # Cost to contact one at-risk user
revenue_per_retention = 50   # Revenue saved per successfully retained user
revenue_lost_per_churn = 240 # Revenue lost per missed churner
```

Adjust these in `calculate_business_metrics()` to match your actual numbers before drawing conclusions from the ROI output.

## Saving and loading the model

The trained model saves to `models/mlp_churn_model.h5` and the scaler to `models/feature_scaler.pkl` at the end of each run. To load them for inference:

```python
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('models/mlp_churn_model.h5')
scaler = joblib.load('models/feature_scaler.pkl')

X_new_scaled = scaler.transform(X_new)
predictions = (model.predict(X_new_scaled) > 0.5).astype(int)
```

## Project context

Built as the EDA and deep learning project in a 3-month AI internship at SmartED Innovations, assessed by Skill India and NSDC (Sep-Oct 2025). An earlier version of this pipeline using Logistic Regression is in `Spotify Churn Prediction with Deep Learning 0.py` and `1.py` if you want to compare approaches.

## Dependencies

- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `imbalanced-learn`
- `joblib`
