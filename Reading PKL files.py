import joblib
best_churn_model = joblib.load('best_churn_model.pkl')
feature_scaler = joblib.load('feature_scaler.pkl')
print(best_churn_model)
print(feature_scaler)