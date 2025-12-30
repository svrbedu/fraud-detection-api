import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

def train():
    if os.path.exists("fraud_model.joblib"):
        print("Model already exists. Skipping training.")
        return

    print("Training model...")
    # Create dummy training data matching the Transaction schema
    # Features: amount, distance_from_home, hour_of_day
    X = pd.DataFrame({
        'amount': [10.0, 20.0, 1000.0, 2500.0, 5.0, 150.0],
        'distance_from_home': [2.0, 5.0, 100.0, 200.0, 1.0, 50.0],
        'hour_of_day': [10, 14, 3, 2, 12, 23]
    })
    # Labels: 0 = Legit, 1 = Fraud
    y = np.array([0, 0, 1, 1, 0, 1])

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    joblib.dump(model, "fraud_model.joblib")
    print("Model trained and saved to fraud_model.joblib")

if __name__ == "__main__":
    train()