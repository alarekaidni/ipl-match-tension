"""
src/model.py
------------
Machine Learning modeling module for 2nd-innings ball-by-ball win-probability estimation.

Author: Candidate for JioStar Data Analytics Internship
Project: IPL Match Tension & Win-Probability Swing
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve

# Canonical feature list
CORE_FEATURES = [
    "runs_needed",
    "balls_left",
    "wickets_in_hand",
    "current_run_rate",
    "required_run_rate",
    "target"
]


def temporal_train_test_split(
    df: pd.DataFrame, 
    split_season: int = 2015,
    features: list = CORE_FEATURES
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits chase records chronologically by season to prevent intra-match data leakage.
    
    Train: Seasons <= split_season (e.g., 2008 to 2015, 8 seasons)
    Test:  Seasons > split_season  (e.g., 2016 to 2017, 2 seasons)
    """
    train_mask = df["season"] <= split_season
    test_mask = df["season"] > split_season
    
    X_train = df.loc[train_mask, features].copy()
    y_train = df.loc[train_mask, "label"].copy()
    
    X_test = df.loc[test_mask, features].copy()
    y_test = df.loc[test_mask, "label"].copy()
    
    return X_train, X_test, y_train, y_test


def train_logistic_baseline(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> LogisticRegression:
    """Trains a regularized Logistic Regression baseline model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> XGBClassifier:
    """Trains a regularized Gradient Boosted Decision Tree (XGBoost) model."""
    model = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str = "Model"
) -> Dict[str, float]:
    """Computes Log-Loss, Brier Score, Accuracy, and ROC-AUC for probabilistic evaluation."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs),
        "brier_score": brier_score_loss(y_test, probs),
        "roc_auc": roc_auc_score(y_test, probs)
    }
    return metrics


def save_model(model: Any, filepath: str) -> None:
    """Persists model artifact using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Persisted model to: {filepath}")
