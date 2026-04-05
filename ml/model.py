"""
Machine Learning Signal Generation Model

This module implements the SignalModel class that uses ensemble learning methods
(Random Forest, Gradient Boosting) to predict cryptocurrency price direction
based on technical indicators. Includes walk-forward cross-validation and
feature importance analysis.

Author: AlphaPulse Team
License: MIT
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from sklearn.pipeline import Pipeline

from .features import FEATURE_COLS

logger = logging.getLogger("alphapulse.model")

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class SignalModel:
    """
    Machine Learning model for generating trading signals from technical indicators.
    
    Uses ensemble learning methods (Random Forest or Gradient Boosting) to predict
    whether the next price bar will close higher (1) or lower (0) based on a comprehensive
    set of technical indicators. Implements walk-forward cross-validation to prevent
    data leakage and provide realistic performance estimates.
    
    Attributes:
        classifier_name (str): Type of classifier used ('random_forest' or 'gradient_boosting')
        pipeline (sklearn.Pipeline): ML pipeline with scaling and classifier
        feature_importances_ (dict): Feature importance scores (available after training)
        eval_metrics_ (dict): Cross-validation performance metrics
        
    Example:
        >>> model = SignalModel(classifier='random_forest', n_estimators=200)
        >>> metrics = model.train(df_features)
        >>> signal = model.predict_latest(df_features)
        >>> print(f"Signal: {signal['direction']} with {signal['confidence']:.1%} confidence")
    """

    def __init__(
        self,
        classifier: str = "random_forest",
        n_estimators: int = 200,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        """
        Initialize the SignalModel.
        
        Args:
            classifier (str): Type of classifier ('random_forest' or 'gradient_boosting')
            n_estimators (int): Number of trees in the ensemble. Default: 200
            max_depth (int): Maximum depth of trees. Default: 6
            random_state (int): Random seed for reproducibility. Default: 42
            
        Raises:
            ValueError: If classifier is not supported
        """
        self.classifier_name = classifier
        self.random_state = random_state

        clf = self._build_classifier(classifier, n_estimators, max_depth, random_state)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        self.feature_importances_: Optional[Dict[str, float]] = None
        self.eval_metrics_: Optional[Dict[str, float]] = None

    def _build_classifier(self, name: str, n_estimators: int,
                           max_depth: int, seed: int):
        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=seed,
            )
        elif name == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown classifier: {name}. Use 'random_forest' or 'gradient_boosting'.")

    def train(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
        """
        Train the model using walk-forward cross-validation.
        
        Uses TimeSeriesSplit to prevent data leakage by ensuring that
        training data always comes before validation data chronologically.
        
        Args:
            df (pd.DataFrame): Training data with features and target column
            n_splits (int): Number of cross-validation folds. Default: 5
            
        Returns:
            Dict[str, float]: Average cross-validation metrics
            
        Metrics:
            - accuracy: Overall prediction accuracy
            - precision: Positive prediction precision
            - recall: Positive prediction recall
            - f1: F1-score (harmonic mean of precision and recall)
            - roc_auc: Area under ROC curve
            
        Note:
            After training, the model is fitted on the entire dataset
            and feature importances are calculated.
        """
        X = df[FEATURE_COLS].values
        y = df["target"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            self.pipeline.fit(X_tr, y_tr)
            y_pred = self.pipeline.predict(X_val)
            y_prob = self.pipeline.predict_proba(X_val)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_val, y_prob),
            }
            fold_metrics.append(metrics)
            logger.info(f"Fold {fold+1}/{n_splits} — Acc: {metrics['accuracy']:.3f} | AUC: {metrics['roc_auc']:.3f}")

        self.pipeline.fit(X, y)

        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        self.eval_metrics_ = avg_metrics

        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            self.feature_importances_ = dict(
                sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
            )

        logger.info(f"Training complete. Avg metrics: {avg_metrics}")
        return avg_metrics

    def predict_latest(self, df: pd.DataFrame) -> Dict:
        """
        Generate a trading signal for the most recent data point.
        
        Args:
            df (pd.DataFrame): DataFrame with features (must include latest data)
            
        Returns:
            Dict: Signal information containing:
                - direction (str): 'BUY' or 'SELL'
                - confidence (float): Confidence score (0-1)
                - prob_up (float): Probability of price moving up
                - prob_down (float): Probability of price moving down
                
        Example:
            >>> signal = model.predict_latest(df_features)
            >>> if signal['direction'] == 'BUY' and signal['confidence'] > 0.6:
            ...     print("Strong buy signal detected")
        """
        X = df[FEATURE_COLS].values[-1].reshape(1, -1)
        label = int(self.pipeline.predict(X)[0])
        proba = self.pipeline.predict_proba(X)[0]

        return {
            "direction": "BUY" if label == 1 else "SELL",
            "confidence": float(proba[label]),
            "prob_up": float(proba[1]),
            "prob_down": float(proba[0]),
        }

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate probability predictions for the entire dataset.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            
        Returns:
            pd.Series: Probability of price moving up for each timestamp
            
        Note:
            Returns the probability of the positive class (price moving up)
            Useful for backtesting and strategy evaluation.
        """
        X = df[FEATURE_COLS].values
        return pd.Series(
            self.pipeline.predict_proba(X)[:, 1],
            index=df.index,
            name="signal_prob",
        )

    def save(self, name: str = "signal_model"):
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(self.pipeline, path)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, name: str = "signal_model"):
        path = MODEL_DIR / f"{name}.joblib"
        self.pipeline = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self

    def get_feature_importance_df(self) -> pd.DataFrame:
        if not self.feature_importances_:
            raise RuntimeError("Model not trained yet.")
        return pd.DataFrame(
            list(self.feature_importances_.items()),
            columns=["feature", "importance"]
        ).sort_values("importance", ascending=False)
