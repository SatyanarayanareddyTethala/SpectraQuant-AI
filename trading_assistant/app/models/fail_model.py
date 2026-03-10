"""
Failure probability model for predicting trade failures.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class FailureModel:
    """Binary classification model for failure prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize failure model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_type = config.get('type', 'lightgbm')
        self.params = config.get('params', {})
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the failure model.
        
        Args:
            X: Training features
            y: Training target (0=success, 1=failure)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary of training metrics
        """
        self.feature_names = X.columns.tolist()
        
        # Split validation if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
        
        # Train based on model type
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == 'logistic':
            self.model = self._train_logistic(X_train, y_train)
        else:
            # Default to logistic regression
            self.model = self._train_logistic(X_train, y_train)
        
        self.is_trained = True
        
        # Compute metrics
        y_pred_proba = self.predict_proba(X_val)
        metrics = self._compute_metrics(y_val, y_pred_proba)
        
        return metrics
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> lgb.Booster:
        """Train LightGBM classifier"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'is_unbalance': True,
            **self.params
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=params.get('n_estimators', 50),
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        return model
    
    def _train_logistic(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LogisticRegression:
        """Train logistic regression model"""
        params = {
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42,
            'C': self.params.get('C', 1.0)
        }
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict failure probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of failure probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure feature order matches
        if self.feature_names:
            X = X[self.feature_names]
        
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            probas = self.model.predict(X)
        else:
            probas = self.model.predict_proba(X)[:, 1]
        
        return probas
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary failure labels.
        
        Args:
            X: Features to predict on
            threshold: Classification threshold
            
        Returns:
            Array of binary predictions
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def save(self, path: str):
        """Save model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'params': self.params
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.params = model_data.get('params', {})
        self.is_trained = True
    
    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Binary metrics at 0.5 threshold
        y_pred = (y_pred_proba >= 0.5).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision_0_5 = precision_score(y_true, y_pred, zero_division=0)
        recall_0_5 = recall_score(y_true, y_pred, zero_division=0)
        f1_0_5 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision@0.5': precision_0_5,
            'recall@0.5': recall_0_5,
            'f1@0.5': f1_0_5
        }
