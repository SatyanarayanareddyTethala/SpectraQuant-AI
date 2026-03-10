"""
Ranking model for cross-sectional stock ranking.
Supports LightGBM, XGBoost, and sklearn fallback.
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


class RankingModel:
    """Cross-sectional ranking model"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ranking model.
        
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
        Train the ranking model.
        
        Args:
            X: Training features
            y: Training target (ranking scores)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary of training metrics
        """
        self.feature_names = X.columns.tolist()
        
        # Split validation if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
        
        # Train based on model type
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            self.model = self._train_xgboost(X_train, y_train, X_val, y_val)
        else:
            # Fallback to sklearn
            self.model = self._train_sklearn(X_train, y_train)
        
        self.is_trained = True
        
        # Compute metrics
        y_pred = self.predict(X_val)
        metrics = self._compute_metrics(y_val, y_pred)
        
        return metrics
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> lgb.Booster:
        """Train LightGBM model"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            **self.params
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=params.get('n_estimators', 100),
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        return model
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> xgb.Booster:
        """Train XGBoost model"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            **self.params
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=[(dval, 'validation')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        return model
    
    def _train_sklearn(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> GradientBoostingRegressor:
        """Train sklearn gradient boosting model"""
        params = {
            'n_estimators': self.params.get('n_estimators', 100),
            'learning_rate': self.params.get('learning_rate', 0.05),
            'max_depth': self.params.get('max_depth', 5),
            'random_state': 42
        }
        
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure feature order matches
        if self.feature_names:
            X = X[self.feature_names]
        
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            predictions = self.model.predict(X)
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            dmatrix = xgb.DMatrix(X)
            predictions = self.model.predict(dmatrix)
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            importance = self.model.feature_importance(importance_type='gain')
            return dict(zip(self.feature_names, importance))
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            importance = self.model.get_score(importance_type='gain')
            return importance
        else:
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
    
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
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Ranking metrics (Spearman correlation)
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'spearman_correlation': spearman_corr
        }


class EnsembleRankingModel:
    """Ensemble of ranking models for uncertainty estimation"""
    
    def __init__(self, config: Dict[str, Any], n_models: int = 3):
        """
        Initialize ensemble.
        
        Args:
            config: Model configuration
            n_models: Number of models in ensemble
        """
        self.config = config
        self.n_models = n_models
        self.models = [RankingModel(config) for _ in range(n_models)]
        self.is_trained = False
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[Dict[str, float]]:
        """
        Train ensemble with different data subsets.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            List of metrics for each model
        """
        all_metrics = []
        
        for i, model in enumerate(self.models):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # Train model
            metrics = model.train(X_sample, y_sample)
            all_metrics.append(metrics)
        
        self.is_trained = True
        return all_metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (mean predictions, std predictions)
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Compute mean and std
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    def save(self, base_path: str):
        """Save ensemble models"""
        for i, model in enumerate(self.models):
            path = f"{base_path}_model_{i}.pkl"
            model.save(path)
    
    def load(self, base_path: str):
        """Load ensemble models"""
        for i, model in enumerate(self.models):
            path = f"{base_path}_model_{i}.pkl"
            model.load(path)
        self.is_trained = True
