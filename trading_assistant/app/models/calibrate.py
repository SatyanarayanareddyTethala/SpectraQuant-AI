"""
Model calibration using isotonic regression or Platt scaling.
"""
from typing import Literal
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


class ModelCalibrator:
    """Calibrates model predictions to improve probability estimates"""
    
    def __init__(self, method: Literal['isotonic', 'platt'] = 'isotonic'):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'platt')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Fit calibrator on predictions and true labels.
        
        Args:
            y_pred: Uncalibrated predictions
            y_true: True labels
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred, y_true)
        elif self.method == 'platt':
            # Platt scaling: fit logistic regression on logit of predictions
            X = y_pred.reshape(-1, 1)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(X, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions.
        
        Args:
            y_pred: Uncalibrated predictions
            
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'isotonic':
            return self.calibrator.transform(y_pred)
        elif self.method == 'platt':
            X = y_pred.reshape(-1, 1)
            return self.calibrator.predict_proba(X)[:, 1]
    
    def fit_transform(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """
        Fit calibrator and transform predictions.
        
        Args:
            y_pred: Uncalibrated predictions
            y_true: True labels
            
        Returns:
            Calibrated predictions
        """
        self.fit(y_pred, y_true)
        return self.transform(y_pred)
    
    def save(self, path: str):
        """Save calibrator to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        calibrator_data = {
            'method': self.method,
            'calibrator': self.calibrator
        }
        
        with open(path, 'wb') as f:
            pickle.dump(calibrator_data, f)
    
    def load(self, path: str):
        """Load calibrator from disk"""
        with open(path, 'rb') as f:
            calibrator_data = pickle.load(f)
        
        self.method = calibrator_data['method']
        self.calibrator = calibrator_data['calibrator']
        self.is_fitted = True
    
    def evaluate_calibration(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> dict:
        """
        Evaluate calibration quality.
        
        Args:
            y_pred: Predictions to evaluate
            y_true: True labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        # Compute calibration curve
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy='uniform'
        )
        
        # Compute Brier score
        from sklearn.metrics import brier_score_loss
        brier_score = brier_score_loss(y_true, y_pred)
        
        # Compute log loss
        from sklearn.metrics import log_loss
        logloss = log_loss(y_true, y_pred)
        
        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        return {
            'brier_score': brier_score,
            'log_loss': logloss,
            'ece': ece,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
