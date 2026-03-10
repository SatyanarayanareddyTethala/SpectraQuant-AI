"""
Model registry for tracking and managing models.
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from ..db import crud, models


class ModelRegistry:
    """Manages model versions and lifecycle"""
    
    def __init__(self, db_session: Session):
        """
        Initialize model registry.
        
        Args:
            db_session: Database session
        """
        self.db = db_session
    
    def register_model(
        self,
        model_type: str,
        data_window: Dict[str, Any],
        metrics: Dict[str, float],
        model_path: str,
        status: str = 'candidate'
    ) -> int:
        """
        Register a new model in the registry.
        
        Args:
            model_type: Type of model (rank, fail, calibrated)
            data_window: Training data window info
            metrics: Model performance metrics
            model_path: Path to serialized model
            status: Model status (candidate, active, archived, rolled_back)
            
        Returns:
            Model ID
        """
        model = crud.create_model(
            db=self.db,
            model_type=model_type,
            data_window=data_window,
            metrics=metrics,
            model_path=model_path,
            status=status
        )
        
        return model.model_id
    
    def get_active_model(self, model_type: str) -> Optional[models.ModelRegistry]:
        """
        Get currently active model of given type.
        
        Args:
            model_type: Type of model to retrieve
            
        Returns:
            ModelRegistry object or None
        """
        return crud.get_active_model(self.db, model_type)
    
    def promote_model(
        self,
        model_id: int,
        demote_current: bool = True
    ) -> bool:
        """
        Promote a candidate model to active status.
        
        Args:
            model_id: ID of model to promote
            demote_current: Whether to archive current active model
            
        Returns:
            True if successful
        """
        # Get the model to promote
        model = self.db.query(models.ModelRegistry).filter(
            models.ModelRegistry.model_id == model_id
        ).first()
        
        if not model:
            return False
        
        # Demote current active model if requested
        if demote_current:
            current_active = self.get_active_model(model.model_type)
            if current_active:
                crud.update_model_status(
                    self.db,
                    current_active.model_id,
                    'archived'
                )
        
        # Promote new model
        crud.update_model_status(self.db, model_id, 'active')
        
        return True
    
    def rollback_model(
        self,
        model_type: str,
        target_model_id: Optional[int] = None
    ) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            model_type: Type of model to rollback
            target_model_id: Specific model to rollback to, or None for last archived
            
        Returns:
            True if successful
        """
        # Mark current active as rolled_back
        current_active = self.get_active_model(model_type)
        if current_active:
            crud.update_model_status(
                self.db,
                current_active.model_id,
                'rolled_back'
            )
        
        # Find target model
        if target_model_id is None:
            # Get most recent archived model
            target = self.db.query(models.ModelRegistry).filter(
                models.ModelRegistry.model_type == model_type,
                models.ModelRegistry.status == 'archived'
            ).order_by(models.ModelRegistry.created_at.desc()).first()
        else:
            target = self.db.query(models.ModelRegistry).filter(
                models.ModelRegistry.model_id == target_model_id
            ).first()
        
        if not target:
            return False
        
        # Promote target to active
        crud.update_model_status(self.db, target.model_id, 'active')
        
        return True
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[models.ModelRegistry]:
        """
        List models with optional filters.
        
        Args:
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of ModelRegistry objects
        """
        query = self.db.query(models.ModelRegistry)
        
        if model_type:
            query = query.filter(models.ModelRegistry.model_type == model_type)
        
        if status:
            query = query.filter(models.ModelRegistry.status == status)
        
        return query.order_by(
            models.ModelRegistry.created_at.desc()
        ).limit(limit).all()
    
    def get_model_metrics(self, model_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary of metrics or None
        """
        model = self.db.query(models.ModelRegistry).filter(
            models.ModelRegistry.model_id == model_id
        ).first()
        
        if model:
            return model.metrics_json
        
        return None
    
    def compare_models(
        self,
        model_id_1: int,
        model_id_2: int
    ) -> Optional[Dict[str, Any]]:
        """
        Compare metrics between two models.
        
        Args:
            model_id_1: First model ID
            model_id_2: Second model ID
            
        Returns:
            Comparison dictionary or None
        """
        model_1 = self.db.query(models.ModelRegistry).filter(
            models.ModelRegistry.model_id == model_id_1
        ).first()
        
        model_2 = self.db.query(models.ModelRegistry).filter(
            models.ModelRegistry.model_id == model_id_2
        ).first()
        
        if not model_1 or not model_2:
            return None
        
        metrics_1 = model_1.metrics_json
        metrics_2 = model_2.metrics_json
        
        comparison = {
            'model_1_id': model_id_1,
            'model_2_id': model_id_2,
            'metrics_1': metrics_1,
            'metrics_2': metrics_2,
            'differences': {}
        }
        
        # Compute differences for common metrics
        for key in metrics_1:
            if key in metrics_2:
                try:
                    diff = metrics_2[key] - metrics_1[key]
                    pct_change = (diff / metrics_1[key] * 100) if metrics_1[key] != 0 else 0
                    comparison['differences'][key] = {
                        'absolute': diff,
                        'percent': pct_change
                    }
                except:
                    pass
        
        return comparison
