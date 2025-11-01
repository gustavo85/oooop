"""
ml_integration_adapter.py - Adapter for integrating ML Pipeline V5.0 with existing system
Provides backward compatibility with neural_network_optimizer.py interface
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Try to import new ML pipeline
try:
    from ml_pipeline import (
        MLPipeline, GameSession, AdvancedFeatureEngineer,
        create_dummy_sessions, NUMPY_AVAILABLE, SKLEARN_AVAILABLE,
        XGBOOST_AVAILABLE, PYTORCH_AVAILABLE
    )
    from ml_config_loader import get_ml_config
    ML_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML Pipeline V5.0 not available: {e}")
    ML_PIPELINE_AVAILABLE = False

# Fallback to old neural network optimizer
try:
    from neural_network_optimizer import NeuralNetworkOptimizer as OldOptimizer
    OLD_OPTIMIZER_AVAILABLE = True
except ImportError:
    OLD_OPTIMIZER_AVAILABLE = False
    logger.warning("Old neural network optimizer not available")


class MLIntegrationAdapter:
    """
    Adapter that provides unified interface to ML system
    
    Features:
    - Automatic fallback to old optimizer if V5.0 not available
    - Backward compatible API
    - Session data conversion
    - Performance monitoring
    - Graceful degradation
    """
    
    def __init__(self, use_v5: bool = True):
        """
        Args:
            use_v5: Try to use ML Pipeline V5.0. Falls back to old if unavailable.
        """
        self.using_v5 = False
        self.pipeline: Optional[MLPipeline] = None
        self.old_optimizer: Optional[Any] = None
        self.config = None
        
        if use_v5 and ML_PIPELINE_AVAILABLE:
            try:
                self.config = get_ml_config()
                self.pipeline = MLPipeline(model_dir=self.config.model_dir)
                
                # Try to load existing models
                if self.pipeline.load_models():
                    logger.info("✓ ML Pipeline V5.0 loaded with existing models")
                else:
                    logger.info("✓ ML Pipeline V5.0 initialized (no existing models)")
                
                self.using_v5 = True
                
            except Exception as e:
                logger.error(f"Failed to initialize ML Pipeline V5.0: {e}")
                logger.info("Falling back to old optimizer...")
        
        # Fallback to old optimizer
        if not self.using_v5 and OLD_OPTIMIZER_AVAILABLE:
            try:
                self.old_optimizer = OldOptimizer()
                logger.info("✓ Using legacy neural network optimizer")
            except Exception as e:
                logger.error(f"Failed to initialize old optimizer: {e}")
    
    def is_available(self) -> bool:
        """Check if any ML system is available"""
        return self.using_v5 or (self.old_optimizer is not None)
    
    def get_version(self) -> str:
        """Get version of ML system being used"""
        if self.using_v5:
            return "V5.0 (Advanced Pipeline)"
        elif self.old_optimizer:
            return "V4.0 (Legacy)"
        else:
            return "None (ML not available)"
    
    def convert_session_to_v5(self, session_data: Dict[str, Any]) -> GameSession:
        """
        Convert session data dict to GameSession object for V5.0
        
        Args:
            session_data: Dictionary with session data (V4.0 format)
            
        Returns:
            GameSession object for V5.0 pipeline
        """
        # Create GameSession with defaults, then update from session_data
        session = GameSession(
            session_id=session_data.get('session_id', f"session_{int(time.time())}"),
            game_name=session_data.get('game_name', 'Unknown'),
            game_exe=session_data.get('game_exe', 'game.exe'),
            timestamp=session_data.get('timestamp', time.time()),
            duration_seconds=session_data.get('duration_seconds', 300),
        )
        
        # Hardware mapping (V4 -> V5)
        if 'cpu_cores' in session_data:
            session.cpu_cores_physical = session_data['cpu_cores']
        if 'cpu_threads' in session_data:
            session.cpu_cores_logical = session_data['cpu_threads']
        if 'cpu_freq_mhz' in session_data:
            session.cpu_freq_base_mhz = session_data['cpu_freq_mhz']
        if 'ram_gb' in session_data:
            session.ram_capacity_gb = session_data['ram_gb']
        if 'gpu_vram_gb' in session_data:
            session.gpu_vram_gb = session_data['gpu_vram_gb']
        
        # Configuration mapping
        if 'timer_resolution_ms' in session_data:
            session.timer_resolution_ms = session_data['timer_resolution_ms']
        if 'cpu_priority' in session_data:
            session.cpu_priority_class = session_data['cpu_priority']
        if 'gpu_clock_locked' in session_data:
            session.gpu_clock_locked = session_data['gpu_clock_locked']
        if 'memory_optimization_level' in session_data:
            session.memory_optimization_level = session_data['memory_optimization_level']
        if 'network_qos_enabled' in session_data:
            session.network_qos_enabled = session_data['network_qos_enabled']
        if 'core_parking_disabled' in session_data:
            session.core_parking_disabled = session_data['core_parking_disabled']
        
        # Performance metrics mapping
        if 'avg_fps' in session_data:
            session.fps_avg = session_data['avg_fps']
        if 'fps_1_percent_low' in session_data:
            session.fps_p1 = session_data['fps_1_percent_low']
        if 'frame_time_avg_ms' in session_data:
            session.frame_time_avg_ms = session_data['frame_time_avg_ms']
        if 'frame_time_p99_ms' in session_data:
            session.frame_time_p99_ms = session_data['frame_time_p99_ms']
        if 'frame_time_std_dev' in session_data:
            session.frame_time_std_ms = session_data['frame_time_std_dev']
        if 'avg_cpu_usage' in session_data:
            session.cpu_usage_avg = session_data['avg_cpu_usage']
        if 'avg_gpu_usage' in session_data:
            session.gpu_usage_avg = session_data['avg_gpu_usage']
        if 'avg_memory_usage_mb' in session_data:
            session.memory_working_set_mb = session_data['avg_memory_usage_mb']
        if 'stutter_count' in session_data:
            session.stutter_count = session_data['stutter_count']
        if 'stability_score' in session_data:
            # Stability score is derived in V5, but we can set baseline
            pass
        
        return session
    
    def train(self, sessions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train ML models on session data
        
        Args:
            sessions_data: List of session dictionaries
            
        Returns:
            Training results dictionary
        """
        if not self.is_available():
            logger.error("No ML system available for training")
            return {'success': False, 'error': 'ML not available'}
        
        if self.using_v5:
            try:
                # Convert sessions to V5 format
                sessions = [self.convert_session_to_v5(s) for s in sessions_data]
                
                logger.info(f"Training ML Pipeline V5.0 with {len(sessions)} sessions...")
                results = self.pipeline.train_all_models(sessions)
                
                return {
                    'success': True,
                    'version': 'V5.0',
                    'num_sessions': len(sessions),
                    'models_trained': list(results.keys())
                }
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                return {'success': False, 'error': str(e)}
        
        else:
            # Use old optimizer
            try:
                # Convert to old format
                from neural_network_optimizer import GameSession as OldGameSession
                
                old_sessions = []
                for s in sessions_data:
                    old_session = OldGameSession(**s)
                    old_sessions.append(old_session)
                
                self.old_optimizer.train(old_sessions)
                
                return {
                    'success': True,
                    'version': 'V4.0',
                    'num_sessions': len(old_sessions)
                }
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                return {'success': False, 'error': str(e)}
    
    def predict(
        self, 
        session_data: Dict[str, Any],
        return_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Predict performance for a session
        
        Args:
            session_data: Session dictionary
            return_explanation: Include SHAP explanations (V5.0 only)
            
        Returns:
            Predictions dictionary with fps, stability, success probability
        """
        if not self.is_available():
            logger.error("No ML system available for prediction")
            return {'error': 'ML not available'}
        
        if self.using_v5:
            try:
                session = self.convert_session_to_v5(session_data)
                predictions = self.pipeline.predict(session, return_explanation)
                
                # Add metadata
                predictions['version'] = 'V5.0'
                predictions['timestamp'] = time.time()
                
                return predictions
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return {'error': str(e)}
        
        else:
            # Use old optimizer
            try:
                from neural_network_optimizer import GameSession as OldGameSession
                old_session = OldGameSession(**session_data)
                
                fps_pred = self.old_optimizer.predict_fps(old_session)
                stability_pred = self.old_optimizer.predict_stability(old_session)
                
                return {
                    'fps': fps_pred,
                    'stability_score': stability_pred,
                    'version': 'V4.0',
                    'timestamp': time.time()
                }
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return {'error': str(e)}
    
    def get_feature_importance(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top-K most important features
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            List of {'feature': name, 'importance': value} dicts
        """
        if not self.is_available():
            return []
        
        if self.using_v5 and self.pipeline.explainer:
            try:
                # Get SHAP-based importance
                # TODO: Implement global feature importance extraction
                logger.info("Feature importance available via SHAP")
                return []
            except Exception as e:
                logger.error(f"Error getting feature importance: {e}")
                return []
        
        elif self.old_optimizer:
            try:
                # Old optimizer has feature importance
                return self.old_optimizer.get_feature_importance(top_k)
            except Exception as e:
                logger.error(f"Error getting feature importance: {e}")
                return []
        
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of ML system"""
        status = {
            'available': self.is_available(),
            'version': self.get_version(),
            'using_v5': self.using_v5,
            'capabilities': {
                'numpy': NUMPY_AVAILABLE if ML_PIPELINE_AVAILABLE else False,
                'sklearn': SKLEARN_AVAILABLE if ML_PIPELINE_AVAILABLE else False,
                'xgboost': XGBOOST_AVAILABLE if ML_PIPELINE_AVAILABLE else False,
                'pytorch': PYTORCH_AVAILABLE if ML_PIPELINE_AVAILABLE else False,
                'explainability': self.using_v5 and self.pipeline and self.pipeline.explainer is not None
            }
        }
        
        if self.using_v5 and self.pipeline:
            status['models_loaded'] = list(self.pipeline.models.keys())
            status['model_dir'] = str(self.pipeline.model_dir)
        
        return status


# Global adapter instance
_global_adapter: Optional[MLIntegrationAdapter] = None


def get_ml_adapter(use_v5: bool = True) -> MLIntegrationAdapter:
    """
    Get global ML adapter instance
    
    Args:
        use_v5: Try to use V5.0 pipeline. Only used on first call.
        
    Returns:
        MLIntegrationAdapter instance
    """
    global _global_adapter
    
    if _global_adapter is None:
        _global_adapter = MLIntegrationAdapter(use_v5=use_v5)
    
    return _global_adapter


# Convenience functions for backward compatibility

def train_models(sessions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train ML models (backward compatible)"""
    adapter = get_ml_adapter()
    return adapter.train(sessions_data)


def predict_performance(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict performance (backward compatible)"""
    adapter = get_ml_adapter()
    return adapter.predict(session_data, return_explanation=False)


def predict_with_explanation(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict with SHAP explanation (V5.0 only)"""
    adapter = get_ml_adapter()
    return adapter.predict(session_data, return_explanation=True)


def get_ml_status() -> Dict[str, Any]:
    """Get ML system status"""
    adapter = get_ml_adapter()
    return adapter.get_status()
