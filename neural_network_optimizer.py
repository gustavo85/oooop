"""
Neural Network and Explainable AI V4.0
Advanced ML with XGBoost and SHAP interpretability
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from numpy import ndarray
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    ndarray = None  # Fallback for type hints
    logger.warning("numpy not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


@dataclass
class GameSession:
    """Gaming session data for ML training"""
    game_name: str
    game_exe: str
    
    # Hardware
    cpu_cores: int
    cpu_threads: int
    cpu_freq_mhz: int
    ram_gb: int
    gpu_vram_gb: int
    
    # Optimization settings
    timer_resolution_ms: float
    cpu_priority: int  # 0=normal, 1=high, 2=realtime
    gpu_clock_locked: bool
    memory_optimization_level: int
    network_qos_enabled: bool
    core_parking_disabled: bool
    
    # Performance metrics
    avg_fps: float
    fps_1_percent_low: float
    frame_time_avg_ms: float
    frame_time_p99_ms: float
    frame_time_std_dev: float
    
    # System metrics
    avg_cpu_usage: float
    avg_gpu_usage: float
    avg_memory_usage_mb: float
    
    # Stability
    stutter_count: int
    stability_score: float
    
    # Session metadata
    duration_seconds: int
    timestamp: float


class NeuralNetworkOptimizer:
    """
    Advanced neural network for game optimization using XGBoost
    
    Features:
    - Gradient boosting for complex patterns
    - Multi-target prediction (FPS, stability, etc.)
    - Feature importance analysis
    - SHAP-based explainability
    - Incremental learning support
    """
    
    def __init__(self):
        self.model_dir = Path.home() / '.game_optimizer' / 'nn_models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps_model: Optional[Any] = None
        self.stability_model: Optional[Any] = None
        self.explainer: Optional[Any] = None
        
        self.feature_names = [
            'cpu_cores', 'cpu_threads', 'cpu_freq_mhz', 'ram_gb', 'gpu_vram_gb',
            'timer_resolution_ms', 'cpu_priority', 'gpu_clock_locked',
            'memory_optimization_level', 'network_qos_enabled', 'core_parking_disabled'
        ]
        
        self.training_data: List[GameSession] = []
        self.training_data_file = self.model_dir / 'nn_training_data.pkl'
        
        if XGBOOST_AVAILABLE:
            self._load_models()
            self._load_training_data()
    
    def _load_models(self):
        """Load pre-trained models"""
        fps_model_file = self.model_dir / 'xgb_fps_model.json'
        stability_model_file = self.model_dir / 'xgb_stability_model.json'
        
        try:
            if fps_model_file.exists():
                self.fps_model = xgb.XGBRegressor()
                self.fps_model.load_model(fps_model_file)
                logger.info("✓ XGBoost FPS model loaded")
            
            if stability_model_file.exists():
                self.stability_model = xgb.XGBRegressor()
                self.stability_model.load_model(stability_model_file)
                logger.info("✓ XGBoost stability model loaded")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _load_training_data(self):
        """Load historical training data"""
        try:
            if self.training_data_file.exists():
                # Pickle is used here for model persistence - only load trusted data
                with open(self.training_data_file, 'rb') as f:
                    self.training_data = pickle.load(f)
                logger.info(f"Loaded {len(self.training_data)} training sessions")
        except Exception as e:
            logger.debug(f"Error loading training data: {e}")
    
    def _save_training_data(self):
        """Save training data"""
        try:
            with open(self.training_data_file, 'wb') as f:
                pickle.dump(self.training_data, f)
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def add_session(self, session: GameSession):
        """Add a gaming session for training"""
        self.training_data.append(session)
        self._save_training_data()
        logger.debug(f"Added session: {session.game_name}, FPS={session.avg_fps:.1f}")
    
    def prepare_features(self, sessions: List[GameSession]):
        """Prepare feature matrix from sessions"""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for feature preparation")
        
        features = []
        for session in sessions:
            feature_vector = [
                session.cpu_cores,
                session.cpu_threads,
                session.cpu_freq_mhz,
                session.ram_gb,
                session.gpu_vram_gb,
                session.timer_resolution_ms,
                session.cpu_priority,
                1 if session.gpu_clock_locked else 0,
                session.memory_optimization_level,
                1 if session.network_qos_enabled else 0,
                1 if session.core_parking_disabled else 0,
            ]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def train_models(self, min_sessions: int = 20) -> bool:
        """
        Train XGBoost models on collected data
        
        Args:
            min_sessions: Minimum number of sessions required for training
        
        Returns:
            Success status
        """
        if not XGBOOST_AVAILABLE or not NUMPY_AVAILABLE:
            logger.error("XGBoost and numpy are required for training")
            return False
        
        if len(self.training_data) < min_sessions:
            logger.warning(f"Not enough training data: {len(self.training_data)}/{min_sessions} sessions")
            return False
        
        logger.info(f"Training neural network on {len(self.training_data)} sessions...")
        
        # Prepare features and targets
        X = self.prepare_features(self.training_data)
        y_fps = np.array([s.avg_fps for s in self.training_data], dtype=np.float32)
        y_stability = np.array([s.stability_score for s in self.training_data], dtype=np.float32)
        
        # Train FPS model
        logger.info("Training FPS prediction model...")
        self.fps_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            booster='gbtree',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.fps_model.fit(X, y_fps)
        
        # Train stability model
        logger.info("Training stability prediction model...")
        self.stability_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            booster='gbtree',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.stability_model.fit(X, y_stability)
        
        # Save models
        self.fps_model.save_model(self.model_dir / 'xgb_fps_model.json')
        self.stability_model.save_model(self.model_dir / 'xgb_stability_model.json')
        
        # Initialize SHAP explainer
        if SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.fps_model)
            logger.info("✓ SHAP explainer initialized")
        
        logger.info("✓ Neural network training complete")
        return True
    
    def predict(self, session_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict performance for given settings
        
        Args:
            session_params: Dictionary with hardware and optimization settings
        
        Returns:
            Dictionary with predicted FPS and stability
        """
        if not self.fps_model or not self.stability_model:
            logger.error("Models not trained. Call train_models() first")
            return {'fps': 0.0, 'stability': 0.0, 'confidence': 0.0}
        
        if not NUMPY_AVAILABLE:
            logger.error("numpy required for prediction")
            return {'fps': 0.0, 'stability': 0.0, 'confidence': 0.0}
        
        # Prepare feature vector
        features = np.array([[
            session_params.get('cpu_cores', 8),
            session_params.get('cpu_threads', 16),
            session_params.get('cpu_freq_mhz', 3600),
            session_params.get('ram_gb', 16),
            session_params.get('gpu_vram_gb', 8),
            session_params.get('timer_resolution_ms', 0.5),
            session_params.get('cpu_priority', 1),
            1 if session_params.get('gpu_clock_locked', False) else 0,
            session_params.get('memory_optimization_level', 2),
            1 if session_params.get('network_qos_enabled', False) else 0,
            1 if session_params.get('core_parking_disabled', True) else 0,
        ]], dtype=np.float32)
        
        # Predict
        fps_pred = float(self.fps_model.predict(features)[0])
        stability_pred = float(self.stability_model.predict(features)[0])
        
        # Calculate confidence (placeholder - would use prediction intervals in real implementation)
        confidence = 0.85
        
        return {
            'fps': fps_pred,
            'stability': stability_pred,
            'confidence': confidence
        }
    
    def explain_prediction(self, session_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Explain prediction using SHAP values
        
        Args:
            session_params: Dictionary with hardware and optimization settings
        
        Returns:
            Dictionary with SHAP values and feature importance
        """
        if not SHAP_AVAILABLE or not self.explainer:
            logger.warning("SHAP not available. Install with: pip install shap")
            return None
        
        if not NUMPY_AVAILABLE:
            return None
        
        # Prepare features
        features = np.array([[
            session_params.get('cpu_cores', 8),
            session_params.get('cpu_threads', 16),
            session_params.get('cpu_freq_mhz', 3600),
            session_params.get('ram_gb', 16),
            session_params.get('gpu_vram_gb', 8),
            session_params.get('timer_resolution_ms', 0.5),
            session_params.get('cpu_priority', 1),
            1 if session_params.get('gpu_clock_locked', False) else 0,
            session_params.get('memory_optimization_level', 2),
            1 if session_params.get('network_qos_enabled', False) else 0,
            1 if session_params.get('core_parking_disabled', True) else 0,
        ]], dtype=np.float32)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # Create explanation dictionary
        explanation = {
            'feature_names': self.feature_names,
            'feature_values': features[0].tolist(),
            'shap_values': shap_values[0].tolist(),
            'base_value': float(self.explainer.expected_value),
        }
        
        # Sort by absolute SHAP value (importance)
        importance = sorted(
            zip(self.feature_names, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        explanation['feature_importance'] = [
            {'feature': name, 'impact': float(value)}
            for name, value in importance
        ]
        
        return explanation
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get overall feature importance from the model"""
        if not self.fps_model:
            return None
        
        try:
            importance_dict = self.fps_model.get_booster().get_score(importance_type='weight')
            
            # Map feature indices to names
            named_importance = {}
            for i, name in enumerate(self.feature_names):
                feature_key = f'f{i}'
                if feature_key in importance_dict:
                    named_importance[name] = importance_dict[feature_key]
                else:
                    named_importance[name] = 0.0
            
            return named_importance
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None
    
    def optimize_settings(self, hardware_params: Dict[str, Any],
                         target_metric: str = 'fps') -> Dict[str, Any]:
        """
        Find optimal settings for given hardware
        
        Args:
            hardware_params: Fixed hardware parameters (cpu_cores, ram_gb, etc.)
            target_metric: Optimization target ('fps' or 'stability')
        
        Returns:
            Optimized settings dictionary
        """
        if not self.fps_model or not self.stability_model:
            logger.error("Models not trained")
            return {}
        
        logger.info(f"Optimizing for {target_metric}...")
        
        # Define search space for optimizable parameters
        timer_resolutions = [0.5, 1.0]
        cpu_priorities = [0, 1, 2]
        gpu_clock_options = [False, True]
        memory_levels = [0, 1, 2]
        network_qos_options = [False, True]
        core_parking_options = [False, True]
        
        best_score = 0.0
        best_settings = {}
        
        # Grid search over parameter space
        for timer_res in timer_resolutions:
            for cpu_prio in cpu_priorities:
                for gpu_lock in gpu_clock_options:
                    for mem_level in memory_levels:
                        for net_qos in network_qos_options:
                            for core_park in core_parking_options:
                                params = {
                                    **hardware_params,
                                    'timer_resolution_ms': timer_res,
                                    'cpu_priority': cpu_prio,
                                    'gpu_clock_locked': gpu_lock,
                                    'memory_optimization_level': mem_level,
                                    'network_qos_enabled': net_qos,
                                    'core_parking_disabled': core_park,
                                }
                                
                                prediction = self.predict(params)
                                score = prediction[target_metric]
                                
                                if score > best_score:
                                    best_score = score
                                    best_settings = {
                                        'timer_resolution_ms': timer_res,
                                        'cpu_priority': cpu_prio,
                                        'gpu_clock_locked': gpu_lock,
                                        'memory_optimization_level': mem_level,
                                        'network_qos_enabled': net_qos,
                                        'core_parking_disabled': core_park,
                                        'predicted_fps': prediction['fps'],
                                        'predicted_stability': prediction['stability'],
                                        'confidence': prediction['confidence'],
                                    }
        
        logger.info(f"Optimal settings found: predicted {target_metric}={best_score:.2f}")
        return best_settings
    
    def export_model_info(self, output_file: Path) -> bool:
        """Export model information and statistics"""
        try:
            info = {
                'model_type': 'XGBoost Gradient Boosting',
                'training_sessions': len(self.training_data),
                'feature_names': self.feature_names,
                'feature_importance': self.get_feature_importance(),
                'models': {
                    'fps_model_trained': self.fps_model is not None,
                    'stability_model_trained': self.stability_model is not None,
                },
                'shap_available': SHAP_AVAILABLE,
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"Model info exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting model info: {e}")
            return False


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not XGBOOST_AVAILABLE:
        print("ERROR: XGBoost not installed. Install with: pip install xgboost")
        return
    
    nn = NeuralNetworkOptimizer()
    
    # Create sample training data
    print("\n=== Creating sample training data ===")
    for i in range(30):
        session = GameSession(
            game_name="Sample Game",
            game_exe="game.exe",
            cpu_cores=8,
            cpu_threads=16,
            cpu_freq_mhz=3600 + (i * 50),
            ram_gb=16,
            gpu_vram_gb=8,
            timer_resolution_ms=0.5,
            cpu_priority=1,
            gpu_clock_locked=True,
            memory_optimization_level=2,
            network_qos_enabled=True,
            core_parking_disabled=True,
            avg_fps=120 + (i * 2),
            fps_1_percent_low=95 + i,
            frame_time_avg_ms=8.3 - (i * 0.01),
            frame_time_p99_ms=12.0,
            frame_time_std_dev=1.5,
            avg_cpu_usage=60.0,
            avg_gpu_usage=95.0,
            avg_memory_usage_mb=8000,
            stutter_count=5,
            stability_score=85 + (i * 0.3),
            duration_seconds=300,
            timestamp=time.time()
        )
        nn.add_session(session)
    
    # Train models
    print("\n=== Training neural network ===")
    success = nn.train_models()
    
    if success:
        # Make prediction
        print("\n=== Making prediction ===")
        test_params = {
            'cpu_cores': 8,
            'cpu_threads': 16,
            'cpu_freq_mhz': 4000,
            'ram_gb': 16,
            'gpu_vram_gb': 8,
            'timer_resolution_ms': 0.5,
            'cpu_priority': 1,
            'gpu_clock_locked': True,
            'memory_optimization_level': 2,
            'network_qos_enabled': True,
            'core_parking_disabled': True,
        }
        
        prediction = nn.predict(test_params)
        print(f"Predicted FPS: {prediction['fps']:.1f}")
        print(f"Predicted Stability: {prediction['stability']:.1f}")
        print(f"Confidence: {prediction['confidence']:.2f}")
        
        # Explain prediction
        if SHAP_AVAILABLE:
            print("\n=== SHAP Explanation ===")
            explanation = nn.explain_prediction(test_params)
            if explanation:
                print("\nTop 5 features by impact:")
                for item in explanation['feature_importance'][:5]:
                    print(f"  {item['feature']}: {item['impact']:.4f}")


if __name__ == "__main__":
    main()
