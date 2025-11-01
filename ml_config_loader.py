"""
ml_config_loader.py - Configuration Loader for ML Pipeline V5.0
Loads and validates ML configuration from YAML
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available. Using default configuration.")


class MLConfig:
    """
    Configuration manager for ML Pipeline
    
    Loads configuration from ml_config.yaml and provides
    typed access to all settings
    """
    
    DEFAULT_CONFIG = {
        'model_dir': '~/.game_optimizer/ml_models_v5',
        'cache_dir': '~/.game_optimizer/ml_cache',
        'logs_dir': '~/.game_optimizer/ml_logs',
        
        'feature_engineering': {
            'enabled': True,
            'num_features': 96,
            'scaler_type': 'robust'
        },
        
        'deep_learning': {
            'enabled': True,
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.001
        },
        
        'gradient_boosting': {
            'enabled': True,
            'xgboost': {'enabled': True, 'n_estimators': 500},
            'lightgbm': {'enabled': True, 'n_estimators': 500},
            'catboost': {'enabled': True, 'iterations': 500}
        },
        
        'automl': {
            'enabled': False,
            'n_trials': 100
        },
        
        'explainability': {
            'enabled': True,
            'shap': {'enabled': True, 'top_k_features': 5}
        },
        
        'monitoring': {
            'enabled': True,
            'drift_detection': {'enabled': True, 'threshold': 0.2}
        },
        
        'auto_retraining': {
            'enabled': True,
            'triggers': {
                'min_sessions': 500,
                'time_interval': 604800
            }
        },
        
        'validation': {
            'test_size': 0.2,
            'success_criteria': {
                'fps_mape': 0.08,
                'stability_r2': 0.85,
                'success_f1': 0.90
            }
        },
        
        'logging': {
            'level': 'INFO',
            'file': 'ml_pipeline.log'
        },
        
        'random_seeds': {
            'global': 42,
            'train_split': 42
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: Path to ml_config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Try to find config in current directory or package directory
            current_dir = Path(__file__).parent
            config_path = current_dir / 'ml_config.yaml'
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        
        # Start with default config
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Try to load from file
        if YAML_AVAILABLE and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    
                if yaml_config:
                    self._deep_update(self.config, yaml_config)
                    logger.info(f"✓ ML config loaded from {self.config_path}")
                else:
                    logger.warning(f"Empty config file: {self.config_path}. Using defaults.")
                    
            except Exception as e:
                logger.warning(f"Error loading ML config: {e}. Using defaults.")
        else:
            if not YAML_AVAILABLE:
                logger.warning("PyYAML not installed. Using default ML configuration.")
            else:
                logger.info(f"Config file not found: {self.config_path}. Using defaults.")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation
        
        Example:
            config.get('deep_learning.learning_rate')
            config.get('gradient_boosting.xgboost.enabled')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set config value using dot notation
        
        Example:
            config.set('deep_learning.learning_rate', 0.01)
        """
        keys = key_path.split('.')
        target = self.config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    # Convenience properties for common config values
    
    @property
    def model_dir(self) -> Path:
        """Get model directory path"""
        return Path(self.get('model_dir', '~/.game_optimizer/ml_models_v5')).expanduser()
    
    @property
    def is_deep_learning_enabled(self) -> bool:
        """Check if deep learning is enabled"""
        return self.get('deep_learning.enabled', True)
    
    @property
    def is_gradient_boosting_enabled(self) -> bool:
        """Check if gradient boosting is enabled"""
        return self.get('gradient_boosting.enabled', True)
    
    @property
    def is_automl_enabled(self) -> bool:
        """Check if AutoML is enabled"""
        return self.get('automl.enabled', False)
    
    @property
    def is_explainability_enabled(self) -> bool:
        """Check if explainability is enabled"""
        return self.get('explainability.enabled', True)
    
    @property
    def test_size(self) -> float:
        """Get validation test size"""
        return self.get('validation.test_size', 0.2)
    
    @property
    def random_seed(self) -> int:
        """Get global random seed"""
        return self.get('random_seeds.global', 42)
    
    @property
    def batch_size(self) -> int:
        """Get training batch size"""
        return self.get('deep_learning.batch_size', 64)
    
    @property
    def learning_rate(self) -> float:
        """Get learning rate"""
        return self.get('deep_learning.learning_rate', 0.001)
    
    @property
    def epochs(self) -> int:
        """Get number of training epochs"""
        return self.get('deep_learning.epochs', 100)
    
    def __repr__(self) -> str:
        return f"MLConfig(config_path={self.config_path})"


# Global config instance
_global_config: Optional[MLConfig] = None


def get_ml_config(config_path: Optional[Path] = None) -> MLConfig:
    """
    Get global ML config instance
    
    Args:
        config_path: Optional path to config file. Only used on first call.
        
    Returns:
        MLConfig instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = MLConfig(config_path)
    
    return _global_config


def reload_ml_config(config_path: Optional[Path] = None):
    """
    Reload ML configuration from file
    
    Args:
        config_path: Optional new config path
    """
    global _global_config
    _global_config = MLConfig(config_path)
    logger.info("ML configuration reloaded")
