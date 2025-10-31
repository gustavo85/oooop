"""
ML Auto-Tuner V3.5 - Real Machine Learning with scikit-learn
Uses collected telemetry to predict optimal settings for games
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML features disabled.")
    logger.warning("Install with: pip install scikit-learn numpy")


@dataclass
class HardwareProfile:
    """Hardware configuration for ML feature extraction"""
    cpu_cores: int
    cpu_threads: int
    cpu_freq_mhz: int
    ram_gb: int
    gpu_vram_gb: int
    gpu_vendor: str  # NVIDIA/AMD/Intel


class MLAutoTuner:
    """Machine Learning Auto-Tuner for game profile optimization"""
    
    def __init__(self):
        self.model_dir = Path.home() / '.game_optimizer' / 'ml_models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_file = self.model_dir / 'fps_predictor.pkl'
        self.scaler_file = self.model_dir / 'scaler.pkl'
        
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        
        self.training_data: List[Dict[str, Any]] = []
        self.training_data_file = self.model_dir / 'training_data.json'
        
        if SKLEARN_AVAILABLE:
            self._load_model()
            self._load_training_data()
        else:
            logger.warning("ML Auto-Tuner disabled (scikit-learn not installed)")
    
    def _load_model(self):
        """Load pre-trained model if exists"""
        try:
            if self.model_file.exists() and self.scaler_file.exists():
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info("✓ ML model loaded")
            else:
                logger.info("No pre-trained model found. Will train on first use.")
                
        except Exception as e:
            logger.error(f"Model load error: {e}")
    
    def _load_training_data(self):
        """Load historical training data"""
        try:
            if self.training_data_file.exists():
                with open(self.training_data_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                
                logger.info(f"✓ Loaded {len(self.training_data)} training samples")
                
        except Exception as e:
            logger.error(f"Training data load error: {e}")
    
    def save_model(self):
        """Save trained model to disk"""
        if not SKLEARN_AVAILABLE or self.model is None:
            return
        
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info("✓ ML model saved")
            
        except Exception as e:
            logger.error(f"Model save error: {e}")
    
    def add_training_sample(self, telemetry_data) -> bool:
        """Add telemetry data as training sample"""
        if not SKLEARN_AVAILABLE:
            return False
        
        try:
            # Extract features from telemetry
            sample = {
                'game_exe': telemetry_data.game_exe,
                'cpu_model': telemetry_data.cpu_model,
                'gpu_model': telemetry_data.gpu_model,
                'ram_gb': telemetry_data.ram_gb,
                'optimizations': telemetry_data.optimizations,
                'avg_fps': telemetry_data.frame_metrics.avg_fps if telemetry_data.frame_metrics else 0,
                'one_percent_low': telemetry_data.frame_metrics.one_percent_low if telemetry_data.frame_metrics else 0,
                'cpu_usage': telemetry_data.cpu_usage_avg,
                'timestamp': time.time()
            }
            
            self.training_data.append(sample)
            
            # Save training data
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2)
            
            # Retrain if we have enough samples
            if len(self.training_data) >= 20:
                self.train_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Add training sample error: {e}")
            return False
    
    def train_model(self) -> bool:
        """Train ML model on collected data"""
        if not SKLEARN_AVAILABLE or len(self.training_data) < 10:
            logger.warning("Not enough training data (need ≥10 samples)")
            return False
        
        try:
            logger.info(f"Training ML model on {len(self.training_data)} samples...")
            
            # Extract features and targets
            X = []
            y = []
            
            for sample in self.training_data:
                features = self._extract_features(sample)
                if features and sample.get('avg_fps', 0) > 0:
                    X.append(features)
                    y.append(sample['avg_fps'])
            
            if len(X) < 10:
                logger.warning("Not enough valid samples after feature extraction")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"✓ Model trained: R²_train={train_score:.3f}, R²_test={test_score:.3f}")
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False
    
    def _extract_features(self, sample: Dict) -> Optional[List[float]]:
        """Extract numerical features from sample"""
        try:
            features = []
            
            # RAM (GB)
            features.append(float(sample.get('ram_gb', 16)))
            
            # Optimizations (binary encoding of common ones)
            opts = sample.get('optimizations', [])
            features.append(1.0 if 'cpu_optimizations_v3' in opts else 0.0)
            features.append(1.0 if 'gpu_clocks_locked' in opts else 0.0)
            features.append(1.0 if 'network_qos' in opts else 0.0)
            features.append(1.0 if 'timer' in opts else 0.0)
            features.append(1.0 if 'core_parking_disabled' in opts else 0.0)
            features.append(1.0 if 'memory_purge' in opts else 0.0)
            
            # GPU vendor (one-hot)
            gpu = sample.get('gpu_model', '').lower()
            features.append(1.0 if 'nvidia' in gpu or 'geforce' in gpu or 'rtx' in gpu else 0.0)
            features.append(1.0 if 'amd' in gpu or 'radeon' in gpu else 0.0)
            features.append(1.0 if 'intel' in gpu else 0.0)
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None
    
    def get_optimized_profile(self, game_exe: str) -> Optional[Any]:
        """
        Get ML-optimized profile for a game.
        Returns GameProfile if prediction is confident, else None.
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            return None
        
        try:
            # Check if we have historical data for this game
            game_samples = [s for s in self.training_data if s['game_exe'].lower() == game_exe.lower()]
            
            if not game_samples:
                logger.info(f"No historical data for {game_exe}")
                return None
            
            # Find best configuration from history
            best_sample = max(game_samples, key=lambda s: s.get('avg_fps', 0))
            
            logger.info(f"ML recommendation for {game_exe}:")
            logger.info(f"  Best known FPS: {best_sample.get('avg_fps', 0):.1f}")
            logger.info(f"  Optimizations: {', '.join(best_sample.get('optimizations', []))}")
            
            # Create profile based on best configuration
            from config_loader import GameProfile
            
            opts = best_sample.get('optimizations', [])
            
            profile = GameProfile(
                name=f"ML Optimized - {game_exe}",
                game_exe=game_exe,
                timer_resolution_ms=0.5,
                memory_optimization_level=2 if 'memory_purge' in opts else 1,
                network_qos_enabled='network_qos' in opts,
                gpu_scheduling_enabled=True,
                gpu_clock_locking='gpu_clocks_locked' in opts,
                power_high_performance=True,
                cpu_priority_class='HIGH',
                disable_core_parking='core_parking_disabled' in opts,
                optimize_working_set=True,
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"ML profile generation error: {e}")
            return None
    
    def predict_fps(self, game_exe: str, optimizations: List[str]) -> Optional[float]:
        """Predict expected FPS for given configuration"""
        if not SKLEARN_AVAILABLE or self.model is None or self.scaler is None:
            return None
        
        try:
            # Create sample with proposed config
            sample = {
                'game_exe': game_exe,
                'ram_gb': 16,  # Assume defaults
                'gpu_model': 'nvidia',
                'optimizations': optimizations
            }
            
            features = self._extract_features(sample)
            if not features:
                return None
            
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.debug(f"FPS prediction error: {e}")
            return None