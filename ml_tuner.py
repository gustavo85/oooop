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
    from sklearn.linear_model import SGDRegressor  # For incremental learning
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
    gpu_driver_version: Optional[str] = None  # NEW: Driver version for stability tracking


class MLAutoTuner:
    """Machine Learning Auto-Tuner for game profile optimization with dual-model prediction"""
    
    def __init__(self):
        self.model_dir = Path.home() / '.game_optimizer' / 'ml_models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_file = self.model_dir / 'fps_predictor.pkl'
        self.scaler_file = self.model_dir / 'scaler.pkl'
        self.incremental_model_file = self.model_dir / 'incremental_model.pkl'
        self.stability_model_file = self.model_dir / 'stability_predictor.pkl'  # NEW: Stability model
        
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.incremental_model: Optional[Any] = None  # For online learning
        self.stability_model: Optional[Any] = None  # NEW: For stability prediction
        
        self.training_data: List[Dict[str, Any]] = []
        self.training_data_file = self.model_dir / 'training_data.json'
        
        self.use_incremental_learning = True  # Enable incremental learning by default
        
        if SKLEARN_AVAILABLE:
            self._load_model()
            self._load_training_data()
        else:
            logger.warning("ML Auto-Tuner disabled (scikit-learn not installed)")
    
    def _load_model(self):
        """Load pre-trained models (FPS and stability) if they exist"""
        try:
            if self.model_file.exists() and self.scaler_file.exists():
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info("✓ ML FPS model loaded")
            else:
                logger.info("No pre-trained FPS model found. Will train on first use.")
            
            # Load incremental model if exists
            if self.incremental_model_file.exists():
                with open(self.incremental_model_file, 'rb') as f:
                    self.incremental_model = pickle.load(f)
                logger.info("✓ Incremental learning model loaded")
            else:
                # Initialize incremental learning model
                if SKLEARN_AVAILABLE:
                    self.incremental_model = SGDRegressor(
                        loss='squared_error',
                        penalty='l2',
                        alpha=0.0001,
                        learning_rate='adaptive',
                        eta0=0.01,
                        max_iter=1000,
                        warm_start=True  # Enable incremental learning
                    )
                    logger.info("✓ Incremental learning model initialized")
            
            # NEW: Load stability model if exists
            if self.stability_model_file.exists():
                with open(self.stability_model_file, 'rb') as f:
                    self.stability_model = pickle.load(f)
                logger.info("✓ Stability prediction model loaded")
            else:
                # Initialize stability model
                if SKLEARN_AVAILABLE:
                    from sklearn.linear_model import LogisticRegression
                    self.stability_model = LogisticRegression(
                        penalty='l2',
                        C=1.0,
                        max_iter=1000,
                        random_state=42
                    )
                    logger.info("✓ Stability prediction model initialized")
                
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
        """Save trained models (FPS and stability) to disk"""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            if self.model is not None:
                with open(self.model_file, 'wb') as f:
                    pickle.dump(self.model, f)
            
            if self.scaler is not None:
                with open(self.scaler_file, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            if self.incremental_model is not None:
                with open(self.incremental_model_file, 'wb') as f:
                    pickle.dump(self.incremental_model, f)
            
            # NEW: Save stability model
            if self.stability_model is not None:
                with open(self.stability_model_file, 'wb') as f:
                    pickle.dump(self.stability_model, f)
            
            logger.info("✓ ML models saved")
            
        except Exception as e:
            logger.error(f"Model save error: {e}")
    
    def add_training_sample(self, telemetry_data) -> bool:
        """Add telemetry data as training sample with incremental learning"""
        if not SKLEARN_AVAILABLE:
            return False
        
        try:
            # Extract features from telemetry including DPC latency and stutter count
            sample = {
                'game_exe': telemetry_data.game_exe,
                'cpu_model': telemetry_data.cpu_model,
                'gpu_model': telemetry_data.gpu_model,
                'ram_gb': telemetry_data.ram_gb,
                'optimizations': telemetry_data.optimizations,
                'avg_fps': telemetry_data.frame_metrics.avg_fps if telemetry_data.frame_metrics else 0,
                'one_percent_low': telemetry_data.frame_metrics.one_percent_low if telemetry_data.frame_metrics else 0,
                'stutter_count': telemetry_data.frame_metrics.stutter_count if telemetry_data.frame_metrics else 0,
                'cpu_usage': telemetry_data.cpu_usage_avg,
                'dpc_latency_avg_us': telemetry_data.dpc_latency_avg_us,
                'dpc_spikes_count': telemetry_data.dpc_spikes_count,
                'timestamp': time.time()
            }
            
            self.training_data.append(sample)
            
            # Save training data
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2)
            
            # Incremental learning: update model immediately with new sample
            if self.use_incremental_learning and self.incremental_model is not None:
                self._partial_fit_sample(sample)
            
            # Full retrain if we have enough samples
            if len(self.training_data) >= 20:
                self.train_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Add training sample error: {e}")
            return False
    
    def _partial_fit_sample(self, sample: Dict[str, Any]) -> bool:
        """Incrementally update model with a single sample (online learning)"""
        try:
            features = self._extract_features(sample)
            avg_fps = sample.get('avg_fps', 0)
            
            if not features or avg_fps <= 0:
                return False
            
            # Quality score: FPS - (stutter_count * 0.1) - (dpc_latency * 0.01)
            # This teaches the model that stutters and DPC latency are bad
            stutter_count = sample.get('stutter_count', 0)
            dpc_latency = sample.get('dpc_latency_avg_us', 0)
            quality_score = avg_fps - (stutter_count * 0.1) - (dpc_latency * 0.01)
            
            X = np.array([features])
            y = np.array([quality_score])
            
            # Scale if scaler exists, otherwise fit new scaler
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                # Partial fit scaler (update running statistics)
                try:
                    self.scaler.partial_fit(X)
                except AttributeError:
                    # StandardScaler doesn't have partial_fit, just transform
                    pass
                X_scaled = self.scaler.transform(X)
            
            # Partial fit the incremental model
            self.incremental_model.partial_fit(X_scaled, y)
            
            # Save updated models
            self.save_model()
            
            logger.info(f"✓ Incremental learning: updated model with new sample (quality={quality_score:.1f})")
            return True
            
        except Exception as e:
            logger.debug(f"Partial fit error: {e}")
            return False
    
    def train_model(self) -> bool:
        """Train both FPS and stability ML models on collected data"""
        if not SKLEARN_AVAILABLE or len(self.training_data) < 10:
            logger.warning("Not enough training data (need ≥10 samples)")
            return False
        
        try:
            logger.info(f"Training ML models on {len(self.training_data)} samples...")
            
            # Extract features and targets - separate datasets for FPS and stability
            X_fps = []
            y_fps = []
            
            X_stability = []
            y_stability = []
            
            for sample in self.training_data:
                features = self._extract_features(sample)
                if not features:
                    continue
                
                # For FPS training: only use successful optimizations with valid FPS
                if not sample.get('optimization_failed', False) and sample.get('avg_fps', 0) > 0:
                    X_fps.append(features)
                    y_fps.append(sample['avg_fps'])
                
                # For stability training: use ALL samples (especially failed ones are valuable)
                X_stability.append(features)
                # Stability target: 1 if unstable (high DPC latency OR failed optimization), 0 if stable
                dpc_max = sample.get('dpc_latency_max_us', 0)
                opt_failed = sample.get('optimization_failed', False)
                is_unstable = 1 if (dpc_max > 1000 or opt_failed) else 0
                y_stability.append(is_unstable)
            
            if len(X_fps) < 10:
                logger.warning("Not enough valid FPS samples after feature extraction")
                return False
            
            # Convert to numpy arrays
            X_fps = np.array(X_fps)
            y_fps = np.array(y_fps)
            
            # Split FPS data
            X_fps_train, X_fps_test, y_fps_train, y_fps_test = train_test_split(
                X_fps, y_fps, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_fps_train_scaled = self.scaler.fit_transform(X_fps_train)
            X_fps_test_scaled = self.scaler.transform(X_fps_test)
            
            # Train FPS model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_fps_train_scaled, y_fps_train)
            
            # Evaluate FPS model
            train_score = self.model.score(X_fps_train_scaled, y_fps_train)
            test_score = self.model.score(X_fps_test_scaled, y_fps_test)
            
            logger.info(f"✓ FPS model trained: R²_train={train_score:.3f}, R²_test={test_score:.3f}")
            
            # Train stability model (only if we have both classes and enough samples)
            if len(X_stability) >= 10:
                X_stability = np.array(X_stability)
                y_stability = np.array(y_stability)
                
                # Check if we have both classes
                unique_labels = np.unique(y_stability)
                if len(unique_labels) > 1:
                    # Split stability data with stratification
                    X_stab_train, X_stab_test, y_stab_train, y_stab_test = train_test_split(
                        X_stability, y_stability, test_size=0.2, random_state=42, stratify=y_stability)
                    
                    # Use same scaler (already fitted on FPS data)
                    X_stab_train_scaled = self.scaler.transform(X_stab_train)
                    X_stab_test_scaled = self.scaler.transform(X_stab_test)
                    
                    from sklearn.linear_model import LogisticRegression
                    self.stability_model = LogisticRegression(
                        penalty='l2',
                        C=1.0,
                        max_iter=1000,
                        random_state=42
                    )
                    
                    self.stability_model.fit(X_stab_train_scaled, y_stab_train)
                    
                    stab_score = self.stability_model.score(X_stab_test_scaled, y_stab_test)
                    logger.info(f"✓ Stability model trained: Accuracy={stab_score:.3f}")
                else:
                    logger.info("⚠️  Not enough diversity in stability labels, skipping stability model training")
            else:
                logger.info("⚠️  Not enough stability samples, skipping stability model training")
            
            # Save models
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False
    
    def _extract_features(self, sample: Dict) -> Optional[List[float]]:
        """Extract numerical features from sample including temperatures, driver version, and stability metrics"""
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
            
            # DPC latency (normalized to 0-1 range, 0=good, 1=bad)
            dpc_latency_us = float(sample.get('dpc_latency_avg_us', 0))
            # Normalize: 0-1000μs -> 0-1
            features.append(min(dpc_latency_us / 1000.0, 1.0))
            
            # Stutter count (normalized, log scale to handle large values)
            stutter_count = float(sample.get('stutter_count', 0))
            # Log normalize: helps with large outliers
            features.append(min(np.log1p(stutter_count) / 10.0, 1.0))
            
            # NEW: CPU temperature (normalized to 0-1, 0=cold, 1=hot)
            cpu_temp = float(sample.get('cpu_temp_avg', 50))
            # Normalize: 30-100°C -> 0-1
            features.append(max(0.0, min((cpu_temp - 30) / 70.0, 1.0)))
            
            # NEW: GPU temperature (normalized to 0-1, 0=cold, 1=hot)
            gpu_temp = float(sample.get('gpu_temp_avg', 50))
            # Normalize: 30-100°C -> 0-1
            features.append(max(0.0, min((gpu_temp - 30) / 70.0, 1.0)))
            
            # NEW: GPU driver version (encoded as hash to avoid version comparison issues)
            # Use simple hash-based encoding for driver version as categorical feature
            driver_version = sample.get('gpu_driver_version', '0.0')
            try:
                # Hash the driver version string and normalize to 0-1 range
                # This treats each version as a unique category without assuming ordering
                driver_hash = hash(driver_version) % 10000  # Mod to keep it reasonable
                driver_numeric = driver_hash / 10000.0
            except (ValueError, AttributeError, TypeError):
                driver_numeric = 0.0
            features.append(driver_numeric)
            
            # NEW: Max DPC latency (normalized, indicates worst-case stability)
            dpc_latency_max_us = float(sample.get('dpc_latency_max_us', 0))
            # Normalize: 0-2000μs -> 0-1
            features.append(min(dpc_latency_max_us / 2000.0, 1.0))
            
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
    
    def generate_ml_profile(self, game_exe: str, hardware_context: Dict[str, Any]) -> Optional[Any]:
        """
        Generate ML-optimized profile with dual-model prediction and safety filtering.
        
        Args:
            game_exe: Game executable name
            hardware_context: Dict with cpu_temp_avg, gpu_temp_avg, gpu_driver_version, etc.
            
        Returns:
            GameProfile if safe and confident, None otherwise
        """
        if not SKLEARN_AVAILABLE or self.model is None or self.scaler is None:
            logger.warning("ML models not available for profile generation")
            return None
        
        try:
            from config_loader import GameProfile
            
            # Test different optimization configurations
            base_optimizations = ['cpu_optimizations_v3', 'timer', 'power_plan']
            aggressive_optimizations = base_optimizations + ['gpu_clocks_locked', 'core_parking_disabled', 'network_qos', 'memory_purge']
            conservative_optimizations = base_optimizations.copy()
            
            # Build sample with current hardware context
            base_sample = {
                'game_exe': game_exe,
                'ram_gb': hardware_context.get('ram_gb', 16),
                'gpu_model': hardware_context.get('gpu_model', 'nvidia'),
                'cpu_temp_avg': hardware_context.get('cpu_temp_avg', 50),
                'gpu_temp_avg': hardware_context.get('gpu_temp_avg', 50),
                'gpu_driver_version': hardware_context.get('gpu_driver_version', '0.0'),
                'dpc_latency_avg_us': hardware_context.get('dpc_latency_avg_us', 0),
                'dpc_latency_max_us': hardware_context.get('dpc_latency_max_us', 0),
            }
            
            # Predict FPS for aggressive config
            aggressive_sample = {**base_sample, 'optimizations': aggressive_optimizations}
            aggressive_features = self._extract_features(aggressive_sample)
            
            if not aggressive_features:
                logger.warning("Failed to extract features for ML profile generation")
                return None
            
            X_aggressive = np.array([aggressive_features])
            X_aggressive_scaled = self.scaler.transform(X_aggressive)
            
            # Predict FPS
            predicted_fps = self.model.predict(X_aggressive_scaled)[0]
            
            # Calculate confidence (variance from RandomForest)
            confidence = 1.0  # Default high confidence
            if hasattr(self.model, 'estimators_'):
                # Get predictions from all trees
                tree_predictions = [tree.predict(X_aggressive_scaled)[0] for tree in self.model.estimators_]
                prediction_std = np.std(tree_predictions)
                # High std = low confidence
                confidence = max(0.0, 1.0 - (prediction_std / predicted_fps) if predicted_fps > 0 else 0.0)
            
            logger.info(f"ML prediction: {predicted_fps:.1f} FPS (confidence: {confidence:.2f})")
            
            # Predict stability risk
            stability_risk = 0.0
            if self.stability_model is not None:
                try:
                    # Get probability of instability (class 1)
                    risk_proba = self.stability_model.predict_proba(X_aggressive_scaled)[0]
                    stability_risk = risk_proba[1] if len(risk_proba) > 1 else 0.0
                    logger.info(f"Stability risk: {stability_risk:.2f}")
                except Exception as e:
                    logger.debug(f"Stability prediction error: {e}")
            
            # Safety filters
            use_conservative = False
            
            # Filter 1: High stability risk
            if stability_risk > 0.6:
                logger.warning("⚠️  High stability risk detected, using conservative profile")
                use_conservative = True
            
            # Filter 2: Low confidence
            if confidence < 0.5:
                logger.warning("⚠️  Low prediction confidence, using conservative profile")
                use_conservative = True
            
            # Filter 3: High GPU temperature (heuristic safety rule)
            gpu_temp = hardware_context.get('gpu_temp_avg', 0)
            if gpu_temp > 85:
                logger.warning(f"⚠️  High GPU temperature ({gpu_temp}°C), disabling GPU clock locking")
                use_conservative = True
            
            # Select optimization set
            if use_conservative:
                selected_opts = conservative_optimizations
            else:
                selected_opts = aggressive_optimizations
            
            # Build profile
            profile = GameProfile(
                name=f"ML Optimized - {game_exe}",
                game_exe=game_exe,
                timer_resolution_ms=0.5 if not use_conservative else 1.0,
                memory_optimization_level=2 if 'memory_purge' in selected_opts else 1,
                network_qos_enabled='network_qos' in selected_opts,
                gpu_scheduling_enabled=True,
                gpu_clock_locking='gpu_clocks_locked' in selected_opts and gpu_temp <= 85,
                power_high_performance=True,
                cpu_priority_class='HIGH',
                disable_core_parking='core_parking_disabled' in selected_opts,
                optimize_working_set=True,
            )
            
            logger.info(f"✓ Generated {'conservative' if use_conservative else 'aggressive'} ML profile")
            logger.info(f"  Optimizations: {', '.join(selected_opts)}")
            
            return profile
            
        except Exception as e:
            logger.error(f"ML profile generation error: {e}")
            return None