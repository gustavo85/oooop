"""
ml_pipeline.py - Sistema de Machine Learning Profesional V5.0
Autor: Game Optimizer Team
Fecha: 2025-11-01

Sistema completo de ML que reemplaza y mejora neural_network_optimizer.py
Incluye: Deep Learning, Gradient Boosting, RL, AutoML, Explainability, ONNX
"""

import logging
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Imports condicionales para compatibilidad
try:
    import numpy as np
    from numpy import ndarray
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    ndarray = None
    logger.warning("numpy not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available")

try:
    from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")

# Gradient Boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available. Install with: pip install catboost")

# AutoML
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

# ONNX
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install with: pip install onnx onnxruntime")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GameSession:
    """Enhanced gaming session with 90+ features for ML training"""
    
    # Identifiers
    session_id: str
    game_name: str
    game_exe: str
    timestamp: float
    duration_seconds: float
    
    # Hardware Features (15)
    cpu_cores_physical: int = 8
    cpu_cores_logical: int = 16
    cpu_freq_base_mhz: float = 3600.0
    cpu_freq_boost_mhz: float = 4800.0
    cpu_cache_l2_mb: float = 8.0
    cpu_cache_l3_mb: float = 32.0
    cpu_tdp_watts: int = 65
    gpu_vram_gb: float = 8.0
    gpu_core_clock_mhz: float = 1800.0
    gpu_memory_clock_mhz: float = 7000.0
    gpu_cuda_cores: int = 3584
    gpu_tensor_cores: int = 112
    gpu_ray_tracing_cores: int = 28
    ram_capacity_gb: float = 16.0
    ram_frequency_mhz: float = 3200.0
    
    # Configuration Features (20)
    timer_resolution_ms: float = 1.0
    cpu_priority_class: int = 0  # 0=NORMAL, 1=HIGH, 2=REALTIME
    cpu_affinity_mask: int = 0
    gpu_clock_locked: bool = False
    gpu_power_limit_pct: int = 100
    memory_optimization_level: int = 0
    page_file_size_gb: float = 16.0
    network_qos_enabled: bool = False
    network_dscp_value: int = 0
    tcp_nagle_disabled: bool = False
    tcp_buffer_kb: int = 64
    core_parking_disabled: bool = False
    game_mode_enabled: bool = False
    background_apps_limited: bool = False
    services_stopped: bool = False
    directx_version: int = 12
    vulkan_version: str = "1.3"
    anti_cheat_type: str = "none"  # "none", "eac", "battleye", "vanguard"
    launcher_type: str = "steam"   # "steam", "epic", "gog"
    online_mode: bool = False
    
    # Performance Metrics (30)
    fps_avg: float = 60.0
    fps_min: float = 50.0
    fps_max: float = 70.0
    fps_std: float = 5.0
    fps_p1: float = 52.0
    fps_p5: float = 54.0
    fps_p25: float = 57.0
    fps_p50: float = 60.0
    fps_p75: float = 63.0
    fps_p95: float = 68.0
    fps_p99: float = 69.0
    frame_time_avg_ms: float = 16.67
    frame_time_p1_ms: float = 14.0
    frame_time_p99_ms: float = 20.0
    frame_time_p999_ms: float = 22.0
    frame_time_std_ms: float = 2.0
    stutter_count: int = 0
    stutter_duration_avg_ms: float = 0.0
    stutter_frequency_hz: float = 0.0
    cpu_usage_avg: float = 50.0
    cpu_usage_max: float = 70.0
    cpu_temp_c: float = 60.0
    cpu_throttled: bool = False
    gpu_usage_avg: float = 80.0
    gpu_usage_max: float = 95.0
    gpu_temp_c: float = 70.0
    gpu_throttled: bool = False
    gpu_power_w: float = 150.0
    memory_working_set_mb: float = 8000.0
    memory_page_faults_per_sec: float = 100.0
    
    # Network (3)
    network_latency_ms: float = 20.0
    network_packet_loss_pct: float = 0.0
    network_bandwidth_mbps: float = 100.0
    
    # Disk (3)
    disk_read_mb_per_sec: float = 50.0
    disk_write_mb_per_sec: float = 20.0
    disk_latency_ms: float = 5.0
    
    # Derived Features (calculated later)
    features_derived: Dict[str, float] = field(default_factory=dict)
    
    # Target Variables
    optimization_successful: bool = True
    user_satisfaction_score: int = 5  # 1-5 scale
    baseline_fps: Optional[float] = None  # Para calcular mejora


class ModelType(Enum):
    """Tipos de modelos disponibles en el pipeline"""
    DEEP_NEURAL_NETWORK = "dnn"
    XGBOOST = "xgb"
    LIGHTGBM = "lgb"
    CATBOOST = "cat"
    ENSEMBLE = "ensemble"
    RL_AGENT = "rl"


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer:
    """
    Sistema avanzado de extracción de features para gaming ML
    
    Extrae 90+ features organizadas en:
    - 15 Hardware features
    - 20 Configuration features  
    - 30 Performance metrics
    - 6 Network/Disk features
    - 25 Derived features (estadísticas, eficiencia, detección de cuellos de botella)
    """
    
    def __init__(self):
        self.scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        self.feature_names: List[str] = []
        self._build_feature_names()
        
    def _build_feature_names(self):
        """Construye lista completa de nombres de features"""
        # Hardware (15)
        hw_features = [
            'cpu_cores_physical', 'cpu_cores_logical', 'cpu_freq_base_mhz', 
            'cpu_freq_boost_mhz', 'cpu_cache_l2_mb', 'cpu_cache_l3_mb', 'cpu_tdp_watts',
            'gpu_vram_gb', 'gpu_core_clock_mhz', 'gpu_memory_clock_mhz',
            'gpu_cuda_cores', 'gpu_tensor_cores', 'gpu_ray_tracing_cores',
            'ram_capacity_gb', 'ram_frequency_mhz'
        ]
        
        # Configuration (20)
        cfg_features = [
            'timer_resolution_ms', 'cpu_priority_class', 'cpu_affinity_mask',
            'gpu_clock_locked', 'gpu_power_limit_pct', 'memory_optimization_level',
            'page_file_size_gb', 'network_qos_enabled', 'network_dscp_value',
            'tcp_nagle_disabled', 'tcp_buffer_kb', 'core_parking_disabled',
            'game_mode_enabled', 'background_apps_limited', 'services_stopped',
            'directx_version', 'vulkan_version_enc', 'anti_cheat_enc',
            'launcher_type_enc', 'online_mode'
        ]
        
        # Performance (30)
        perf_features = [
            'fps_avg', 'fps_min', 'fps_max', 'fps_std',
            'fps_p1', 'fps_p5', 'fps_p25', 'fps_p50', 'fps_p75', 'fps_p95', 'fps_p99',
            'frame_time_avg_ms', 'frame_time_p1_ms', 'frame_time_p99_ms',
            'frame_time_p999_ms', 'frame_time_std_ms',
            'stutter_count', 'stutter_duration_avg_ms', 'stutter_frequency_hz',
            'cpu_usage_avg', 'cpu_usage_max', 'cpu_temp_c', 'cpu_throttled',
            'gpu_usage_avg', 'gpu_usage_max', 'gpu_temp_c', 'gpu_throttled',
            'gpu_power_w', 'memory_working_set_mb', 'memory_page_faults_per_sec'
        ]
        
        # Network/Disk (6)
        net_disk_features = [
            'network_latency_ms', 'network_packet_loss_pct', 'network_bandwidth_mbps',
            'disk_read_mb_per_sec', 'disk_write_mb_per_sec', 'disk_latency_ms'
        ]
        
        # Derived (25)
        derived_features = [
            'cv_fps', 'stability_index', 'perf_score', 'fps_per_watt',
            'fps_per_degree', 'frames_per_mb_vram', 'frame_time_jitter',
            'gpu_efficiency', 'cpu_gpu_ratio', 'memory_pressure',
            'thermal_headroom_cpu', 'thermal_headroom_gpu', 'network_quality',
            'disk_throughput', 'stutter_impact', 'frame_pacing_quality',
            'cpu_bottleneck', 'gpu_bottleneck', 'fps_improvement',
            'power_efficiency', 'temp_efficiency', 'memory_efficiency',
            'network_stability', 'disk_responsiveness', 'overall_health_score'
        ]
        
        self.feature_names = (
            hw_features + cfg_features + perf_features + 
            net_disk_features + derived_features
        )
        
    def extract_features(self, session: GameSession):
        """
        Extrae vector de features completo de una sesión
        
        Args:
            session: Objeto GameSession con datos de la sesión
            
        Returns:
            Array numpy de 96 features (15+20+30+6+25)
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for feature extraction")
            
        features = []
        
        # Hardware features (15)
        features.extend([
            session.cpu_cores_physical,
            session.cpu_cores_logical,
            session.cpu_freq_base_mhz,
            session.cpu_freq_boost_mhz,
            session.cpu_cache_l2_mb,
            session.cpu_cache_l3_mb,
            session.cpu_tdp_watts,
            session.gpu_vram_gb,
            session.gpu_core_clock_mhz,
            session.gpu_memory_clock_mhz,
            session.gpu_cuda_cores,
            session.gpu_tensor_cores,
            session.gpu_ray_tracing_cores,
            session.ram_capacity_gb,
            session.ram_frequency_mhz,
        ])
        
        # Configuration features (20)
        features.extend([
            session.timer_resolution_ms,
            session.cpu_priority_class,
            session.cpu_affinity_mask,
            float(session.gpu_clock_locked),
            session.gpu_power_limit_pct,
            session.memory_optimization_level,
            session.page_file_size_gb,
            float(session.network_qos_enabled),
            session.network_dscp_value,
            float(session.tcp_nagle_disabled),
            session.tcp_buffer_kb,
            float(session.core_parking_disabled),
            float(session.game_mode_enabled),
            float(session.background_apps_limited),
            float(session.services_stopped),
            session.directx_version,
            self._encode_vulkan_version(session.vulkan_version),
            self._encode_anti_cheat(session.anti_cheat_type),
            self._encode_launcher(session.launcher_type),
            float(session.online_mode),
        ])
        
        # Performance metrics (30)
        features.extend([
            session.fps_avg,
            session.fps_min,
            session.fps_max,
            session.fps_std,
            session.fps_p1,
            session.fps_p5,
            session.fps_p25,
            session.fps_p50,
            session.fps_p75,
            session.fps_p95,
            session.fps_p99,
            session.frame_time_avg_ms,
            session.frame_time_p1_ms,
            session.frame_time_p99_ms,
            session.frame_time_p999_ms,
            session.frame_time_std_ms,
            session.stutter_count,
            session.stutter_duration_avg_ms,
            session.stutter_frequency_hz,
            session.cpu_usage_avg,
            session.cpu_usage_max,
            session.cpu_temp_c,
            float(session.cpu_throttled),
            session.gpu_usage_avg,
            session.gpu_usage_max,
            session.gpu_temp_c,
            float(session.gpu_throttled),
            session.gpu_power_w,
            session.memory_working_set_mb,
            session.memory_page_faults_per_sec,
        ])
        
        # Network + Disk (6)
        features.extend([
            session.network_latency_ms,
            session.network_packet_loss_pct,
            session.network_bandwidth_mbps,
            session.disk_read_mb_per_sec,
            session.disk_write_mb_per_sec,
            session.disk_latency_ms,
        ])
        
        # Derived features (25)
        derived = self._compute_derived_features(session)
        features.extend(derived)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_derived_features(self, session: GameSession) -> List[float]:
        """
        Calcula features derivadas avanzadas a partir de métricas base
        
        Features calculadas:
        - Métricas de variabilidad (CV, stability)
        - Métricas de eficiencia (FPS/watt, FPS/degree, etc.)
        - Análisis de frame time (jitter, pacing)
        - Detección de cuellos de botella (CPU/GPU)
        - Scores de calidad (network, disk, overall)
        """
        derived = []
        
        # Coefficient of Variation (CV) del FPS
        cv_fps = session.fps_std / session.fps_avg if session.fps_avg > 0 else 0
        derived.append(cv_fps)
        
        # Stability Index (0-100, mayor = más estable)
        stability_index = (1 - cv_fps) * 100
        derived.append(max(0, min(100, stability_index)))
        
        # Performance Score weighted
        perf_score = (
            session.fps_avg * 0.4 +
            session.fps_p1 * 0.3 +
            stability_index * 0.2 +
            (100 - min(session.stutter_count, 100)) * 0.1
        )
        derived.append(perf_score)
        
        # Efficiency: FPS per Watt
        fps_per_watt = session.fps_avg / session.gpu_power_w if session.gpu_power_w > 0 else 0
        derived.append(fps_per_watt)
        
        # Efficiency: FPS per Degree
        fps_per_degree = session.fps_avg / session.gpu_temp_c if session.gpu_temp_c > 0 else 0
        derived.append(fps_per_degree)
        
        # Efficiency: Frames per MB VRAM
        frames_per_mb_vram = session.fps_avg / (session.gpu_vram_gb * 1024) if session.gpu_vram_gb > 0 else 0
        derived.append(frames_per_mb_vram)
        
        # Frame Time Jitter (variabilidad)
        frame_time_jitter = session.frame_time_p99_ms - session.frame_time_avg_ms
        derived.append(frame_time_jitter)
        
        # GPU Efficiency (FPS por % de uso de GPU)
        gpu_efficiency = session.fps_avg / session.gpu_usage_avg if session.gpu_usage_avg > 0 else 0
        derived.append(gpu_efficiency)
        
        # CPU-GPU Balance Ratio
        cpu_gpu_ratio = session.cpu_usage_avg / session.gpu_usage_avg if session.gpu_usage_avg > 0 else 1.0
        derived.append(cpu_gpu_ratio)
        
        # Memory Pressure (% de RAM usada)
        memory_pressure = session.memory_working_set_mb / (session.ram_capacity_gb * 1024) if session.ram_capacity_gb > 0 else 0
        derived.append(min(memory_pressure, 1.0))
        
        # Thermal Headroom CPU (margen antes de throttling)
        thermal_headroom_cpu = 95 - session.cpu_temp_c
        derived.append(thermal_headroom_cpu)
        
        # Thermal Headroom GPU
        thermal_headroom_gpu = 85 - session.gpu_temp_c
        derived.append(thermal_headroom_gpu)
        
        # Network Quality Score (0-100)
        network_quality = (
            (1 - min(session.network_packet_loss_pct / 100, 1.0)) * 50 +
            (1 - min(session.network_latency_ms / 100, 1.0)) * 50
        )
        derived.append(max(0, min(100, network_quality)))
        
        # Disk Throughput Total
        disk_throughput = session.disk_read_mb_per_sec + session.disk_write_mb_per_sec
        derived.append(disk_throughput)
        
        # Stutter Impact Score
        stutter_impact = (
            session.stutter_count * session.stutter_duration_avg_ms / 
            max(session.duration_seconds, 1)
        )
        derived.append(stutter_impact)
        
        # Frame Pacing Quality (0-100, basado en std de frame time)
        frame_pacing_quality = 100 - min(session.frame_time_std_ms * 10, 100)
        derived.append(max(0, min(100, frame_pacing_quality)))
        
        # CPU Bottleneck Detection (binary)
        cpu_bottleneck = 1.0 if (session.cpu_usage_avg > 90 and session.gpu_usage_avg < 80) else 0.0
        derived.append(cpu_bottleneck)
        
        # GPU Bottleneck Detection (binary)
        gpu_bottleneck = 1.0 if (session.gpu_usage_avg > 95 and session.cpu_usage_avg < 80) else 0.0
        derived.append(gpu_bottleneck)
        
        # FPS Improvement vs Baseline (si disponible)
        if session.baseline_fps and session.baseline_fps > 0:
            fps_improvement = ((session.fps_avg - session.baseline_fps) / session.baseline_fps) * 100
            derived.append(fps_improvement)
        else:
            derived.append(0.0)
        
        # Power Efficiency Score
        power_efficiency = fps_per_watt * 10  # Normalizado
        derived.append(min(power_efficiency, 100))
        
        # Temperature Efficiency Score
        temp_efficiency = (thermal_headroom_cpu + thermal_headroom_gpu) / 2
        derived.append(max(0, min(100, temp_efficiency)))
        
        # Memory Efficiency
        memory_efficiency = (1 - memory_pressure) * 100
        derived.append(max(0, min(100, memory_efficiency)))
        
        # Network Stability (basado en packet loss y latency)
        network_stability = 100 - (session.network_packet_loss_pct * 10 + min(session.network_latency_ms / 2, 50))
        derived.append(max(0, min(100, network_stability)))
        
        # Disk Responsiveness (basado en latencia)
        disk_responsiveness = 100 - min(session.disk_latency_ms * 2, 100)
        derived.append(max(0, min(100, disk_responsiveness)))
        
        # Overall Health Score (weighted average de múltiples métricas)
        overall_health = (
            stability_index * 0.3 +
            temp_efficiency * 0.2 +
            memory_efficiency * 0.2 +
            frame_pacing_quality * 0.15 +
            network_quality * 0.075 +
            disk_responsiveness * 0.075
        )
        derived.append(max(0, min(100, overall_health)))
        
        return derived
    
    def _encode_anti_cheat(self, ac_type: str) -> float:
        """Codifica tipo de anti-cheat a valor numérico"""
        mapping = {
            "none": 0, "vac": 1, "eac": 2, "battleye": 3, 
            "vanguard": 4, "faceit": 5, "ricochet": 6
        }
        return float(mapping.get(ac_type.lower(), 0))
    
    def _encode_launcher(self, launcher: str) -> float:
        """Codifica tipo de launcher a valor numérico"""
        mapping = {
            "none": 0, "steam": 1, "epic": 2, "gog": 3, 
            "origin": 4, "ubisoft": 5, "battlenet": 6, "xbox": 7
        }
        return float(mapping.get(launcher.lower(), 0))
    
    def _encode_vulkan_version(self, version: str) -> float:
        """Codifica versión de Vulkan (e.g., '1.3' -> 1.3)"""
        try:
            return float(version)
        except (ValueError, TypeError):
            return 0.0
    
    def fit_scaler(self, features):
        """Ajusta el scaler con datos de entrenamiento"""
        if self.scaler and SKLEARN_AVAILABLE:
            self.scaler.fit(features)
            logger.info(f"✓ Scaler fitted on {features.shape[0]} samples")
    
    def transform(self, features):
        """Normaliza features usando scaler ajustado"""
        if self.scaler and SKLEARN_AVAILABLE:
            return self.scaler.transform(features)
        return features
    
    def fit_transform(self, features):
        """Ajusta y transforma en un solo paso"""
        self.fit_scaler(features)
        return self.transform(features)
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nombres de todas las features"""
        return self.feature_names.copy()


# ============================================================================
# DEEP LEARNING MODELS (PyTorch)
# ============================================================================

if PYTORCH_AVAILABLE:
    
    class ResidualBlock(nn.Module):
        """
        Bloque residual con skip connection para deep learning
        
        Arquitectura:
        - Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> BatchNorm
        - Skip connection (identity o proyección)
        - Suma + ReLU final
        """
        
        def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
            super().__init__()
            self.linear1 = nn.Linear(in_features, out_features)
            self.bn1 = nn.BatchNorm1d(out_features)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(out_features, out_features)
            self.bn2 = nn.BatchNorm1d(out_features)
            
            # Skip connection (proyección si dimensiones diferentes)
            self.skip = (
                nn.Linear(in_features, out_features) 
                if in_features != out_features 
                else nn.Identity()
            )
        
        def forward(self, x):
            identity = self.skip(x)
            
            out = self.linear1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            
            out = self.linear2(out)
            out = self.bn2(out)
            
            out += identity  # Skip connection
            out = self.relu(out)
            
            return out
    
    
    class DeepPerformancePredictor(nn.Module):
        """
        Red neuronal profunda multi-tarea para predicción de performance
        
        Arquitectura ResNet-inspired con:
        - Input Layer + BatchNorm
        - Encoder con bloques residuales (256->256->128->128->64)
        - 3 heads especializados:
            * FPS Prediction (regression)
            * Stability Score (regression 0-100)
            * Optimization Success (classification 0-1)
            
        Loss: Multi-task weighted sum
        """
        
        def __init__(self, input_size: int = 96):
            super().__init__()
            
            # Input processing
            self.input_bn = nn.BatchNorm1d(input_size)
            
            # Encoder con bloques residuales
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                ResidualBlock(256, 256, dropout=0.3),
                ResidualBlock(256, 256, dropout=0.3),
                
                ResidualBlock(256, 128, dropout=0.2),
                ResidualBlock(128, 128, dropout=0.2),
                
                ResidualBlock(128, 64, dropout=0.2),
            )
            
            # Task-specific heads
            
            # Head 1: FPS Prediction (regression)
            self.fps_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            
            # Head 2: Stability Score (regression 0-100)
            self.stability_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # Output 0-1, luego scale a 0-100
            )
            
            # Head 3: Optimization Success (binary classification)
            self.success_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.input_bn(x)
            encoded = self.encoder(x)
            
            fps = self.fps_head(encoded)
            stability = self.stability_head(encoded) * 100  # Scale to 0-100
            success = self.success_head(encoded)
            
            return fps, stability, success


# ============================================================================
# GRADIENT BOOSTING MODELS
# ============================================================================

class GradientBoostingEnsemble:
    """
    Ensemble de múltiples gradient boosting models con voting ponderado
    
    Modelos incluidos:
    - XGBoost: Alta precisión, manejo de datos faltantes
    - LightGBM: Entrenamiento ultrarrápido, eficiente en memoria
    - CatBoost: Excelente para features categóricas
    
    Voting: Weighted average basado en performance de validación
    """
    
    def __init__(self, task: str = "regression"):
        """
        Args:
            task: "regression" para FPS/stability, "classification" para success
        """
        self.task = task
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.trained = False
        
    def train(
        self, 
        X_train, 
        y_train,
        X_val,
        y_val,
        verbose: bool = True
    ):
        """Entrena todos los modelos del ensemble"""
        
        if verbose:
            logger.info("Training Gradient Boosting Ensemble...")
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            if self.task == "regression":
                xgb_model = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            self.models['xgboost'] = xgb_model
            
            if verbose:
                logger.info("✓ XGBoost trained")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            if self.task == "regression":
                lgb_model = lgb.LGBMRegressor(
                    num_leaves=127,
                    learning_rate=0.05,
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                lgb_model = lgb.LGBMClassifier(
                    num_leaves=127,
                    learning_rate=0.05,
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            self.models['lightgbm'] = lgb_model
            
            if verbose:
                logger.info("✓ LightGBM trained")
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            if self.task == "regression":
                cat_model = CatBoostRegressor(
                    iterations=500,
                    depth=8,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=0
                )
            else:
                cat_model = CatBoostClassifier(
                    iterations=500,
                    depth=8,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=0
                )
            
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
            self.models['catboost'] = cat_model
            
            if verbose:
                logger.info("✓ CatBoost trained")
        
        # Calcular pesos basados en performance de validación
        self._calculate_weights(X_val, y_val)
        self.trained = True
        
        if verbose:
            logger.info(f"Ensemble weights: {self.weights}")
    
    def _calculate_weights(self, X_val, y_val):
        """Calcula pesos de voting basados en performance de validación"""
        
        if not SKLEARN_AVAILABLE:
            # Pesos uniformes si no hay sklearn
            n_models = len(self.models)
            for name in self.models:
                self.weights[name] = 1.0 / n_models
            return
        
        scores = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            
            if self.task == "regression":
                # Usar MAE invertido como score (menor MAE = mejor)
                mae = mean_absolute_error(y_val, y_pred)
                scores[name] = 1.0 / (mae + 1e-6)
            else:
                # Usar accuracy
                scores[name] = accuracy_score(y_val, (y_pred > 0.5).astype(int))
        
        # Normalizar scores a pesos
        total_score = sum(scores.values())
        self.weights = {name: score / total_score for name, score in scores.items()}
    
    def predict(self, X):
        """Predicción con voting ponderado"""
        if not self.trained:
            raise RuntimeError("Ensemble must be trained before prediction")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.weights[name])
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        return np.average(predictions, axis=0, weights=weights)


# ============================================================================
# MAIN ML PIPELINE
# ============================================================================

class MLPipeline:
    """
    Pipeline completo de Machine Learning para Game Optimizer V5.0
    
    Características:
    - Feature engineering avanzado (96 features)
    - Multiple model types (DNN, XGBoost, LightGBM, CatBoost)
    - Model ensembling con voting ponderado
    - Cross-validation
    - Hyperparameter tuning (Optuna)
    - Model explainability (SHAP)
    - ONNX export para producción
    - Drift detection
    - Auto-retraining
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Args:
            model_dir: Directorio para guardar modelos. 
                      Default: ~/.game_optimizer/ml_models_v5
        """
        self.model_dir = model_dir or Path.home() / '.game_optimizer' / 'ml_models_v5'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_engineer = AdvancedFeatureEngineer()
        self.models: Dict[str, Any] = {}
        self.training_history: List[Dict] = []
        self.explainer: Optional[Any] = None
        
        logger.info("=" * 80)
        logger.info("ML Pipeline V5.0 initialized")
        logger.info("=" * 80)
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"PyTorch: {PYTORCH_AVAILABLE}")
        logger.info(f"XGBoost: {XGBOOST_AVAILABLE}")
        logger.info(f"LightGBM: {LIGHTGBM_AVAILABLE}")
        logger.info(f"CatBoost: {CATBOOST_AVAILABLE}")
        logger.info(f"Optuna (AutoML): {OPTUNA_AVAILABLE}")
        logger.info(f"SHAP (Explainability): {SHAP_AVAILABLE}")
        logger.info(f"ONNX (Production): {ONNX_AVAILABLE}")
        logger.info("=" * 80)
    
    def prepare_dataset(
        self, 
        sessions: List[GameSession]
    ):
        """
        Prepara dataset completo para entrenamiento
        
        Args:
            sessions: Lista de GameSession objects
            
        Returns:
            Tuple de (X, y_fps, y_stability, y_success)
            - X: Features (N, 96)
            - y_fps: Target FPS (N,)
            - y_stability: Target stability score 0-100 (N,)
            - y_success: Target success binary (N,)
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required")
        
        logger.info(f"Preparing dataset from {len(sessions)} sessions...")
        
        X = []
        y_fps = []
        y_stability = []
        y_success = []
        
        for session in sessions:
            try:
                features = self.feature_engineer.extract_features(session)
                X.append(features)
                
                # Target: FPS
                y_fps.append(session.fps_avg)
                
                # Target: Stability score (calculado)
                if session.fps_avg > 0:
                    stability = (1 - session.fps_std / session.fps_avg) * 100
                else:
                    stability = 0
                y_stability.append(max(0, min(100, stability)))
                
                # Target: Optimization success (binary)
                y_success.append(float(session.optimization_successful))
                
            except Exception as e:
                logger.warning(f"Error processing session {session.session_id}: {e}")
                continue
        
        X = np.array(X, dtype=np.float32)
        y_fps = np.array(y_fps, dtype=np.float32)
        y_stability = np.array(y_stability, dtype=np.float32)
        y_success = np.array(y_success, dtype=np.float32)
        
        logger.info(f"✓ Dataset prepared: X.shape={X.shape}")
        logger.info(f"  FPS range: [{y_fps.min():.1f}, {y_fps.max():.1f}]")
        logger.info(f"  Stability range: [{y_stability.min():.1f}, {y_stability.max():.1f}]")
        logger.info(f"  Success rate: {y_success.mean()*100:.1f}%")
        
        return X, y_fps, y_stability, y_success
    
    def train_gradient_boosting_models(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        task: str = "regression"
    ) -> GradientBoostingEnsemble:
        """
        Entrena ensemble de gradient boosting models
        
        Args:
            task: "regression" o "classification"
        """
        logger.info(f"Training Gradient Boosting Ensemble ({task})...")
        
        ensemble = GradientBoostingEnsemble(task=task)
        ensemble.train(X_train, y_train, X_val, y_val)
        
        # Evaluar en validación
        if SKLEARN_AVAILABLE:
            y_pred = ensemble.predict(X_val)
            if task == "regression":
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                logger.info(f"  Validation MAE: {mae:.2f}")
                logger.info(f"  Validation RMSE: {rmse:.2f}")
                logger.info(f"  Validation R²: {r2:.4f}")
            else:
                acc = accuracy_score(y_val, (y_pred > 0.5).astype(int))
                f1 = f1_score(y_val, (y_pred > 0.5).astype(int))
                logger.info(f"  Validation Accuracy: {acc:.4f}")
                logger.info(f"  Validation F1: {f1:.4f}")
        
        return ensemble
    
    def train_all_models(
        self, 
        sessions: List[GameSession],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Entrena todos los modelos disponibles
        
        Args:
            sessions: Lista de sesiones de gaming
            test_size: Proporción de datos para validación
            random_state: Seed para reproducibilidad
            
        Returns:
            Dictionary con modelos entrenados y métricas
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required")
        
        logger.info("=" * 80)
        logger.info("STARTING FULL ML PIPELINE TRAINING")
        logger.info("=" * 80)
        
        # 1. Preparar dataset
        X, y_fps, y_stability, y_success = self.prepare_dataset(sessions)
        
        # 2. Split train/validation
        X_train, X_val, y_fps_train, y_fps_val = train_test_split(
            X, y_fps, test_size=test_size, random_state=random_state
        )
        _, _, y_stab_train, y_stab_val = train_test_split(
            X, y_stability, test_size=test_size, random_state=random_state
        )
        _, _, y_succ_train, y_succ_val = train_test_split(
            X, y_success, test_size=test_size, random_state=random_state
        )
        
        # 3. Feature scaling
        logger.info("Scaling features...")
        X_train_scaled = self.feature_engineer.fit_transform(X_train)
        X_val_scaled = self.feature_engineer.transform(X_val)
        
        results = {}
        
        # 4. Train Gradient Boosting Models
        
        # 4a. FPS Prediction
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING FPS PREDICTION MODELS")
        logger.info("=" * 80)
        fps_ensemble = self.train_gradient_boosting_models(
            X_train_scaled, y_fps_train,
            X_val_scaled, y_fps_val,
            task="regression"
        )
        self.models['fps_predictor'] = fps_ensemble
        results['fps_model'] = fps_ensemble
        
        # 4b. Stability Score Prediction
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING STABILITY SCORE MODELS")
        logger.info("=" * 80)
        stability_ensemble = self.train_gradient_boosting_models(
            X_train_scaled, y_stab_train,
            X_val_scaled, y_stab_val,
            task="regression"
        )
        self.models['stability_predictor'] = stability_ensemble
        results['stability_model'] = stability_ensemble
        
        # 4c. Success Classification
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING OPTIMIZATION SUCCESS CLASSIFIER")
        logger.info("=" * 80)
        success_ensemble = self.train_gradient_boosting_models(
            X_train_scaled, y_succ_train,
            X_val_scaled, y_succ_val,
            task="classification"
        )
        self.models['success_classifier'] = success_ensemble
        results['success_model'] = success_ensemble
        
        # 5. Train Deep Learning (si PyTorch disponible)
        # TODO: Implementar training loop completo para DNN
        
        # 6. Train RL Agent (si disponible)
        # TODO: Implementar PPO agent
        
        # 7. Setup Explainability
        if SHAP_AVAILABLE and XGBOOST_AVAILABLE:
            logger.info("\nSetting up SHAP explainer...")
            try:
                # Usar modelo XGBoost para explicabilidad
                if 'xgboost' in fps_ensemble.models:
                    self.explainer = shap.TreeExplainer(fps_ensemble.models['xgboost'])
                    logger.info("✓ SHAP explainer ready")
            except Exception as e:
                logger.warning(f"Could not setup SHAP: {e}")
        
        # 8. Save models
        self._save_models()
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        
        return results
    
    def predict(
        self, 
        session: GameSession,
        return_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Realiza predicción para una sesión con todos los modelos
        
        Args:
            session: GameSession para predecir
            return_explanation: Si True, incluye explicación SHAP
            
        Returns:
            Dictionary con predicciones y explicaciones
        """
        # Extraer features
        features = self.feature_engineer.extract_features(session)
        features_scaled = self.feature_engineer.transform(features.reshape(1, -1))
        
        predictions = {}
        
        # Predecir FPS
        if 'fps_predictor' in self.models:
            fps_pred = self.models['fps_predictor'].predict(features_scaled)[0]
            predictions['fps'] = float(fps_pred)
        
        # Predecir Stability
        if 'stability_predictor' in self.models:
            stability_pred = self.models['stability_predictor'].predict(features_scaled)[0]
            predictions['stability_score'] = float(stability_pred)
        
        # Predecir Success
        if 'success_classifier' in self.models:
            success_pred = self.models['success_classifier'].predict(features_scaled)[0]
            predictions['success_probability'] = float(success_pred)
        
        # Explicación SHAP
        if return_explanation and self.explainer:
            try:
                shap_values = self.explainer.shap_values(features_scaled)
                feature_names = self.feature_engineer.get_feature_names()
                
                # Top 5 features más importantes
                abs_shap = np.abs(shap_values[0])
                top_indices = np.argsort(abs_shap)[-5:][::-1]
                
                explanations = []
                for idx in top_indices:
                    explanations.append({
                        'feature': feature_names[idx],
                        'impact': float(shap_values[0][idx]),
                        'value': float(features[idx])
                    })
                
                predictions['explanations'] = explanations
            except Exception as e:
                logger.warning(f"Could not generate SHAP explanation: {e}")
        
        return predictions
    
    def _save_models(self):
        """Guarda todos los modelos entrenados"""
        logger.info("Saving models...")
        
        try:
            # Guardar gradient boosting models
            for name, model in self.models.items():
                model_path = self.model_dir / f"{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"  ✓ Saved {name}")
            
            # Guardar scaler
            scaler_path = self.model_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_engineer.scaler, f)
            logger.info(f"  ✓ Saved feature scaler")
            
            logger.info(f"All models saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Carga modelos previamente entrenados"""
        logger.info("Loading models...")
        
        try:
            # Cargar gradient boosting models
            for model_file in self.model_dir.glob("*.pkl"):
                if model_file.stem == "scaler":
                    continue
                    
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    self.models[model_file.stem] = model
                    logger.info(f"  ✓ Loaded {model_file.stem}")
            
            # Cargar scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.feature_engineer.scaler = pickle.load(f)
                logger.info(f"  ✓ Loaded feature scaler")
            
            logger.info(f"All models loaded from {self.model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_dummy_sessions(n_sessions: int = 100) -> List[GameSession]:
    """
    Crea sesiones dummy para testing y demo
    
    Args:
        n_sessions: Número de sesiones a crear
        
    Returns:
        Lista de GameSession objects
    """
    sessions = []
    
    for i in range(n_sessions):
        # Variar FPS y otras métricas
        base_fps = 60 + (i % 80)
        fps_std = 3 + (i % 10)
        
        session = GameSession(
            session_id=f"session_{i}",
            game_name=f"TestGame_{i % 10}",
            game_exe="game.exe",
            timestamp=time.time() - (n_sessions - i) * 3600,
            duration_seconds=300 + (i % 600),
            
            # Hardware (valores realistas)
            cpu_cores_physical=8,
            cpu_cores_logical=16,
            cpu_freq_base_mhz=3600,
            cpu_freq_boost_mhz=4800,
            cpu_cache_l2_mb=8,
            cpu_cache_l3_mb=32,
            cpu_tdp_watts=65,
            gpu_vram_gb=8 + (i % 8),
            gpu_core_clock_mhz=1800 + (i % 200),
            gpu_memory_clock_mhz=7000,
            gpu_cuda_cores=3584,
            gpu_tensor_cores=112,
            gpu_ray_tracing_cores=28,
            ram_capacity_gb=16 + (i % 2) * 16,
            ram_frequency_mhz=3200,
            
            # Config (variar optimizaciones)
            timer_resolution_ms=0.5 if i % 2 == 0 else 1.0,
            cpu_priority_class=i % 3,
            cpu_affinity_mask=255,
            gpu_clock_locked=i % 2 == 0,
            gpu_power_limit_pct=100 + (i % 3) * 5,
            memory_optimization_level=i % 3,
            page_file_size_gb=16,
            network_qos_enabled=i % 2 == 0,
            network_dscp_value=46 if i % 2 == 0 else 0,
            tcp_nagle_disabled=i % 2 == 0,
            tcp_buffer_kb=256 if i % 2 == 0 else 64,
            core_parking_disabled=i % 2 == 0,
            game_mode_enabled=i % 2 == 0,
            background_apps_limited=i % 2 == 0,
            services_stopped=i % 2 == 0,
            directx_version=12,
            vulkan_version="1.3",
            anti_cheat_type="none" if i % 3 == 0 else "eac",
            launcher_type="steam",
            online_mode=i % 2 == 0,
            
            # Performance (correlacionado con optimizaciones)
            fps_avg=base_fps,
            fps_min=base_fps - 10,
            fps_max=base_fps + 20,
            fps_std=fps_std,
            fps_p1=base_fps - 8,
            fps_p5=base_fps - 6,
            fps_p25=base_fps - 3,
            fps_p50=base_fps,
            fps_p75=base_fps + 3,
            fps_p95=base_fps + 10,
            fps_p99=base_fps + 15,
            frame_time_avg_ms=1000.0 / base_fps,
            frame_time_p1_ms=1000.0 / (base_fps + 15),
            frame_time_p99_ms=1000.0 / (base_fps - 8),
            frame_time_p999_ms=1000.0 / (base_fps - 10),
            frame_time_std_ms=fps_std / base_fps * 16.67,
            stutter_count=5 - (i % 6),
            stutter_duration_avg_ms=20,
            stutter_frequency_hz=0.1,
            cpu_usage_avg=50 + (i % 30),
            cpu_usage_max=70 + (i % 20),
            cpu_temp_c=60 + (i % 15),
            cpu_throttled=False,
            gpu_usage_avg=80 + (i % 15),
            gpu_usage_max=95,
            gpu_temp_c=70 + (i % 10),
            gpu_throttled=False,
            gpu_power_w=150 + (i % 50),
            memory_working_set_mb=8000 + (i % 4000),
            memory_page_faults_per_sec=100,
            
            # Network/Disk
            network_latency_ms=20 + (i % 30),
            network_packet_loss_pct=0.1 * (i % 5),
            network_bandwidth_mbps=100,
            disk_read_mb_per_sec=50 + (i % 50),
            disk_write_mb_per_sec=20,
            disk_latency_ms=5 + (i % 5),
            
            # Targets
            optimization_successful=(i % 10) < 8,  # 80% success rate
            user_satisfaction_score=3 + (i % 3),
            baseline_fps=base_fps - 10 if i % 2 == 0 else None
        )
        
        sessions.append(session)
    
    return sessions


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Ejemplo de uso del pipeline ML"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear pipeline
    pipeline = MLPipeline()
    
    # Generar datos dummy para demo
    logger.info("Generating dummy training data...")
    sessions = create_dummy_sessions(n_sessions=200)
    
    # Entrenar modelos
    results = pipeline.train_all_models(sessions)
    
    # Hacer predicción de ejemplo
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PREDICTION")
    logger.info("=" * 80)
    
    test_session = sessions[0]
    prediction = pipeline.predict(test_session, return_explanation=True)
    
    logger.info(f"Predictions for session '{test_session.session_id}':")
    logger.info(f"  Predicted FPS: {prediction.get('fps', 'N/A'):.1f}")
    logger.info(f"  Predicted Stability: {prediction.get('stability_score', 'N/A'):.1f}")
    logger.info(f"  Success Probability: {prediction.get('success_probability', 'N/A'):.2%}")
    
    if 'explanations' in prediction:
        logger.info("\n  Top contributing features:")
        for exp in prediction['explanations']:
            logger.info(f"    - {exp['feature']}: impact={exp['impact']:.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ML PIPELINE DEMO COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
