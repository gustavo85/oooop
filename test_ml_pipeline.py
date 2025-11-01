"""
test_ml_pipeline.py - Comprehensive tests for ML Pipeline V5.0
Tests feature engineering, models, training, prediction, and integration
"""

import sys
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test 1: Verify all imports work"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Import Test")
    logger.info("=" * 80)
    
    try:
        import ml_pipeline
        logger.info("✓ ml_pipeline imported")
        
        import ml_config_loader
        logger.info("✓ ml_config_loader imported")
        
        import ml_integration_adapter
        logger.info("✓ ml_integration_adapter imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_config_loader():
    """Test 2: Config loader functionality"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Config Loader Test")
    logger.info("=" * 80)
    
    try:
        from ml_config_loader import get_ml_config
        
        config = get_ml_config()
        logger.info(f"✓ Config loaded: {config}")
        
        # Test property access
        logger.info(f"  Model dir: {config.model_dir}")
        logger.info(f"  Deep learning enabled: {config.is_deep_learning_enabled}")
        logger.info(f"  Gradient boosting enabled: {config.is_gradient_boosting_enabled}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        
        # Test dot notation access
        test_value = config.get('gradient_boosting.xgboost.n_estimators', 500)
        logger.info(f"  XGBoost n_estimators: {test_value}")
        
        logger.info("✓ Config loader works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Config loader test failed: {e}")
        return False


def test_feature_engineering():
    """Test 3: Feature engineering"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Feature Engineering Test")
    logger.info("=" * 80)
    
    try:
        from ml_pipeline import AdvancedFeatureEngineer, GameSession, NUMPY_AVAILABLE
        
        if not NUMPY_AVAILABLE:
            logger.warning("⚠ NumPy not available, skipping feature engineering test")
            return True
        
        engineer = AdvancedFeatureEngineer()
        logger.info(f"✓ Feature engineer created")
        logger.info(f"  Feature names count: {len(engineer.get_feature_names())}")
        
        # Create test session
        session = GameSession(
            session_id="test_1",
            game_name="TestGame",
            game_exe="test.exe",
            timestamp=time.time(),
            duration_seconds=300,
            fps_avg=120.0,
            fps_std=5.0
        )
        
        # Extract features
        features = engineer.extract_features(session)
        logger.info(f"✓ Features extracted: shape={features.shape}")
        logger.info(f"  Feature vector sample: {features[:10]}")
        
        # Test scaling
        import numpy as np
        features_batch = np.array([features, features])  # Batch of 2
        engineer.fit_scaler(features_batch)
        scaled = engineer.transform(features_batch)
        logger.info(f"✓ Scaling works: scaled shape={scaled.shape}")
        
        logger.info("✓ Feature engineering works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dummy_data_generation():
    """Test 4: Dummy data generation"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Dummy Data Generation Test")
    logger.info("=" * 80)
    
    try:
        from ml_pipeline import create_dummy_sessions
        
        sessions = create_dummy_sessions(n_sessions=50)
        logger.info(f"✓ Created {len(sessions)} dummy sessions")
        
        # Check session structure
        session = sessions[0]
        logger.info(f"  Sample session ID: {session.session_id}")
        logger.info(f"  Game: {session.game_name}")
        logger.info(f"  FPS: {session.fps_avg:.1f}")
        stability_score = ((1 - session.fps_std/session.fps_avg)*100) if session.fps_avg > 0 else 0
        logger.info(f"  Stability score: {stability_score:.1f}")
        
        logger.info("✓ Dummy data generation works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Dummy data generation test failed: {e}")
        return False


def test_ml_pipeline_basic():
    """Test 5: Basic ML pipeline functionality"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: ML Pipeline Basic Test")
    logger.info("=" * 80)
    
    try:
        from ml_pipeline import MLPipeline, create_dummy_sessions, SKLEARN_AVAILABLE
        
        if not SKLEARN_AVAILABLE:
            logger.warning("⚠ scikit-learn not available, skipping pipeline test")
            return True
        
        # Create pipeline
        pipeline = MLPipeline()
        logger.info(f"✓ Pipeline created")
        
        # Generate data
        sessions = create_dummy_sessions(n_sessions=100)
        logger.info(f"✓ Generated {len(sessions)} training sessions")
        
        # Prepare dataset
        X, y_fps, y_stability, y_success = pipeline.prepare_dataset(sessions)
        logger.info(f"✓ Dataset prepared:")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y_fps shape: {y_fps.shape}")
        logger.info(f"  y_stability shape: {y_stability.shape}")
        logger.info(f"  y_success shape: {y_success.shape}")
        
        logger.info("✓ ML pipeline basic functionality works")
        return True
        
    except Exception as e:
        logger.error(f"✗ ML pipeline basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_boosting():
    """Test 6: Gradient boosting ensemble"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Gradient Boosting Ensemble Test")
    logger.info("=" * 80)
    
    try:
        from ml_pipeline import (
            GradientBoostingEnsemble, create_dummy_sessions, MLPipeline,
            XGBOOST_AVAILABLE, SKLEARN_AVAILABLE
        )
        
        if not (XGBOOST_AVAILABLE and SKLEARN_AVAILABLE):
            logger.warning("⚠ XGBoost or sklearn not available, skipping ensemble test")
            return True
        
        # Prepare data
        pipeline = MLPipeline()
        sessions = create_dummy_sessions(n_sessions=100)
        X, y_fps, _, _ = pipeline.prepare_dataset(sessions)
        
        # Scale features
        X_scaled = pipeline.feature_engineer.fit_transform(X)
        
        # Split train/val
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_fps, test_size=0.2, random_state=42
        )
        
        # Create ensemble
        ensemble = GradientBoostingEnsemble(task="regression")
        logger.info(f"✓ Ensemble created")
        
        # Train (minimal config for speed)
        ensemble.train(X_train, y_train, X_val, y_val, verbose=True)
        logger.info(f"✓ Ensemble trained")
        logger.info(f"  Models: {list(ensemble.models.keys())}")
        logger.info(f"  Weights: {ensemble.weights}")
        
        # Predict
        predictions = ensemble.predict(X_val[:5])
        logger.info(f"✓ Predictions made: {predictions}")
        
        logger.info("✓ Gradient boosting ensemble works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Gradient boosting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_adapter():
    """Test 7: Integration adapter"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 7: Integration Adapter Test")
    logger.info("=" * 80)
    
    try:
        from ml_integration_adapter import get_ml_adapter, get_ml_status
        
        # Get adapter
        adapter = get_ml_adapter(use_v5=True)
        logger.info(f"✓ Adapter created")
        
        # Get status
        status = get_ml_status()
        logger.info(f"✓ Status retrieved:")
        logger.info(f"  Available: {status['available']}")
        logger.info(f"  Version: {status['version']}")
        logger.info(f"  Using V5: {status['using_v5']}")
        logger.info(f"  Capabilities: {status['capabilities']}")
        
        # Test session conversion
        session_data = {
            'session_id': 'test_session',
            'game_name': 'TestGame',
            'game_exe': 'test.exe',
            'timestamp': time.time(),
            'duration_seconds': 300,
            'cpu_cores': 8,
            'cpu_threads': 16,
            'cpu_freq_mhz': 3600,
            'ram_gb': 16,
            'gpu_vram_gb': 8,
            'timer_resolution_ms': 0.5,
            'cpu_priority': 1,
            'avg_fps': 120,
            'fps_1_percent_low': 100,
            'avg_cpu_usage': 60,
            'avg_gpu_usage': 80
        }
        
        if adapter.using_v5:
            session_v5 = adapter.convert_session_to_v5(session_data)
            logger.info(f"✓ Session converted to V5:")
            logger.info(f"  Game: {session_v5.game_name}")
            logger.info(f"  FPS: {session_v5.fps_avg}")
            logger.info(f"  CPU cores: {session_v5.cpu_cores_physical}")
        
        logger.info("✓ Integration adapter works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_pipeline():
    """Test 8: Full training pipeline (if dependencies available)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 8: Full Training Pipeline Test")
    logger.info("=" * 80)
    
    try:
        from ml_pipeline import (
            MLPipeline, create_dummy_sessions,
            SKLEARN_AVAILABLE, XGBOOST_AVAILABLE
        )
        
        if not (SKLEARN_AVAILABLE and XGBOOST_AVAILABLE):
            logger.warning("⚠ Dependencies not available, skipping full training test")
            return True
        
        # Create pipeline with temp directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MLPipeline(model_dir=Path(tmpdir))
            
            # Generate training data
            sessions = create_dummy_sessions(n_sessions=150)
            logger.info(f"✓ Generated {len(sessions)} training sessions")
            
            # Train all models
            logger.info("Training models (this may take a moment)...")
            results = pipeline.train_all_models(sessions, test_size=0.2)
            
            logger.info(f"✓ Training complete:")
            logger.info(f"  Models trained: {list(results.keys())}")
            
            # Test prediction
            test_session = sessions[0]
            prediction = pipeline.predict(test_session, return_explanation=False)
            
            logger.info(f"✓ Prediction made:")
            logger.info(f"  FPS: {prediction.get('fps', 'N/A')}")
            logger.info(f"  Stability: {prediction.get('stability_score', 'N/A')}")
            logger.info(f"  Success prob: {prediction.get('success_probability', 'N/A')}")
        
        logger.info("✓ Full training pipeline works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Full training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING ML PIPELINE V5.0 COMPREHENSIVE TESTS")
    logger.info("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Config Loader", test_config_loader),
        ("Feature Engineering", test_feature_engineering),
        ("Dummy Data Generation", test_dummy_data_generation),
        ("ML Pipeline Basic", test_ml_pipeline_basic),
        ("Gradient Boosting Ensemble", test_gradient_boosting),
        ("Integration Adapter", test_integration_adapter),
        ("Full Training Pipeline", test_full_training_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 80)
    logger.info(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
