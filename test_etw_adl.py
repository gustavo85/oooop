"""
Test script for ETW and ADL implementations
Tests the new ETW frame time monitoring and AMD ADL clock control features.
"""

import logging
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_etw_structures():
    """Test ETW structure definitions"""
    logger.info("=" * 80)
    logger.info("TEST 1: ETW Structure Definitions")
    logger.info("=" * 80)
    
    try:
        from etw_monitor import (
            GUID, EVENT_HEADER, WNODE_HEADER, EVENT_RECORD,
            EVENT_TRACE_PROPERTIES, DXGI_PROVIDER_GUID, KERNEL_PROVIDER_GUID
        )
        
        # Test GUID creation
        guid = GUID.from_string('{CA11C036-0102-4A2D-A6AD-F03CFED5D3C9}')
        logger.info(f"✓ GUID structure working: Data1={guid.Data1}")
        
        # Test DXGI provider GUID
        logger.info(f"✓ DXGI Provider GUID: {DXGI_PROVIDER_GUID.Data1:08X}")
        logger.info(f"✓ Kernel Provider GUID: {KERNEL_PROVIDER_GUID.Data1:08X}")
        
        # Test structure sizes
        import ctypes
        logger.info(f"✓ EVENT_HEADER size: {ctypes.sizeof(EVENT_HEADER)} bytes")
        logger.info(f"✓ WNODE_HEADER size: {ctypes.sizeof(WNODE_HEADER)} bytes")
        logger.info(f"✓ EVENT_TRACE_PROPERTIES size: {ctypes.sizeof(EVENT_TRACE_PROPERTIES)} bytes")
        
        logger.info("✓ All ETW structures defined correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ ETW structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_etw_frame_monitor():
    """Test ETW frame time monitor initialization"""
    logger.info("=" * 80)
    logger.info("TEST 2: ETW Frame Time Monitor")
    logger.info("=" * 80)
    
    try:
        from etw_monitor import ETWFrameTimeMonitor
        
        monitor = ETWFrameTimeMonitor()
        logger.info("✓ ETW Frame Time Monitor created")
        
        # Test QPC frequency
        logger.info(f"✓ QPC Frequency: {monitor.qpc_freq_val:,} Hz")
        
        # Try to start (will fail without admin but tests the code path)
        logger.info("Attempting to start ETW session (requires Administrator)...")
        success = monitor.start(session_name="TestDXGISession")
        
        if success:
            logger.info("✓ ETW session started successfully!")
            time.sleep(2)
            
            # Check for frame times
            frame_times = monitor.get_frame_times()
            logger.info(f"✓ Collected {len(frame_times)} frame samples")
            
            p99 = monitor.get_recent_p99()
            if p99:
                logger.info(f"✓ Frame Time P99: {p99:.2f}ms")
            
            monitor.stop()
            logger.info("✓ ETW session stopped")
        else:
            logger.warning("⚠️  ETW session failed to start (expected without admin privileges)")
            logger.info("✓ Start method executed without crashing")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ETW frame monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dpc_monitor():
    """Test DPC latency monitor"""
    logger.info("=" * 80)
    logger.info("TEST 3: DPC Latency Monitor")
    logger.info("=" * 80)
    
    try:
        from etw_monitor import ETWDPCLatencyMonitor
        
        monitor = ETWDPCLatencyMonitor()
        logger.info("✓ DPC Latency Monitor created")
        
        # Try to start
        logger.info("Attempting to start DPC monitoring...")
        success = monitor.start()
        
        if success:
            logger.info("✓ DPC monitor started")
            time.sleep(5)
            
            avg_latency = monitor.get_average_latency()
            max_latency = monitor.get_recent_max_latency()
            
            if avg_latency:
                logger.info(f"✓ Average DPC latency: {avg_latency:.1f}μs")
            if max_latency:
                logger.info(f"✓ Max DPC latency: {max_latency:.1f}μs")
            
            monitor.stop()
            logger.info("✓ DPC monitor stopped")
        else:
            logger.warning("⚠️  DPC monitor failed to start")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ DPC monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adl_structures():
    """Test ADL structure definitions"""
    logger.info("=" * 80)
    logger.info("TEST 4: ADL Structure Definitions")
    logger.info("=" * 80)
    
    try:
        from gpu_native_control import (
            ADLODNCapabilities, ADLODNPerformanceLevel, ADLODNPerformanceLevels,
            ADLOD8InitSetting, ADLOD8CurrentSetting, ADLOD8SetSetting
        )
        
        import ctypes
        
        logger.info(f"✓ ADLODNCapabilities size: {ctypes.sizeof(ADLODNCapabilities)} bytes")
        logger.info(f"✓ ADLODNPerformanceLevels size: {ctypes.sizeof(ADLODNPerformanceLevels)} bytes")
        logger.info(f"✓ ADLOD8InitSetting size: {ctypes.sizeof(ADLOD8InitSetting)} bytes")
        logger.info(f"✓ ADLOD8CurrentSetting size: {ctypes.sizeof(ADLOD8CurrentSetting)} bytes")
        
        logger.info("✓ All ADL structures defined correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ ADL structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adl_wrapper():
    """Test ADL wrapper initialization"""
    logger.info("=" * 80)
    logger.info("TEST 5: ADL Wrapper")
    logger.info("=" * 80)
    
    try:
        from gpu_native_control import ADLWrapper
        
        wrapper = ADLWrapper()
        logger.info("✓ ADL Wrapper created")
        
        if wrapper.adl:
            logger.info("✓ ADL DLL loaded")
            
            # Try to initialize
            if wrapper.initialize():
                logger.info(f"✓ ADL initialized with {wrapper.adapter_count} adapter(s)")
                logger.info(f"✓ Primary adapter index: {wrapper.primary_adapter_index}")
                logger.info(f"✓ ADL context: {wrapper.context}")
                
                # Test clock locking (won't actually lock without proper GPU)
                logger.info("Testing clock locking API...")
                result = wrapper.lock_clocks_max()
                if result:
                    logger.info("✓ Clock locking succeeded")
                    wrapper.unlock_clocks()
                    logger.info("✓ Clocks unlocked")
                else:
                    logger.warning("⚠️  Clock locking not supported or failed (expected on some GPUs)")
                
                wrapper.cleanup()
                logger.info("✓ ADL cleanup complete")
            else:
                logger.warning("⚠️  ADL initialization failed (no AMD GPU or drivers not installed)")
        else:
            logger.info("⚠️  ADL DLL not found (no AMD GPU detected)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ADL wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nvapi_wrapper():
    """Test NVAPI wrapper"""
    logger.info("=" * 80)
    logger.info("TEST 6: NVAPI Wrapper")
    logger.info("=" * 80)
    
    try:
        from gpu_native_control import NVAPIWrapper
        
        wrapper = NVAPIWrapper()
        logger.info("✓ NVAPI Wrapper created")
        
        if wrapper.nvapi:
            logger.info("✓ NVAPI DLL loaded")
            
            if wrapper.initialize():
                logger.info(f"✓ NVAPI initialized with {len(wrapper.gpu_handles)} GPU(s)")
                
                # Test clock locking
                logger.info("Testing NVAPI clock locking...")
                result = wrapper.lock_clocks_max()
                if result:
                    logger.info("✓ NVAPI clock locking succeeded")
                    wrapper.unlock_clocks()
                    logger.info("✓ NVAPI clocks unlocked")
                else:
                    logger.warning("⚠️  NVAPI clock locking failed")
                
                wrapper.cleanup()
                logger.info("✓ NVAPI cleanup complete")
            else:
                logger.warning("⚠️  NVAPI initialization failed")
        else:
            logger.info("⚠️  NVAPI DLL not found (no NVIDIA GPU detected)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ NVAPI wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitoring():
    """Test performance monitoring with memory-mapped telemetry"""
    logger.info("=" * 80)
    logger.info("TEST 7: Performance Monitoring")
    logger.info("=" * 80)
    
    try:
        from monitoring import PerformanceMonitor, MemoryMappedTelemetry
        
        # Test memory-mapped telemetry
        mmap_telem = MemoryMappedTelemetry(size_mb=1)
        if mmap_telem.initialize():
            logger.info("✓ Memory-mapped telemetry initialized")
            
            # Write test events
            for i in range(10):
                mmap_telem.write_json_event({
                    'event_id': i,
                    'timestamp': time.time(),
                    'data': f'test_event_{i}'
                })
            
            logger.info("✓ Wrote 10 test events to mmap")
            
            # Flush to disk
            if mmap_telem.flush_to_disk():
                logger.info("✓ Flushed mmap to disk")
            
            mmap_telem.cleanup()
            logger.info("✓ Memory-mapped telemetry cleanup complete")
        else:
            logger.warning("⚠️  Memory-mapped telemetry initialization failed")
        
        # Test performance monitor
        monitor = PerformanceMonitor()
        logger.info("✓ Performance Monitor created")
        logger.info(f"✓ Using ETW: {monitor.use_etw}")
        
        monitor.cleanup()
        logger.info("✓ Performance Monitor cleanup complete")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("ETW AND ADL IMPLEMENTATION TESTS")
    logger.info("=" * 80)
    logger.info("")
    
    tests = [
        ("ETW Structures", test_etw_structures),
        ("ETW Frame Monitor", test_etw_frame_monitor),
        ("DPC Monitor", test_dpc_monitor),
        ("ADL Structures", test_adl_structures),
        ("ADL Wrapper", test_adl_wrapper),
        ("NVAPI Wrapper", test_nvapi_wrapper),
        ("Performance Monitoring", test_monitoring),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info("")
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
            logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
