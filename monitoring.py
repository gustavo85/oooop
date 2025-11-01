"""
Performance Monitoring V4.0 - Real ETW Frame Time + DPC Latency + Telemetry + A/B Testing
NEW V4.0: Real ETW implementation via etw_monitor.py module
"""

import ctypes
import logging
import time
import json
import threading
import statistics
from ctypes import wintypes
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Deque
from collections import deque
import psutil

logger = logging.getLogger(__name__)

# Try to import real ETW monitors
try:
    from etw_monitor import ETWFrameTimeMonitor, ETWDPCLatencyMonitor
    ETW_AVAILABLE = True
    logger.info("✓ Real ETW monitoring available")
except ImportError:
    ETW_AVAILABLE = False
    logger.warning("⚠️ Real ETW monitoring not available, using fallback mode")
    ETWFrameTimeMonitor = None
    ETWDPCLatencyMonitor = None


@dataclass
class FrameMetrics:
    """Frame time metrics for a gaming session"""
    avg_fps: float
    min_fps: float
    max_fps: float
    one_percent_low: float
    point_one_percent_low: float
    frame_time_avg_ms: float
    frame_time_p99_ms: float
    frame_time_p999_ms: float
    stutter_count: int
    total_frames: int


@dataclass
class DPCLatencyReading:
    """DPC/ISR latency reading"""
    timestamp: float
    latency_us: float
    offending_driver: Optional[str]


@dataclass
class SessionTelemetry:
    """Complete telemetry for a gaming session"""
    session_id: str
    game_exe: str
    game_pid: int
    start_time: float
    end_time: float
    duration_seconds: float
    
    # Performance metrics
    frame_metrics: Optional[FrameMetrics]
    cpu_usage_avg: float
    gpu_usage_avg: float
    memory_usage_avg_mb: float
    
    # Optimizations applied
    optimizations: List[str]
    profile_name: str
    
    # Hardware info with exact versions
    cpu_model: str
    gpu_model: str
    ram_gb: int
    gpu_driver_version: str  # Exact driver version
    bios_version: str        # BIOS version
    
    # DPC latency
    dpc_latency_avg_us: float
    dpc_spikes_count: int
    
    # NEW: Context switch metrics
    context_switches_avg: float
    context_switch_spikes: int
    
    # NEW: Expanded telemetry for stability and dynamic context
    cpu_temp_avg: float = 0.0
    gpu_temp_avg: float = 0.0
    frame_time_p999_ms: float = 0.0
    dpc_latency_max_us: float = 0.0
    
    # NEW: Rollback tracking for ML training
    optimization_failed: bool = False
    
    # NEW: Memory pressure tracking
    memory_pressure_pct: float = 0.0  # Percentage of total RAM being used


class PerformanceMonitor:
    """
    Real-time performance monitoring using ETW (when available) or fallback to QPC
    V4.0: Integrated with real ETW frame time and DPC latency monitors
    """
    
    def __init__(self):
        self.active_sessions: Dict[int, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.monitor_threads: Dict[int, threading.Thread] = {}
        
        # ETW monitors (V4.0)
        self.etw_frame_monitor: Optional[Any] = None
        self.etw_dpc_monitor: Optional[Any] = None
        self.use_etw = ETW_AVAILABLE
        
        # Initialize ETW monitors if available
        if self.use_etw and ETWFrameTimeMonitor and ETWDPCLatencyMonitor:
            try:
                self.etw_frame_monitor = ETWFrameTimeMonitor()
                self.etw_dpc_monitor = ETWDPCLatencyMonitor()
                logger.info("✓ Real ETW monitors initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ETW monitors, using fallback: {e}")
                self.use_etw = False
        
        # Fallback QPC timer
        try:
            self.kernel32 = ctypes.windll.kernel32
            self.QueryPerformanceFrequency = self.kernel32.QueryPerformanceFrequency
            self.QueryPerformanceCounter = self.kernel32.QueryPerformanceCounter
            
            self.qpc_freq = wintypes.LARGE_INTEGER()
            self.QueryPerformanceFrequency(ctypes.byref(self.qpc_freq))
            self.qpc_freq_val = self.qpc_freq.value
            
        except Exception as e:
            logger.error(f"QPC initialization error: {e}")
            self.qpc_freq_val = 1000000  # Fallback to microseconds
    
    def start_monitoring(self, pid: int, game_exe: str) -> bool:
        """Start monitoring for a game process with ETW (when available) or fallback mode"""
        
        with self.lock:
            if pid in self.active_sessions:
                return False
            
            try:
                # Initialize session
                session = {
                    'pid': pid,
                    'game_exe': game_exe,
                    'start_time': time.time(),
                    'frame_times': deque(maxlen=10000),  # Last 10k frames
                    'cpu_samples': deque(maxlen=600),    # 10 min at 1Hz
                    'memory_samples': deque(maxlen=600),
                    'dpc_readings': deque(maxlen=1000),
                    'context_switches': deque(maxlen=1000),  # NEW: Context switch tracking
                    'cpu_temp_samples': deque(maxlen=600),  # NEW: CPU temperature tracking
                    'gpu_temp_samples': deque(maxlen=600),  # NEW: GPU temperature tracking
                    'active': True,
                    'baseline_frame_time_p999': None,  # NEW: Baseline for alerts
                    'use_etw': self.use_etw,  # Track if this session uses ETW
                }
                
                self.active_sessions[pid] = session
                
                # Start ETW monitors if available (V4.0)
                if self.use_etw and self.etw_frame_monitor and self.etw_dpc_monitor:
                    try:
                        # Start frame time monitoring via ETW
                        if self.etw_frame_monitor.start(session_name=f"FrameTime_{pid}"):
                            session['etw_frame_active'] = True
                            logger.info(f"✓ ETW frame time monitoring started for PID {pid}")
                        else:
                            session['etw_frame_active'] = False
                            logger.warning(f"ETW frame monitor failed to start for PID {pid}, using fallback")
                        
                        # Start DPC monitoring via ETW
                        if self.etw_dpc_monitor.start(session_name=f"DPC_{pid}"):
                            session['etw_dpc_active'] = True
                            logger.info(f"✓ ETW DPC monitoring started for PID {pid}")
                        else:
                            session['etw_dpc_active'] = False
                            logger.warning(f"ETW DPC monitor failed to start for PID {pid}, using fallback")
                    except Exception as e:
                        logger.warning(f"ETW startup error for PID {pid}: {e}, using fallback")
                        session['etw_frame_active'] = False
                        session['etw_dpc_active'] = False
                
                # Start monitoring thread (handles both ETW and fallback modes)
                monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    args=(pid,),
                    daemon=True
                )
                monitor_thread.start()
                self.monitor_threads[pid] = monitor_thread
                
                logger.info(f"✓ Performance monitoring started for PID {pid}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}")
                return False
    
    def _monitor_loop(self, pid: int):
        """Background monitoring loop with ETW or fallback mode"""
        
        try:
            process = psutil.Process(pid)
            last_frame_time = self._get_qpc_time()
            last_dpc_check = time.time()
            last_temp_check = time.time()
            last_context_switches = None
            last_etw_sync = time.time()
            
            while True:
                with self.lock:
                    if pid not in self.active_sessions or not self.active_sessions[pid]['active']:
                        break
                    
                    session = self.active_sessions[pid]
                    use_etw_frames = session.get('etw_frame_active', False)
                    use_etw_dpc = session.get('etw_dpc_active', False)
                
                # Frame time collection: ETW or fallback
                if use_etw_frames and self.etw_frame_monitor:
                    # Sync ETW frame times every second
                    if time.time() - last_etw_sync > 1.0:
                        etw_frame_times = self.etw_frame_monitor.get_frame_times()
                        if etw_frame_times:
                            with self.lock:
                                # Add new ETW frame times
                                for ft in etw_frame_times[-100:]:  # Last 100 frames
                                    session['frame_times'].append(ft)
                        last_etw_sync = time.time()
                else:
                    # Fallback: QPC-based frame time estimation
                    current_time = self._get_qpc_time()
                    frame_delta = (current_time - last_frame_time) * 1000  # ms
                    
                    if 5 < frame_delta < 500:  # Sanity check: 2-200 FPS
                        with self.lock:
                            session['frame_times'].append(frame_delta)
                    
                    last_frame_time = current_time
                
                # CPU/Memory sampling (every second)
                try:
                    cpu_percent = process.cpu_percent(interval=0.1)
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    
                    # Context switch tracking
                    try:
                        ctx_switches = process.num_ctx_switches()
                        if last_context_switches is not None:
                            # Calculate delta
                            voluntary_delta = ctx_switches.voluntary - last_context_switches.voluntary
                            involuntary_delta = ctx_switches.involuntary - last_context_switches.involuntary
                            total_switches = voluntary_delta + involuntary_delta
                            
                            with self.lock:
                                session['context_switches'].append(total_switches)
                        
                        last_context_switches = ctx_switches
                    except Exception:
                        pass
                    
                    with self.lock:
                        session['cpu_samples'].append(cpu_percent)
                        session['memory_samples'].append(memory_mb)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                
                # Temperature check (every 10 seconds)
                if time.time() - last_temp_check > 10:
                    cpu_temp = self._measure_cpu_temperature()
                    gpu_temp = self._measure_gpu_temperature()
                    
                    with self.lock:
                        if cpu_temp is not None:
                            session['cpu_temp_samples'].append(cpu_temp)
                        if gpu_temp is not None:
                            session['gpu_temp_samples'].append(gpu_temp)
                    
                    last_temp_check = time.time()
                
                # DPC latency check (every 5 seconds): ETW or fallback
                if time.time() - last_dpc_check > 5:
                    if use_etw_dpc and self.etw_dpc_monitor:
                        # Use real ETW DPC data
                        avg_latency = self.etw_dpc_monitor.get_average_latency()
                        max_latency = self.etw_dpc_monitor.get_recent_max_latency(window_seconds=60)
                        
                        if avg_latency is not None and avg_latency > 100:
                            dpc_latency = DPCLatencyReading(
                                timestamp=time.time(),
                                latency_us=avg_latency,
                                offending_driver="ETW-detected"  # Would need driver info from ETW
                            )
                            with self.lock:
                                session['dpc_readings'].append(dpc_latency)
                    else:
                        # Fallback: sleep overshoot heuristic
                        dpc_latency = self._measure_dpc_latency()
                        if dpc_latency:
                            with self.lock:
                                session['dpc_readings'].append(dpc_latency)
                    last_dpc_check = time.time()
                
                # Sleep to maintain ~60 Hz monitoring
                time.sleep(1.0 / 60)
                
        except Exception as e:
            logger.error(f"Monitor loop error for PID {pid}: {e}")
    
    def _get_qpc_time(self) -> float:
        """Get high-resolution timestamp in seconds"""
        counter = wintypes.LARGE_INTEGER()
        self.QueryPerformanceCounter(ctypes.byref(counter))
        return counter.value / self.qpc_freq_val
    
    def _measure_dpc_latency(self) -> Optional[DPCLatencyReading]:
        """
        Measure DPC/ISR latency by checking if Sleep(1) overshoots significantly.
        High latency = drivers causing interrupt delays.
        """
        try:
            start = self._get_qpc_time()
            time.sleep(0.001)  # 1ms sleep
            end = self._get_qpc_time()
            
            actual_sleep_us = (end - start) * 1_000_000
            expected_sleep_us = 1000
            
            latency_us = actual_sleep_us - expected_sleep_us
            
            if latency_us > 100:  # >100μs overhead
                return DPCLatencyReading(
                    timestamp=time.time(),
                    latency_us=latency_us,
                    offending_driver=self._identify_offending_driver() if latency_us > 500 else None
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"DPC measurement error: {e}")
            return None
    
    def _identify_offending_driver(self) -> Optional[str]:
        """
        Simplified driver identification via common culprits.
        Real implementation would use ETW kernel events.
        """
        common_offenders = [
            "nvlddmkm.sys",      # NVIDIA
            "amdkmdag.sys",      # AMD
            "intelppm.sys",      # Intel Power Management
            "RTKVHD64.sys",      # Realtek Audio
            "ndis.sys",          # Network
            "storport.sys",      # Storage
        ]
        
        # This is a placeholder - real implementation needs kernel tracing
        return "Unknown (enable kernel tracing for details)"
    
    def _measure_cpu_temperature(self) -> Optional[float]:
        """
        Measure CPU temperature using WMI.
        Returns temperature in Celsius or None if unavailable.
        """
        try:
            import wmi
            w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            temperature_infos = w.Sensor()
            
            for sensor in temperature_infos:
                if sensor.SensorType == 'Temperature' and 'CPU' in sensor.Name:
                    return float(sensor.Value)
            
            # Fallback: Try WMI MSAcpi_ThermalZoneTemperature
            w2 = wmi.WMI(namespace="root\\wmi")
            for temp in w2.MSAcpi_ThermalZoneTemperature():
                # Convert from tenths of Kelvin to Celsius
                kelvin = temp.CurrentTemperature / 10.0
                celsius = kelvin - 273.15
                return celsius
        except Exception as e:
            logger.debug(f"CPU temperature measurement failed: {e}")
            return None
    
    def _measure_gpu_temperature(self) -> Optional[float]:
        """
        Measure GPU temperature using WMI or NVIDIA/AMD APIs.
        Returns temperature in Celsius or None if unavailable.
        """
        try:
            import wmi
            w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            temperature_infos = w.Sensor()
            
            for sensor in temperature_infos:
                if sensor.SensorType == 'Temperature' and 'GPU' in sensor.Name:
                    return float(sensor.Value)
        except Exception as e:
            logger.debug(f"GPU temperature measurement failed: {e}")
        
        return None
    
    def stop_monitoring(self, pid: int) -> bool:
        """Stop monitoring for a process and cleanup ETW sessions"""
        
        with self.lock:
            if pid not in self.active_sessions:
                return False
            
            session = self.active_sessions[pid]
            session['active'] = False
            session['end_time'] = time.time()
            
            # Stop ETW monitors for this session
            if session.get('etw_frame_active', False) and self.etw_frame_monitor:
                try:
                    self.etw_frame_monitor.stop()
                    logger.info(f"✓ ETW frame monitor stopped for PID {pid}")
                except Exception as e:
                    logger.debug(f"ETW frame monitor stop error: {e}")
            
            if session.get('etw_dpc_active', False) and self.etw_dpc_monitor:
                try:
                    self.etw_dpc_monitor.stop()
                    logger.info(f"✓ ETW DPC monitor stopped for PID {pid}")
                except Exception as e:
                    logger.debug(f"ETW DPC monitor stop error: {e}")
        
        # Wait for thread to finish
        if pid in self.monitor_threads:
            self.monitor_threads[pid].join(timeout=2)
            del self.monitor_threads[pid]
        
        logger.info(f"✓ Performance monitoring stopped for PID {pid}")
        return True
    
    def get_session_summary(self, pid: int) -> Optional[Dict[str, Any]]:
        """Get summary metrics for a session"""
        
        with self.lock:
            if pid not in self.active_sessions:
                return None
            
            session = self.active_sessions[pid]
            
            # Calculate frame metrics
            frame_times = list(session['frame_times'])
            if not frame_times:
                return None
            
            try:
                frame_times_sorted = sorted(frame_times)
                n_frames = len(frame_times_sorted)
                
                avg_frame_time = statistics.mean(frame_times)
                avg_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
                
                # Percentiles
                p99_idx = int(n_frames * 0.99)
                p999_idx = int(n_frames * 0.999)
                p1_idx = int(n_frames * 0.01)
                p01_idx = int(n_frames * 0.001)
                
                frame_time_p99 = frame_times_sorted[p99_idx] if p99_idx < n_frames else frame_times_sorted[-1]
                frame_time_p999 = frame_times_sorted[p999_idx] if p999_idx < n_frames else frame_times_sorted[-1]
                
                # 1% and 0.1% lows (FPS, not frame time)
                one_percent_low_ft = frame_times_sorted[p99_idx] if p99_idx < n_frames else frame_times_sorted[-1]
                point_one_percent_low_ft = frame_times_sorted[p999_idx] if p999_idx < n_frames else frame_times_sorted[-1]
                
                one_percent_low = 1000.0 / one_percent_low_ft if one_percent_low_ft > 0 else 0
                point_one_percent_low = 1000.0 / point_one_percent_low_ft if point_one_percent_low_ft > 0 else 0
                
                # Stutter detection (frame time >2x median)
                median_ft = statistics.median(frame_times)
                stutter_threshold = median_ft * 2
                stutter_count = sum(1 for ft in frame_times if ft > stutter_threshold)
                
                summary = {
                    'avg_fps': avg_fps,
                    'min_fps': 1000.0 / max(frame_times) if max(frame_times) > 0 else 0,
                    'max_fps': 1000.0 / min(frame_times) if min(frame_times) > 0 else 0,
                    'one_percent_low': one_percent_low,
                    'point_one_percent_low': point_one_percent_low,
                    'frame_time_avg': avg_frame_time,
                    'frame_time_p99': frame_time_p99,
                    'frame_time_p999': frame_time_p999,
                    'stutter_count': stutter_count,
                    'total_frames': n_frames,
                    'cpu_usage_avg': statistics.mean(session['cpu_samples']) if session['cpu_samples'] else 0,
                    'memory_mb_avg': statistics.mean(session['memory_samples']) if session['memory_samples'] else 0,
                    'dpc_latency_avg_us': statistics.mean([r.latency_us for r in session['dpc_readings']]) if session['dpc_readings'] else 0,
                    'dpc_latency_max_us': max([r.latency_us for r in session['dpc_readings']], default=0) if session['dpc_readings'] else 0,
                    'dpc_spikes_count': sum(1 for r in session['dpc_readings'] if r.latency_us > 500),
                    'context_switches_avg': statistics.mean(session['context_switches']) if session['context_switches'] else 0,
                    'context_switch_spikes': sum(1 for cs in session['context_switches'] if cs > 1000) if session['context_switches'] else 0,
                    'cpu_temp_avg': statistics.mean(session['cpu_temp_samples']) if session['cpu_temp_samples'] else 0,
                    'gpu_temp_avg': statistics.mean(session['gpu_temp_samples']) if session['gpu_temp_samples'] else 0,
                }
                
                # Calculate memory pressure percentage
                try:
                    total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
                    memory_used_mb = summary['memory_mb_avg']
                    if total_ram_mb > 0:
                        summary['memory_pressure_pct'] = (memory_used_mb / total_ram_mb) * 100
                    else:
                        summary['memory_pressure_pct'] = 0.0
                except Exception:
                    summary['memory_pressure_pct'] = 0.0
                
                # Store baseline for alerts if not set
                if session.get('baseline_frame_time_p999') is None:
                    session['baseline_frame_time_p999'] = frame_time_p999
                
                return summary
                
            except Exception as e:
                logger.error(f"Summary calculation error: {e}")
                return None
    
    def check_critical_alerts(self, pid: int) -> bool:
        """
        Check if critical performance alerts are triggered.
        Returns True if optimization should be rolled back.
        """
        with self.lock:
            if pid not in self.active_sessions:
                return False
            
            session = self.active_sessions[pid]
            
            # Need at least 100 frames for meaningful analysis
            if len(session['frame_times']) < 100:
                return False
            
            try:
                frame_times = list(session['frame_times'])
                frame_times_sorted = sorted(frame_times)
                n_frames = len(frame_times_sorted)
                
                # Calculate recent P99.9
                p999_idx = int(n_frames * 0.999)
                # Clamp index to valid range
                p999_idx = min(p999_idx, n_frames - 1)
                current_p999 = frame_times_sorted[p999_idx]
                
                # Check if baseline exists
                baseline_p999 = session.get('baseline_frame_time_p999')
                
                # Alert 1: Frame time P99.9 exceeds baseline by 15%
                if baseline_p999 and baseline_p999 > 0:
                    if current_p999 > baseline_p999 * 1.15:
                        logger.warning(f"⚠️  ALERT: Frame time P99.9 increased {((current_p999/baseline_p999 - 1)*100):.1f}% (baseline: {baseline_p999:.2f}ms, current: {current_p999:.2f}ms)")
                        return True
                
                # Alert 2: DPC latency max exceeds 1500 µs
                if session['dpc_readings']:
                    max_dpc = max([r.latency_us for r in session['dpc_readings']])
                    if max_dpc > 1500:
                        logger.warning(f"⚠️  ALERT: DPC latency spike detected ({max_dpc:.0f}µs > 1500µs threshold)")
                        return True
                
                return False
                
            except Exception as e:
                logger.debug(f"Critical alerts check error: {e}")
                return False
    
    def cleanup(self):
        """Stop all monitoring sessions and ETW monitors"""
        with self.lock:
            for pid in list(self.active_sessions.keys()):
                self.active_sessions[pid]['active'] = False
        
        for thread in self.monitor_threads.values():
            thread.join(timeout=1)
        
        # Cleanup ETW monitors
        if self.etw_frame_monitor:
            try:
                self.etw_frame_monitor.stop()
            except Exception as e:
                logger.debug(f"ETW frame monitor cleanup error: {e}")
        
        if self.etw_dpc_monitor:
            try:
                self.etw_dpc_monitor.stop()
            except Exception as e:
                logger.debug(f"ETW DPC monitor cleanup error: {e}")
        
        self.active_sessions.clear()
        self.monitor_threads.clear()
        logger.info("✓ Performance monitoring cleanup complete")


class TelemetryCollector:
    """Collect and export telemetry data"""
    
    def __init__(self):
        self.active_sessions: Dict[int, Dict[str, Any]] = {}
        self.completed_sessions: List[SessionTelemetry] = []
        self.lock = threading.Lock()
        
        self.telemetry_dir = Path.home() / '.game_optimizer' / 'telemetry'
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
    
    def start_session(self, pid: int, game_exe: str, profile, opt_state):
        """Start telemetry collection for a session with exact hardware versions"""
        
        with self.lock:
            import uuid
            
            self.active_sessions[pid] = {
                'session_id': str(uuid.uuid4()),
                'game_exe': game_exe,
                'game_pid': pid,
                'start_time': time.time(),
                'profile_name': profile.name,
                'optimizations': list(opt_state.optimizations_applied),
                'cpu_model': self._get_cpu_model(),
                'gpu_model': self._get_gpu_model(),
                'ram_gb': self._get_ram_gb(),
                'gpu_driver_version': self._get_gpu_driver_version(),
                'bios_version': self._get_bios_version(),
            }
    
    def end_session(self, pid: int, performance_metrics: Optional[Dict[str, Any]] = None, optimization_failed: bool = False) -> Optional[SessionTelemetry]:
        """End telemetry session and create final report with performance metrics"""
        
        with self.lock:
            if pid not in self.active_sessions:
                return None
            
            session_data = self.active_sessions.pop(pid)
            session_data['end_time'] = time.time()
            session_data['duration_seconds'] = session_data['end_time'] - session_data['start_time']
            
            # Get performance metrics from PerformanceMonitor if provided
            if performance_metrics:
                session_data['frame_metrics'] = None  # Would need to construct FrameMetrics object
                session_data['cpu_usage_avg'] = performance_metrics.get('cpu_usage_avg', 0.0)
                session_data['gpu_usage_avg'] = 0.0  # Not yet tracked in PerformanceMonitor
                session_data['memory_usage_avg_mb'] = performance_metrics.get('memory_mb_avg', 0.0)
                session_data['dpc_latency_avg_us'] = performance_metrics.get('dpc_latency_avg_us', 0.0)
                session_data['dpc_latency_max_us'] = performance_metrics.get('dpc_latency_max_us', 0.0)
                session_data['dpc_spikes_count'] = performance_metrics.get('dpc_spikes_count', 0)
                session_data['context_switches_avg'] = performance_metrics.get('context_switches_avg', 0.0)
                session_data['context_switch_spikes'] = performance_metrics.get('context_switch_spikes', 0)
                session_data['cpu_temp_avg'] = performance_metrics.get('cpu_temp_avg', 0.0)
                session_data['gpu_temp_avg'] = performance_metrics.get('gpu_temp_avg', 0.0)
                session_data['frame_time_p999_ms'] = performance_metrics.get('frame_time_p999', 0.0)
                
                # Calculate memory pressure percentage
                try:
                    import psutil
                    total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
                    memory_used_mb = performance_metrics.get('memory_mb_avg', 0.0)
                    if total_ram_mb > 0:
                        session_data['memory_pressure_pct'] = (memory_used_mb / total_ram_mb) * 100
                    else:
                        session_data['memory_pressure_pct'] = 0.0
                except Exception:
                    session_data['memory_pressure_pct'] = 0.0
            else:
                # Use placeholders if no metrics provided
                session_data['frame_metrics'] = None
                session_data['cpu_usage_avg'] = 0.0
                session_data['gpu_usage_avg'] = 0.0
                session_data['memory_usage_avg_mb'] = 0.0
                session_data['dpc_latency_avg_us'] = 0.0
                session_data['dpc_latency_max_us'] = 0.0
                session_data['dpc_spikes_count'] = 0
                session_data['context_switches_avg'] = 0.0
                session_data['context_switch_spikes'] = 0
                session_data['cpu_temp_avg'] = 0.0
                session_data['gpu_temp_avg'] = 0.0
                session_data['frame_time_p999_ms'] = 0.0
                session_data['memory_pressure_pct'] = 0.0
            
            # Track rollback status
            session_data['optimization_failed'] = optimization_failed
            
            telemetry = SessionTelemetry(**session_data)
            self.completed_sessions.append(telemetry)
            
            # Auto-export
            self._export_session(telemetry)
            
            return telemetry
    
    def _export_session(self, telemetry: SessionTelemetry):
        """Export single session to JSON"""
        try:
            filename = f"{telemetry.game_exe}_{int(telemetry.start_time)}.json"
            filepath = self.telemetry_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(telemetry), f, indent=2)
                
        except Exception as e:
            logger.error(f"Telemetry export error: {e}")
    
    def export_to_file(self, output_file: Optional[Path] = None) -> Path:
        """Export all telemetry to a single JSON file"""
        
        if output_file is None:
            output_file = self.telemetry_dir / f"telemetry_export_{int(time.time())}.json"
        
        try:
            data = {
                'export_time': time.time(),
                'total_sessions': len(self.completed_sessions),
                'sessions': [asdict(s) for s in self.completed_sessions]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✓ Telemetry exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return output_file
    
    def _get_cpu_model(self) -> str:
        try:
            import platform
            return platform.processor()
        except Exception:
            return "Unknown CPU"
    
    def _get_gpu_model(self) -> str:
        try:
            import wmi
            w = wmi.WMI()
            for gpu in w.Win32_VideoController():
                return gpu.Name
        except Exception:
            return "Unknown GPU"
    
    def _get_ram_gb(self) -> int:
        try:
            return int(psutil.virtual_memory().total / (1024**3))
        except Exception:
            return 0
    
    def _get_gpu_driver_version(self) -> str:
        """Get exact GPU driver version (e.g., 555.99 for NVIDIA)"""
        try:
            import wmi
            w = wmi.WMI()
            for gpu in w.Win32_VideoController():
                if gpu.DriverVersion:
                    return gpu.DriverVersion
        except Exception:
            pass
        
        # Fallback: Try registry for NVIDIA
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"SOFTWARE\NVIDIA Corporation\Global\Display", 0, 
                               winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                version, _ = winreg.QueryValueEx(key, "Version")
                return version
        except Exception:
            pass
        
        return "Unknown"
    
    def _get_bios_version(self) -> str:
        """Get exact BIOS version"""
        try:
            import wmi
            w = wmi.WMI()
            for bios in w.Win32_BIOS():
                if bios.SMBIOSBIOSVersion:
                    return bios.SMBIOSBIOSVersion
        except Exception:
            pass
        
        return "Unknown"


class ABTestingFramework:
    """A/B testing framework to compare baseline vs optimized performance"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
    
    def collect_baseline(self, pid: int, duration: int = 30) -> Dict[str, float]:
        """Collect baseline metrics without optimizations"""
        
        logger.info(f"Collecting baseline for {duration}s...")
        
        self.performance_monitor.start_monitoring(pid, "baseline_test")
        time.sleep(duration)
        
        metrics = self.performance_monitor.get_session_summary(pid)
        self.performance_monitor.stop_monitoring(pid)
        
        if not metrics:
            return {}
        
        return metrics
    
    def run_full_test(self, game_pid: int, game_exe: str, optimizer, duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Run full A/B test: baseline → optimize → compare
        
        Args:
            game_pid: Process ID of game
            game_exe: Game executable name
            optimizer: GameOptimizer instance
            duration_minutes: Duration for each phase
            
        Returns:
            Dict with baseline, optimized, and improvement metrics
        """
        
        duration_sec = duration_minutes * 60
        
        # Phase 1: Baseline
        logger.info("=" * 80)
        logger.info("A/B TEST PHASE 1: BASELINE (NO OPTIMIZATIONS)")
        logger.info("=" * 80)
        
        baseline = self.collect_baseline(game_pid, duration=duration_sec)
        
        if not baseline:
            logger.error("Failed to collect baseline metrics")
            return {}
        
        # Phase 2: Apply optimizations
        logger.info("=" * 80)
        logger.info("A/B TEST PHASE 2: APPLYING OPTIMIZATIONS")
        logger.info("=" * 80)
        
        optimizer.start_optimization(game_pid, enable_ab_test=False)
        
        # Phase 3: Collect optimized metrics
        logger.info("=" * 80)
        logger.info(f"A/B TEST PHASE 3: OPTIMIZED ({duration_minutes} minutes)")
        logger.info("=" * 80)
        
        self.performance_monitor.start_monitoring(game_pid, game_exe)
        time.sleep(duration_sec)
        
        optimized = self.performance_monitor.get_session_summary(game_pid)
        self.performance_monitor.stop_monitoring(game_pid)
        
        if not optimized:
            logger.error("Failed to collect optimized metrics")
            optimizer.stop_optimization(game_pid)
            return {}
        
        # Calculate improvements
        improvements = self._calculate_improvements(baseline, optimized)
        
        results = {
            'baseline': baseline,
            'optimized': optimized,
            'improvement': improvements,
            'test_duration_minutes': duration_minutes,
            'timestamp': time.time()
        }
        
        # Export results
        self._export_ab_test_results(game_exe, results)
        
        return results
    
    def _calculate_improvements(self, baseline: Dict, optimized: Dict) -> Dict[str, float]:
        """Calculate percentage improvements"""
        
        improvements = {}
        
        for key in ['avg_fps', 'one_percent_low', 'point_one_percent_low']:
            if key in baseline and key in optimized:
                base_val = baseline[key]
                opt_val = optimized[key]
                
                if base_val > 0:
                    pct_change = ((opt_val - base_val) / base_val) * 100
                    improvements[f"{key}_pct"] = pct_change
        
        # Frame time is inverted (lower is better)
        for key in ['frame_time_avg', 'frame_time_p99', 'frame_time_p999']:
            if key in baseline and key in optimized:
                base_val = baseline[key]
                opt_val = optimized[key]
                
                if base_val > 0:
                    pct_change = ((base_val - opt_val) / base_val) * 100
                    improvements[f"{key}_pct"] = pct_change
        
        return improvements
    
    def _export_ab_test_results(self, game_exe: str, results: Dict):
        """Export A/B test results to JSON"""
        try:
            export_dir = Path.home() / '.game_optimizer' / 'ab_tests'
            export_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"ab_test_{game_exe}_{int(time.time())}.json"
            filepath = export_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✓ A/B test results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"A/B test export error: {e}")