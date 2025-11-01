"""
Automated Benchmarking System V4.0
Comprehensive performance testing and analysis framework
"""

import logging
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available for benchmarking")


@dataclass
class BenchmarkMetrics:
    """Performance metrics collected during benchmark"""
    timestamp: float
    fps: float
    frame_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: float = 0.0
    gpu_temp_celsius: float = 0.0
    power_draw_watts: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark results"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # FPS statistics
    avg_fps: float
    min_fps: float
    max_fps: float
    fps_1_percent_low: float
    fps_0_1_percent_low: float
    fps_std_dev: float
    
    # Frame time statistics
    avg_frame_time_ms: float
    p95_frame_time_ms: float
    p99_frame_time_ms: float
    frame_time_std_dev_ms: float
    
    # System statistics
    avg_cpu_usage: float
    avg_memory_usage_mb: float
    avg_gpu_usage: float
    avg_power_draw: float
    
    # Stability metrics
    frame_time_variance: float
    stability_score: float  # 0-100
    stutter_count: int
    
    # Raw data
    metrics: List[BenchmarkMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


class PerformanceCollector:
    """Collects performance metrics during benchmark"""
    
    def __init__(self, target_pid: Optional[int] = None):
        self.target_pid = target_pid
        self.collecting = False
        self.metrics: List[BenchmarkMetrics] = []
        self.collection_interval = 0.1  # 100ms
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start collecting metrics"""
        if self.collecting:
            logger.warning("Already collecting metrics")
            return
        
        self.collecting = True
        self.metrics = []
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info("Performance collection started")
    
    def stop(self):
        """Stop collecting metrics"""
        if not self.collecting:
            return
        
        self.collecting = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(f"Performance collection stopped. Collected {len(self.metrics)} samples")
    
    def _collect_loop(self):
        """Main collection loop"""
        while self.collecting:
            try:
                metric = self._collect_single_metric()
                if metric:
                    self.metrics.append(metric)
            except Exception as e:
                logger.debug(f"Error collecting metric: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_single_metric(self) -> Optional[BenchmarkMetrics]:
        """Collect a single metric snapshot"""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            # Get process info if target PID is specified
            if self.target_pid:
                try:
                    process = psutil.Process(self.target_pid)
                    cpu_usage = process.cpu_percent(interval=0.01)
                    memory_usage = process.memory_info().rss / (1024 ** 2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cpu_usage = psutil.cpu_percent(interval=0.01)
                    memory_usage = psutil.virtual_memory().used / (1024 ** 2)
            else:
                cpu_usage = psutil.cpu_percent(interval=0.01)
                memory_usage = psutil.virtual_memory().used / (1024 ** 2)
            
            # Estimate FPS from frame time (placeholder - real implementation would use ETW)
            # For now, simulate based on CPU usage (higher usage = lower FPS)
            estimated_fps = max(30, 144 - (cpu_usage * 0.5))
            frame_time = 1000.0 / estimated_fps if estimated_fps > 0 else 16.67
            
            return BenchmarkMetrics(
                timestamp=time.time(),
                fps=estimated_fps,
                frame_time_ms=frame_time,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage,
                gpu_usage_percent=0.0,  # Would require GPU monitoring
                gpu_temp_celsius=0.0,
                power_draw_watts=0.0
            )
        except Exception as e:
            logger.debug(f"Error in metric collection: {e}")
            return None


class BenchmarkAnalyzer:
    """Analyzes benchmark metrics and generates results"""
    
    @staticmethod
    def analyze(metrics: List[BenchmarkMetrics], test_name: str, 
                start_time: datetime, end_time: datetime) -> BenchmarkResult:
        """Analyze metrics and generate benchmark result"""
        if not metrics:
            raise ValueError("No metrics to analyze")
        
        duration = (end_time - start_time).total_seconds()
        
        # Extract data series
        fps_values = [m.fps for m in metrics]
        frame_times = [m.frame_time_ms for m in metrics]
        cpu_values = [m.cpu_usage_percent for m in metrics]
        memory_values = [m.memory_usage_mb for m in metrics]
        gpu_values = [m.gpu_usage_percent for m in metrics]
        power_values = [m.power_draw_watts for m in metrics]
        
        # Calculate FPS statistics
        avg_fps = statistics.mean(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
        fps_std_dev = statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0
        
        # Calculate percentile lows
        sorted_fps = sorted(fps_values)
        fps_1_percent_low = BenchmarkAnalyzer._percentile_low(sorted_fps, 1.0)
        fps_0_1_percent_low = BenchmarkAnalyzer._percentile_low(sorted_fps, 0.1)
        
        # Calculate frame time statistics
        avg_frame_time = statistics.mean(frame_times)
        sorted_frame_times = sorted(frame_times)
        p95_frame_time = BenchmarkAnalyzer._percentile(sorted_frame_times, 95.0)
        p99_frame_time = BenchmarkAnalyzer._percentile(sorted_frame_times, 99.0)
        frame_time_std_dev = statistics.stdev(frame_times) if len(frame_times) > 1 else 0.0
        
        # Calculate system statistics
        avg_cpu = statistics.mean(cpu_values)
        avg_memory = statistics.mean(memory_values)
        avg_gpu = statistics.mean(gpu_values)
        avg_power = statistics.mean(power_values)
        
        # Calculate stability metrics
        frame_time_variance = statistics.variance(frame_times) if len(frame_times) > 1 else 0.0
        stutter_count = BenchmarkAnalyzer._count_stutters(frame_times)
        stability_score = BenchmarkAnalyzer._calculate_stability_score(
            frame_time_std_dev, stutter_count, len(metrics)
        )
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            avg_fps=avg_fps,
            min_fps=min_fps,
            max_fps=max_fps,
            fps_1_percent_low=fps_1_percent_low,
            fps_0_1_percent_low=fps_0_1_percent_low,
            fps_std_dev=fps_std_dev,
            avg_frame_time_ms=avg_frame_time,
            p95_frame_time_ms=p95_frame_time,
            p99_frame_time_ms=p99_frame_time,
            frame_time_std_dev_ms=frame_time_std_dev,
            avg_cpu_usage=avg_cpu,
            avg_memory_usage_mb=avg_memory,
            avg_gpu_usage=avg_gpu,
            avg_power_draw=avg_power,
            frame_time_variance=frame_time_variance,
            stability_score=stability_score,
            stutter_count=stutter_count,
            metrics=metrics
        )
    
    @staticmethod
    def _percentile(sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values"""
        if not sorted_values:
            return 0.0
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    @staticmethod
    def _percentile_low(sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile low (average of bottom percentile)"""
        if not sorted_values:
            return 0.0
        count = max(1, int((percentile / 100.0) * len(sorted_values)))
        return statistics.mean(sorted_values[:count])
    
    @staticmethod
    def _count_stutters(frame_times: List[float], threshold_multiplier: float = 2.0) -> int:
        """Count frame time stutters"""
        if len(frame_times) < 2:
            return 0
        
        median_ft = statistics.median(frame_times)
        threshold = median_ft * threshold_multiplier
        
        stutter_count = 0
        for ft in frame_times:
            if ft > threshold:
                stutter_count += 1
        
        return stutter_count
    
    @staticmethod
    def _calculate_stability_score(std_dev: float, stutter_count: int, 
                                   total_samples: int) -> float:
        """Calculate stability score (0-100, higher is better)"""
        # Base score on standard deviation (lower is better)
        std_dev_score = max(0, 100 - (std_dev * 2))
        
        # Penalty for stutters
        stutter_ratio = stutter_count / max(total_samples, 1)
        stutter_penalty = stutter_ratio * 50
        
        score = max(0, min(100, std_dev_score - stutter_penalty))
        return round(score, 2)


class AutomatedBenchmark:
    """
    Automated benchmarking system with comprehensive analysis
    
    Features:
    - Automated performance testing
    - Real-time metric collection
    - Statistical analysis
    - Comparison between runs
    - Historical tracking
    """
    
    def __init__(self):
        self.results_dir = Path.home() / '.game_optimizer' / 'benchmarks'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_collector: Optional[PerformanceCollector] = None
        self.benchmark_history: List[BenchmarkResult] = []
        
        self._load_history()
    
    def run_benchmark(self, test_name: str, duration_seconds: int = 60,
                     target_pid: Optional[int] = None) -> BenchmarkResult:
        """
        Run automated benchmark
        
        Args:
            test_name: Name of the benchmark test
            duration_seconds: Duration of the test in seconds
            target_pid: Target process PID (optional)
        
        Returns:
            BenchmarkResult with complete analysis
        """
        logger.info(f"Starting benchmark: {test_name} (duration: {duration_seconds}s)")
        
        start_time = datetime.now()
        
        # Start collecting metrics
        collector = PerformanceCollector(target_pid)
        collector.start()
        
        # Wait for benchmark duration
        logger.info(f"Collecting metrics for {duration_seconds} seconds...")
        time.sleep(duration_seconds)
        
        # Stop collection
        collector.stop()
        end_time = datetime.now()
        
        # Analyze results
        logger.info("Analyzing benchmark results...")
        result = BenchmarkAnalyzer.analyze(
            collector.metrics,
            test_name,
            start_time,
            end_time
        )
        
        # Save result
        self._save_result(result)
        self.benchmark_history.append(result)
        
        logger.info(f"Benchmark complete: Avg FPS={result.avg_fps:.1f}, "
                   f"1% Low={result.fps_1_percent_low:.1f}, "
                   f"Stability={result.stability_score:.1f}")
        
        return result
    
    def run_ab_test(self, baseline_name: str, optimized_name: str,
                   duration_seconds: int = 60, target_pid: Optional[int] = None) -> Dict[str, Any]:
        """
        Run A/B comparison test
        
        Args:
            baseline_name: Name for baseline test
            optimized_name: Name for optimized test
            duration_seconds: Duration per test
            target_pid: Target process PID
        
        Returns:
            Comparison results
        """
        logger.info(f"Starting A/B test: {baseline_name} vs {optimized_name}")
        
        # Run baseline
        logger.info("Running baseline benchmark...")
        baseline = self.run_benchmark(baseline_name, duration_seconds, target_pid)
        
        logger.info("Waiting 10 seconds before optimized run...")
        time.sleep(10)
        
        # Run optimized
        logger.info("Running optimized benchmark...")
        optimized = self.run_benchmark(optimized_name, duration_seconds, target_pid)
        
        # Calculate improvements
        comparison = {
            'baseline': baseline.to_dict(),
            'optimized': optimized.to_dict(),
            'improvements': {
                'avg_fps_delta': optimized.avg_fps - baseline.avg_fps,
                'avg_fps_pct': ((optimized.avg_fps - baseline.avg_fps) / baseline.avg_fps * 100),
                '1_percent_low_delta': optimized.fps_1_percent_low - baseline.fps_1_percent_low,
                '1_percent_low_pct': ((optimized.fps_1_percent_low - baseline.fps_1_percent_low) / 
                                     baseline.fps_1_percent_low * 100),
                'stability_delta': optimized.stability_score - baseline.stability_score,
                'stutter_reduction': baseline.stutter_count - optimized.stutter_count,
            }
        }
        
        logger.info(f"A/B Test Results:")
        logger.info(f"  FPS Improvement: {comparison['improvements']['avg_fps_pct']:.1f}%")
        logger.info(f"  1% Low Improvement: {comparison['improvements']['1_percent_low_pct']:.1f}%")
        logger.info(f"  Stability Improvement: {comparison['improvements']['stability_delta']:.1f}")
        
        return comparison
    
    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to file"""
        try:
            filename = f"benchmark_{result.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.debug(f"Benchmark result saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")
    
    def _load_history(self):
        """Load benchmark history"""
        try:
            for filepath in sorted(self.results_dir.glob("benchmark_*.json")):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Note: Full reconstruction would require converting back to dataclass
                    # For now, just log that we found files
                    logger.debug(f"Found benchmark file: {filepath.name}")
        except Exception as e:
            logger.debug(f"Error loading benchmark history: {e}")
    
    def get_latest_results(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get latest benchmark results"""
        return [r.to_dict() for r in self.benchmark_history[-count:]]
    
    def export_report(self, output_file: Path) -> bool:
        """Export comprehensive benchmark report"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_benchmarks': len(self.benchmark_history),
                'recent_results': self.get_latest_results(20)
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Benchmark report exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return False


def main():
    """Example usage of automated benchmarking"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    benchmark = AutomatedBenchmark()
    
    # Run a simple benchmark
    print("Running 10-second benchmark test...")
    result = benchmark.run_benchmark("Test Benchmark", duration_seconds=10)
    
    print("\nBenchmark Results:")
    print(f"  Average FPS: {result.avg_fps:.1f}")
    print(f"  1% Low FPS: {result.fps_1_percent_low:.1f}")
    print(f"  0.1% Low FPS: {result.fps_0_1_percent_low:.1f}")
    print(f"  Avg Frame Time: {result.avg_frame_time_ms:.2f}ms")
    print(f"  P95 Frame Time: {result.p95_frame_time_ms:.2f}ms")
    print(f"  P99 Frame Time: {result.p99_frame_time_ms:.2f}ms")
    print(f"  Stability Score: {result.stability_score:.1f}/100")
    print(f"  Stutter Count: {result.stutter_count}")


if __name__ == "__main__":
    main()
