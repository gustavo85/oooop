"""
Utilities V4.0 - Common utilities, error handling, and retry logic
Provides reusable patterns for robust operation
"""

import logging
import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def retry_on_exception(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator to retry a function on exception with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay (exponential backoff)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function(exception, attempt_number)
        
    Example:
        @retry_on_exception(max_attempts=3, delay=1.0)
        def unstable_operation():
            # Code that might fail transiently
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logger.warning(
                                f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                        
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            # If we get here, all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


@contextmanager
def error_handler(
    operation_name: str,
    reraise: bool = False,
    log_level: int = logging.ERROR
):
    """
    Context manager for consistent error handling and logging.
    
    Args:
        operation_name: Name of the operation for logging
        reraise: Whether to reraise the exception after logging
        log_level: Logging level for errors
        
    Example:
        with error_handler("GPU initialization", reraise=False):
            # Code that might fail
            initialize_gpu()
            # Continue execution even if exception occurs
    
    Note: This does not return a value. To handle return values, use try-except directly.
    """
    try:
        yield
    except Exception as e:
        logger.log(
            log_level,
            f"Error in {operation_name}: {type(e).__name__}: {e}",
            exc_info=True
        )
        
        if reraise:
            raise


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent repeated failures.
    
    States:
    - CLOSED: Normal operation, failures counted
    - OPEN: Too many failures, operation blocked
    - HALF_OPEN: Testing if service recovered
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        
        @breaker.call
        def unreliable_operation():
            # Operation that might fail repeatedly
            pass
    """
    
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = self.CLOSED
    
    def call(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == self.OPEN:
                if self.last_failure_time and time.time() - self.last_failure_time >= self.timeout:
                    logger.info(f"Circuit breaker for {func.__name__} entering HALF_OPEN state")
                    self.state = self.HALF_OPEN
                else:
                    time_remaining = self.timeout - (time.time() - self.last_failure_time) if self.last_failure_time else self.timeout
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker OPEN for {func.__name__}. "
                        f"Try again in {time_remaining:.0f}s"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == self.HALF_OPEN:
            logger.info("Circuit breaker recovered, returning to CLOSED state")
        
        self.failure_count = 0
        self.state = self.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker threshold reached ({self.failure_count} failures), "
                f"opening circuit for {self.timeout}s"
            )
            self.state = self.OPEN
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.CLOSED
        logger.info("Circuit breaker manually reset")


class ResourceManager:
    """
    Context manager for consistent resource cleanup.
    
    Example:
        class GPUController(ResourceManager):
            def _acquire(self):
                self.gpu = initialize_gpu()
            
            def _release(self):
                cleanup_gpu(self.gpu)
        
        with GPUController() as gpu_ctrl:
            gpu_ctrl.gpu.do_something()
    """
    
    def __init__(self):
        self._acquired = False
    
    def _acquire(self):
        """Override to implement resource acquisition"""
        pass
    
    def _release(self):
        """Override to implement resource release"""
        pass
    
    def __enter__(self):
        """Acquire resource"""
        try:
            self._acquire()
            self._acquired = True
            return self
        except Exception as e:
            logger.error(f"Failed to acquire resource: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resource"""
        if self._acquired:
            try:
                self._release()
            except Exception as e:
                logger.error(f"Error releasing resource: {e}", exc_info=True)
            finally:
                self._acquired = False
        
        # Return False to propagate exceptions
        return False


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return on division by zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between min and max.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(max_value, value))


def format_bytes(bytes_value: int) -> str:
    """
    Format byte count as human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds as human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


class PerformanceTimer:
    """
    Simple performance timer for profiling code sections.
    
    Example:
        with PerformanceTimer("Expensive operation") as timer:
            # ... code to time ...
            pass
        # Automatically logs elapsed time
    """
    
    def __init__(self, name: str, log_level: int = logging.DEBUG):
        """
        Initialize performance timer.
        
        Args:
            name: Name of the operation being timed
            log_level: Logging level for results
        """
        self.name = name
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        """Start timer"""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log result"""
        self.elapsed = time.perf_counter() - self.start_time
        logger.log(
            self.log_level,
            f"{self.name} completed in {self.elapsed*1000:.2f}ms"
        )
        return False


def validate_range(
    value: Any,
    min_value: Any,
    max_value: Any,
    name: str = "value"
) -> None:
    """
    Validate that a value is within a range, raising ValueError if not.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the value for error message
        
    Raises:
        ValueError: If value is outside range
    """
    if not (min_value <= value <= max_value):
        raise ValueError(
            f"{name} must be between {min_value} and {max_value}, got {value}"
        )


def validate_type(value: Any, expected_type: Type, name: str = "value") -> None:
    """
    Validate that a value is of expected type, raising TypeError if not.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Name of the value for error message
        
    Raises:
        TypeError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


# Example usage demonstrations
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Example 1: Retry decorator
    @retry_on_exception(max_attempts=3, delay=0.5)
    def unreliable_function():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success!"
    
    print("\n=== Example 1: Retry Decorator ===")
    try:
        result = unreliable_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Example 2: Error handler
    print("\n=== Example 2: Error Handler ===")
    with error_handler("test operation", reraise=False):
        print("This will succeed")
    print("Operation completed")
    
    # Example 3: Circuit breaker
    print("\n=== Example 3: Circuit Breaker ===")
    breaker = CircuitBreaker(failure_threshold=3, timeout=5)
    
    @breaker.call
    def failing_operation():
        raise Exception("This always fails")
    
    for i in range(5):
        try:
            failing_operation()
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
    
    # Example 4: Performance timer
    print("\n=== Example 4: Performance Timer ===")
    with PerformanceTimer("Test operation"):
        time.sleep(0.1)
    
    # Example 5: Utility functions
    print("\n=== Example 5: Utility Functions ===")
    print(f"Format bytes: {format_bytes(1536000000)}")
    print(f"Format duration: {format_duration(3665)}")
    print(f"Safe divide: {safe_divide(10, 0, default=-1)}")
    print(f"Clamp value: {clamp(150, 0, 100)}")
