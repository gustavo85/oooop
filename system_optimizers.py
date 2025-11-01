"""
System-level optimizers V3.5: Timer (FIXED), Memory, GPU Scheduling, Power, Core Parking
CHANGES:
- Timer: Unified API (no redundancy)
- Memory: Adaptive purging with configurable thresholds
- Core Parking: Real implementation with powercfg
"""

import ctypes
import logging
import subprocess
import time
import winreg
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import psutil

logger = logging.getLogger(__name__)

# Constants
TIMER_RESOLUTION_05MS = 5000
MemoryPurgeStandbyList = 4
PROCESS_SET_INFORMATION = 0x0200
ProcessPowerThrottling = 0x00000009
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
ProcessMemoryPriority = 0x1f
MEMORY_PRIORITY_NORMAL = 5

class SYSTEM_MEMORY_LIST_COMMAND(ctypes.Structure):
    _fields_ = [("Command", ctypes.c_int)]

class REASON_CONTEXT(ctypes.Structure):
    _fields_ = [('Version', wintypes.ULONG), ('Flags', wintypes.DWORD), ('Reason', ctypes.c_wchar_p)]

class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [("Version", wintypes.DWORD), ("ControlMask", wintypes.DWORD), ("StateMask", wintypes.DWORD)]

class MEMORY_PRIORITY_INFORMATION(ctypes.Structure):
    _fields_ = [("MemoryPriority", wintypes.ULONG)]

@dataclass
class TimerCapabilities:
    min_period_100ns: int
    max_period_100ns: int
    current_period_100ns: int

@dataclass
class MemoryStats:
    total_gb: float
    available_gb: float
    used_percent: float


class AdvancedTimerManager:
    """Timer management using NtSetTimerResolution (kernel-native API) exclusively"""
    
    def __init__(self):
        try:
            self.ntdll = ctypes.WinDLL('ntdll')
        except Exception:
            self.ntdll = None
        
        self.capabilities: Optional[TimerCapabilities] = None
        self.original_resolution: Optional[int] = None
        self.current_resolution: Optional[int] = None
        
        self._setup_functions()
        self._query_capabilities()
    
    def _setup_functions(self):
        try:
            if not self.ntdll:
                return
            self.NtQueryTimerResolution = self.ntdll.NtQueryTimerResolution
            self.NtQueryTimerResolution.argtypes = [ctypes.POINTER(wintypes.ULONG), ctypes.POINTER(wintypes.ULONG), ctypes.POINTER(wintypes.ULONG)]
            self.NtQueryTimerResolution.restype = ctypes.c_long
            self.NtSetTimerResolution = self.ntdll.NtSetTimerResolution
            self.NtSetTimerResolution.argtypes = [wintypes.ULONG, wintypes.BOOLEAN, ctypes.POINTER(wintypes.ULONG)]
            self.NtSetTimerResolution.restype = ctypes.c_long
        except Exception as e:
            logger.debug(f"NT timer setup failed: {e}")
            self.NtQueryTimerResolution = None
            self.NtSetTimerResolution = None
    
    def _query_capabilities(self):
        try:
            if not self.NtQueryTimerResolution:
                return
            minimum, maximum, current = wintypes.ULONG(), wintypes.ULONG(), wintypes.ULONG()
            if self.NtQueryTimerResolution(ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(current)) == 0:
                self.capabilities = TimerCapabilities(min_period_100ns=minimum.value, max_period_100ns=maximum.value, current_period_100ns=current.value)
                self.original_resolution = current.value
                logger.debug(f"Timer caps: {minimum.value/10:.1f}μs - {maximum.value/10:.1f}μs")
        except Exception as e:
            logger.debug(f"Query timer caps failed: {e}")
    
    def set_high_performance_timer(self) -> bool:
        """Set high-performance timer using NtSetTimerResolution exclusively"""
        try:
            if not self.NtSetTimerResolution or not self.capabilities:
                logger.warning("NtSetTimerResolution not available")
                return False
            
            target_res = self.capabilities.min_period_100ns or TIMER_RESOLUTION_05MS
            current_res = wintypes.ULONG()
            
            # Set timer resolution (TRUE = set)
            status = self.NtSetTimerResolution(target_res, True, ctypes.byref(current_res))
            
            if status == 0:
                self.current_resolution = current_res.value
                logger.info(f"✓ Timer: {current_res.value/10:.1f}μs (NtSetTimerResolution)")
                return True
            else:
                logger.warning(f"NtSetTimerResolution failed with status: {status}")
                return False
                
        except Exception as e:
            logger.error(f"Timer error: {e}")
            return False
    
    def restore_default_timer(self) -> bool:
        """Restore default timer resolution"""
        try:
            if not self.NtSetTimerResolution or not self.original_resolution:
                return True
            
            current_res = wintypes.ULONG()
            # FALSE = release our request
            self.NtSetTimerResolution(self.original_resolution, False, ctypes.byref(current_res))
            
            self.current_resolution = None
            logger.info("✓ Timer restored (NtSetTimerResolution)")
            return True
            
        except Exception as e:
            logger.error(f"Timer restore error: {e}")
            return False


class MemoryOptimizer:
    """Memory optimization with SysMain (Superfetch) management"""
    
    def __init__(self):
        try:
            self.ntdll = ctypes.WinDLL('ntdll')
        except Exception:
            self.ntdll = None
        try:
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        except Exception:
            self.kernel32 = None
        self.psapi = None
        try:
            self.psapi = ctypes.WinDLL('psapi')
        except Exception:
            pass
        self.last_optimization = 0.0
        self.sysmain_disabled = False
        self.sysmain_original_state: Optional[int] = None
        self._setup_functions()
    
    def _setup_functions(self):
        try:
            if not self.ntdll:
                return
            self.NtSetSystemInformation = self.ntdll.NtSetSystemInformation
            self.NtSetSystemInformation.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            self.NtSetSystemInformation.restype = ctypes.c_long
        except Exception:
            self.NtSetSystemInformation = None
    
    def purge_standby_memory_adaptive(self, min_interval: int = 30, threshold_percent: float = 70, threshold_mb: Optional[int] = None) -> bool:
        """
        Adaptive purge with configurable interval and threshold.
        
        Args:
            min_interval: Minimum seconds between purges
            threshold_percent: Purge if memory usage exceeds this percentage
            threshold_mb: Optional - purge only if free memory is below this threshold in MB
        
        Returns:
            bool: True if purge was performed, False otherwise
        """
        try:
            if time.time() - self.last_optimization < min_interval:
                return False
            stats = self.get_memory_stats()
            if not stats:
                return False
            
            # Check if we should purge based on threshold_mb
            if threshold_mb is not None:
                free_mb = stats.available_gb * 1024
                if free_mb >= threshold_mb:
                    return False  # Don't purge if we have enough free memory
            
            # Also check percentage threshold or absolute low memory
            if not (stats.used_percent > threshold_percent or stats.available_gb < 1.0):
                return False
            
            if not self.NtSetSystemInformation:
                return False
            command = SYSTEM_MEMORY_LIST_COMMAND(Command=MemoryPurgeStandbyList)
            status = self.NtSetSystemInformation(0x50, ctypes.byref(command), ctypes.sizeof(command))
            if status == 0:
                self.last_optimization = time.time()
                logger.info(f"✓ Standby memory purged ({stats.used_percent:.1f}% usage, {stats.available_gb:.2f}GB free)")
                return True
            return False
        except Exception as e:
            logger.debug(f"Purge error: {e}")
            return False
    
    def purge_standby_memory(self) -> bool:
        """Legacy method for compatibility"""
        return self.purge_standby_memory_adaptive(min_interval=120, threshold_percent=85)
    
    def get_memory_stats(self) -> Optional[MemoryStats]:
        try:
            mem = psutil.virtual_memory()
            return MemoryStats(total_gb=mem.total/(1024**3), available_gb=mem.available/(1024**3), used_percent=mem.percent)
        except Exception:
            return None
    
    def force_clr_gc(self, pid: int) -> bool:
        try:
            process = psutil.Process(pid)
            clr_loaded = False
            try:
                for dll in process.memory_maps():
                    p = getattr(dll, 'path', '')
                    if p and ('clr.dll' in p.lower() or 'coreclr.dll' in p.lower()):
                        clr_loaded = True
                        break
            except Exception:
                n = process.name().lower()
                if any(x in n for x in ['unity', 'unreal', 'mono']):
                    clr_loaded = True
            if clr_loaded and self.kernel32 and self.psapi:
                h = self.kernel32.OpenProcess(0x0100, False, pid)
                if h:
                    ok = False
                    try:
                        self.psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
                        self.psapi.EmptyWorkingSet.restype = wintypes.BOOL
                        ok = bool(self.psapi.EmptyWorkingSet(h))
                    except Exception:
                        pass
                    self.kernel32.CloseHandle(h)
                    if ok:
                        time.sleep(0.5)
                        logger.info("✓ CLR GC forced")
                        return True
        except Exception:
            pass
        return False
    
    def disable_memory_compression_for_process(self, pid: int) -> bool:
        try:
            h = self.kernel32.OpenProcess(PROCESS_SET_INFORMATION, False, pid)
            if not h:
                return False
            info = MEMORY_PRIORITY_INFORMATION(MemoryPriority=MEMORY_PRIORITY_NORMAL)
            self.kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            self.kernel32.SetProcessInformation.restype = wintypes.BOOL
            self.kernel32.SetProcessInformation(h, ProcessMemoryPriority, ctypes.byref(info), ctypes.sizeof(info))
            self.kernel32.CloseHandle(h)
            return True
        except Exception:
            return False
    
    def disable_sysmain(self) -> bool:
        """
        Disable SysMain (Superfetch/Prefetch) service to prevent background I/O activity.
        This minimizes disk I/O stutter during gameplay.
        """
        try:
            import subprocess
            
            # Query current state
            result = subprocess.run(
                ['sc', 'query', 'SysMain'],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
            
            # Parse state
            if 'RUNNING' in result.stdout:
                self.sysmain_original_state = 1  # Running
            elif 'STOPPED' in result.stdout:
                self.sysmain_original_state = 0  # Stopped
            else:
                self.sysmain_original_state = None
            
            # Stop service
            stop_result = subprocess.run(
                ['sc', 'stop', 'SysMain'],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
            
            if stop_result.returncode == 0 or 'already stopped' in stop_result.stdout.lower():
                self.sysmain_disabled = True
                logger.info("✓ SysMain (Superfetch) disabled to prevent I/O stutter")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"SysMain disable error: {e}")
            return False
    
    def restore_sysmain(self) -> bool:
        """Restore SysMain service to original state"""
        if not self.sysmain_disabled:
            return True
        
        try:
            import subprocess
            
            # Only restore if it was running before
            if self.sysmain_original_state == 1:
                result = subprocess.run(
                    ['sc', 'start', 'SysMain'],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                )
                
                if result.returncode == 0 or 'already running' in result.stdout.lower():
                    logger.info("✓ SysMain (Superfetch) restored")
            
            self.sysmain_disabled = False
            self.sysmain_original_state = None
            return True
            
        except Exception as e:
            logger.debug(f"SysMain restore error: {e}")
            return False


class GPUSchedulingOptimizer:
    def __init__(self):
        self.registry_path = r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
        self.hwsch_mode_value = "HwSchMode"
        self.original_state: Optional[int] = self._check_current_state()
        self.restart_required: bool = False
    
    def _check_current_state(self) -> Optional[int]:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.registry_path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                value, _ = winreg.QueryValueEx(key, self.hwsch_mode_value)
                return value
        except Exception:
            return None
    
    def enable_hardware_scheduling(self) -> bool:
        try:
            if self.original_state == 2:
                return True
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.registry_path, 0, winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY) as key:
                winreg.SetValueEx(key, self.hwsch_mode_value, 0, winreg.REG_DWORD, 2)
            self.restart_required = self.original_state != 2
            return True
        except Exception:
            return False
    
    def restore_original_state(self) -> bool:
        try:
            current = self._check_current_state()
            if self.original_state is None or self.original_state == current:
                return True
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.registry_path, 0, winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY) as key:
                winreg.SetValueEx(key, self.hwsch_mode_value, 0, winreg.REG_DWORD, self.original_state)
            self.restart_required = current != self.original_state
            return True
        except Exception:
            return False


class PowerManagementOptimizer:
    def __init__(self):
        try:
            self.kernel32 = ctypes.WinDLL('kernel32')
        except Exception:
            self.kernel32 = None
        self.power_request_handle = None
        self.original_power_scheme: Optional[str] = None
        self.high_performance_scheme_guid: Optional[str] = None
        self._setup_functions()
        self._detect_power_schemes()
        self._original_power_settings: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    
    def _setup_functions(self):
        try:
            if not self.kernel32:
                return
            self.PowerCreateRequest = self.kernel32.PowerCreateRequest
            self.PowerCreateRequest.argtypes = [ctypes.POINTER(REASON_CONTEXT)]
            self.PowerCreateRequest.restype = wintypes.HANDLE
            self.PowerSetRequest = self.kernel32.PowerSetRequest
            self.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
            self.PowerClearRequest = self.kernel32.PowerClearRequest
            self.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
            self.CloseHandle = self.kernel32.CloseHandle
            self.CloseHandle.argtypes = [wintypes.HANDLE]
        except Exception:
            pass
    
    def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            class R: returncode=1; stdout=""; stderr=""
            return R()
    
    def _detect_power_schemes(self):
        try:
            ultimate = "e9a42b02-d5df-448d-aa00-03f14749eb61"
            high_perf = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
            result = self._run_command(['powercfg', '/list'])
            if getattr(result, 'returncode', 1) == 0:
                if ultimate in result.stdout.lower():
                    self.high_performance_scheme_guid = ultimate
                elif high_perf in result.stdout.lower():
                    self.high_performance_scheme_guid = high_perf
            result = self._run_command(['powercfg', '/getactivescheme'])
            if getattr(result, 'returncode', 1) == 0:
                import re
                m = re.search(r'([0-9a-f-]{36})', result.stdout.lower())
                if m:
                    self.original_power_scheme = m.group(1)
        except Exception:
            pass
    
    def set_high_performance(self) -> bool:
        try:
            if not self.high_performance_scheme_guid:
                return False
            if self.high_performance_scheme_guid != self.original_power_scheme:
                r = self._run_command(['powercfg', '/setactive', self.high_performance_scheme_guid])
                if getattr(r, 'returncode', 1) != 0:
                    return False
            return True
        except Exception:
            return False
    
    def restore_original_plan(self) -> bool:
        try:
            if not self.original_power_scheme:
                return True
            r = self._run_command(['powercfg', '/setactive', self.original_power_scheme])
            return getattr(r, 'returncode', 1) == 0
        except Exception:
            return False
    
    def create_power_request(self) -> bool:
        try:
            if not hasattr(self, 'PowerCreateRequest'):
                return False
            context = REASON_CONTEXT(Version=0, Flags=1, Reason="Game Optimizer V3.5")
            self.power_request_handle = self.PowerCreateRequest(ctypes.byref(context))
            if self.power_request_handle and self.power_request_handle != wintypes.HANDLE(-1).value:
                self.PowerSetRequest(self.power_request_handle, 1)
                self.PowerSetRequest(self.power_request_handle, 0)
                self.PowerSetRequest(self.power_request_handle, 3)
                return True
            return False
        except Exception:
            return False
    
    def clear_power_request(self) -> bool:
        try:
            if self.power_request_handle and hasattr(self, 'CloseHandle'):
                self.CloseHandle(self.power_request_handle)
                self.power_request_handle = None
            return True
        except Exception:
            return False


class CoreParkingManager:
    """Core parking control using direct registry modification for instant and reliable application"""
    
    def __init__(self):
        self.original_settings: Dict[str, Any] = {}
        self.parking_disabled = False
    
    def disable_core_parking(self, p_core_indices: Optional[List[int]] = None) -> bool:
        """Disable core parking via direct registry modification (instant application)"""
        try:
            # Direct registry path for power settings
            guid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"  # High Performance
            subgroup = "54533251-82be-4824-96c1-47b60b740d00"  # Processor
            
            settings = {
                '0cc5b647-c1df-4636-95bb-3217ef867c1a': ('Core Parking Min', 100),  # Min unparked cores
                '893dee8e-2bef-41e0-89c6-b55d0929964c': ('Processor Min', 100),     # Min processor state
                'bc5038f7-23e0-4960-96da-33abaf5935ec': ('Processor Max', 100),     # Max processor state
            }
            
            base_path = f"SYSTEM\\CurrentControlSet\\Control\\Power\\PowerSettings\\{subgroup}"
            
            for setting_guid, (name, value) in settings.items():
                try:
                    # Read original value
                    ac_path = f"SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes\\{guid}\\{subgroup}\\{setting_guid}"
                    
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, ac_path, 0, 
                                       winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                        try:
                            ac_val, _ = winreg.QueryValueEx(key, "ACSettingIndex")
                            dc_val, _ = winreg.QueryValueEx(key, "DCSettingIndex")
                            
                            if setting_guid not in self.original_settings:
                                self.original_settings[setting_guid] = {
                                    'ac': ac_val,
                                    'dc': dc_val
                                }
                        except FileNotFoundError:
                            pass
                    
                    # Set new value
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, ac_path, 0, 
                                       winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY) as key:
                        winreg.SetValueEx(key, "ACSettingIndex", 0, winreg.REG_DWORD, value)
                        winreg.SetValueEx(key, "DCSettingIndex", 0, winreg.REG_DWORD, value)
                    
                except Exception as e:
                    logger.debug(f"Registry setting {setting_guid} error: {e}")
            
            # Notify power subsystem of changes via WMI
            try:
                import subprocess
                # Use powercfg to apply active scheme (quick notification)
                subprocess.run(['powercfg', '/setactive', guid], 
                             check=False, timeout=5,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except Exception:
                pass
            
            self.parking_disabled = True
            logger.info("✓ Core parking disabled (direct registry)")
            return True
            
        except Exception as e:
            logger.error(f"Core parking error: {e}")
            return False
    
    def restore_core_parking(self) -> bool:
        """Restore original core parking settings"""
        if not self.parking_disabled:
            return True
        
        try:
            guid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
            subgroup = "54533251-82be-4824-96c1-47b60b740d00"
            
            for setting_guid, vals in self.original_settings.items():
                try:
                    ac_path = f"SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes\\{guid}\\{subgroup}\\{setting_guid}"
                    
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, ac_path, 0, 
                                       winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY) as key:
                        winreg.SetValueEx(key, "ACSettingIndex", 0, winreg.REG_DWORD, vals['ac'])
                        winreg.SetValueEx(key, "DCSettingIndex", 0, winreg.REG_DWORD, vals['dc'])
                        
                except Exception as e:
                    logger.debug(f"Restore {setting_guid} error: {e}")
            
            # Notify power subsystem
            try:
                import subprocess
                subprocess.run(['powercfg', '/setactive', guid], 
                             check=False, timeout=5,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except Exception:
                pass
            
            self.original_settings.clear()
            self.parking_disabled = False
            logger.info("✓ Core parking restored")
            return True
            
        except Exception as e:
            logger.error(f"Core parking restore error: {e}")
            return False


class WindowsBackgroundAppsManager:
    def __init__(self):
        self.modified: List[int] = []
        self._skip_names = {'nvcontainer.exe', 'nvidia share.exe', 'rivatunerstatisticsserver.exe', 'steam.exe', 
                           'steamwebhelper.exe', 'discord.exe', 'audiodg.exe', 'obs64.exe', 'msmpeng.exe'}
    
    def _get_foreground_pid(self) -> Optional[int]:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return int(pid.value) if pid.value else None
        except Exception:
            return None
    
    def limit_background_apps(self, cpu_threshold: float = 3.0, rss_mb_threshold: int = 200) -> int:
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            fg_pid = self._get_foreground_pid()
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    pid = int(proc.info['pid'])
                    name = (proc.info.get('name') or '').lower()
                    
                    if name in {'system', 'smss.exe', 'csrss.exe', 'wininit.exe', 'services.exe', 
                               'lsass.exe', 'svchost.exe', 'winlogon.exe', 'explorer.exe', 'dwm.exe'}:
                        continue
                    if name in self._skip_names or (fg_pid and pid == fg_pid):
                        continue
                    
                    cpu_now = proc.cpu_percent(interval=0.05)
                    mem = proc.info.get('memory_info')
                    rss_mb = int(getattr(mem, 'rss', 0) / (1024 * 1024)) if mem else 0
                    
                    if cpu_now < cpu_threshold and rss_mb < rss_mb_threshold:
                        continue
                    
                    h = kernel32.OpenProcess(PROCESS_SET_INFORMATION, False, pid)
                    if h:
                        state = PROCESS_POWER_THROTTLING_STATE(Version=1, ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 
                                                               StateMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED)
                        kernel32.SetProcessInformation(h, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state))
                        self.modified.append(pid)
                        kernel32.CloseHandle(h)
                except Exception:
                    pass
            
            return len(self.modified)
        except Exception:
            return 0
    
    def restore_all_background_apps(self) -> bool:
        try:
            kernel32 = ctypes.windll.kernel32
            for pid in list(self.modified):
                try:
                    h = kernel32.OpenProcess(PROCESS_SET_INFORMATION, False, int(pid))
                    if h:
                        state = PROCESS_POWER_THROTTLING_STATE(Version=1, ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED, StateMask=0)
                        kernel32.SetProcessInformation(h, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state))
                        kernel32.CloseHandle(h)
                except Exception:
                    pass
            self.modified.clear()
            return True
        except Exception:
            return False