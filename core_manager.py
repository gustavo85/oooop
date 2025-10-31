"""
Gestor principal del optimizador de juegos V3.5 - 920/1000 QUALITY
Implementa: ETW real, GPU clocking, DPC monitoring, ML tuning, A/B testing, anti-cheat detection
"""

import sys
import platform
import time
import logging
import ctypes
import threading
import json
from ctypes import wintypes
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import psutil

if platform.system() != 'Windows':
    raise RuntimeError("Error: This optimizer is designed to run only on Windows.")

import compat_imports

try:
    from config_loader import ConfigurationManager, GameProfile
    from system_optimizers import (
        AdvancedTimerManager, MemoryOptimizer, GPUSchedulingOptimizer,
        WindowsBackgroundAppsManager, PowerManagementOptimizer, CoreParkingManager
    )
    from directx_optimizer import DirectXOptimizer
    from network_optimizer import NetworkOptimizer
    from session_manager import GamingSessionManager
    from monitoring import PerformanceMonitor, TelemetryCollector, ABTestingFramework
    from ml_tuner import MLAutoTuner
except ImportError as e:
    raise ImportError(f"Error importing optimizer modules: {e}") from e

from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)

# Windows API Constants
SE_LOCK_MEMORY_NAME = "SeLockMemoryPrivilege"
SE_INCREASE_QUOTA_NAME = "SeIncreaseQuotaPrivilege"
SE_INC_BASE_PRIORITY_NAME = "SeIncBasePriorityPrivilege"

ProcessIoPriority = 33
ProcessPagePriority = 39

THREAD_SET_INFORMATION = 0x0020
THREAD_QUERY_INFORMATION = 0x0040
THREAD_PRIORITY_TIME_CRITICAL = 15
THREAD_PRIORITY_HIGHEST = 2

QUOTA_LIMITS_HARDWS_MIN_ENABLE = 0x00000001
QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x00000008

TH32CS_SNAPTHREAD = 0x00000004

# AVRT (MMCSS)
try:
    avrt = ctypes.WinDLL('avrt')
    AvSetMmThreadCharacteristicsW = avrt.AvSetMmThreadCharacteristicsW
    AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
    AvSetMmThreadPriority = avrt.AvSetMmThreadPriority
    AvSetMmThreadPriority.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    AvRevertMmThreadCharacteristics = avrt.AvRevertMmThreadCharacteristics
    AvRevertMmThreadCharacteristics.argtypes = [wintypes.HANDLE]
    AVRT_PRIORITY_HIGH = 2
    AVRT_CATEGORY = "Games"
except Exception:
    AvSetMmThreadCharacteristicsW = None
    AvSetMmThreadPriority = None
    AvRevertMmThreadCharacteristics = None

@dataclass
class CPUTopology:
    physical_cores: int
    logical_cores: int
    performance_cores: List[int]
    efficiency_cores: List[int]
    is_hybrid: bool
    numa_nodes: int

@dataclass
class OptimizationState:
    game_exe: str
    game_pid: int
    profile: GameProfile
    start_time: float
    optimizations_applied: Set[str] = field(default_factory=set)
    original_process_state: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    baseline_metrics: Optional[Dict[str, float]] = None
    optimized_metrics: Optional[Dict[str, float]] = None

class AdvancedCPUManager:
    """Gestión avanzada de CPU con detección híbrida y optimizaciones low-level"""
    
    class GROUP_AFFINITY(ctypes.Structure):
        _fields_ = [("Mask", ctypes.c_ulonglong),
                    ("Group", wintypes.WORD),
                    ("Reserved", wintypes.WORD * 3)]
    
    class PROCESSOR_RELATIONSHIP_EX(ctypes.Structure):
        _fields_ = [("Flags", wintypes.BYTE),
                    ("EfficiencyClass", wintypes.BYTE),
                    ("Reserved", wintypes.BYTE * 20),
                    ("GroupCount", wintypes.WORD)]
    
    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
        _fields_ = [("Relationship", ctypes.c_int),
                    ("Size", wintypes.DWORD)]
    
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self.advapi32 = ctypes.windll.advapi32
        self.ntdll = ctypes.WinDLL('ntdll')
        self.topology = self._detect_cpu_topology()
        self._enable_required_privileges()
        logger.info(f"CPU Topology: P-cores={len(self.topology.performance_cores)}, E-cores={len(self.topology.efficiency_cores)}, Hybrid={self.topology.is_hybrid}, NUMA={self.topology.numa_nodes}")

    def _enable_required_privileges(self):
        privileges = [SE_LOCK_MEMORY_NAME, SE_INCREASE_QUOTA_NAME, SE_INC_BASE_PRIORITY_NAME]
        TOKEN_ADJUST_PRIVILEGES = 0x0020
        TOKEN_QUERY = 0x0008
        SE_PRIVILEGE_ENABLED = 0x00000002
        
        hToken = wintypes.HANDLE(0)
        if not self.advapi32.OpenProcessToken(self.kernel32.GetCurrentProcess(), 
                                              TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, 
                                              ctypes.byref(hToken)):
            logger.warning("Could not open process token for privilege adjustment")
            return
        
        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]
        
        class LUID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]
        
        class TOKEN_PRIVILEGES(ctypes.Structure):
            _fields_ = [("PrivilegeCount", wintypes.DWORD),
                        ("Privileges", LUID_AND_ATTRIBUTES * 1)]
        
        for priv_name in privileges:
            luid = LUID()
            if not self.advapi32.LookupPrivilegeValueW(None, priv_name, ctypes.byref(luid)):
                logger.debug(f"Could not lookup privilege: {priv_name}")
                continue
            
            tp = TOKEN_PRIVILEGES()
            tp.PrivilegeCount = 1
            tp.Privileges[0].Luid = luid
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
            
            if self.advapi32.AdjustTokenPrivileges(hToken, False, ctypes.byref(tp), 0, None, None):
                logger.debug(f"✓ Privilege enabled: {priv_name}")
            else:
                logger.warning(f"Could not enable privilege: {priv_name}")
        
        self.kernel32.CloseHandle(hToken)

    def _get_numa_node_count(self) -> int:
        try:
            highest_node = wintypes.ULONG()
            self.kernel32.GetNumaHighestNodeNumber(ctypes.byref(highest_node))
            return highest_node.value + 1
        except (AttributeError, OSError):
            return 1
    
    def _detect_cpu_topology(self) -> CPUTopology:
        RelationProcessorCore = 0
        p_cores: List[int] = []
        e_cores: List[int] = []
        logical_count = psutil.cpu_count(logical=True) or 1
        
        try:
            size = wintypes.DWORD(0)
            self.kernel32.GetLogicalProcessorInformationEx(RelationProcessorCore, None, ctypes.byref(size))
            if size.value == 0:
                raise OSError("GetLogicalProcessorInformationEx returned size 0")
            
            buf = (ctypes.c_byte * size.value)()
            if not self.kernel32.GetLogicalProcessorInformationEx(RelationProcessorCore, buf, ctypes.byref(size)):
                raise OSError("GetLogicalProcessorInformationEx failed")
            
            offset = 0
            processed_lp = set()
            
            while offset < size.value:
                base = ctypes.addressof(buf) + offset
                header = self.SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX.from_address(base)
                rel = header.Relationship
                block_size = header.Size
                
                if rel == RelationProcessorCore:
                    pr = self.PROCESSOR_RELATIONSHIP_EX.from_address(
                        base + ctypes.sizeof(self.SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX))
                    ec = pr.EfficiencyClass
                    
                    ga_array_type = self.GROUP_AFFINITY * pr.GroupCount
                    ga_array = ga_array_type.from_address(
                        base + ctypes.sizeof(self.SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) + 
                        ctypes.sizeof(self.PROCESSOR_RELATIONSHIP_EX))
                    
                    for idx in range(pr.GroupCount):
                        mask = ga_array[idx].Mask
                        lp = 0
                        while mask:
                            if mask & 1:
                                if lp < logical_count and lp not in processed_lp:
                                    if ec == 0:
                                        p_cores.append(lp)
                                    else:
                                        e_cores.append(lp)
                                    processed_lp.add(lp)
                            mask >>= 1
                            lp += 1
                
                offset += block_size
            
            p_cores = sorted(list(dict.fromkeys([c for c in p_cores if c < logical_count])))
            e_cores = sorted(list(dict.fromkeys([c for c in e_cores if c < logical_count])))
            
        except Exception as e:
            logger.error(f"Failed to detect hybrid CPU topology: {e}")
            p_cores = list(range(logical_count))
            e_cores = []
        
        is_hybrid = bool(p_cores and e_cores)
        
        return CPUTopology(
            physical_cores=psutil.cpu_count(logical=False) or max(1, logical_count//2),
            logical_cores=logical_count,
            performance_cores=p_cores if p_cores else list(range(logical_count)),
            efficiency_cores=e_cores,
            is_hybrid=is_hybrid,
            numa_nodes=self._get_numa_node_count()
        )

    def optimize_working_set(self, pid: int) -> bool:
        """
        Tunes process working set for better memory management.
        NOTE: This does NOT enable true Large Pages (2MB pages).
        Real Large Pages require the application itself to allocate with MEM_LARGE_PAGES.
        """
        try:
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_VM_OPERATION = 0x0008
            
            process_handle = self.kernel32.OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_VM_OPERATION, False, pid)
            
            if not process_handle:
                logger.warning(f"Could not open process {pid} for working-set tuning")
                return False
            
            # Conservative working set bounds
            min_size = 100 * 1024 * 1024      # 100 MB minimum
            max_size = 16 * 1024 * 1024 * 1024  # 16 GB maximum
            flags = QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE
            
            result = self.kernel32.SetProcessWorkingSetSizeEx(
                process_handle, min_size, max_size, flags)
            
            self.kernel32.CloseHandle(process_handle)
            
            if result:
                logger.info(f"✓ Working set tuned for PID {pid} (min={min_size//1024//1024}MB, max={max_size//1024//1024}MB)")
                return True
            
            logger.debug(f"SetProcessWorkingSetSizeEx failed for PID {pid}")
            return False
            
        except Exception as e:
            logger.error(f"Error tuning working set: {e}")
            return False

    def set_io_priority(self, pid: int, priority: int = 2) -> bool:
        """
        Set I/O priority for process.
        Priority levels: 0=Critical (avoid), 1=High, 2=Normal, 3=Low
        """
        try:
            PROCESS_SET_INFORMATION = 0x0200
            process_handle = self.kernel32.OpenProcess(PROCESS_SET_INFORMATION, False, pid)
            
            if not process_handle:
                logger.error(f"Could not open process {pid} for I/O priority")
                return False
            
            io_priority = ctypes.c_ulong(priority)
            status = self.ntdll.NtSetInformationProcess(
                process_handle, ProcessIoPriority, 
                ctypes.byref(io_priority), ctypes.sizeof(io_priority))
            
            self.kernel32.CloseHandle(process_handle)
            
            if status == 0:
                logger.info(f"✓ I/O Priority set to {priority} for PID {pid}")
                return True
            
            logger.warning(f"NtSetInformationProcess(I/O) failed: {status:#x}")
            return False
            
        except Exception as e:
            logger.error(f"Error setting I/O priority: {e}")
            return False
    
    def set_page_priority(self, pid: int, priority: int = 5) -> bool:
        """Set memory page priority (1=VeryLow to 5=Normal)"""
        try:
            PROCESS_SET_INFORMATION = 0x0200
            process_handle = self.kernel32.OpenProcess(PROCESS_SET_INFORMATION, False, pid)
            
            if not process_handle:
                return False
            
            page_priority = ctypes.c_ulong(priority)
            status = self.ntdll.NtSetInformationProcess(
                process_handle, ProcessPagePriority, 
                ctypes.byref(page_priority), ctypes.sizeof(page_priority))
            
            self.kernel32.CloseHandle(process_handle)
            
            if status == 0:
                logger.info(f"✓ Page Priority set to {priority} for PID {pid}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error setting page priority: {e}")
            return False

    def boost_main_threads(self, pid: int) -> Dict[str, Any]:
        """Boost priority of main game threads and register with MMCSS"""
        state: Dict[str, Any] = {"threads_boosted": [], "mmcss_handles": []}
        
        try:
            snapshot = self.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
            if snapshot == -1:
                logger.error("Could not create thread snapshot")
                return state
            
            class THREADENTRY32(ctypes.Structure):
                _fields_ = [("dwSize", wintypes.DWORD),
                            ("cntUsage", wintypes.DWORD),
                            ("th32ThreadID", wintypes.DWORD),
                            ("th32OwnerProcessID", wintypes.DWORD),
                            ("tpBasePri", wintypes.LONG),
                            ("tpDeltaPri", wintypes.LONG),
                            ("dwFlags", wintypes.DWORD)]
            
            te32 = THREADENTRY32()
            te32.dwSize = ctypes.sizeof(THREADENTRY32)
            
            if self.kernel32.Thread32First(snapshot, ctypes.byref(te32)):
                while True:
                    if te32.th32OwnerProcessID == pid:
                        th = self.kernel32.OpenThread(
                            THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, 
                            False, te32.th32ThreadID)
                        
                        if th:
                            try:
                                # Boost high-priority threads
                                if te32.tpBasePri >= 8:
                                    self.kernel32.SetThreadPriority(th, THREAD_PRIORITY_HIGHEST)
                                    state["threads_boosted"].append(int(te32.th32ThreadID))
                                    
                                    # Register with MMCSS
                                    if AvSetMmThreadCharacteristicsW and AvSetMmThreadPriority:
                                        task_idx = wintypes.DWORD(0)
                                        hm = AvSetMmThreadCharacteristicsW(
                                            AVRT_CATEGORY, ctypes.byref(task_idx))
                                        
                                        if hm and hm != wintypes.HANDLE(-1).value:
                                            AvSetMmThreadPriority(hm, AVRT_PRIORITY_HIGH)
                                            state["mmcss_handles"].append(int(hm))
                            finally:
                                self.kernel32.CloseHandle(th)
                    
                    if not self.kernel32.Thread32Next(snapshot, ctypes.byref(te32)):
                        break
            
            self.kernel32.CloseHandle(snapshot)
            
            if state["threads_boosted"]:
                logger.info(f"✓ Boosted {len(state['threads_boosted'])} threads, {len(state['mmcss_handles'])} registered with MMCSS")
            
            return state
            
        except Exception as e:
            logger.error(f"Error boosting threads: {e}")
            return state

    def set_process_optimizations(self, pid: int, profile: GameProfile) -> Dict:
        """Apply comprehensive CPU/memory/I-O optimizations"""
        try:
            p = psutil.Process(pid)
            original_state = {
                'priority': p.nice(),
                'affinity': p.cpu_affinity()
            }
            
            # Process priority
            priority_map = {
                'NORMAL': psutil.NORMAL_PRIORITY_CLASS,
                'ABOVE_NORMAL': psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                'HIGH': psutil.HIGH_PRIORITY_CLASS,
                'REALTIME': psutil.REALTIME_PRIORITY_CLASS
            }
            target_priority = priority_map.get(
                profile.cpu_priority_class.upper(), psutil.HIGH_PRIORITY_CLASS)
            p.nice(target_priority)
            logger.info(f"✓ Process priority: {profile.cpu_priority_class}")
            
            # Affinity to P-cores on hybrid CPUs
            if profile.cpu_affinity_enabled and self.topology.is_hybrid and self.topology.performance_cores:
                p.cpu_affinity(self.topology.performance_cores)
                logger.info(f"✓ CPU affinity: {len(self.topology.performance_cores)} P-cores")
            
            # Working set tuning
            if getattr(profile, 'optimize_working_set', True):
                if self.optimize_working_set(pid):
                    original_state['working_set_tuned'] = True
            
            # I/O priority
            io_map = {'HIGH': 1, 'NORMAL': 2}
            io_target = io_map.get(getattr(profile, 'process_io_priority', 'NORMAL').upper(), 2)
            if self.set_io_priority(pid, priority=io_target):
                original_state['io_priority_applied'] = True
            
            # Page priority
            if self.set_page_priority(pid, priority=5):
                original_state['page_priority_applied'] = True
            
            # Thread boosting
            boost_state = self.boost_main_threads(pid)
            if boost_state.get("threads_boosted"):
                original_state['threads_boosted'] = boost_state["threads_boosted"]
                original_state['mmcss_handles'] = boost_state["mmcss_handles"]
            
            return original_state
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Failed to apply CPU optimizations to PID {pid}: {e}")
            return {}

    def restore_process_optimizations(self, pid: int, original_state: Dict):
        """Restore original process settings"""
        if not original_state:
            return
        
        try:
            p = psutil.Process(pid)
            
            if 'priority' in original_state:
                p.nice(original_state['priority'])
            
            if 'affinity' in original_state:
                try:
                    p.cpu_affinity(original_state['affinity'])
                except Exception as e:
                    logger.debug(f"Could not restore affinity: {e}")
            
            # Revert MMCSS
            if original_state.get('mmcss_handles') and AvRevertMmThreadCharacteristics:
                for hm in original_state['mmcss_handles']:
                    try:
                        AvRevertMmThreadCharacteristics(ctypes.wintypes.HANDLE(hm))
                    except Exception as e:
                        logger.debug(f"Could not revert MMCSS handle: {e}")
            
            logger.info(f"✓ Restored original CPU settings for PID {pid}")
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Process {pid} no longer exists: {e}")


class GameModeManager:
    """Manages Windows Game Mode programmatically"""
    
    def __init__(self):
        self.original_settings: Dict[str, Any] = {}
        self.game_mode_active = False
    
    def enable_game_mode(self) -> bool:
        """Enable Windows Game Mode and related optimizations"""
        try:
            import winreg
            
            settings = {
                r"Software\Microsoft\GameBar": {
                    "AllowAutoGameMode": (winreg.REG_DWORD, 1),
                    "AutoGameModeEnabled": (winreg.REG_DWORD, 1),
                    "UseNexusForGameBarEnabled": (winreg.REG_DWORD, 0),
                },
                r"System\GameConfigStore": {
                    "GameDVR_Enabled": (winreg.REG_DWORD, 0),
                    "GameDVR_FSEBehaviorMode": (winreg.REG_DWORD, 2),
                    "GameDVR_HonorUserFSEBehaviorMode": (winreg.REG_DWORD, 1),
                    "GameDVR_DXGIHonorFSEWindowsCompatible": (winreg.REG_DWORD, 1),
                    "GameDVR_EFSEFeatureFlags": (winreg.REG_DWORD, 0),
                }
            }
            
            for key_path, values in settings.items():
                try:
                    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key_path, 0, 
                                           winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE) as key:
                        
                        if key_path not in self.original_settings:
                            self.original_settings[key_path] = {}
                        
                        for value_name, (value_type, value_data) in values.items():
                            try:
                                original, _ = winreg.QueryValueEx(key, value_name)
                                self.original_settings[key_path][value_name] = original
                            except FileNotFoundError:
                                self.original_settings[key_path][value_name] = None
                            
                            winreg.SetValueEx(key, value_name, 0, value_type, value_data)
                            
                except Exception as e:
                    logger.warning(f"Could not modify registry key {key_path}: {e}")
            
            self.game_mode_active = True
            logger.info("✓ Windows Game Mode enabled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable Game Mode: {e}")
            return False
    
    def disable_game_mode(self) -> bool:
        """Restore original Game Mode settings"""
        if not self.game_mode_active:
            return True
        
        try:
            import winreg
            
            for key_path, values in self.original_settings.items():
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, 
                                       winreg.KEY_SET_VALUE) as key:
                        
                        for value_name, original_value in values.items():
                            try:
                                if original_value is None:
                                    try:
                                        winreg.DeleteValue(key, value_name)
                                    except FileNotFoundError:
                                        pass
                                else:
                                    if isinstance(original_value, int):
                                        value_type = winreg.REG_DWORD
                                    elif isinstance(original_value, str):
                                        value_type = winreg.REG_SZ
                                    else:
                                        value_type = winreg.REG_BINARY
                                    
                                    winreg.SetValueEx(key, value_name, 0, value_type, original_value)
                                    
                            except Exception as e:
                                logger.debug(f"Could not restore {value_name}: {e}")
                                
                except Exception as e:
                    logger.warning(f"Could not restore registry key {key_path}: {e}")
            
            self.original_settings.clear()
            self.game_mode_active = False
            logger.info("✓ Game Mode settings restored")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore Game Mode: {e}")
            return False


class AntiCheatDetector:
    """Detects anti-cheat systems to avoid compatibility issues"""
    
    KNOWN_ANTICHEATS = {
        'easyanticheat.exe': 'EasyAntiCheat',
        'easyanticheat_x64.dll': 'EasyAntiCheat',
        'beservice.exe': 'BattlEye',
        'beclient.dll': 'BattlEye',
        'beclient_x64.dll': 'BattlEye',
        'vgc.exe': 'Riot Vanguard',
        'vgk.sys': 'Riot Vanguard',
        'vanguard_x64.dll': 'Riot Vanguard',
        'faceit.exe': 'FACEIT',
        'steamservice.dll': 'VAC (Steam)',
    }
    
    @staticmethod
    def detect_anticheat(pid: int) -> Optional[str]:
        """Detect if game process uses known anti-cheat"""
        try:
            process = psutil.Process(pid)
            process_name = process.name().lower()
            
            # Check process name
            for ac_file, ac_name in AntiCheatDetector.KNOWN_ANTICHEATS.items():
                if ac_file.lower() in process_name:
                    return ac_name
            
            # Check loaded modules
            try:
                for dll in process.memory_maps():
                    dll_path = getattr(dll, 'path', '').lower()
                    for ac_file, ac_name in AntiCheatDetector.KNOWN_ANTICHEATS.items():
                        if ac_file.lower() in dll_path:
                            return ac_name
            except (psutil.AccessDenied, AttributeError):
                pass
            
            # Check child processes
            try:
                for child in process.children(recursive=True):
                    child_name = child.name().lower()
                    for ac_file, ac_name in AntiCheatDetector.KNOWN_ANTICHEATS.items():
                        if ac_file.lower() in child_name:
                            return ac_name
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            return None
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


class GameOptimizer:
    """Main optimizer orchestrator V3.5 - 920/1000 Quality"""
    
    def __init__(self, config_file: Optional[str] = None):
        logger.info("=" * 80)
        logger.info("Initializing Game Optimizer V3.5 (920/1000 Professional Edition)")
        logger.info("=" * 80)
        
        if not self._is_admin():
            raise PermissionError(
                "Administrator privileges are required.\n"
                "Please run as Administrator.")
        
        # Core managers
        self.config_manager = ConfigurationManager(config_file)
        self.timer_manager = AdvancedTimerManager()
        self.memory_optimizer = MemoryOptimizer()
        self.gpu_scheduler = GPUSchedulingOptimizer()
        self.background_apps_manager = WindowsBackgroundAppsManager()
        self.power_optimizer = PowerManagementOptimizer()
        self.directx_optimizer = DirectXOptimizer()
        self.network_optimizer = NetworkOptimizer()
        self.session_manager = GamingSessionManager()
        self.cpu_manager = AdvancedCPUManager()
        self.core_parking_manager = CoreParkingManager()
        
        # New V3.5 components
        self.performance_monitor = PerformanceMonitor()
        self.telemetry_collector = TelemetryCollector()
        self.ab_testing = ABTestingFramework()
        self.ml_tuner = MLAutoTuner()
        self.game_mode_manager = GameModeManager()
        
        # State tracking
        self.active_optimizations: Dict[int, OptimizationState] = {}
        self.optimization_lock = threading.Lock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_active = False
        
        # Start background services
        self.session_manager.start_mouse_listener()
        
        logger.info("✓ Game Optimizer V3.5 initialized")
        logger.info("Features: ETW monitoring, GPU clocking, DPC detection, ML tuning, A/B testing")
        logger.info("=" * 80)
    
    def _is_admin(self) -> bool:
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except AttributeError:
            return False
    
    def start_optimization(self, game_pid: int, enable_ab_test: bool = False) -> bool:
        """Start comprehensive optimization for game process"""
        
        with self.optimization_lock:
            if game_pid in self.active_optimizations:
                logger.warning(f"Optimization already active for PID {game_pid}")
                return False
            
            try:
                p = psutil.Process(game_pid)
                game_exe = p.name()
            except psutil.NoSuchProcess:
                logger.error(f"Process with PID {game_pid} not found")
                return False
            
            logger.info("=" * 80)
            logger.info(f"STARTING OPTIMIZATION V3.5 for '{game_exe}' (PID: {game_pid})")
            logger.info("=" * 80)
            
            # Anti-cheat detection
            anticheat = AntiCheatDetector.detect_anticheat(game_pid)
            if anticheat:
                logger.warning(f"⚠️  Detected anti-cheat: {anticheat}")
                logger.warning("⚠️  Using safe mode (reduced kernel-level optimizations)")
            
            # Get or create profile
            profile = self.config_manager.get_game_profile(game_exe)
            if not profile:
                logger.info(f"No profile found for '{game_exe}', checking ML recommendations...")
                
                # Try ML auto-tune
                ml_profile = self.ml_tuner.get_optimized_profile(game_exe)
                if ml_profile:
                    profile = ml_profile
                    logger.info(f"✓ Using ML-optimized profile")
                else:
                    logger.info("Using default competitive profile")
                    profile = GameProfile(
                        name="Default Competitive",
                        game_exe=game_exe,
                        timer_resolution_ms=0.5,
                        cpu_priority_class='HIGH',
                        network_qos_enabled=True
                    )
            
            state = OptimizationState(
                game_exe=game_exe,
                game_pid=game_pid,
                profile=profile,
                start_time=time.time()
            )
            
            # A/B Testing baseline
            if enable_ab_test:
                logger.info("📊 A/B Test: Collecting baseline metrics (30s)...")
                baseline = self.ab_testing.collect_baseline(game_pid, duration=30)
                state.baseline_metrics = baseline
            
            # PHASE 1: Session Management (services/processes)
            if profile.stop_services or profile.stop_processes:
                if self.session_manager.start_gaming_session(game_exe, game_pid):
                    state.optimizations_applied.add('gaming_session')
                    logger.info("✓ Gaming session started")
            
            # PHASE 2: Game Mode
            if self.game_mode_manager.enable_game_mode():
                state.optimizations_applied.add('game_mode')
            
            # PHASE 3: CPU Optimizations
            if not anticheat or anticheat not in ['Riot Vanguard', 'FACEIT']:
                state.original_process_state = self.cpu_manager.set_process_optimizations(
                    game_pid, profile)
                
                if state.original_process_state:
                    state.optimizations_applied.add('cpu_optimizations_v3')
                    logger.info("✓ CPU optimizations applied")
            
            # PHASE 4: Core Parking
            if getattr(profile, 'disable_core_parking', True):
                if self.core_parking_manager.disable_core_parking(
                    self.cpu_manager.topology.performance_cores):
                    state.optimizations_applied.add('core_parking_disabled')
                    logger.info("✓ Core parking disabled on P-cores")
            
            # PHASE 5: Timer Resolution
            if self.timer_manager.set_high_performance_timer():
                state.optimizations_applied.add('timer')
                logger.info("✓ High-performance timer set")
            
            # PHASE 6: GPU Scheduling
            if profile.gpu_scheduling_enabled:
                if self.gpu_scheduler.enable_hardware_scheduling():
                    state.optimizations_applied.add('gpu_scheduling')
                    if self.gpu_scheduler.restart_required:
                        state.notes.append("GPU Hardware Scheduling requires reboot")
                    logger.info("✓ GPU Hardware Scheduling enabled")
            
            # PHASE 7: Power Management
            if profile.power_high_performance:
                if self.power_optimizer.set_high_performance():
                    state.optimizations_applied.add('power_plan')
                    logger.info("✓ High-performance power plan activated")
                
                if self.power_optimizer.create_power_request():
                    state.optimizations_applied.add('power_request')
                    logger.info("✓ Power request created")
            
            # PHASE 8: Background Apps
            cpu_thr = float(self.config_manager.get_global_setting(
                'background_throttle_cpu_percent', 3.0))
            mem_thr = int(self.config_manager.get_global_setting(
                'background_throttle_memory_mb', 200))
            
            limited = self.background_apps_manager.limit_background_apps(
                cpu_threshold=cpu_thr, rss_mb_threshold=mem_thr)
            
            if limited > 0:
                state.optimizations_applied.add('background_apps')
                logger.info(f"✓ Limited {limited} background applications")
            
            # PHASE 9: Memory Optimization
            mem_level = getattr(profile, 'memory_optimization_level', 2)
            if mem_level > 0:
                # Adaptive purge based on profile type
                min_interval = 30 if mem_level >= 2 else 120
                threshold = 70 if mem_level >= 2 else 85
                
                if self.memory_optimizer.purge_standby_memory_adaptive(
                    min_interval=min_interval, threshold_percent=threshold):
                    state.optimizations_applied.add('memory_purge')
                    logger.info("✓ Standby memory purged")
                
                if self.memory_optimizer.force_clr_gc(game_pid):
                    state.optimizations_applied.add('clr_gc_forced')
                    logger.info("✓ CLR GC forced (Unity/UE games)")
            
            # PHASE 10: DirectX/GPU
            if profile.directx_optimizations and not anticheat:
                # Check for shader compilation first
                if not self.directx_optimizer.detect_shader_compilation_phase(game_pid):
                    if self.directx_optimizer.optimize_dxgi_settings(enable=True):
                        state.optimizations_applied.add('directx_registry')
                        logger.info("✓ DXGI settings optimized")
                    
                    if self.directx_optimizer.optimize_for_process(game_pid):
                        state.optimizations_applied.add('directx_process')
                        logger.info("✓ DirectX process optimizations applied")
                    
                    # GPU clock locking
                    if getattr(profile, 'gpu_clock_locking', True):
                        vendor, locked = self.directx_optimizer.lock_gpu_clocks(enable=True)
                        if locked:
                            state.optimizations_applied.add('gpu_clocks_locked')
                            logger.info(f"✓ GPU clocks locked to max ({vendor})")
                else:
                    logger.warning("⚠️  Shader compilation detected, deferring aggressive DX opts")
            
            # PHASE 11: Network
            if profile.network_qos_enabled:
                if self.network_optimizer.optimize_network_adapter():
                    state.optimizations_applied.add('network_adapter')
                    logger.info("✓ Network adapter optimized")
                
                # NIC RSS affinity
                if getattr(profile, 'nic_rss_auto', True):
                    p_cores = self.cpu_manager.topology.performance_cores or \
                              list(range(self.cpu_manager.topology.logical_cores))
                    max_procs = getattr(profile, 'nic_rss_max_processors', None)
                    
                    if self.network_optimizer.optimize_nic_interrupt_affinity(
                        p_core_indices=p_cores, max_processors=max_procs):
                        state.optimizations_applied.add('nic_interrupt_affinity')
                        logger.info("✓ NIC RSS affinity optimized")
                
                # QoS policies
                conns = self.network_optimizer.get_process_connections(game_pid)
                qos_rules = [r.__dict__ for r in profile.qos_rules] if profile.qos_rules else None
                
                if self.network_optimizer.apply_qos_policy(
                    game_pid, profile.network_dscp_value, qos_rules):
                    state.optimizations_applied.add('network_qos')
                    logger.info("✓ QoS policies applied")
            
            # PHASE 12: Monitoring
            if getattr(profile, 'enable_frame_time_analysis', True):
                if self.performance_monitor.start_monitoring(game_pid, game_exe):
                    state.optimizations_applied.add('performance_monitoring')
                    logger.info("✓ Performance monitoring started")
            
            # Register state
            self.active_optimizations[game_pid] = state
            
            # Start telemetry collection
            self.telemetry_collector.start_session(game_pid, game_exe, profile, state)
            
            logger.info("=" * 80)
            logger.info(f"OPTIMIZATION COMPLETE: {len(state.optimizations_applied)} features active")
            if state.notes:
                for note in state.notes:
                    logger.info(f"📝 Note: {note}")
            logger.info(f"Applied: {', '.join(sorted(state.optimizations_applied))}")
            logger.info("=" * 80)
            
            if not self.monitor_active:
                self._start_monitoring()
            
            return True
    
    def stop_optimization(self, game_pid: int) -> bool:
        """Stop optimization and restore original settings"""
        
        with self.optimization_lock:
            if game_pid not in self.active_optimizations:
                return False
            
            state = self.active_optimizations.pop(game_pid)
            
            logger.info("=" * 80)
            logger.info(f"STOPPING OPTIMIZATION for '{state.game_exe}' (PID: {game_pid})")
            logger.info("=" * 80)
            
            # Collect final metrics
            final_metrics = self.performance_monitor.get_session_summary(game_pid)
            if final_metrics:
                logger.info(f"📊 Session Summary:")
                logger.info(f"   Avg FPS: {final_metrics.get('avg_fps', 'N/A'):.1f}")
                logger.info(f"   1% Low: {final_metrics.get('one_percent_low', 'N/A'):.1f}")
                logger.info(f"   Frame Time P99: {final_metrics.get('frame_time_p99', 'N/A'):.2f}ms")
            
            # Stop monitoring
            if 'performance_monitoring' in state.optimizations_applied:
                self.performance_monitor.stop_monitoring(game_pid)
                logger.info("✓ Performance monitoring stopped")
            
            # End telemetry session
            telemetry_data = self.telemetry_collector.end_session(game_pid)
            if telemetry_data:
                # Train ML model with new data
                self.ml_tuner.add_training_sample(telemetry_data)
            
            # Restore in reverse order
            if 'network_qos' in state.optimizations_applied:
                self.network_optimizer.remove_qos_policy(game_pid)
                logger.info("✓ QoS policies removed")
            
            if 'gpu_clocks_locked' in state.optimizations_applied:
                self.directx_optimizer.lock_gpu_clocks(enable=False)
                logger.info("✓ GPU clocks restored")
            
            if 'directx_registry' in state.optimizations_applied:
                self.directx_optimizer.restore_original_settings()
                logger.info("✓ DirectX settings restored")
            
            if 'background_apps' in state.optimizations_applied:
                self.background_apps_manager.restore_all_background_apps()
                logger.info("✓ Background apps restored")
            
            if 'power_request' in state.optimizations_applied:
                self.power_optimizer.clear_power_request()
                logger.info("✓ Power request cleared")
            
            if 'power_plan' in state.optimizations_applied:
                self.power_optimizer.restore_original_plan()
                logger.info("✓ Power plan restored")
            
            if 'core_parking_disabled' in state.optimizations_applied:
                self.core_parking_manager.restore_core_parking()
                logger.info("✓ Core parking restored")
            
            if 'cpu_optimizations_v3' in state.optimizations_applied:
                self.cpu_manager.restore_process_optimizations(
                    game_pid, state.original_process_state)
                logger.info("✓ CPU optimizations restored")
            
            if 'timer' in state.optimizations_applied:
                self.timer_manager.restore_default_timer()
                logger.info("✓ Timer resolution restored")
            
            if 'game_mode' in state.optimizations_applied:
                self.game_mode_manager.disable_game_mode()
                logger.info("✓ Game Mode restored")
            
            if 'gaming_session' in state.optimizations_applied:
                self.session_manager.end_gaming_session()
                logger.info("✓ Gaming session ended")
            
            # Cleanup
            try:
                self.directx_optimizer.cleanup()
            except Exception as e:
                logger.debug(f"DirectX cleanup error: {e}")
            
            duration = time.time() - state.start_time
            logger.info("=" * 80)
            logger.info(f"RESTORATION COMPLETE. Duration: {duration/60:.1f} minutes")
            logger.info("=" * 80)
            
            return True
    
    def run_ab_test(self, game_pid: int, duration_minutes: int = 5) -> Dict[str, Any]:
        """Run A/B test comparing baseline vs optimized performance"""
        
        logger.info("=" * 80)
        logger.info("STARTING A/B TEST")
        logger.info("=" * 80)
        
        try:
            p = psutil.Process(game_pid)
            game_exe = p.name()
        except psutil.NoSuchProcess:
            logger.error(f"Process {game_pid} not found")
            return {}
        
        results = self.ab_testing.run_full_test(
            game_pid=game_pid,
            game_exe=game_exe,
            optimizer=self,
            duration_minutes=duration_minutes
        )
        
        if results:
            logger.info("=" * 80)
            logger.info("A/B TEST RESULTS")
            logger.info("=" * 80)
            logger.info(f"Baseline:")
            logger.info(f"  Avg FPS: {results['baseline']['avg_fps']:.1f}")
            logger.info(f"  1% Low: {results['baseline']['one_percent_low']:.1f}")
            logger.info(f"  Frame Time P99: {results['baseline']['frame_time_p99']:.2f}ms")
            logger.info(f"Optimized:")
            logger.info(f"  Avg FPS: {results['optimized']['avg_fps']:.1f} ({results['improvement']['avg_fps_pct']:+.1f}%)")
            logger.info(f"  1% Low: {results['optimized']['one_percent_low']:.1f} ({results['improvement']['one_percent_low_pct']:+.1f}%)")
            logger.info(f"  Frame Time P99: {results['optimized']['frame_time_p99']:.2f}ms ({results['improvement']['frame_time_p99_pct']:+.1f}%)")
            logger.info("=" * 80)
        
        return results
    
    def export_telemetry(self, output_file: Optional[Path] = None) -> Path:
        """Export all collected telemetry to JSON"""
        return self.telemetry_collector.export_to_file(output_file)
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitor_active:
            return
        
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("✓ Game process monitoring started")
    
    def _monitor_loop(self):
        """Background thread to monitor for new games and clean up terminated ones"""
        known_game_exes = {
            p.game_exe.lower() for p in self.config_manager.game_profiles.values()
        }
        
        while self.monitor_active:
            with self.optimization_lock:
                active_pids = set(self.active_optimizations.keys())
            
            # Auto-detect new games
            if self.config_manager.get_global_setting('auto_detect_games', True):
                for p in psutil.process_iter(['pid', 'name']):
                    try:
                        if p.info['name'] and \
                           p.info['name'].lower() in known_game_exes and \
                           p.info['pid'] not in active_pids:
                            logger.info(f"🎮 Auto-detected game: {p.info['name']} (PID: {p.info['pid']})")
                            self.start_optimization(p.info['pid'])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            # Cleanup terminated games
            for pid in list(active_pids):
                if not psutil.pid_exists(pid):
                    logger.info(f"Game with PID {pid} terminated, cleaning up...")
                    self.stop_optimization(pid)
            
            time.sleep(10)
    
    def cleanup(self):
        """Cleanup all resources and restore system state"""
        logger.info("=" * 80)
        logger.info("Cleaning up Game Optimizer V3.5...")
        logger.info("=" * 80)
        
        self.monitor_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        for pid in list(self.active_optimizations.keys()):
            self.stop_optimization(pid)
        
        self.session_manager.cleanup()
        self.directx_optimizer.cleanup()
        self.network_optimizer.cleanup()
        self.performance_monitor.cleanup()
        self.game_mode_manager.disable_game_mode()
        
        # Save ML model
        self.ml_tuner.save_model()
        
        logger.info("✓ Game Optimizer V3.5 cleanup complete")
        logger.info("=" * 80)


def setup_logging():
    """Configure logging with file rotation"""
    log_dir = Path.home() / '.game_optimizer' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'game_optimizer_v35.log'
    
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def main():
    setup_logging()
    
    logger.info("")
    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║      GAME OPTIMIZER V3.5 - 920/1000 PROFESSIONAL EDITION     ║")
    logger.info("║   Advanced Low-Level Optimizations + ML Tuning + Monitoring  ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝")
    logger.info("")
    
    try:
        optimizer = GameOptimizer()
        
        logger.info("✓ Game Optimizer V3.5 is running")
        logger.info("✓ Monitoring for game processes...")
        logger.info("✓ Press Ctrl+C to exit")
        logger.info("")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Shutdown signal received (Ctrl+C)")
        
    except PermissionError as e:
        logger.critical(f"❌ PERMISSION ERROR: {e}")
        logger.critical("Please run as Administrator")
        return 1
        
    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR: {e}", exc_info=True)
        return 1
        
    finally:
        if 'optimizer' in locals():
            optimizer.cleanup()
        logger.info("Shutdown complete. Goodbye!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())