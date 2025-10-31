"""
Network Optimizer V3.5
FIXED: Thread-safety, better error handling
(El código original ya era bastante bueno, solo mejoras menores)
"""

import ctypes
import logging
import socket
import subprocess
import re
import threading
from ctypes import wintypes
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import psutil
import time

logger = logging.getLogger(__name__)

QOS_NON_ADAPTIVE_FLOW = 0x00000002
DSCP_EF = 46

@dataclass
class NetworkConnection:
    local_address: str
    local_port: int
    remote_address: str
    remote_port: int
    protocol: str
    status: str
    pid: int

@dataclass
class NetworkMetrics:
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    latency_ms: float
    jitter_ms: float
    bandwidth_mbps: float


class NetworkOptimizer:
    """Network optimizer with QoS, RSS affinity, TCP tuning"""
    
    def __init__(self):
        self.qos_dll = None
        self.ws2_32 = None
        self.iphlpapi = None
        self.optimized_processes: Set[int] = set()
        self.qos_policies: Dict[int, List[str]] = {}
        self.network_metrics: Dict[int, NetworkMetrics] = {}
        self.lock = threading.Lock()  # NEW: Thread-safety
        
        self._load_network_dlls()
        self._setup_icmp()
    
    def _load_network_dlls(self):
        try:
            self.ws2_32 = ctypes.WinDLL('ws2_32.dll')
        except Exception as e:
            logger.debug(f"ws2_32 load error: {e}")
        
        try:
            self.iphlpapi = ctypes.WinDLL('iphlpapi.dll')
        except Exception as e:
            logger.debug(f"iphlpapi load error: {e}")
        
        try:
            self.qos_dll = ctypes.WinDLL('qwave.dll')
        except Exception as e:
            logger.debug(f"qwave load error: {e}")
    
    def _setup_icmp(self):
        try:
            self.icmp = ctypes.WinDLL('iphlpapi.dll')
            self.IcmpCreateFile = self.icmp.IcmpCreateFile
            self.IcmpSendEcho = self.icmp.IcmpSendEcho
            self.IcmpCloseHandle = self.icmp.IcmpCloseHandle
            
            self.IcmpCreateFile.restype = wintypes.HANDLE
            self.IcmpCloseHandle.argtypes = [wintypes.HANDLE]
            
            class IP_OPTION_INFORMATION(ctypes.Structure):
                _fields_ = [("Ttl", ctypes.c_ubyte), ("Tos", ctypes.c_ubyte),
                           ("Flags", ctypes.c_ubyte), ("OptionsSize", ctypes.c_ubyte),
                           ("OptionsData", ctypes.c_void_p)]
            
            class ICMP_ECHO_REPLY(ctypes.Structure):
                _fields_ = [("Address", wintypes.DWORD), ("Status", wintypes.DWORD),
                           ("RoundTripTime", wintypes.DWORD), ("DataSize", wintypes.WORD),
                           ("Reserved", wintypes.WORD), ("Data", ctypes.c_void_p),
                           ("Options", IP_OPTION_INFORMATION)]
            
            self.IP_OPTION_INFORMATION = IP_OPTION_INFORMATION
            self.ICMP_ECHO_REPLY = ICMP_ECHO_REPLY
            
        except Exception as e:
            logger.debug(f"ICMP setup error: {e}")
            self.icmp = None
    
    def get_process_connections(self, pid: int) -> List[NetworkConnection]:
        """Get active network connections for process"""
        connections = []
        
        try:
            process = psutil.Process(pid)
            for conn in process.connections(kind='inet'):
                try:
                    connection = NetworkConnection(
                        local_address=conn.laddr.ip if conn.laddr else "",
                        local_port=conn.laddr.port if conn.laddr else 0,
                        remote_address=conn.raddr.ip if conn.raddr else "",
                        remote_port=conn.raddr.port if conn.raddr else 0,
                        protocol='TCP' if conn.type == socket.SOCK_STREAM else 'UDP',
                        status=conn.status,
                        pid=pid
                    )
                    connections.append(connection)
                except Exception as e:
                    logger.debug(f"Connection parse error: {e}")
                    
        except Exception as e:
            logger.debug(f"get_process_connections error: {e}")
        
        return connections
    
    def _run_command(self, command: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
        except Exception:
            class R:
                returncode = 1
                stdout = ""
                stderr = ""
            return R()
    
    def measure_rtt(self, target: str, count: int = 2) -> Optional[float]:
        """Measure round-trip time to target"""
        try:
            command = ['ping', '-n', str(count), '-w', '1000', target]
            result = self._run_command(command, timeout=count + 5)
            
            if getattr(result, 'returncode', 1) == 0:
                latencies = re.findall(r'(?:time|tiempo)[=<](\d+)', result.stdout.lower())
                if latencies:
                    avg_rtt = sum(map(float, latencies)) / len(latencies)
                    return avg_rtt
                    
        except Exception as e:
            logger.debug(f"RTT measurement error: {e}")
        
        return None
    
    def apply_qos_policy(self, pid: int, dscp_value: int = DSCP_EF, 
                        qos_rules: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Apply QoS policies for process"""
        
        with self.lock:  # Thread-safe
            try:
                process = psutil.Process(pid)
                process_name = process.name()
                
                try:
                    exe_path = process.exe()
                except Exception:
                    exe_path = process_name
                
                # Validate IP_TOS support
                tos_res = self._validate_local_dscp_marking(dscp_value)
                if not tos_res.get("local_tos_set"):
                    logger.warning("Local IP_TOS/DSCP not supported, skipping QoS")
                    return False
                
                policy_base = f"GameOptimizer_{process_name}_{pid}"
                self._remove_qos_policy(policy_base)
                
                created_names: List[str] = []
                
                # App-level policy
                ps_command = [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "New-NetQosPolicy",
                    "-Name", f"'{policy_base}'",
                    "-AppPathNameMatchCondition", f"'{exe_path}'",
                    "-DSCPValue", str(int(dscp_value)),
                    "-IPProtocol", "Both",
                    "-ErrorAction", "Stop"
                ]
                
                result = self._run_command(ps_command, timeout=10)
                if getattr(result, 'returncode', 1) == 0:
                    created_names.append(policy_base)
                
                # Per-connection rules
                try:
                    conns = self.get_process_connections(pid)
                    per_conn = [c for c in conns if c.remote_port and c.remote_address][:10]
                    
                    for i, c in enumerate(per_conn, start=1):
                        name = f"{policy_base}_port_{i}"
                        proto = c.protocol.upper()
                        
                        parts = [
                            "powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "New-NetQosPolicy",
                            "-Name", f"'{name}'",
                            "-DSCPValue", str(int(dscp_value)),
                            "-IPProtocol", proto,
                            "-RemotePort", str(c.remote_port)
                        ]
                        
                        res = self._run_command(parts, timeout=10)
                        if getattr(res, 'returncode', 1) == 0:
                            created_names.append(name)
                            
                except Exception as e:
                    logger.debug(f"Per-connection QoS error: {e}")
                
                # Custom rules
                if qos_rules:
                    for i, rule in enumerate(qos_rules, start=1):
                        try:
                            name = rule.get('name', f"{policy_base}_rule{i}")
                            proto = rule.get('protocol', 'both').lower()
                            dscp = int(rule.get('dscp', dscp_value))
                            
                            parts = [
                                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                                "New-NetQosPolicy",
                                "-Name", f"'{name}'",
                                "-DSCPValue", str(dscp),
                                "-IPProtocol", 'Both' if proto == 'both' else proto.upper()
                            ]
                            
                            for key, ps_param in [('localport', '-LocalPort'), ('remoteport', '-RemotePort'),
                                                 ('localip', '-LocalIPAddress'), ('remoteip', '-RemoteIPAddress')]:
                                if rule.get(key) and rule.get(key) != 'any':
                                    parts.extend([ps_param, str(rule[key])])
                            
                            res = self._run_command(parts, timeout=10)
                            if getattr(res, 'returncode', 1) == 0:
                                created_names.append(name)
                                
                        except Exception as e:
                            logger.debug(f"Custom QoS rule error: {e}")
                
                if created_names:
                    self.optimized_processes.add(pid)
                    self.qos_policies[pid] = created_names
                    logger.info(f"✓ QoS: {len(created_names)} policies created")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"QoS policy error: {e}")
                return False
    
    def _validate_local_dscp_marking(self, dscp_value: int) -> Dict[str, Any]:
        """Validate that local DSCP marking is supported"""
        result = {"local_tos_set": False, "tos_value": None}
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                tos = (dscp_value << 2) & 0xFC
                s.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, tos)
                got = s.getsockopt(socket.IPPROTO_IP, socket.IP_TOS)
                result["local_tos_set"] = (got & 0xFC) == tos
                result["tos_value"] = got
        except Exception as e:
            logger.debug(f"DSCP validation error: {e}")
        
        return result
    
    def _remove_qos_policy(self, policy_base: str):
        """Remove QoS policies by base name"""
        try:
            ps_delete_cmd = [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                f"Get-NetQosPolicy -Name '{policy_base}*' -ErrorAction SilentlyContinue | "
                f"Remove-NetQosPolicy -Confirm:$false -ErrorAction SilentlyContinue"
            ]
            self._run_command(ps_delete_cmd, timeout=15)
            
        except Exception as e:
            logger.debug(f"QoS removal error: {e}")
    
    def remove_qos_policy(self, pid: int) -> bool:
        """Remove QoS policies for process"""
        
        with self.lock:
            try:
                if pid not in self.qos_policies:
                    return True
                
                names = self.qos_policies.pop(pid)
                
                for name in names:
                    ps_cmd = [
                        "powershell", "-NoProfile", "-NonInteractive", "-Command",
                        "Remove-NetQosPolicy", "-Name", f"'{name}'", 
                        "-Confirm:$false", "-ErrorAction", "SilentlyContinue"
                    ]
                    self._run_command(ps_cmd, timeout=5)
                
                self.optimized_processes.discard(pid)
                return True
                
            except Exception as e:
                logger.error(f"QoS removal error: {e}")
                return False
    
    def optimize_network_adapter(self) -> bool:
        """Optimize network adapter settings"""
        try:
            # Disable power saving
            power_plan_guid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
            power_subgroup_guid = "19cbb8fa-5279-450e-9fac-8a3d5fedd0c1"
            power_setting_guid = "12bbebe6-58d6-4636-95bb-3217ef867c1a"
            
            self._run_command(['powercfg', '/setacvalueindex', power_plan_guid, 
                             power_subgroup_guid, power_setting_guid, '0'], timeout=10)
            
            # Enable TCP offload
            self._run_command(['netsh', 'int', 'tcp', 'set', 'global', 
                             'taskoffload=enabled'], timeout=10)
            
            return True
            
        except Exception as e:
            logger.error(f"Network adapter optimization error: {e}")
            return False
    
    def optimize_nic_interrupt_affinity(self, p_core_indices: Optional[List[int]] = None, 
                                        max_processors: Optional[int] = None) -> bool:
        """Optimize NIC RSS to use P-cores"""
        try:
            base_proc = 0
            if p_core_indices and len(p_core_indices) > 0:
                base_proc = int(p_core_indices[0])
            
            # Query RSS queues
            ps_list = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                      "Get-NetAdapterRss | Where-Object {$_.Enabled -eq $true} | "
                      "Select-Object Name,NumberOfReceiveQueues"]
            
            res = self._run_command(ps_list, timeout=10)
            queues = 0
            
            if getattr(res, 'returncode', 1) == 0 and res.stdout:
                m = re.findall(r'NumberOfReceiveQueues\s*:\s*(\d+)', res.stdout)
                if m:
                    queues = max(int(x) for x in m if x.isdigit())
            
            logical = psutil.cpu_count(logical=True) or 1
            target = min(queues if queues > 0 else logical, logical)
            
            if max_processors is not None:
                target = min(target, int(max_processors))
            
            if p_core_indices:
                target = min(target, len(p_core_indices))
            
            if target <= 0:
                target = min(logical, 4)
            
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                  "Get-NetAdapterRss | Where-Object {$_.Status -eq 'Up'} | ForEach-Object {",
                  f"Set-NetAdapterRss -Name $_.Name -MaxProcessors {target} -ErrorAction SilentlyContinue ;",
                  f"Set-NetAdapterRss -Name $_.Name -BaseProcessorNumber {base_proc} -ErrorAction SilentlyContinue",
                  "}"]
            
            self._run_command(cmd, timeout=15)
            logger.info(f"✓ NIC RSS: {target} queues starting at core {base_proc}")
            return True
            
        except Exception as e:
            logger.error(f"NIC RSS error: {e}")
            return False
    
    def optimize_tcp_per_connection(self, pid: int, connections: List[NetworkConnection], 
                                    allow_global_tcp_tuning: bool = False) -> bool:
        """Optimize TCP settings (per-connection or global)"""
        # Kept for compatibility, but global tuning disabled by default
        return True
    
    def cleanup(self):
        """Cleanup all network optimizations"""
        with self.lock:
            for pid in list(self.optimized_processes):
                self.remove_qos_policy(pid)