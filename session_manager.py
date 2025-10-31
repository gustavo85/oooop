import ctypes
import logging
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import win32service
import win32serviceutil
import pywintypes
import getpass

logger = logging.getLogger(__name__)

class ServiceState(Enum):
    STOPPED = win32service.SERVICE_STOPPED
    START_PENDING = win32service.SERVICE_START_PENDING
    STOP_PENDING = win32service.SERVICE_STOP_PENDING
    RUNNING = win32service.SERVICE_RUNNING
    CONTINUE_PENDING = win32service.SERVICE_CONTINUE_PENDING
    PAUSE_PENDING = win32service.SERVICE_PAUSE_PENDING
    PAUSED = win32service.SERVICE_PAUSED
    UNKNOWN = 0

@dataclass
class ServiceInfo:
    name: str
    display_name: str
    status: ServiceState
    startup_type: str
    pid: Optional[int] = None

@dataclass
class ProcessInfo:
    pid: int
    name: str
    exe_path: str
    memory_mb: float
    cpu_percent: float

@dataclass
class SessionState:
    game_exe: str
    game_pid: int
    start_time: float
    stopped_services: List[str]
    stopped_processes: List[int]
    active: bool = True

class GamingSessionManager:
    CRITICAL_SERVICES = {
        'AudioSrv', 'AudioEndpointBuilder', 'Dhcp', 'Dnscache', 'EventLog',
        'EventSystem', 'lmhosts', 'LanmanServer', 'LanmanWorkstation', 'mpssvc',
        'nsi', 'PlugPlay', 'Power', 'ProfSvc', 'RpcSs', 'SamSs', 'Schedule',
        'SENS', 'ShellHWDetection', 'Spooler', 'SysMain', 'Themes', 'UserManager',
        'Winmgmt', 'WinDefend', 'wscsvc', 'WSearch', 'wuauserv', 'BFE', 'DPS',
        'CryptSvc', 'Netman', 'NlaSvc', 'Wcmsvc',
    }
    CRITICAL_PROCESSES = {
        'system', 'registry', 'smss.exe', 'csrss.exe', 'wininit.exe',
        'services.exe', 'lsass.exe', 'svchost.exe', 'winlogon.exe',
        'explorer.exe', 'dwm.exe', 'taskhostw.exe', 'sihost.exe',
        'runtimebroker.exe', 'ctfmon.exe', 'conhost.exe', 'fontdrvhost.exe',
        'audiodg.exe',
    }
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path.home() / '.game_optimizer' / 'session_config.json'
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.services_to_stop: List[str] = []
        self.processes_to_stop: List[str] = []
        self.active_session: Optional[SessionState] = None
        self.session_lock = threading.Lock()
        self.mouse_listener_thread: Optional[threading.Thread] = None
        self.mouse_listener_active = False
        self._mouse_listener = None
        self.ui_lock = threading.Lock()
        self.action_history: List[Dict] = []
        self._load_configuration()
        self._validate_configuration()
    def _load_configuration(self):
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.services_to_stop = config.get('services_to_stop', [])
                    self.processes_to_stop = config.get('processes_to_stop', [])
            else:
                self._set_default_configuration()
                self._save_configuration()
        except Exception as e:
            logger.debug(f"Session config load error: {e}")
            self._set_default_configuration()
    def _set_default_configuration(self):
        try:
            self.services_to_stop = [
                'DiagTrack', 'dmwappushservice', 'WbioSrvc', 'TabletInputService',
                'wisvc', 'RetailDemo', 'PhoneSvc', 'MapsBroker', 'lfsvc',
                'SharedAccess', 'PrintNotify', 'Fax', 'XblAuthManager', 'XblGameSave',
                'XboxNetApiSvc', 'XboxGipSvc',
            ]
            self.processes_to_stop = [
                'chrome.exe', 'firefox.exe', 'msedge.exe', 'opera.exe', 'brave.exe',
                'spotify.exe', 'Discord.exe', 'Skype.exe', 'Teams.exe', 'OneDrive.exe',
                'Dropbox.exe', 'GoogleDrive.exe', 'Telegram.exe', 'WhatsApp.exe',
                'Slack.exe', 'Notion.exe',
            ]
        except Exception:
            pass
    def _save_configuration(self):
        try:
            config = {
                'services_to_stop': sorted(list(set(self.services_to_stop))),
                'processes_to_stop': sorted(list(set(self.processes_to_stop))),
                'version': '2.0',
                'last_modified': time.time()
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.debug(f"Session config save error: {e}")
    def _validate_configuration(self):
        try:
            original_services_count = len(self.services_to_stop)
            critical_services_lower = {s.lower() for s in self.CRITICAL_SERVICES}
            self.services_to_stop = [s for s in self.services_to_stop if s.lower() not in critical_services_lower]
            if len(self.services_to_stop) < original_services_count:
                self._save_configuration()
            original_processes_count = len(self.processes_to_stop)
            critical_processes_lower = {p.lower() for p in self.CRITICAL_PROCESSES}
            self.processes_to_stop = [p for p in self.processes_to_stop if p.lower() not in critical_processes_lower]
            if len(self.processes_to_stop) < original_processes_count:
                self._save_configuration()
        except Exception:
            pass
    def get_service_info(self, service_name: str) -> Optional[ServiceInfo]:
        hscm = None
        hs = None
        try:
            hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ENUMERATE_SERVICE)
            hs = win32serviceutil.SmartOpenService(hscm, service_name, win32service.SERVICE_QUERY_STATUS | win32service.SERVICE_QUERY_CONFIG)
            status = win32service.QueryServiceStatusEx(hs)
            config = win32service.QueryServiceConfig(hs)
            startup_map = {
                win32service.SERVICE_AUTO_START: 'Automatic',
                win32service.SERVICE_BOOT_START: 'Boot',
                win32service.SERVICE_DEMAND_START: 'Manual',
                win32service.SERVICE_DISABLED: 'Disabled',
                win32service.SERVICE_SYSTEM_START: 'System',
            }
            return ServiceInfo(
                name=service_name,
                display_name=config[8],
                status=ServiceState(status['CurrentState']),
                startup_type=startup_map.get(config[1], 'Unknown'),
                pid=status.get('ProcessId')
            )
        except Exception:
            return None
        finally:
            try:
                if hs:
                    win32service.CloseServiceHandle(hs)
            except Exception:
                pass
            try:
                if hscm:
                    win32service.CloseServiceHandle(hscm)
            except Exception:
                pass
    def _stop_dependent_services(self, hs) -> List[str]:
        stopped = []
        try:
            deps = win32service.EnumDependentServices(hs, win32service.SERVICE_ACTIVE)
            for dep_name, dep_display, dep_status in deps:
                try:
                    if dep_name:
                        if self.stop_service(dep_name):
                            stopped.append(dep_name)
                except Exception:
                    pass
        except pywintypes.error:
            pass
        except Exception:
            pass
        return stopped
    def _wait_service_state(self, hs, desired_state: int, timeout: int = 30) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                status = win32service.QueryServiceStatusEx(hs)
                if status.get('CurrentState') == desired_state:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False
    def stop_service(self, service_name: str, timeout: int = 30) -> bool:
        """API pública que mantiene compatibilidad booleana."""
        try:
            hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
            hs = win32serviceutil.SmartOpenService(hscm, service_name, win32service.SERVICE_STOP | win32service.SERVICE_QUERY_STATUS | win32service.SERVICE_ENUMERATE_DEPENDENTS)
            try:
                status = win32service.QueryServiceStatusEx(hs)
                current = status.get('CurrentState')
                if current == win32service.SERVICE_STOPPED:
                    return True
                self._stop_dependent_services(hs)
                self.action_history.append({'type': 'service', 'name': service_name, 'previous_state': ServiceState(current).name, 'timestamp': time.time()})
                win32service.ControlService(hs, win32service.SERVICE_CONTROL_STOP)
                ok = self._wait_service_state(hs, win32service.SERVICE_STOPPED, timeout)
                return ok
            finally:
                try:
                    win32service.CloseServiceHandle(hs)
                except Exception:
                    pass
                try:
                    win32service.CloseServiceHandle(hscm)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"stop_service error {service_name}: {e}")
            return False
    def _stop_service_record(self, service_name: str, timeout: int = 30) -> Tuple[bool, bool]:
        """Detiene un servicio y devuelve (stopped_ok, was_running)."""
        try:
            hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
            hs = win32serviceutil.SmartOpenService(hscm, service_name, win32service.SERVICE_STOP | win32service.SERVICE_QUERY_STATUS | win32service.SERVICE_ENUMERATE_DEPENDENTS)
            try:
                status = win32service.QueryServiceStatusEx(hs)
                current = status.get('CurrentState')
                was_running = (current != win32service.SERVICE_STOPPED)
                if not was_running:
                    return True, False
                self._stop_dependent_services(hs)
                self.action_history.append({'type': 'service', 'name': service_name, 'previous_state': ServiceState(current).name, 'timestamp': time.time()})
                win32service.ControlService(hs, win32service.SERVICE_CONTROL_STOP)
                ok = self._wait_service_state(hs, win32service.SERVICE_STOPPED, timeout)
                return ok, was_running and ok
            finally:
                try:
                    win32service.CloseServiceHandle(hs)
                except Exception:
                    pass
                try:
                    win32service.CloseServiceHandle(hscm)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"_stop_service_record error {service_name}: {e}")
            return False, False
    def start_service(self, service_name: str, timeout: int = 30) -> bool:
        try:
            hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
            hs = win32serviceutil.SmartOpenService(hscm, service_name, win32service.SERVICE_START | win32service.SERVICE_QUERY_STATUS)
            try:
                status = win32service.QueryServiceStatusEx(hs)
                current = status.get('CurrentState')
                if current == win32service.SERVICE_RUNNING:
                    return True
                win32service.StartService(hs, None)
                ok = self._wait_service_state(hs, win32service.SERVICE_RUNNING, timeout)
                return ok
            finally:
                try:
                    win32service.CloseServiceHandle(hs)
                except Exception:
                    pass
                try:
                    win32service.CloseServiceHandle(hscm)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"start_service error {service_name}: {e}")
            return False
    def _get_foreground_pid(self) -> Optional[int]:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return int(pid.value) if pid.value else None
        except Exception:
            return None
    def stop_process(self, process_name: str) -> List[int]:
        try:
            critical = {p.lower() for p in self.CRITICAL_PROCESSES}
            if process_name.lower() in critical:
                return []
            stopped_pids = []
            procs_to_wait: List[psutil.Process] = []
            current_user = getpass.getuser().lower()
            foreground_pid = self._get_foreground_pid()
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'username']):
                try:
                    if (proc.info['name'] or '').lower() == process_name.lower():
                        if foreground_pid and int(proc.info['pid']) == foreground_pid:
                            continue
                        uname = (proc.info.get('username') or '').lower()
                        if uname and current_user not in uname:
                            continue
                        pid = int(proc.info['pid'])
                        self.action_history.append({'type': 'process', 'name': process_name, 'pid': pid, 'exe_path': proc.info.get('exe', 'N/A'), 'timestamp': time.time()})
                        proc.terminate()
                        stopped_pids.append(pid)
                        procs_to_wait.append(proc)
                except Exception:
                    pass
            if procs_to_wait:
                gone, alive = psutil.wait_procs(procs_to_wait, timeout=3)
                for p in alive:
                    try:
                        p.kill()
                    except Exception:
                        pass
                return [p.pid for p in gone] + [p.pid for p in alive]
            return stopped_pids
        except Exception as e:
            logger.debug(f"stop_process error {process_name}: {e}")
            return []
    def start_gaming_session(self, game_exe: str, game_pid: int) -> bool:
        with self.session_lock:
            try:
                if self.active_session:
                    return False
                self.action_history.clear()
                stopped_services = []
                stopped_processes = []
                for service_name in self.services_to_stop:
                    try:
                        ok, was_running = self._stop_service_record(service_name)
                        if ok and was_running:
                            stopped_services.append(service_name)
                    except Exception:
                        pass
                for process_name in self.processes_to_stop:
                    try:
                        pids = self.stop_process(process_name)
                        stopped_processes.extend(pids)
                    except Exception:
                        pass
                self.active_session = SessionState(
                    game_exe=game_exe, game_pid=game_pid, start_time=time.time(),
                    stopped_services=stopped_services, stopped_processes=stopped_processes
                )
                return True
            except Exception as e:
                logger.debug(f"start_gaming_session error: {e}")
                try:
                    self._rollback_actions()
                except Exception:
                    pass
                return False
    def end_gaming_session(self) -> bool:
        with self.session_lock:
            try:
                if not self.active_session:
                    return False
                try:
                    for service_name in self.active_session.stopped_services:
                        try:
                            self.start_service(service_name)
                        except Exception:
                            pass
                except Exception:
                    pass
                self.active_session = None
                self.action_history.clear()
                return True
            except Exception:
                return False
    def _rollback_actions(self):
        try:
            for action in reversed(self.action_history):
                try:
                    if action['type'] == 'service' and action.get('previous_state') == 'RUNNING':
                        self.start_service(action['name'])
                except Exception:
                    pass
            self.action_history.clear()
        except Exception:
            pass
    def is_session_active(self) -> bool:
        try:
            return self.active_session is not None and self.active_session.active
        except Exception:
            return False
    def get_session_info(self) -> Optional[Dict]:
        try:
            if not self.active_session:
                return None
            return {
                'game_exe': self.active_session.game_exe,
                'game_pid': self.active_session.game_pid,
                'duration_seconds': time.time() - self.active_session.start_time,
                'stopped_services_count': len(self.active_session.stopped_services),
                'stopped_processes_count': len(self.active_session.stopped_processes),
                'active': self.active_session.active
            }
        except Exception:
            return None
    def show_configuration_ui(self):
        if not self.ui_lock.acquire(blocking=False):
            return
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox
            root = tk.Tk()
            root.title("Gaming Session Manager - Configuration")
            def on_closing():
                try:
                    self.ui_lock.release()
                except Exception:
                    pass
                try:
                    root.destroy()
                except Exception:
                    pass
            root.protocol("WM_DELETE_WINDOW", on_closing)
            def save_and_close():
                try:
                    self.services_to_stop = [s.strip() for s in services_textbox.get('1.0', tk.END).strip().split('\n') if s.strip()]
                    self.processes_to_stop = [p.strip() for p in processes_textbox.get('1.0', tk.END).strip().split('\n') if p.strip()]
                    self._validate_configuration()
                    self._save_configuration()
                    messagebox.showinfo("Success", "Configuration saved and validated successfully!")
                except Exception:
                    pass
                on_closing()
            main_frame = ttk.Frame(root, padding="10")
            main_frame.grid(row=0, column=0, sticky="nsew")
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)
            ttk.Label(main_frame, text="Services to Stop (one per line):").grid(row=0, column=0, sticky="w", pady=5)
            services_textbox = tk.Text(main_frame, width=40, height=15)
            services_textbox.grid(row=1, column=0, sticky="nsew")
            services_textbox.insert('1.0', '\n'.join(self.services_to_stop))
            ttk.Label(main_frame, text="Processes to Stop (one per line):").grid(row=0, column=1, sticky="w", pady=5, padx=5)
            processes_textbox = tk.Text(main_frame, width=40, height=15)
            processes_textbox.grid(row=1, column=1, sticky="nsew", padx=5)
            processes_textbox.insert('1.0', '\n'.join(self.processes_to_stop))
            main_frame.columnconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(1, weight=1)
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=2, column=0, columnspan=2, pady=10)
            ttk.Button(button_frame, text="Save & Close", command=save_and_close).pack(side="left", padx=5)
            ttk.Button(button_frame, text="Cancel", command=on_closing).pack(side="left", padx=5)
            root.mainloop()
        except Exception:
            try:
                self.ui_lock.release()
            except Exception:
                pass
    def start_mouse_listener(self):
        if self.mouse_listener_active:
            return
        try:
            from pynput import mouse
            self.mouse_listener_active = True
            middle_button_pressed_time = None
            def on_click(x, y, button, pressed):
                nonlocal middle_button_pressed_time
                try:
                    if button == mouse.Button.middle:
                        if pressed:
                            middle_button_pressed_time = time.time()
                        else:
                            if middle_button_pressed_time and (time.time() - middle_button_pressed_time >= 2.0):
                                threading.Thread(target=self.show_configuration_ui, daemon=True).start()
                            middle_button_pressed_time = None
                except Exception:
                    pass
            self._mouse_listener = mouse.Listener(on_click=on_click)
            self._mouse_listener.start()
        except Exception:
            pass
    def stop_mouse_listener(self):
        if not self.mouse_listener_active:
            return
        self.mouse_listener_active = False
        try:
            if self._mouse_listener:
                self._mouse_listener.stop()
                self._mouse_listener = None
        except Exception:
            pass
    def cleanup(self):
        try:
            if self.is_session_active():
                self.end_gaming_session()
        except Exception:
            pass
        try:
            self.stop_mouse_listener()
        except Exception:
            pass