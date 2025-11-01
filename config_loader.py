"""
Configuration management V3.5
ADDED: Fields for working set, core parking, GPU clocking, telemetry
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import time

logger = logging.getLogger(__name__)

@dataclass
class QoSRule:
    name: str
    protocol: str = 'both'
    dscp: int = 46
    localport: Any = 'any'
    remoteport: Any = 'any'
    localip: Optional[str] = None
    remoteip: Optional[str] = None

@dataclass
class GameProfile:
    name: str
    game_exe: str
    
    # Timer
    timer_resolution_ms: float = 0.5
    
    # Memory
    memory_optimization_level: int = 2
    optimize_working_set: bool = True  # RENAMED from large_pages_enabled
    
    # Network
    network_qos_enabled: bool = True
    network_dscp_value: int = 46
    qos_rules: List[QoSRule] = field(default_factory=list)
    nic_rss_auto: bool = True
    nic_rss_max_processors: Optional[int] = None
    disable_nagle: bool = False  # NEW: Disable Nagle's algorithm for lower latency
    tcp_buffer_tuning: bool = False  # NEW: Tune TCP buffer sizes
    
    # GPU
    gpu_scheduling_enabled: bool = True
    gpu_clock_locking: bool = True  # NEW
    directx_optimizations: bool = True
    
    # Power
    power_high_performance: bool = True
    
    # System
    stop_services: bool = True
    stop_processes: bool = True
    
    # CPU
    cpu_affinity_enabled: bool = True
    cpu_priority_class: str = 'HIGH'
    process_io_priority: str = 'NORMAL'
    disable_core_parking: bool = True  # NEW
    
    # Monitoring
    enable_frame_time_analysis: bool = True
    enable_telemetry: bool = True  # NEW
    
    # ML
    ml_auto_tune_enabled: bool = False  # NEW


class ConfigurationManager:
    CONFIG_VERSION = "3.5"
    
    def __init__(self, config_file: Optional[str] = None, validate: bool = True):
        self.config_dir = Path.home() / '.game_optimizer'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = Path(config_file) if config_file else self.config_dir / 'config.json'
        self.global_config: Dict[str, Any] = {}
        self.game_profiles: Dict[str, GameProfile] = {}
        self.validate_on_load = validate
        
        self._load_configuration()
    
    def _load_configuration(self):
        try:
            if self.config_file.exists():
                # Validate config file structure if validation enabled
                if self.validate_on_load:
                    try:
                        from config_validator import ConfigValidator
                        is_valid, errors = ConfigValidator.validate_config_file(self.config_file)
                        if not is_valid:
                            logger.warning(f"Configuration file has validation errors: {errors}")
                    except ImportError:
                        logger.debug("config_validator not available, skipping file validation")
                
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                config_version = data.get('version', '1.0')
                if config_version != self.CONFIG_VERSION:
                    data = self._migrate_config(data, config_version)
                    self._save_migrated_data(data)
                
                self.global_config = data.get('global', {})
                
                # Validate global config if enabled
                if self.validate_on_load:
                    try:
                        from config_validator import ConfigValidator
                        is_valid, errors = ConfigValidator.validate_global_config(self.global_config)
                        if not is_valid:
                            logger.warning(f"Global config validation errors: {errors}")
                    except ImportError:
                        pass
                
                profiles_data = data.get('game_profiles', {})
                
                for game_exe, profile_data in profiles_data.items():
                    try:
                        # Convert QoS rules
                        if 'qos_rules' in profile_data and profile_data['qos_rules']:
                            profile_data['qos_rules'] = [QoSRule(**rule) for rule in profile_data['qos_rules']]
                        
                        profile = GameProfile(**profile_data)
                        
                        # Validate and sanitize profile if enabled
                        if self.validate_on_load:
                            try:
                                from config_validator import ConfigValidator
                                profile = ConfigValidator.sanitize_profile(profile)
                                is_valid, errors = ConfigValidator.validate_profile(profile)
                                if not is_valid:
                                    logger.warning(f"Profile '{profile.name}' validation errors: {errors}")
                            except ImportError:
                                pass
                        
                        self.game_profiles[game_exe.lower()] = profile
                        
                    except Exception as e:
                        logger.warning(f"Failed to load profile for {game_exe}: {e}")
            else:
                self._set_default_configuration()
                self.save_configuration()
                
        except Exception as e:
            logger.error(f"Config load error: {e}")
            self._set_default_configuration()
            self.save_configuration()
    
    def _migrate_config(self, data: Dict, from_version: str) -> Dict:
        """Migrate old config to V3.5"""
        try:
            default = GameProfile(name="", game_exe="")
            
            for profile_data in data.get('game_profiles', {}).values():
                # Rename large_pages_enabled → optimize_working_set
                if 'large_pages_enabled' in profile_data:
                    profile_data['optimize_working_set'] = profile_data.pop('large_pages_enabled')
                
                # Add new fields with defaults
                for key, value in asdict(default).items():
                    profile_data.setdefault(key, value)
            
            # Global defaults
            data.setdefault('global', {})
            data['global'].setdefault('auto_detect_games', True)
            data['global'].setdefault('start_minimized', False)
            data['global'].setdefault('log_level', 'INFO')
            data['global'].setdefault('enable_telemetry', True)
            data['global'].setdefault('background_throttle_cpu_percent', 3.0)
            data['global'].setdefault('background_throttle_memory_mb', 200)
            data['global'].setdefault('telemetry_export_interval_minutes', 60)
            
            data['version'] = self.CONFIG_VERSION
            
            logger.info(f"Migrated config from {from_version} to {self.CONFIG_VERSION}")
            return data
            
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return data
    
    def _set_default_configuration(self):
        self.global_config = {
            'auto_detect_games': True,
            'start_minimized': False,
            'log_level': 'INFO',
            'enable_telemetry': True,
            'background_throttle_cpu_percent': 3.0,
            'background_throttle_memory_mb': 200,
            'telemetry_export_interval_minutes': 60,
        }
        
        self._create_default_profiles()
    
    def _create_default_profiles(self):
        self.game_profiles = {}
        
        # Competitive FPS
        competitive = GameProfile(
            name="Competitive FPS",
            game_exe="csgo.exe",
            timer_resolution_ms=0.5,
            memory_optimization_level=2,
            network_qos_enabled=True,
            network_dscp_value=46,
            cpu_priority_class='HIGH',
            process_io_priority='HIGH',
            optimize_working_set=True,
            gpu_clock_locking=True,
            disable_core_parking=True,
        )
        self.game_profiles[competitive.game_exe.lower()] = competitive
        
        # AAA Single-player
        singleplayer = GameProfile(
            name="AAA Single Player",
            game_exe="cyberpunk2077.exe",
            timer_resolution_ms=1.0,
            memory_optimization_level=1,
            network_qos_enabled=False,
            cpu_priority_class='ABOVE_NORMAL',
            process_io_priority='NORMAL',
            optimize_working_set=True,
            gpu_clock_locking=False,
            disable_core_parking=False,
        )
        self.game_profiles[singleplayer.game_exe.lower()] = singleplayer
    
    def save_configuration(self):
        try:
            data = {
                'version': self.CONFIG_VERSION,
                'global': self.global_config,
                'game_profiles': {}
            }
            
            for game_exe, profile in self.game_profiles.items():
                profile_dict = asdict(profile)
                profile_dict['qos_rules'] = [asdict(rule) for rule in profile.qos_rules]
                data['game_profiles'][game_exe] = profile_dict
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Config save error: {e}")
    
    def _save_migrated_data(self, data: Dict):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Migrated data save error: {e}")
    
    def get_game_profile(self, game_exe: str) -> Optional[GameProfile]:
        game_exe_lower = game_exe.lower()
        
        if game_exe_lower in self.game_profiles:
            return self.game_profiles[game_exe_lower]
        
        # Partial match
        for key, profile in self.game_profiles.items():
            if key in game_exe_lower:
                return profile
        
        return None
    
    def create_game_profile(self, profile: GameProfile) -> bool:
        try:
            # Validate profile before saving
            if self.validate_on_load:
                try:
                    from config_validator import ConfigValidator
                    profile = ConfigValidator.sanitize_profile(profile)
                    is_valid, errors = ConfigValidator.validate_profile(profile)
                    if not is_valid:
                        logger.error(f"Profile validation failed: {errors}")
                        return False
                except ImportError:
                    logger.debug("config_validator not available, skipping validation")
            
            self.game_profiles[profile.game_exe.lower()] = profile
            self.save_configuration()
            return True
        except Exception as e:
            logger.error(f"Failed to create profile: {e}")
            return False
    
    def delete_game_profile(self, game_exe: str) -> bool:
        try:
            game_exe_lower = game_exe.lower()
            if game_exe_lower in self.game_profiles:
                del self.game_profiles[game_exe_lower]
                self.save_configuration()
                return True
            return False
        except Exception:
            return False
    
    def get_global_setting(self, key: str, default: Any = None) -> Any:
        return self.global_config.get(key, default)
    
    def set_global_setting(self, key: str, value: Any):
        self.global_config[key] = value
        self.save_configuration()