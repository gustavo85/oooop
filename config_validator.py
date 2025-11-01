"""
Configuration Validator V4.0
Validates configuration files and values to prevent runtime errors
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import fields
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates game profiles and global configuration"""
    
    # Valid ranges for numeric settings
    VALID_RANGES = {
        'timer_resolution_ms': (0.5, 2.0),
        'memory_optimization_level': (0, 2),
        'network_dscp_value': (0, 63),
        'background_throttle_cpu_percent': (0.1, 100.0),
        'background_throttle_memory_mb': (10, 10000),
        'telemetry_export_interval_minutes': (1, 1440),
    }
    
    # Valid string values
    VALID_VALUES = {
        'cpu_priority_class': ['NORMAL', 'ABOVE_NORMAL', 'HIGH', 'REALTIME'],
        'process_io_priority': ['NORMAL', 'HIGH', 'LOW'],
        'log_level': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    }
    
    @staticmethod
    def validate_profile(profile: Any) -> Tuple[bool, List[str]]:
        """
        Validate a game profile configuration.
        
        Args:
            profile: GameProfile instance to validate
            
        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        errors = []
        
        # Validate name and game_exe
        if not profile.name or len(profile.name.strip()) == 0:
            errors.append("Profile name cannot be empty")
        
        if not profile.game_exe or len(profile.game_exe.strip()) == 0:
            errors.append("Game executable name cannot be empty")
        elif not profile.game_exe.endswith('.exe'):
            errors.append(f"Game executable must end with '.exe': {profile.game_exe}")
        
        # Validate numeric ranges
        for field_name, (min_val, max_val) in ConfigValidator.VALID_RANGES.items():
            if hasattr(profile, field_name):
                value = getattr(profile, field_name)
                if value is not None and not (min_val <= value <= max_val):
                    errors.append(
                        f"{field_name} must be between {min_val} and {max_val}, got {value}"
                    )
        
        # Validate string values
        for field_name, valid_values in ConfigValidator.VALID_VALUES.items():
            if hasattr(profile, field_name):
                value = getattr(profile, field_name)
                if value and value.upper() not in [v.upper() for v in valid_values]:
                    errors.append(
                        f"{field_name} must be one of {valid_values}, got '{value}'"
                    )
        
        # Validate boolean fields
        bool_fields = [
            'optimize_working_set', 'network_qos_enabled', 'gpu_scheduling_enabled',
            'gpu_clock_locking', 'directx_optimizations', 'power_high_performance',
            'stop_services', 'stop_processes', 'cpu_affinity_enabled',
            'disable_core_parking', 'enable_frame_time_analysis', 'enable_telemetry',
            'ml_auto_tune_enabled', 'disable_nagle', 'tcp_buffer_tuning', 'nic_rss_auto'
        ]
        
        for field_name in bool_fields:
            if hasattr(profile, field_name):
                value = getattr(profile, field_name)
                if value is not None and not isinstance(value, bool):
                    errors.append(f"{field_name} must be boolean (true/false), got {type(value)}")
        
        # Validate QoS rules if present
        if hasattr(profile, 'qos_rules') and profile.qos_rules:
            for i, rule in enumerate(profile.qos_rules):
                if not hasattr(rule, 'name') or not rule.name:
                    errors.append(f"QoS rule {i} must have a name")
                
                if hasattr(rule, 'dscp') and not (0 <= rule.dscp <= 63):
                    errors.append(f"QoS rule {i} DSCP must be between 0 and 63, got {rule.dscp}")
                
                if hasattr(rule, 'protocol') and rule.protocol not in ['tcp', 'udp', 'both']:
                    errors.append(f"QoS rule {i} protocol must be 'tcp', 'udp', or 'both'")
        
        # Validate NIC RSS settings
        if hasattr(profile, 'nic_rss_max_processors') and profile.nic_rss_max_processors is not None:
            if not (1 <= profile.nic_rss_max_processors <= 64):
                errors.append(f"nic_rss_max_processors must be between 1 and 64")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Profile validation failed for '{profile.name}': {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return is_valid, errors
    
    @staticmethod
    def validate_global_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate global configuration.
        
        Args:
            config: Global configuration dictionary
            
        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        errors = []
        
        # Validate boolean settings
        bool_settings = ['auto_detect_games', 'start_minimized', 'enable_telemetry']
        for setting in bool_settings:
            if setting in config and not isinstance(config[setting], bool):
                errors.append(f"{setting} must be boolean, got {type(config[setting])}")
        
        # Validate numeric ranges
        for setting, (min_val, max_val) in ConfigValidator.VALID_RANGES.items():
            if setting in config:
                value = config[setting]
                if not isinstance(value, (int, float)):
                    errors.append(f"{setting} must be numeric, got {type(value)}")
                elif not (min_val <= value <= max_val):
                    errors.append(
                        f"{setting} must be between {min_val} and {max_val}, got {value}"
                    )
        
        # Validate log level
        if 'log_level' in config:
            valid_levels = ConfigValidator.VALID_VALUES['log_level']
            if config['log_level'].upper() not in valid_levels:
                errors.append(f"log_level must be one of {valid_levels}")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Global config validation failed: {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return is_valid, errors
    
    @staticmethod
    def validate_config_file(config_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate configuration file exists and is readable.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        errors = []
        
        if not config_path.exists():
            errors.append(f"Configuration file does not exist: {config_path}")
            return False, errors
        
        if not config_path.is_file():
            errors.append(f"Configuration path is not a file: {config_path}")
            return False, errors
        
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict):
                errors.append("Configuration file must contain a JSON object")
            else:
                if 'version' not in data:
                    errors.append("Configuration file missing 'version' field")
                
                if 'global' in data and not isinstance(data['global'], dict):
                    errors.append("'global' section must be a dictionary")
                
                if 'game_profiles' in data and not isinstance(data['game_profiles'], dict):
                    errors.append("'game_profiles' section must be a dictionary")
        
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            errors.append(f"Error reading configuration file: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def sanitize_profile(profile: Any) -> Any:
        """
        Sanitize profile values to ensure they're within valid ranges.
        Modifies profile in-place and returns it.
        
        Args:
            profile: GameProfile instance to sanitize
            
        Returns:
            Sanitized profile
        """
        # Clamp numeric values to valid ranges
        for field_name, (min_val, max_val) in ConfigValidator.VALID_RANGES.items():
            if hasattr(profile, field_name):
                value = getattr(profile, field_name)
                if value is not None:
                    clamped = max(min_val, min(max_val, value))
                    if clamped != value:
                        logger.info(f"Clamping {field_name} from {value} to {clamped}")
                        setattr(profile, field_name, clamped)
        
        # Force valid string values
        for field_name, valid_values in ConfigValidator.VALID_VALUES.items():
            if hasattr(profile, field_name):
                value = getattr(profile, field_name)
                if value and value.upper() not in [v.upper() for v in valid_values]:
                    default = valid_values[0]
                    logger.warning(f"Invalid {field_name} '{value}', using default '{default}'")
                    setattr(profile, field_name, default)
        
        # Ensure game_exe ends with .exe
        if hasattr(profile, 'game_exe') and profile.game_exe:
            if not profile.game_exe.lower().endswith('.exe'):
                profile.game_exe = f"{profile.game_exe}.exe"
                logger.info(f"Added .exe extension to game_exe: {profile.game_exe}")
        
        return profile
    
    @staticmethod
    def get_validation_report(profile: Any) -> str:
        """
        Get a human-readable validation report for a profile.
        
        Args:
            profile: GameProfile instance to validate
            
        Returns:
            Formatted validation report string
        """
        is_valid, errors = ConfigValidator.validate_profile(profile)
        
        report = []
        report.append(f"Validation Report for Profile: {profile.name}")
        report.append("=" * 60)
        
        if is_valid:
            report.append("✅ Profile is valid - no errors found")
        else:
            report.append(f"❌ Profile has {len(errors)} error(s):")
            for i, error in enumerate(errors, 1):
                report.append(f"  {i}. {error}")
        
        report.append("")
        report.append("Configuration Summary:")
        report.append(f"  Name: {profile.name}")
        report.append(f"  Game: {profile.game_exe}")
        report.append(f"  Timer Resolution: {profile.timer_resolution_ms}ms")
        report.append(f"  Memory Level: {profile.memory_optimization_level}")
        report.append(f"  CPU Priority: {profile.cpu_priority_class}")
        report.append(f"  GPU Clock Lock: {profile.gpu_clock_locking}")
        report.append(f"  Network QoS: {profile.network_qos_enabled}")
        
        return "\n".join(report)


def validate_and_fix_config(config_manager) -> bool:
    """
    Validate all profiles in a configuration manager and attempt to fix issues.
    
    Args:
        config_manager: ConfigurationManager instance
        
    Returns:
        True if all profiles are valid (after fixes), False otherwise
    """
    all_valid = True
    
    # Validate global config
    is_valid, errors = ConfigValidator.validate_global_config(config_manager.global_config)
    if not is_valid:
        logger.error("Global configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        all_valid = False
    
    # Validate and sanitize each profile
    for game_exe, profile in config_manager.game_profiles.items():
        # First try to sanitize
        profile = ConfigValidator.sanitize_profile(profile)
        config_manager.game_profiles[game_exe] = profile
        
        # Then validate
        is_valid, errors = ConfigValidator.validate_profile(profile)
        if not is_valid:
            logger.error(f"Profile '{profile.name}' validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            all_valid = False
    
    # Save if any fixes were applied
    if all_valid:
        config_manager.save_configuration()
        logger.info("✅ All configurations validated and saved")
    
    return all_valid


if __name__ == "__main__":
    # Example usage
    from config_loader import ConfigurationManager, GameProfile
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config_manager = ConfigurationManager()
    
    # Validate all profiles
    print("\nValidating all profiles...")
    all_valid = validate_and_fix_config(config_manager)
    
    if all_valid:
        print("\n✅ All profiles are valid!")
    else:
        print("\n❌ Some profiles have errors. Check logs for details.")
    
    # Example: Validate a specific profile
    if config_manager.game_profiles:
        profile = list(config_manager.game_profiles.values())[0]
        print("\n" + ConfigValidator.get_validation_report(profile))
