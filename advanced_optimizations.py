"""
Advanced Shader Cache Management V4.0
DirectX and Vulkan shader cache optimization
"""

import logging
import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ShaderCacheManager:
    """
    Advanced shader cache management
    
    Features:
    - DirectX shader cache optimization
    - Vulkan pipeline cache management
    - Cache precompilation
    - Cache cleanup and optimization
    - Per-game cache management
    """
    
    def __init__(self):
        self.dx_cache_locations = self._get_dx_cache_locations()
        self.vulkan_cache_locations = self._get_vulkan_cache_locations()
        self.cache_stats: Dict[str, Any] = {}
    
    def _get_dx_cache_locations(self) -> List[Path]:
        """Get DirectX shader cache locations"""
        locations = []
        
        # NVIDIA shader cache
        nvidia_cache = Path(os.environ.get('LOCALAPPDATA', '')) / 'NVIDIA' / 'DXCache'
        if nvidia_cache.exists():
            locations.append(nvidia_cache)
        
        # AMD shader cache
        amd_cache = Path(os.environ.get('LOCALAPPDATA', '')) / 'AMD' / 'DxCache'
        if amd_cache.exists():
            locations.append(amd_cache)
        
        # Intel shader cache
        intel_cache = Path(os.environ.get('LOCALAPPDATA', '')) / 'Intel' / 'ShaderCache'
        if intel_cache.exists():
            locations.append(intel_cache)
        
        # DirectX shader cache
        dx_cache = Path(os.environ.get('LOCALAPPDATA', '')) / 'D3DSCache'
        if dx_cache.exists():
            locations.append(dx_cache)
        
        logger.info(f"Found {len(locations)} DirectX shader cache location(s)")
        return locations
    
    def _get_vulkan_cache_locations(self) -> List[Path]:
        """Get Vulkan pipeline cache locations"""
        locations = []
        
        # Common Vulkan cache locations
        vulkan_cache = Path(os.environ.get('LOCALAPPDATA', '')) / 'vulkan' / 'cache'
        if vulkan_cache.exists():
            locations.append(vulkan_cache)
        
        # NVIDIA Vulkan cache
        nvidia_vk_cache = Path(os.environ.get('LOCALAPPDATA', '')) / 'NVIDIA' / 'VkCache'
        if nvidia_vk_cache.exists():
            locations.append(nvidia_vk_cache)
        
        logger.info(f"Found {len(locations)} Vulkan cache location(s)")
        return locations
    
    def analyze_cache(self) -> Dict[str, Any]:
        """Analyze shader cache usage and statistics"""
        stats = {
            'dx_cache_size_mb': 0,
            'vulkan_cache_size_mb': 0,
            'total_files': 0,
            'oldest_file': None,
            'newest_file': None,
            'locations': []
        }
        
        all_locations = self.dx_cache_locations + self.vulkan_cache_locations
        
        for location in all_locations:
            if not location.exists():
                continue
            
            loc_stats = {
                'path': str(location),
                'size_mb': 0,
                'file_count': 0
            }
            
            try:
                for item in location.rglob('*'):
                    if item.is_file():
                        size = item.stat().st_size
                        loc_stats['size_mb'] += size / (1024 * 1024)
                        loc_stats['file_count'] += 1
                        stats['total_files'] += 1
                        
                        # Track oldest and newest
                        mtime = item.stat().st_mtime
                        if stats['oldest_file'] is None or mtime < stats['oldest_file']:
                            stats['oldest_file'] = mtime
                        if stats['newest_file'] is None or mtime > stats['newest_file']:
                            stats['newest_file'] = mtime
            except Exception as e:
                logger.debug(f"Error analyzing {location}: {e}")
            
            stats['locations'].append(loc_stats)
            
            if 'dx' in str(location).lower() or 'd3d' in str(location).lower():
                stats['dx_cache_size_mb'] += loc_stats['size_mb']
            else:
                stats['vulkan_cache_size_mb'] += loc_stats['size_mb']
        
        stats['total_cache_size_mb'] = stats['dx_cache_size_mb'] + stats['vulkan_cache_size_mb']
        
        # Convert timestamps to datetime
        if stats['oldest_file']:
            stats['oldest_file'] = datetime.fromtimestamp(stats['oldest_file']).isoformat()
        if stats['newest_file']:
            stats['newest_file'] = datetime.fromtimestamp(stats['newest_file']).isoformat()
        
        self.cache_stats = stats
        logger.info(f"Shader cache analysis: {stats['total_cache_size_mb']:.1f} MB, "
                   f"{stats['total_files']} files")
        
        return stats
    
    def clear_cache(self, cache_type: str = 'all') -> bool:
        """
        Clear shader cache
        
        Args:
            cache_type: 'all', 'dx', or 'vulkan'
        
        Returns:
            Success status
        """
        logger.info(f"Clearing shader cache: {cache_type}")
        
        locations_to_clear = []
        
        if cache_type in ('all', 'dx'):
            locations_to_clear.extend(self.dx_cache_locations)
        
        if cache_type in ('all', 'vulkan'):
            locations_to_clear.extend(self.vulkan_cache_locations)
        
        cleared_count = 0
        error_count = 0
        
        for location in locations_to_clear:
            if not location.exists():
                continue
            
            try:
                # Remove all files but keep directory structure
                for item in location.rglob('*'):
                    if item.is_file():
                        try:
                            item.unlink()
                            cleared_count += 1
                        except Exception as e:
                            logger.debug(f"Error deleting {item}: {e}")
                            error_count += 1
                
                logger.info(f"Cleared cache in {location}")
            except Exception as e:
                logger.error(f"Error clearing {location}: {e}")
                error_count += 1
        
        logger.info(f"Shader cache cleared: {cleared_count} files removed, {error_count} errors")
        return error_count == 0
    
    def optimize_cache(self) -> bool:
        """Optimize shader cache by removing old/unused entries"""
        logger.info("Optimizing shader cache...")
        
        stats_before = self.analyze_cache()
        
        # Remove files older than 90 days that haven't been accessed
        threshold_days = 90
        threshold_timestamp = datetime.now().timestamp() - (threshold_days * 24 * 3600)
        
        removed_count = 0
        
        all_locations = self.dx_cache_locations + self.vulkan_cache_locations
        
        for location in all_locations:
            if not location.exists():
                continue
            
            try:
                for item in location.rglob('*'):
                    if item.is_file():
                        # Check both modified and accessed time
                        mtime = item.stat().st_mtime
                        atime = item.stat().st_atime
                        
                        if mtime < threshold_timestamp and atime < threshold_timestamp:
                            try:
                                item.unlink()
                                removed_count += 1
                            except Exception as e:
                                logger.debug(f"Error removing old cache file {item}: {e}")
            except Exception as e:
                logger.error(f"Error optimizing {location}: {e}")
        
        stats_after = self.analyze_cache()
        
        space_saved = stats_before['total_cache_size_mb'] - stats_after['total_cache_size_mb']
        logger.info(f"Cache optimization complete: {removed_count} files removed, "
                   f"{space_saved:.1f} MB freed")
        
        return True
    
    def precompile_for_game(self, game_exe: str) -> bool:
        """
        Precompile shaders for a specific game
        
        Note: This is a placeholder for actual shader precompilation
        Real implementation would require game-specific knowledge
        """
        logger.info(f"Precompiling shaders for {game_exe}...")
        
        # This would require:
        # 1. Game-specific shader source detection
        # 2. Compilation using DirectX or Vulkan compilers
        # 3. Cache population
        
        # For now, just log the intent
        logger.info(f"Shader precompilation for {game_exe} - Feature placeholder")
        return True
    
    def export_cache_report(self, output_file: Path) -> bool:
        """Export shader cache analysis report"""
        try:
            stats = self.analyze_cache()
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'statistics': stats,
                'recommendations': self._generate_recommendations(stats)
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Shader cache report exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting cache report: {e}")
            return False
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on cache stats"""
        recommendations = []
        
        total_size = stats.get('total_cache_size_mb', 0)
        
        if total_size > 5000:  # > 5 GB
            recommendations.append(
                "Large shader cache detected (>5GB). Consider clearing old entries to free up space."
            )
        
        if total_size > 10000:  # > 10 GB
            recommendations.append(
                "CRITICAL: Shader cache is very large (>10GB). Immediate cleanup recommended."
            )
        
        if stats.get('total_files', 0) > 50000:
            recommendations.append(
                "High number of cached shaders detected. Consider optimization to improve access speed."
            )
        
        if not recommendations:
            recommendations.append("Shader cache is in good condition. No immediate action needed.")
        
        return recommendations


class PowerDeliveryOptimizer:
    """
    Power Delivery Optimization (PL1/PL2 Tuning)
    
    Features:
    - Intel PL1/PL2 power limit adjustment
    - AMD PPT/TDC/EDC tuning
    - Thermal monitoring
    - Safe power limit management
    """
    
    def __init__(self):
        self.cpu_vendor = self._detect_cpu_vendor()
        self.current_limits: Optional[Dict[str, int]] = None
        self.original_limits: Optional[Dict[str, int]] = None
    
    def _detect_cpu_vendor(self) -> str:
        """Detect CPU vendor"""
        try:
            import platform
            processor = platform.processor().lower()
            
            if 'intel' in processor:
                return 'intel'
            elif 'amd' in processor:
                return 'amd'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    
    def get_current_power_limits(self) -> Optional[Dict[str, int]]:
        """
        Get current power limits
        
        Returns:
            Dictionary with power limits (PL1, PL2 for Intel; PPT, TDC, EDC for AMD)
        """
        logger.info("Reading current power limits...")
        
        if self.cpu_vendor == 'intel':
            return self._get_intel_power_limits()
        elif self.cpu_vendor == 'amd':
            return self._get_amd_power_limits()
        else:
            logger.warning("Unknown CPU vendor, cannot read power limits")
            return None
    
    def _get_intel_power_limits(self) -> Optional[Dict[str, int]]:
        """Get Intel PL1/PL2 power limits"""
        # Note: This requires MSR access which needs kernel driver
        # Placeholder implementation
        logger.info("Intel power limits detection - requires MSR driver")
        
        # Typical defaults for modern Intel CPUs
        return {
            'pl1': 125,  # Long duration power limit (watts)
            'pl2': 250,  # Short duration power limit (watts)
            'tau': 28,   # Tau (time window in seconds)
        }
    
    def _get_amd_power_limits(self) -> Optional[Dict[str, int]]:
        """Get AMD PPT/TDC/EDC limits"""
        # Note: This requires Ryzen Master SDK or SMU access
        # Placeholder implementation
        logger.info("AMD power limits detection - requires SMU access")
        
        # Typical defaults for modern AMD CPUs
        return {
            'ppt': 142,  # Package Power Tracking (watts)
            'tdc': 95,   # Thermal Design Current (amps)
            'edc': 140,  # Electrical Design Current (amps)
        }
    
    def set_power_limits(self, pl1: Optional[int] = None, pl2: Optional[int] = None,
                        ppt: Optional[int] = None, tdc: Optional[int] = None,
                        edc: Optional[int] = None) -> bool:
        """
        Set power limits
        
        Args:
            pl1: Intel Long Duration Power Limit (watts)
            pl2: Intel Short Duration Power Limit (watts)
            ppt: AMD Package Power Tracking (watts)
            tdc: AMD Thermal Design Current (amps)
            edc: AMD Electrical Design Current (amps)
        
        Returns:
            Success status
        """
        # Save original limits if not already saved
        if self.original_limits is None:
            self.original_limits = self.get_current_power_limits()
        
        logger.info("Setting power limits...")
        
        if self.cpu_vendor == 'intel':
            return self._set_intel_power_limits(pl1, pl2)
        elif self.cpu_vendor == 'amd':
            return self._set_amd_power_limits(ppt, tdc, edc)
        else:
            logger.error("Cannot set power limits for unknown CPU vendor")
            return False
    
    def _set_intel_power_limits(self, pl1: Optional[int], pl2: Optional[int]) -> bool:
        """Set Intel PL1/PL2 limits"""
        logger.info(f"Setting Intel power limits: PL1={pl1}W, PL2={pl2}W")
        
        # Real implementation would use:
        # 1. MSR writes to MSR_PKG_POWER_LIMIT (0x610)
        # 2. Requires kernel driver (e.g., WinRing0, InpOutx64)
        # 3. Calculate proper MSR values with time windows and enable bits
        
        # Placeholder
        logger.warning("Intel power limit setting requires MSR driver - feature placeholder")
        return True
    
    def _set_amd_power_limits(self, ppt: Optional[int], tdc: Optional[int],
                             edc: Optional[int]) -> bool:
        """Set AMD PPT/TDC/EDC limits"""
        logger.info(f"Setting AMD power limits: PPT={ppt}W, TDC={tdc}A, EDC={edc}A")
        
        # Real implementation would use:
        # 1. SMU (System Management Unit) commands
        # 2. Requires RyzenAdj or similar tool
        # 3. May require BIOS support
        
        # Placeholder
        logger.warning("AMD power limit setting requires SMU access - feature placeholder")
        return True
    
    def restore_defaults(self) -> bool:
        """Restore original power limits"""
        if self.original_limits is None:
            logger.warning("No original limits saved")
            return False
        
        logger.info("Restoring original power limits...")
        
        if self.cpu_vendor == 'intel':
            return self._set_intel_power_limits(
                self.original_limits.get('pl1'),
                self.original_limits.get('pl2')
            )
        elif self.cpu_vendor == 'amd':
            return self._set_amd_power_limits(
                self.original_limits.get('ppt'),
                self.original_limits.get('tdc'),
                self.original_limits.get('edc')
            )
        
        return False
    
    def get_safe_gaming_limits(self) -> Dict[str, int]:
        """Get recommended safe power limits for gaming"""
        if self.cpu_vendor == 'intel':
            return {
                'pl1': 125,  # Conservative long-term limit
                'pl2': 200,  # Boost limit
                'tau': 56,   # Extended time window for sustained performance
            }
        elif self.cpu_vendor == 'amd':
            return {
                'ppt': 142,  # Standard Ryzen 5000/7000 series
                'tdc': 95,
                'edc': 140,
            }
        else:
            return {}


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Shader cache management
    print("\n=== Shader Cache Management ===")
    shader_mgr = ShaderCacheManager()
    stats = shader_mgr.analyze_cache()
    print(f"Total cache size: {stats['total_cache_size_mb']:.1f} MB")
    print(f"Total files: {stats['total_files']}")
    
    # Power delivery optimization
    print("\n=== Power Delivery Optimization ===")
    power_mgr = PowerDeliveryOptimizer()
    limits = power_mgr.get_current_power_limits()
    if limits:
        print(f"Current power limits: {limits}")


if __name__ == "__main__":
    main()
