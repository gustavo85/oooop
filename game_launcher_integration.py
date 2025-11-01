"""
Game Launcher Integration V4.0
Supports Steam, Epic Games Store, GOG Galaxy, and other launchers
"""

import logging
import json
import winreg
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import subprocess
import os

logger = logging.getLogger(__name__)

try:
    import vdf
    VDF_AVAILABLE = True
except ImportError:
    VDF_AVAILABLE = False
    logger.warning("vdf not available. Steam library parsing limited. Install with: pip install vdf")


@dataclass
class GameInfo:
    """Information about an installed game"""
    name: str
    app_id: str  # Launcher-specific ID
    install_path: Path
    executable: str
    launcher: str  # steam, epic, gog, etc.
    is_installed: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SteamIntegration:
    """Steam launcher integration"""
    
    def __init__(self):
        self.steam_path: Optional[Path] = None
        self.library_folders: List[Path] = []
        self.installed_games: Dict[str, GameInfo] = {}
        
        self._detect_steam()
    
    def _detect_steam(self):
        """Detect Steam installation"""
        try:
            # Try to find Steam via registry
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SOFTWARE\WOW6432Node\Valve\Steam") as key:
                install_path = winreg.QueryValueEx(key, "InstallPath")[0]
                self.steam_path = Path(install_path)
                logger.info(f"✓ Steam detected at: {self.steam_path}")
                
                # Find library folders
                self._find_library_folders()
                return True
        except (WindowsError, FileNotFoundError):
            # Try alternative registry path
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                  r"Software\Valve\Steam") as key:
                    install_path = winreg.QueryValueEx(key, "SteamPath")[0]
                    self.steam_path = Path(install_path)
                    logger.info(f"✓ Steam detected at: {self.steam_path}")
                    self._find_library_folders()
                    return True
            except (WindowsError, FileNotFoundError):
                logger.info("Steam not detected")
                return False
    
    def _find_library_folders(self):
        """Find all Steam library folders"""
        if not self.steam_path:
            return
        
        # Add main Steam folder
        self.library_folders.append(self.steam_path)
        
        # Parse libraryfolders.vdf
        library_vdf = self.steam_path / "steamapps" / "libraryfolders.vdf"
        
        if library_vdf.exists() and VDF_AVAILABLE:
            try:
                with open(library_vdf, 'r', encoding='utf-8') as f:
                    data = vdf.load(f)
                
                # Parse library folders
                if 'libraryfolders' in data:
                    for key, value in data['libraryfolders'].items():
                        if isinstance(value, dict) and 'path' in value:
                            lib_path = Path(value['path'])
                            if lib_path.exists():
                                self.library_folders.append(lib_path)
                                logger.debug(f"Found Steam library: {lib_path}")
            except Exception as e:
                logger.debug(f"Error parsing libraryfolders.vdf: {e}")
        
        logger.info(f"Found {len(self.library_folders)} Steam library folder(s)")
    
    def scan_installed_games(self) -> Dict[str, GameInfo]:
        """Scan for installed Steam games"""
        self.installed_games = {}
        
        for library in self.library_folders:
            steamapps = library / "steamapps"
            if not steamapps.exists():
                continue
            
            # Scan .acf manifest files
            for acf_file in steamapps.glob("appmanifest_*.acf"):
                try:
                    game_info = self._parse_acf_manifest(acf_file, library)
                    if game_info:
                        self.installed_games[game_info.app_id] = game_info
                except Exception as e:
                    logger.debug(f"Error parsing {acf_file}: {e}")
        
        logger.info(f"Found {len(self.installed_games)} installed Steam games")
        return self.installed_games
    
    def _parse_acf_manifest(self, acf_file: Path, library_path: Path) -> Optional[GameInfo]:
        """Parse Steam app manifest file"""
        if not VDF_AVAILABLE:
            # Basic parsing without vdf library
            return self._parse_acf_basic(acf_file, library_path)
        
        try:
            with open(acf_file, 'r', encoding='utf-8') as f:
                data = vdf.load(f)
            
            if 'AppState' not in data:
                return None
            
            app_state = data['AppState']
            app_id = app_state.get('appid', '')
            name = app_state.get('name', 'Unknown')
            install_dir = app_state.get('installdir', '')
            
            if not app_id or not install_dir:
                return None
            
            install_path = library_path / "steamapps" / "common" / install_dir
            
            # Try to find executable
            executable = self._find_game_executable(install_path, name)
            
            return GameInfo(
                name=name,
                app_id=app_id,
                install_path=install_path,
                executable=executable or "unknown.exe",
                launcher="steam",
                metadata={'manifest_file': str(acf_file)}
            )
        except Exception as e:
            logger.debug(f"Error parsing ACF with vdf: {e}")
            return None
    
    def _parse_acf_basic(self, acf_file: Path, library_path: Path) -> Optional[GameInfo]:
        """Basic ACF parsing without vdf library"""
        try:
            with open(acf_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple parsing
            app_id = None
            name = None
            install_dir = None
            
            for line in content.split('\n'):
                line = line.strip()
                if '"appid"' in line:
                    app_id = line.split('"')[3]
                elif '"name"' in line:
                    name = line.split('"')[3]
                elif '"installdir"' in line:
                    install_dir = line.split('"')[3]
            
            if app_id and install_dir:
                install_path = library_path / "steamapps" / "common" / install_dir
                executable = self._find_game_executable(install_path, name or "Unknown")
                
                return GameInfo(
                    name=name or f"App {app_id}",
                    app_id=app_id,
                    install_path=install_path,
                    executable=executable or "unknown.exe",
                    launcher="steam"
                )
        except Exception as e:
            logger.debug(f"Basic ACF parsing error: {e}")
        
        return None
    
    def _find_game_executable(self, install_path: Path, game_name: str) -> Optional[str]:
        """Find game executable in install directory"""
        if not install_path.exists():
            return None
        
        # Common executable patterns
        patterns = [
            f"{game_name}.exe",
            f"{game_name.replace(' ', '')}.exe",
            f"{game_name.replace(' ', '_')}.exe",
            "*.exe"
        ]
        
        for pattern in patterns:
            for exe_file in install_path.glob(pattern):
                if exe_file.is_file():
                    return exe_file.name
        
        return None
    
    def launch_game(self, app_id: str) -> bool:
        """Launch a Steam game"""
        try:
            steam_url = f"steam://rungameid/{app_id}"
            subprocess.Popen(['cmd', '/c', 'start', steam_url], shell=True)
            logger.info(f"Launching Steam game: {app_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to launch Steam game {app_id}: {e}")
            return False


class EpicGamesIntegration:
    """Epic Games Store integration"""
    
    def __init__(self):
        self.epic_path: Optional[Path] = None
        self.installed_games: Dict[str, GameInfo] = {}
        
        self._detect_epic()
    
    def _detect_epic(self):
        """Detect Epic Games Store installation"""
        try:
            # Epic stores manifests in ProgramData
            manifests_path = Path(os.environ.get('PROGRAMDATA', 'C:\\ProgramData')) / \
                           "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
            
            if manifests_path.exists():
                self.epic_path = manifests_path
                logger.info(f"✓ Epic Games Store detected")
                return True
        except Exception as e:
            logger.debug(f"Epic detection error: {e}")
        
        logger.info("Epic Games Store not detected")
        return False
    
    def scan_installed_games(self) -> Dict[str, GameInfo]:
        """Scan for installed Epic Games"""
        self.installed_games = {}
        
        if not self.epic_path or not self.epic_path.exists():
            return self.installed_games
        
        for manifest_file in self.epic_path.glob("*.item"):
            try:
                game_info = self._parse_epic_manifest(manifest_file)
                if game_info:
                    self.installed_games[game_info.app_id] = game_info
            except Exception as e:
                logger.debug(f"Error parsing Epic manifest {manifest_file}: {e}")
        
        logger.info(f"Found {len(self.installed_games)} installed Epic Games")
        return self.installed_games
    
    def _parse_epic_manifest(self, manifest_file: Path) -> Optional[GameInfo]:
        """Parse Epic Games manifest"""
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            app_name = data.get('AppName', '')
            display_name = data.get('DisplayName', 'Unknown')
            install_location = data.get('InstallLocation', '')
            launch_executable = data.get('LaunchExecutable', '')
            
            if not app_name or not install_location:
                return None
            
            install_path = Path(install_location)
            
            return GameInfo(
                name=display_name,
                app_id=app_name,
                install_path=install_path,
                executable=launch_executable or "unknown.exe",
                launcher="epic",
                metadata={'manifest_file': str(manifest_file)}
            )
        except Exception as e:
            logger.debug(f"Error parsing Epic manifest: {e}")
            return None
    
    def launch_game(self, app_id: str) -> bool:
        """Launch an Epic Games Store game"""
        try:
            epic_url = f"com.epicgames.launcher://apps/{app_id}?action=launch&silent=true"
            subprocess.Popen(['cmd', '/c', 'start', epic_url], shell=True)
            logger.info(f"Launching Epic game: {app_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to launch Epic game {app_id}: {e}")
            return False


class GOGIntegration:
    """GOG Galaxy integration"""
    
    def __init__(self):
        self.gog_path: Optional[Path] = None
        self.installed_games: Dict[str, GameInfo] = {}
        
        self._detect_gog()
    
    def _detect_gog(self):
        """Detect GOG Galaxy installation"""
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                              r"SOFTWARE\WOW6432Node\GOG.com\GalaxyClient\paths") as key:
                client_path = winreg.QueryValueEx(key, "client")[0]
                self.gog_path = Path(client_path).parent
                logger.info(f"✓ GOG Galaxy detected at: {self.gog_path}")
                return True
        except (WindowsError, FileNotFoundError):
            logger.info("GOG Galaxy not detected")
            return False
    
    def scan_installed_games(self) -> Dict[str, GameInfo]:
        """Scan for installed GOG games"""
        self.installed_games = {}
        
        # GOG games are registered in the registry
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                              r"SOFTWARE\WOW6432Node\GOG.com\Games") as key:
                i = 0
                while True:
                    try:
                        game_id = winreg.EnumKey(key, i)
                        game_info = self._get_gog_game_info(game_id)
                        if game_info:
                            self.installed_games[game_id] = game_info
                        i += 1
                    except WindowsError:
                        break
        except (WindowsError, FileNotFoundError):
            pass
        
        logger.info(f"Found {len(self.installed_games)} installed GOG games")
        return self.installed_games
    
    def _get_gog_game_info(self, game_id: str) -> Optional[GameInfo]:
        """Get GOG game information from registry"""
        try:
            key_path = rf"SOFTWARE\WOW6432Node\GOG.com\Games\{game_id}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                name = winreg.QueryValueEx(key, "gameName")[0]
                install_path = Path(winreg.QueryValueEx(key, "path")[0])
                exe_file = winreg.QueryValueEx(key, "exe")[0]
                
                return GameInfo(
                    name=name,
                    app_id=game_id,
                    install_path=install_path,
                    executable=exe_file,
                    launcher="gog"
                )
        except (WindowsError, FileNotFoundError, OSError) as e:
            logger.debug(f"Error reading GOG game {game_id}: {e}")
            return None
    
    def launch_game(self, app_id: str) -> bool:
        """Launch a GOG game"""
        try:
            if self.gog_path and (self.gog_path / "GalaxyClient.exe").exists():
                subprocess.Popen([
                    str(self.gog_path / "GalaxyClient.exe"),
                    f"/gameId={app_id}",
                    "/command=runGame"
                ])
                logger.info(f"Launching GOG game: {app_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to launch GOG game {app_id}: {e}")
        return False


class GameLauncherManager:
    """
    Unified game launcher manager
    
    Features:
    - Auto-detection of Steam, Epic, GOG
    - Game library scanning
    - Unified game launching
    - Profile integration
    """
    
    def __init__(self):
        self.steam = SteamIntegration()
        self.epic = EpicGamesIntegration()
        self.gog = GOGIntegration()
        
        self.all_games: Dict[str, GameInfo] = {}
    
    def scan_all_games(self) -> Dict[str, GameInfo]:
        """Scan all launchers for installed games"""
        logger.info("Scanning all game launchers...")
        
        self.all_games = {}
        
        # Scan each launcher
        steam_games = self.steam.scan_installed_games()
        epic_games = self.epic.scan_installed_games()
        gog_games = self.gog.scan_installed_games()
        
        # Combine with unique keys
        for game_id, game in steam_games.items():
            self.all_games[f"steam_{game_id}"] = game
        
        for game_id, game in epic_games.items():
            self.all_games[f"epic_{game_id}"] = game
        
        for game_id, game in gog_games.items():
            self.all_games[f"gog_{game_id}"] = game
        
        logger.info(f"Total games found: {len(self.all_games)}")
        return self.all_games
    
    def launch_game(self, launcher_game_id: str) -> bool:
        """Launch a game by its launcher_game_id"""
        if launcher_game_id not in self.all_games:
            logger.error(f"Game not found: {launcher_game_id}")
            return False
        
        game = self.all_games[launcher_game_id]
        
        if game.launcher == "steam":
            return self.steam.launch_game(game.app_id)
        elif game.launcher == "epic":
            return self.epic.launch_game(game.app_id)
        elif game.launcher == "gog":
            return self.gog.launch_game(game.app_id)
        else:
            logger.error(f"Unknown launcher: {game.launcher}")
            return False
    
    def get_games_list(self) -> List[GameInfo]:
        """Get sorted list of all games"""
        return sorted(self.all_games.values(), key=lambda g: g.name)
    
    def find_game_by_executable(self, executable_name: str) -> Optional[GameInfo]:
        """Find game by executable name"""
        exe_lower = executable_name.lower()
        
        for game in self.all_games.values():
            if game.executable.lower() == exe_lower:
                return game
        
        return None


def main():
    """Example usage of game launcher integration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = GameLauncherManager()
    
    print("Scanning for installed games...")
    games = manager.scan_all_games()
    
    print(f"\nFound {len(games)} games:")
    for game in sorted(manager.get_games_list(), key=lambda g: g.name)[:20]:
        print(f"  [{game.launcher.upper()}] {game.name}")
        print(f"    Path: {game.install_path}")
        print(f"    Exe: {game.executable}")
        print()


if __name__ == "__main__":
    main()
