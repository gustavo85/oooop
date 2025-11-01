"""
Multi-Language Support System V4.0
Internationalization (i18n) for Game Optimizer
Supports: English, Spanish, Portuguese, Chinese, Japanese
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"


class TranslationManager:
    """
    Manages translations for the application
    
    Features:
    - Multiple language support
    - Fallback to English
    - Dynamic language switching
    - Translation caching
    """
    
    # Built-in translations
    TRANSLATIONS = {
        Language.ENGLISH: {
            # General
            "app_title": "Game Optimizer V4.0 - Professional Edition",
            "app_subtitle": "Advanced Low-Level Game Performance Optimization",
            
            # Menu
            "menu_file": "File",
            "menu_tools": "Tools",
            "menu_help": "Help",
            "menu_import_config": "Import Configuration",
            "menu_export_config": "Export Configuration",
            "menu_exit": "Exit",
            "menu_run_benchmark": "Run Benchmark",
            "menu_diagnostics": "System Diagnostics",
            "menu_user_guide": "User Guide",
            "menu_about": "About",
            
            # Tabs
            "tab_dashboard": "Dashboard",
            "tab_profiles": "Game Profiles",
            "tab_monitor": "System Monitor",
            "tab_processes": "Process Explorer",
            "tab_ml": "ML Management",
            "tab_analytics": "Analytics",
            "tab_benchmark": "Benchmark",
            "tab_settings": "Settings",
            "tab_help": "Help",
            
            # Dashboard
            "dashboard_welcome": "Welcome to Game Optimizer",
            "dashboard_stats": "System Statistics",
            "dashboard_cpu": "CPU Usage",
            "dashboard_memory": "Memory Usage",
            "dashboard_profiles_count": "Game Profiles",
            "dashboard_sessions": "Active Sessions",
            "dashboard_quick_actions": "Quick Actions",
            
            # Buttons
            "btn_new_profile": "New Profile",
            "btn_start_optimization": "Start Optimization",
            "btn_stop_optimization": "Stop Optimization",
            "btn_run_benchmark": "Run Benchmark",
            "btn_save": "Save",
            "btn_cancel": "Cancel",
            "btn_apply": "Apply",
            "btn_refresh": "Refresh",
            "btn_export": "Export",
            "btn_import": "Import",
            "btn_delete": "Delete",
            "btn_edit": "Edit",
            
            # Settings
            "settings_language": "Language",
            "settings_power": "Power Delivery Optimization",
            "settings_pl1": "PL1 (Long Duration Power)",
            "settings_pl2": "PL2 (Short Duration Power)",
            "settings_shader_cache": "Advanced Shader Cache Management",
            "settings_shader_precompile": "Enable Shader Pre-compilation",
            "settings_shader_optimize": "Optimize Shader Cache on Startup",
            "settings_clear_cache": "Clear Shader Cache",
            "settings_launcher_integration": "Game Launcher Integration",
            "settings_steam": "Steam Integration",
            "settings_epic": "Epic Games Integration",
            "settings_gog": "GOG Galaxy Integration",
            
            # Benchmark
            "benchmark_config": "Benchmark Configuration",
            "benchmark_duration": "Duration (seconds)",
            "benchmark_target": "Target Game",
            "benchmark_mode": "Test Mode",
            "benchmark_results": "Benchmark Results",
            "benchmark_start": "Starting benchmark...",
            "benchmark_complete": "Benchmark completed!",
            "benchmark_avg_fps": "Average FPS",
            "benchmark_low_fps": "1% Low FPS",
            "benchmark_frame_time": "Frame Time (avg)",
            
            # Messages
            "msg_success": "Success",
            "msg_error": "Error",
            "msg_warning": "Warning",
            "msg_confirm": "Confirm",
            "msg_save_success": "Settings saved successfully",
            "msg_profile_created": "Profile created successfully",
            "msg_profile_deleted": "Profile deleted",
            "msg_cache_cleared": "Shader cache cleared successfully",
            
            # Status
            "status_ready": "Ready",
            "status_optimizing": "Optimizing...",
            "status_benchmarking": "Running benchmark...",
            "status_loading": "Loading...",
        },
        
        Language.SPANISH: {
            # General
            "app_title": "Game Optimizer V4.0 - Edición Profesional",
            "app_subtitle": "Optimización Avanzada de Rendimiento de Juegos de Bajo Nivel",
            
            # Menu
            "menu_file": "Archivo",
            "menu_tools": "Herramientas",
            "menu_help": "Ayuda",
            "menu_import_config": "Importar Configuración",
            "menu_export_config": "Exportar Configuración",
            "menu_exit": "Salir",
            "menu_run_benchmark": "Ejecutar Benchmark",
            "menu_diagnostics": "Diagnósticos del Sistema",
            "menu_user_guide": "Guía del Usuario",
            "menu_about": "Acerca de",
            
            # Tabs
            "tab_dashboard": "Tablero",
            "tab_profiles": "Perfiles de Juegos",
            "tab_monitor": "Monitor del Sistema",
            "tab_processes": "Explorador de Procesos",
            "tab_ml": "Gestión ML",
            "tab_analytics": "Analítica",
            "tab_benchmark": "Benchmark",
            "tab_settings": "Configuración",
            "tab_help": "Ayuda",
            
            # Dashboard
            "dashboard_welcome": "Bienvenido a Game Optimizer",
            "dashboard_stats": "Estadísticas del Sistema",
            "dashboard_cpu": "Uso de CPU",
            "dashboard_memory": "Uso de Memoria",
            "dashboard_profiles_count": "Perfiles de Juegos",
            "dashboard_sessions": "Sesiones Activas",
            "dashboard_quick_actions": "Acciones Rápidas",
            
            # Buttons
            "btn_new_profile": "Nuevo Perfil",
            "btn_start_optimization": "Iniciar Optimización",
            "btn_stop_optimization": "Detener Optimización",
            "btn_run_benchmark": "Ejecutar Benchmark",
            "btn_save": "Guardar",
            "btn_cancel": "Cancelar",
            "btn_apply": "Aplicar",
            "btn_refresh": "Actualizar",
            "btn_export": "Exportar",
            "btn_import": "Importar",
            "btn_delete": "Eliminar",
            "btn_edit": "Editar",
            
            # Settings
            "settings_language": "Idioma",
            "settings_power": "Optimización de Entrega de Energía",
            "settings_pl1": "PL1 (Potencia de Larga Duración)",
            "settings_pl2": "PL2 (Potencia de Corta Duración)",
            "settings_shader_cache": "Gestión Avanzada de Caché de Shaders",
            "settings_shader_precompile": "Habilitar Precompilación de Shaders",
            "settings_shader_optimize": "Optimizar Caché de Shaders al Iniciar",
            "settings_clear_cache": "Limpiar Caché de Shaders",
            "settings_launcher_integration": "Integración con Lanzadores de Juegos",
            "settings_steam": "Integración con Steam",
            "settings_epic": "Integración con Epic Games",
            "settings_gog": "Integración con GOG Galaxy",
            
            # Benchmark
            "benchmark_config": "Configuración de Benchmark",
            "benchmark_duration": "Duración (segundos)",
            "benchmark_target": "Juego Objetivo",
            "benchmark_mode": "Modo de Prueba",
            "benchmark_results": "Resultados del Benchmark",
            "benchmark_start": "Iniciando benchmark...",
            "benchmark_complete": "¡Benchmark completado!",
            "benchmark_avg_fps": "FPS Promedio",
            "benchmark_low_fps": "FPS Mínimo 1%",
            "benchmark_frame_time": "Tiempo de Cuadro (promedio)",
            
            # Messages
            "msg_success": "Éxito",
            "msg_error": "Error",
            "msg_warning": "Advertencia",
            "msg_confirm": "Confirmar",
            "msg_save_success": "Configuración guardada exitosamente",
            "msg_profile_created": "Perfil creado exitosamente",
            "msg_profile_deleted": "Perfil eliminado",
            "msg_cache_cleared": "Caché de shaders limpiado exitosamente",
            
            # Status
            "status_ready": "Listo",
            "status_optimizing": "Optimizando...",
            "status_benchmarking": "Ejecutando benchmark...",
            "status_loading": "Cargando...",
        },
        
        Language.PORTUGUESE: {
            # General
            "app_title": "Game Optimizer V4.0 - Edição Profissional",
            "app_subtitle": "Otimização Avançada de Desempenho de Jogos de Baixo Nível",
            
            # Menu
            "menu_file": "Arquivo",
            "menu_tools": "Ferramentas",
            "menu_help": "Ajuda",
            "menu_import_config": "Importar Configuração",
            "menu_export_config": "Exportar Configuração",
            "menu_exit": "Sair",
            "menu_run_benchmark": "Executar Benchmark",
            "menu_diagnostics": "Diagnósticos do Sistema",
            "menu_user_guide": "Guia do Usuário",
            "menu_about": "Sobre",
            
            # Tabs
            "tab_dashboard": "Painel",
            "tab_profiles": "Perfis de Jogos",
            "tab_monitor": "Monitor do Sistema",
            "tab_processes": "Explorador de Processos",
            "tab_ml": "Gerenciamento ML",
            "tab_analytics": "Análises",
            "tab_benchmark": "Benchmark",
            "tab_settings": "Configurações",
            "tab_help": "Ajuda",
            
            # Dashboard
            "dashboard_welcome": "Bem-vindo ao Game Optimizer",
            "dashboard_stats": "Estatísticas do Sistema",
            "dashboard_cpu": "Uso de CPU",
            "dashboard_memory": "Uso de Memória",
            "dashboard_profiles_count": "Perfis de Jogos",
            "dashboard_sessions": "Sessões Ativas",
            "dashboard_quick_actions": "Ações Rápidas",
            
            # Buttons
            "btn_new_profile": "Novo Perfil",
            "btn_start_optimization": "Iniciar Otimização",
            "btn_stop_optimization": "Parar Otimização",
            "btn_run_benchmark": "Executar Benchmark",
            "btn_save": "Salvar",
            "btn_cancel": "Cancelar",
            "btn_apply": "Aplicar",
            "btn_refresh": "Atualizar",
            "btn_export": "Exportar",
            "btn_import": "Importar",
            "btn_delete": "Excluir",
            "btn_edit": "Editar",
            
            # Settings
            "settings_language": "Idioma",
            "settings_power": "Otimização de Entrega de Energia",
            "settings_pl1": "PL1 (Potência de Longa Duração)",
            "settings_pl2": "PL2 (Potência de Curta Duração)",
            "settings_shader_cache": "Gerenciamento Avançado de Cache de Shaders",
            "settings_shader_precompile": "Ativar Pré-compilação de Shaders",
            "settings_shader_optimize": "Otimizar Cache de Shaders na Inicialização",
            "settings_clear_cache": "Limpar Cache de Shaders",
            "settings_launcher_integration": "Integração com Lançadores de Jogos",
            "settings_steam": "Integração com Steam",
            "settings_epic": "Integração com Epic Games",
            "settings_gog": "Integração com GOG Galaxy",
            
            # Benchmark
            "benchmark_config": "Configuração do Benchmark",
            "benchmark_duration": "Duração (segundos)",
            "benchmark_target": "Jogo Alvo",
            "benchmark_mode": "Modo de Teste",
            "benchmark_results": "Resultados do Benchmark",
            "benchmark_start": "Iniciando benchmark...",
            "benchmark_complete": "Benchmark concluído!",
            "benchmark_avg_fps": "FPS Médio",
            "benchmark_low_fps": "FPS Mínimo 1%",
            "benchmark_frame_time": "Tempo de Quadro (médio)",
            
            # Messages
            "msg_success": "Sucesso",
            "msg_error": "Erro",
            "msg_warning": "Aviso",
            "msg_confirm": "Confirmar",
            "msg_save_success": "Configurações salvas com sucesso",
            "msg_profile_created": "Perfil criado com sucesso",
            "msg_profile_deleted": "Perfil excluído",
            "msg_cache_cleared": "Cache de shaders limpo com sucesso",
            
            # Status
            "status_ready": "Pronto",
            "status_optimizing": "Otimizando...",
            "status_benchmarking": "Executando benchmark...",
            "status_loading": "Carregando...",
        },
        
        Language.CHINESE: {
            # General
            "app_title": "游戏优化器 V4.0 - 专业版",
            "app_subtitle": "高级底层游戏性能优化",
            
            # Menu
            "menu_file": "文件",
            "menu_tools": "工具",
            "menu_help": "帮助",
            "menu_import_config": "导入配置",
            "menu_export_config": "导出配置",
            "menu_exit": "退出",
            "menu_run_benchmark": "运行基准测试",
            "menu_diagnostics": "系统诊断",
            "menu_user_guide": "用户指南",
            "menu_about": "关于",
            
            # Tabs
            "tab_dashboard": "仪表板",
            "tab_profiles": "游戏配置文件",
            "tab_monitor": "系统监视器",
            "tab_processes": "进程浏览器",
            "tab_ml": "机器学习管理",
            "tab_analytics": "分析",
            "tab_benchmark": "基准测试",
            "tab_settings": "设置",
            "tab_help": "帮助",
            
            # Dashboard
            "dashboard_welcome": "欢迎使用游戏优化器",
            "dashboard_stats": "系统统计",
            "dashboard_cpu": "CPU使用率",
            "dashboard_memory": "内存使用率",
            "dashboard_profiles_count": "游戏配置文件",
            "dashboard_sessions": "活动会话",
            "dashboard_quick_actions": "快速操作",
            
            # Buttons
            "btn_new_profile": "新建配置文件",
            "btn_start_optimization": "开始优化",
            "btn_stop_optimization": "停止优化",
            "btn_run_benchmark": "运行基准测试",
            "btn_save": "保存",
            "btn_cancel": "取消",
            "btn_apply": "应用",
            "btn_refresh": "刷新",
            "btn_export": "导出",
            "btn_import": "导入",
            "btn_delete": "删除",
            "btn_edit": "编辑",
            
            # Settings
            "settings_language": "语言",
            "settings_power": "电源输送优化",
            "settings_pl1": "PL1 (长时功率)",
            "settings_pl2": "PL2 (短时功率)",
            "settings_shader_cache": "高级着色器缓存管理",
            "settings_shader_precompile": "启用着色器预编译",
            "settings_shader_optimize": "启动时优化着色器缓存",
            "settings_clear_cache": "清除着色器缓存",
            "settings_launcher_integration": "游戏启动器集成",
            "settings_steam": "Steam 集成",
            "settings_epic": "Epic Games 集成",
            "settings_gog": "GOG Galaxy 集成",
            
            # Benchmark
            "benchmark_config": "基准测试配置",
            "benchmark_duration": "持续时间（秒）",
            "benchmark_target": "目标游戏",
            "benchmark_mode": "测试模式",
            "benchmark_results": "基准测试结果",
            "benchmark_start": "开始基准测试...",
            "benchmark_complete": "基准测试完成！",
            "benchmark_avg_fps": "平均FPS",
            "benchmark_low_fps": "1%最低FPS",
            "benchmark_frame_time": "帧时间（平均）",
            
            # Messages
            "msg_success": "成功",
            "msg_error": "错误",
            "msg_warning": "警告",
            "msg_confirm": "确认",
            "msg_save_success": "设置保存成功",
            "msg_profile_created": "配置文件创建成功",
            "msg_profile_deleted": "配置文件已删除",
            "msg_cache_cleared": "着色器缓存清除成功",
            
            # Status
            "status_ready": "就绪",
            "status_optimizing": "优化中...",
            "status_benchmarking": "运行基准测试中...",
            "status_loading": "加载中...",
        },
        
        Language.JAPANESE: {
            # General
            "app_title": "ゲームオプティマイザー V4.0 - プロフェッショナル版",
            "app_subtitle": "高度な低レベルゲームパフォーマンス最適化",
            
            # Menu
            "menu_file": "ファイル",
            "menu_tools": "ツール",
            "menu_help": "ヘルプ",
            "menu_import_config": "設定をインポート",
            "menu_export_config": "設定をエクスポート",
            "menu_exit": "終了",
            "menu_run_benchmark": "ベンチマークを実行",
            "menu_diagnostics": "システム診断",
            "menu_user_guide": "ユーザーガイド",
            "menu_about": "について",
            
            # Tabs
            "tab_dashboard": "ダッシュボード",
            "tab_profiles": "ゲームプロファイル",
            "tab_monitor": "システムモニター",
            "tab_processes": "プロセスエクスプローラー",
            "tab_ml": "ML管理",
            "tab_analytics": "アナリティクス",
            "tab_benchmark": "ベンチマーク",
            "tab_settings": "設定",
            "tab_help": "ヘルプ",
            
            # Dashboard
            "dashboard_welcome": "ゲームオプティマイザーへようこそ",
            "dashboard_stats": "システム統計",
            "dashboard_cpu": "CPU使用率",
            "dashboard_memory": "メモリ使用率",
            "dashboard_profiles_count": "ゲームプロファイル",
            "dashboard_sessions": "アクティブセッション",
            "dashboard_quick_actions": "クイックアクション",
            
            # Buttons
            "btn_new_profile": "新しいプロファイル",
            "btn_start_optimization": "最適化を開始",
            "btn_stop_optimization": "最適化を停止",
            "btn_run_benchmark": "ベンチマークを実行",
            "btn_save": "保存",
            "btn_cancel": "キャンセル",
            "btn_apply": "適用",
            "btn_refresh": "更新",
            "btn_export": "エクスポート",
            "btn_import": "インポート",
            "btn_delete": "削除",
            "btn_edit": "編集",
            
            # Settings
            "settings_language": "言語",
            "settings_power": "電力供給の最適化",
            "settings_pl1": "PL1（長時間電力）",
            "settings_pl2": "PL2（短時間電力）",
            "settings_shader_cache": "高度なシェーダーキャッシュ管理",
            "settings_shader_precompile": "シェーダーのプリコンパイルを有効化",
            "settings_shader_optimize": "起動時にシェーダーキャッシュを最適化",
            "settings_clear_cache": "シェーダーキャッシュをクリア",
            "settings_launcher_integration": "ゲームランチャー統合",
            "settings_steam": "Steam統合",
            "settings_epic": "Epic Games統合",
            "settings_gog": "GOG Galaxy統合",
            
            # Benchmark
            "benchmark_config": "ベンチマーク設定",
            "benchmark_duration": "期間（秒）",
            "benchmark_target": "対象ゲーム",
            "benchmark_mode": "テストモード",
            "benchmark_results": "ベンチマーク結果",
            "benchmark_start": "ベンチマーク開始中...",
            "benchmark_complete": "ベンチマーク完了！",
            "benchmark_avg_fps": "平均FPS",
            "benchmark_low_fps": "1%低FPS",
            "benchmark_frame_time": "フレームタイム（平均）",
            
            # Messages
            "msg_success": "成功",
            "msg_error": "エラー",
            "msg_warning": "警告",
            "msg_confirm": "確認",
            "msg_save_success": "設定が正常に保存されました",
            "msg_profile_created": "プロファイルが正常に作成されました",
            "msg_profile_deleted": "プロファイルが削除されました",
            "msg_cache_cleared": "シェーダーキャッシュが正常にクリアされました",
            
            # Status
            "status_ready": "準備完了",
            "status_optimizing": "最適化中...",
            "status_benchmarking": "ベンチマーク実行中...",
            "status_loading": "読み込み中...",
        },
    }
    
    def __init__(self, default_language: Language = Language.ENGLISH):
        self.current_language = default_language
        self.config_file = Path.home() / '.game_optimizer' / 'language_config.json'
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_language_preference()
    
    def _load_language_preference(self):
        """Load saved language preference"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    lang_code = data.get('language', 'en')
                    for lang in Language:
                        if lang.value == lang_code:
                            self.current_language = lang
                            logger.info(f"Loaded language preference: {lang.name}")
                            break
        except Exception as e:
            logger.debug(f"Error loading language preference: {e}")
    
    def set_language(self, language: Language):
        """Set current language and save preference"""
        self.current_language = language
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({'language': language.value}, f, indent=2)
            logger.info(f"Language changed to: {language.name}")
        except Exception as e:
            logger.error(f"Error saving language preference: {e}")
    
    def get_text(self, key: str, fallback: Optional[str] = None) -> str:
        """
        Get translated text for a key
        
        Args:
            key: Translation key
            fallback: Fallback text if key not found
        
        Returns:
            Translated text or fallback
        """
        # Try current language
        if self.current_language in self.TRANSLATIONS:
            translations = self.TRANSLATIONS[self.current_language]
            if key in translations:
                return translations[key]
        
        # Fallback to English
        if Language.ENGLISH in self.TRANSLATIONS:
            translations = self.TRANSLATIONS[Language.ENGLISH]
            if key in translations:
                return translations[key]
        
        # Return fallback or key itself
        return fallback if fallback else key
    
    def t(self, key: str) -> str:
        """Shorthand for get_text"""
        return self.get_text(key)
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get dictionary of available languages"""
        return {
            Language.ENGLISH.value: "English",
            Language.SPANISH.value: "Español",
            Language.PORTUGUESE.value: "Português",
            Language.CHINESE.value: "中文",
            Language.JAPANESE.value: "日本語",
        }


# Global translation manager instance
_translation_manager: Optional[TranslationManager] = None


def get_translation_manager() -> TranslationManager:
    """Get global translation manager instance"""
    global _translation_manager
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    return _translation_manager


def t(key: str) -> str:
    """Global translation function"""
    return get_translation_manager().get_text(key)


def main():
    """Example usage of translation system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tm = TranslationManager()
    
    print("\n=== Multi-Language Support Demo ===\n")
    
    for language in Language:
        tm.set_language(language)
        print(f"\n{language.name}:")
        print(f"  Title: {tm.t('app_title')}")
        print(f"  Dashboard: {tm.t('tab_dashboard')}")
        print(f"  Start: {tm.t('btn_start_optimization')}")
        print(f"  Status: {tm.t('status_ready')}")


if __name__ == "__main__":
    main()
