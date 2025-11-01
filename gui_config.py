"""
GUI Configuration Panel V3.5 - Tkinter-based settings editor
Provides visual interface for non-technical users
"""

import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Optional
import threading

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Real-time graphs disabled.")


class GameOptimizerGUI:
    """Main GUI window for Game Optimizer configuration"""
    
    def __init__(self, config_manager, optimizer=None):
        self.config_manager = config_manager
        self.optimizer = optimizer
        
        self.root = tk.Tk()
        self.root.title("Game Optimizer V3.5 - Configuration")
        self.root.geometry("900x700")
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tabs
        self.profiles_tab = ttk.Frame(self.notebook)
        self.global_tab = ttk.Frame(self.notebook)
        self.monitoring_tab = ttk.Frame(self.notebook)
        self.telemetry_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.profiles_tab, text="Game Profiles")
        self.notebook.add(self.global_tab, text="Global Settings")
        self.notebook.add(self.monitoring_tab, text="Real-Time Monitoring")
        self.notebook.add(self.telemetry_tab, text="Telemetry & ML")
        
        self._setup_profiles_tab()
        self._setup_global_tab()
        self._setup_monitoring_tab()
        self._setup_telemetry_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_profiles_tab(self):
        """Setup game profiles tab"""
        
        # Left panel: Profile list
        left_frame = ttk.Frame(self.profiles_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        ttk.Label(left_frame, text="Game Profiles:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.profile_listbox = tk.Listbox(left_frame, width=30, height=20)
        self.profile_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.profile_listbox.bind('<<ListboxSelect>>', self._on_profile_select)
        
        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="New Profile", command=self._new_profile).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete", command=self._delete_profile).pack(side=tk.LEFT, padx=2)
        
        # Right panel: Profile editor
        right_frame = ttk.Frame(self.profiles_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(right_frame, text="Profile Settings:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        # Scrollable frame
        canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Profile fields
        self.profile_vars = {}
        
        row = 0
        
        # Name
        ttk.Label(scrollable_frame, text="Profile Name:").grid(row=row, column=0, sticky='w', pady=2)
        self.profile_vars['name'] = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.profile_vars['name'], width=40).grid(row=row, column=1, pady=2)
        row += 1
        
        # Game EXE
        ttk.Label(scrollable_frame, text="Game Executable:").grid(row=row, column=0, sticky='w', pady=2)
        self.profile_vars['game_exe'] = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.profile_vars['game_exe'], width=40).grid(row=row, column=1, pady=2)
        row += 1
        
        # Separator
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # Checkboxes for optimizations
        optimizations = [
            ('optimize_working_set', 'Working Set Tuning'),
            ('network_qos_enabled', 'Network QoS'),
            ('disable_nagle', 'Disable Nagle Algorithm (Low Latency)'),
            ('tcp_buffer_tuning', 'TCP Buffer Tuning'),
            ('gpu_scheduling_enabled', 'GPU Hardware Scheduling'),
            ('gpu_clock_locking', 'GPU Clock Locking'),
            ('power_high_performance', 'High Performance Power Plan'),
            ('cpu_affinity_enabled', 'CPU Affinity (P-cores)'),
            ('disable_core_parking', 'Disable Core Parking'),
            ('directx_optimizations', 'DirectX Optimizations'),
            ('stop_services', 'Stop Background Services'),
            ('stop_processes', 'Stop Background Processes'),
            ('enable_frame_time_analysis', 'Frame Time Monitoring'),
            ('enable_telemetry', 'Telemetry Collection'),
            ('ml_auto_tune_enabled', 'ML Auto-Tuning'),
        ]
        
        ttk.Label(scrollable_frame, text="Optimizations:", font=('Arial', 9, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        for var_name, label_text in optimizations:
            self.profile_vars[var_name] = tk.BooleanVar()
            ttk.Checkbutton(scrollable_frame, text=label_text, variable=self.profile_vars[var_name]).grid(row=row, column=0, columnspan=2, sticky='w', pady=1)
            row += 1
        
        # Numeric settings
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        numeric_settings = [
            ('timer_resolution_ms', 'Timer Resolution (ms):', 0.5),
            ('memory_optimization_level', 'Memory Optimization Level (0-2):', 2),
            ('network_dscp_value', 'Network DSCP Value (0-63):', 46),
        ]
        
        for var_name, label_text, default in numeric_settings:
            ttk.Label(scrollable_frame, text=label_text).grid(row=row, column=0, sticky='w', pady=2)
            self.profile_vars[var_name] = tk.DoubleVar(value=default)
            ttk.Entry(scrollable_frame, textvariable=self.profile_vars[var_name], width=10).grid(row=row, column=1, sticky='w', pady=2)
            row += 1
        
        # String settings
        string_settings = [
            ('cpu_priority_class', 'CPU Priority Class:', 'HIGH'),
            ('process_io_priority', 'I/O Priority:', 'NORMAL'),
        ]
        
        for var_name, label_text, default in string_settings:
            ttk.Label(scrollable_frame, text=label_text).grid(row=row, column=0, sticky='w', pady=2)
            self.profile_vars[var_name] = tk.StringVar(value=default)
            ttk.Entry(scrollable_frame, textvariable=self.profile_vars[var_name], width=20).grid(row=row, column=1, sticky='w', pady=2)
            row += 1
        
        # Save button
        ttk.Button(scrollable_frame, text="Save Profile", command=self._save_profile).grid(row=row, column=0, columnspan=2, pady=10)
        
        # Load profiles into listbox
        self._refresh_profile_list()
    
    def _setup_global_tab(self):
        """Setup global settings tab"""
        
        frame = ttk.Frame(self.global_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.global_vars = {}
        
        row = 0
        
        # Auto-detect games
        self.global_vars['auto_detect_games'] = tk.BooleanVar(
            value=self.config_manager.get_global_setting('auto_detect_games', True))
        ttk.Checkbutton(frame, text="Auto-detect and optimize games", 
                       variable=self.global_vars['auto_detect_games']).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        # Background app thresholds
        ttk.Label(frame, text="Background App CPU Threshold (%):").grid(row=row, column=0, sticky='w', pady=5)
        self.global_vars['background_throttle_cpu_percent'] = tk.DoubleVar(
            value=self.config_manager.get_global_setting('background_throttle_cpu_percent', 3.0))
        ttk.Entry(frame, textvariable=self.global_vars['background_throttle_cpu_percent'], width=10).grid(row=row, column=1, sticky='w', pady=5)
        row += 1
        
        ttk.Label(frame, text="Background App Memory Threshold (MB):").grid(row=row, column=0, sticky='w', pady=5)
        self.global_vars['background_throttle_memory_mb'] = tk.IntVar(
            value=self.config_manager.get_global_setting('background_throttle_memory_mb', 200))
        ttk.Entry(frame, textvariable=self.global_vars['background_throttle_memory_mb'], width=10).grid(row=row, column=1, sticky='w', pady=5)
        row += 1
        
        # Telemetry
        self.global_vars['enable_telemetry'] = tk.BooleanVar(
            value=self.config_manager.get_global_setting('enable_telemetry', True))
        ttk.Checkbutton(frame, text="Enable Telemetry Collection", 
                       variable=self.global_vars['enable_telemetry']).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        # Log level
        ttk.Label(frame, text="Log Level:").grid(row=row, column=0, sticky='w', pady=5)
        self.global_vars['log_level'] = tk.StringVar(
            value=self.config_manager.get_global_setting('log_level', 'INFO'))
        ttk.Combobox(frame, textvariable=self.global_vars['log_level'], 
                    values=['DEBUG', 'INFO', 'WARNING', 'ERROR'], state='readonly', width=15).grid(row=row, column=1, sticky='w', pady=5)
        row += 1
        
        # Save button
        ttk.Button(frame, text="Save Global Settings", command=self._save_global_settings).pack(pady=20)
    
    def _setup_monitoring_tab(self):
        """Setup real-time monitoring tab with frame time P99 graph"""
        
        frame = ttk.Frame(self.monitoring_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Real-Time Performance Monitoring", font=('Arial', 12, 'bold')).pack(pady=10)
        
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure for frame time history
            self.fig = Figure(figsize=(10, 6), dpi=100)
            self.ax_frametime = self.fig.add_subplot(111)
            
            self.ax_frametime.set_title("Frame Time P99 (Last 60 readings)", fontsize=12, fontweight='bold')
            self.ax_frametime.set_ylabel("Frame Time (ms)", fontsize=10)
            self.ax_frametime.set_xlabel("Reading #", fontsize=10)
            self.ax_frametime.grid(True, alpha=0.3)
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initialize data storage for frame time history
            self.frame_time_history = []
            
            # Start update loop
            self._update_monitoring_graphs()
        else:
            ttk.Label(frame, text="Install matplotlib for real-time graphs:\npip install matplotlib").pack(pady=20)
    
    def _setup_telemetry_tab(self):
        """Setup telemetry & ML tab"""
        
        frame = ttk.Frame(self.telemetry_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Telemetry & Machine Learning", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Telemetry stats
        stats_frame = ttk.LabelFrame(frame, text="Telemetry Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.telemetry_stats_label = ttk.Label(stats_frame, text="No data collected yet")
        self.telemetry_stats_label.pack()
        
        # Export button
        ttk.Button(stats_frame, text="Export Telemetry to JSON", command=self._export_telemetry).pack(pady=5)
        
        # ML section
        ml_frame = ttk.LabelFrame(frame, text="Machine Learning", padding=10)
        ml_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(ml_frame, text="The ML model learns from your gaming sessions to suggest optimal settings.").pack(pady=5)
        
        ttk.Button(ml_frame, text="Train ML Model Now", command=self._train_ml_model).pack(pady=5)
        ttk.Button(ml_frame, text="View ML Recommendations", command=self._show_ml_recommendations).pack(pady=5)
    
    def _refresh_profile_list(self):
        """Refresh the profile listbox"""
        self.profile_listbox.delete(0, tk.END)
        
        for game_exe, profile in sorted(self.config_manager.game_profiles.items()):
            self.profile_listbox.insert(tk.END, f"{profile.name} ({game_exe})")
    
    def _on_profile_select(self, event):
        """Handle profile selection"""
        selection = self.profile_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        profile_key = list(sorted(self.config_manager.game_profiles.keys()))[idx]
        profile = self.config_manager.game_profiles[profile_key]
        
        # Load profile into UI
        self.profile_vars['name'].set(profile.name)
        self.profile_vars['game_exe'].set(profile.game_exe)
        
        # Booleans
        for key in ['optimize_working_set', 'network_qos_enabled', 'disable_nagle', 'tcp_buffer_tuning',
                    'gpu_scheduling_enabled', 'gpu_clock_locking', 'power_high_performance', 'cpu_affinity_enabled',
                    'disable_core_parking', 'directx_optimizations', 'stop_services', 
                    'stop_processes', 'enable_frame_time_analysis', 'enable_telemetry', 'ml_auto_tune_enabled']:
            if key in self.profile_vars:
                self.profile_vars[key].set(getattr(profile, key, False))
        
        # Numerics
        self.profile_vars['timer_resolution_ms'].set(profile.timer_resolution_ms)
        self.profile_vars['memory_optimization_level'].set(profile.memory_optimization_level)
        self.profile_vars['network_dscp_value'].set(profile.network_dscp_value)
        
        # Strings
        self.profile_vars['cpu_priority_class'].set(profile.cpu_priority_class)
        self.profile_vars['process_io_priority'].set(getattr(profile, 'process_io_priority', 'NORMAL'))
        
        self.status_var.set(f"Loaded profile: {profile.name}")
    
    def _new_profile(self):
        """Create new profile"""
        from config_loader import GameProfile
        
        profile = GameProfile(name="New Profile", game_exe="game.exe")
        
        # Clear UI
        for var in self.profile_vars.values():
            if isinstance(var, tk.BooleanVar):
                var.set(False)
            elif isinstance(var, (tk.IntVar, tk.DoubleVar)):
                var.set(0)
            else:
                var.set("")
        
        # Set defaults
        self.profile_vars['name'].set("New Profile")
        self.profile_vars['game_exe'].set("game.exe")
        self.profile_vars['timer_resolution_ms'].set(0.5)
        self.profile_vars['memory_optimization_level'].set(2)
        self.profile_vars['network_dscp_value'].set(46)
        self.profile_vars['cpu_priority_class'].set('HIGH')
        self.profile_vars['process_io_priority'].set('NORMAL')
        
        self.status_var.set("New profile created (remember to save)")
    
    def _delete_profile(self):
        """Delete selected profile"""
        selection = self.profile_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a profile to delete")
            return
        
        idx = selection[0]
        profile_key = list(sorted(self.config_manager.game_profiles.keys()))[idx]
        profile = self.config_manager.game_profiles[profile_key]
        
        if messagebox.askyesno("Confirm Delete", f"Delete profile '{profile.name}'?"):
            self.config_manager.delete_game_profile(profile.game_exe)
            self._refresh_profile_list()
            self.status_var.set(f"Deleted profile: {profile.name}")
    
    def _save_profile(self):
        """Save current profile"""
        try:
            from config_loader import GameProfile
            
            # Build profile from UI
            profile = GameProfile(
                name=self.profile_vars['name'].get(),
                game_exe=self.profile_vars['game_exe'].get(),
                timer_resolution_ms=self.profile_vars['timer_resolution_ms'].get(),
                memory_optimization_level=int(self.profile_vars['memory_optimization_level'].get()),
                network_qos_enabled=self.profile_vars['network_qos_enabled'].get(),
                network_dscp_value=int(self.profile_vars['network_dscp_value'].get()),
                gpu_scheduling_enabled=self.profile_vars['gpu_scheduling_enabled'].get(),
                gpu_clock_locking=self.profile_vars['gpu_clock_locking'].get(),
                power_high_performance=self.profile_vars['power_high_performance'].get(),
                cpu_affinity_enabled=self.profile_vars['cpu_affinity_enabled'].get(),
                cpu_priority_class=self.profile_vars['cpu_priority_class'].get(),
                process_io_priority=self.profile_vars['process_io_priority'].get(),
                disable_core_parking=self.profile_vars['disable_core_parking'].get(),
                optimize_working_set=self.profile_vars['optimize_working_set'].get(),
                directx_optimizations=self.profile_vars['directx_optimizations'].get(),
                stop_services=self.profile_vars['stop_services'].get(),
                stop_processes=self.profile_vars['stop_processes'].get(),
                enable_frame_time_analysis=self.profile_vars['enable_frame_time_analysis'].get(),
                enable_telemetry=self.profile_vars['enable_telemetry'].get(),
                ml_auto_tune_enabled=self.profile_vars['ml_auto_tune_enabled'].get(),
            )
            
            self.config_manager.create_game_profile(profile)
            self._refresh_profile_list()
            
            messagebox.showinfo("Success", f"Profile '{profile.name}' saved successfully!")
            self.status_var.set(f"Saved profile: {profile.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile: {e}")
            logger.error(f"Profile save error: {e}")
    
    def _save_global_settings(self):
        """Save global settings"""
        try:
            self.config_manager.set_global_setting('auto_detect_games', 
                                                   self.global_vars['auto_detect_games'].get())
            self.config_manager.set_global_setting('background_throttle_cpu_percent', 
                                                   self.global_vars['background_throttle_cpu_percent'].get())
            self.config_manager.set_global_setting('background_throttle_memory_mb', 
                                                   self.global_vars['background_throttle_memory_mb'].get())
            self.config_manager.set_global_setting('enable_telemetry', 
                                                   self.global_vars['enable_telemetry'].get())
            self.config_manager.set_global_setting('log_level', 
                                                   self.global_vars['log_level'].get())
            
            messagebox.showinfo("Success", "Global settings saved!")
            self.status_var.set("Global settings saved")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def _update_monitoring_graphs(self):
        """Update real-time monitoring graphs with frame time P99 data"""
        if not MATPLOTLIB_AVAILABLE or not self.optimizer:
            return
        
        try:
            # Get active sessions from optimizer
            if hasattr(self.optimizer, 'performance_monitor'):
                # Get all active sessions
                active_sessions = self.optimizer.performance_monitor.active_sessions
                
                if active_sessions:
                    # Get data from first active session (or could cycle through all)
                    pid = list(active_sessions.keys())[0]
                    summary = self.optimizer.performance_monitor.get_session_summary(pid)
                    
                    if summary and 'frame_time_p99' in summary:
                        frame_time_p99 = summary['frame_time_p99']
                        
                        # Add to history (keep last 60 readings)
                        self.frame_time_history.append(frame_time_p99)
                        if len(self.frame_time_history) > 60:
                            self.frame_time_history.pop(0)
                        
                        # Update graph
                        self.ax_frametime.clear()
                        self.ax_frametime.set_title("Frame Time P99 (Last 60 readings)", fontsize=12, fontweight='bold')
                        self.ax_frametime.set_ylabel("Frame Time (ms)", fontsize=10)
                        self.ax_frametime.set_xlabel("Reading #", fontsize=10)
                        self.ax_frametime.grid(True, alpha=0.3)
                        
                        # Plot with color gradient based on performance
                        x = list(range(len(self.frame_time_history)))
                        y = self.frame_time_history
                        
                        # Color: green for good (<20ms), yellow for ok (20-33ms), red for bad (>33ms)
                        colors = []
                        for val in y:
                            if val < 20:
                                colors.append('green')
                            elif val < 33:
                                colors.append('orange')
                            else:
                                colors.append('red')
                        
                        self.ax_frametime.plot(x, y, linewidth=2, color='blue', marker='o', markersize=4)
                        self.ax_frametime.axhline(y=16.67, color='green', linestyle='--', alpha=0.5, label='60 FPS (16.67ms)')
                        self.ax_frametime.axhline(y=33.33, color='orange', linestyle='--', alpha=0.5, label='30 FPS (33.33ms)')
                        self.ax_frametime.legend(loc='upper right')
                        
                        # Add current value annotation
                        if y:
                            self.ax_frametime.annotate(f'{y[-1]:.2f}ms', 
                                                       xy=(len(y)-1, y[-1]),
                                                       xytext=(10, 10),
                                                       textcoords='offset points',
                                                       fontsize=10,
                                                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
                        
                        self.canvas.draw()
            
            # Update every 1 second
            self.root.after(1000, self._update_monitoring_graphs)
            
        except Exception as e:
            logger.debug(f"Graph update error: {e}")
            # Continue updating even on errors
            self.root.after(1000, self._update_monitoring_graphs)
    
    def _export_telemetry(self):
        """Export telemetry to file"""
        try:
            if not self.optimizer or not hasattr(self.optimizer, 'telemetry_collector'):
                messagebox.showwarning("Not Available", "Telemetry collector not initialized")
                return
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filepath:
                output_path = self.optimizer.telemetry_collector.export_to_file(Path(filepath))
                messagebox.showinfo("Success", f"Telemetry exported to:\n{output_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def _train_ml_model(self):
        """Train ML model in background thread"""
        if not self.optimizer or not hasattr(self.optimizer, 'ml_tuner'):
            messagebox.showwarning("Not Available", "ML tuner not initialized")
            return
        
        def train():
            try:
                self.status_var.set("Training ML model...")
                success = self.optimizer.ml_tuner.train_model()
                
                if success:
                    self.status_var.set("ML model training complete!")
                    messagebox.showinfo("Success", "ML model trained successfully!")
                else:
                    self.status_var.set("ML training failed (not enough data?)")
                    messagebox.showwarning("Training Failed", "Not enough training data.\nPlay more games with optimization enabled!")
                    
            except Exception as e:
                self.status_var.set("ML training error")
                messagebox.showerror("Error", f"Training failed: {e}")
        
        threading.Thread(target=train, daemon=True).start()
    
    def _show_ml_recommendations(self):
        """Show ML recommendations for all games with confidence scores"""
        if not self.optimizer or not hasattr(self.optimizer, 'ml_tuner'):
            messagebox.showwarning("Not Available", "ML tuner not initialized")
            return
        
        try:
            recommendations_window = tk.Toplevel(self.root)
            recommendations_window.title("ML Recommendations")
            recommendations_window.geometry("700x500")
            
            # Create text widget with tags for colored confidence
            text_widget = tk.Text(recommendations_window, wrap=tk.WORD, width=80, height=25, font=('Courier', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Configure tags for confidence colors
            text_widget.tag_config('confidence_high', foreground='green', font=('Courier', 10, 'bold'))
            text_widget.tag_config('confidence_medium', foreground='orange', font=('Courier', 10, 'bold'))
            text_widget.tag_config('confidence_low', foreground='red', font=('Courier', 10, 'bold'))
            text_widget.tag_config('header', font=('Courier', 12, 'bold'))
            
            text_widget.insert(tk.END, "ML Recommendations for Your Games\n", 'header')
            text_widget.insert(tk.END, "=" * 70 + "\n\n")
            
            has_recommendations = False
            
            for game_exe in self.config_manager.game_profiles.keys():
                # Get ML profile with confidence score
                result = self.optimizer.ml_tuner.generate_ml_profile(game_exe)
                
                if result:
                    # Check if result is a tuple (profile, confidence)
                    if isinstance(result, tuple) and len(result) == 2:
                        profile, confidence = result
                    else:
                        # Legacy support: just profile, no confidence
                        profile = result
                        confidence = 0.5  # Default medium confidence
                    
                    has_recommendations = True
                    
                    # Game header
                    text_widget.insert(tk.END, f"🎮 {game_exe}\n")
                    text_widget.insert(tk.END, "-" * 70 + "\n")
                    
                    # Confidence score with color coding
                    confidence_pct = confidence * 100
                    text_widget.insert(tk.END, "  Confidence: ")
                    
                    if confidence >= 0.75:
                        confidence_tag = 'confidence_high'
                        confidence_icon = "✓✓✓"
                        risk_level = "Low Risk"
                    elif confidence >= 0.50:
                        confidence_tag = 'confidence_medium'
                        confidence_icon = "✓✓"
                        risk_level = "Medium Risk"
                    else:
                        confidence_tag = 'confidence_low'
                        confidence_icon = "✓"
                        risk_level = "High Risk"
                    
                    text_widget.insert(tk.END, f"{confidence_pct:.1f}% {confidence_icon} ({risk_level})\n", confidence_tag)
                    
                    # Profile details
                    text_widget.insert(tk.END, f"  Profile: {profile.name}\n")
                    text_widget.insert(tk.END, f"  GPU Clock Locking: {'✓ Yes' if profile.gpu_clock_locking else '✗ No'}\n")
                    text_widget.insert(tk.END, f"  Core Parking: {'✓ Disabled' if profile.disable_core_parking else '✗ Enabled'}\n")
                    text_widget.insert(tk.END, f"  Memory Level: {profile.memory_optimization_level}\n")
                    text_widget.insert(tk.END, f"  Timer Resolution: {profile.timer_resolution_ms}ms\n")
                    text_widget.insert(tk.END, f"  CPU Priority: {profile.cpu_priority_class}\n")
                    
                    # Network optimizations
                    if hasattr(profile, 'disable_nagle') and profile.disable_nagle:
                        text_widget.insert(tk.END, f"  TCP Latency Opts: ✓ Enabled (Nagle disabled)\n")
                    if hasattr(profile, 'tcp_buffer_tuning') and profile.tcp_buffer_tuning:
                        text_widget.insert(tk.END, f"  TCP Buffer Tuning: ✓ Enabled\n")
                    
                    text_widget.insert(tk.END, "\n")
            
            if not has_recommendations:
                text_widget.insert(tk.END, "No recommendations available yet.\n")
                text_widget.insert(tk.END, "Play games with optimization enabled to collect training data.\n\n")
                text_widget.insert(tk.END, "Tip: The ML model needs at least 10 gaming sessions to make predictions.")
            
            # Make text widget read-only
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Error showing ML recommendations: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to show recommendations: {e}")
            
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show recommendations: {e}")
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


def main():
    """Standalone GUI launcher"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from config_loader import ConfigurationManager
    
    config_manager = ConfigurationManager()
    gui = GameOptimizerGUI(config_manager)
    
    print("=" * 60)
    print("Game Optimizer V3.5 - Configuration GUI")
    print("=" * 60)
    print("Opening GUI window...")
    
    gui.run()


if __name__ == "__main__":
    main()