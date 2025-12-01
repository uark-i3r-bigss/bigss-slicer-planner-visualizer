import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import geo.core as kg
import transform_parser
from scipy.spatial.transform import Rotation
import os
import re
import time
import yaml
from PIL import Image, ImageTk
import generate_diagram
import logging

class ControlPanel:
    def __init__(self, visualizer):
        self.viz = visualizer
        self.root = tk.Tk()
        
        # Register for visibility updates
        if hasattr(self.viz, 'add_visibility_callback'):
            self.viz.add_visibility_callback(self.on_refresh_diagram)
            
        if hasattr(self.viz, 'set_recording_status_callback'):
            self.viz.set_recording_status_callback(self.update_recording_status)
        
        # Load UI Config
        config_path = os.path.join(os.path.dirname(__file__), "configs", "ui_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.ui_config = yaml.safe_load(f)
        else:
            # Default fallback
            self.ui_config = {
                'window': {'width': 400, 'height': 650, 'title': "SE(3) Controls"},
                'fonts': {'default_size': 10}
            }
            
        self.root.title(self.ui_config['window'].get('title', "SE(3) Controls"))
        w = self.ui_config['window'].get('width', 400)
        h = self.ui_config['window'].get('height', 650)
        self.root.geometry(f"{w}x{h}")
        
        self.editor_window = None # Reference to the popup window
        self.calc_result = None # Store calculator result
        
        self.create_widgets()
        
    def update_logging_status(self, is_logging):
        """Update the logging status indicator."""
        if is_logging:
            self.log_status_label.config(text="● LOGGING", foreground="red")
        else:
            self.log_status_label.config(text="○ LOGGING", foreground="gray")

    def update_recording_status(self, is_recording, filename=None):
        """Update the recording status indicator."""
        if is_recording:
            self.rec_status_label.config(text="● REC", foreground="red")
        else:
            self.rec_status_label.config(text="○ REC", foreground="gray")

    def create_widgets(self):
        # Top Status Bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.log_status_label = ttk.Label(status_frame, text="○ LOGGING", foreground="gray", font=("TkDefaultFont", 10, "bold"))
        self.log_status_label.pack(side=tk.RIGHT, padx=5)
        
        self.rec_status_label = ttk.Label(status_frame, text="○ REC", foreground="gray", font=("TkDefaultFont", 10, "bold"))
        self.rec_status_label.pack(side=tk.RIGHT, padx=5)

        # Notebook for Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: FrameTransform
        self.tab_transform = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_transform, text="FrameTransform")

        # Create a Canvas and Scrollbar for the FrameTransform tab content
        self.canvas = tk.Canvas(self.tab_transform)
        self.scrollbar = ttk.Scrollbar(self.tab_transform, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Force initial width to match canvas
        self.canvas.update_idletasks()
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Tab 3: Vectors
        self.tab_vectors = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_vectors, text="Vectors")
    
        
        # Tab 5: Calculator
        self.tab_calculator = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_calculator, text="Calculator")
        
        # Tab 6: Diagram
        self.tab_diagram = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diagram, text="Diagram")

        # Tab 7: Settings
        self.tab_settings = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_settings, text="Settings")
        
        # --- FrameTransform Tab Content ---
        
        # Object Selection
        self._create_object_selection(self.scrollable_frame)
        
        # I/O & Sync Controls
        self._create_io_controls(self.scrollable_frame)
        
        # Transform Editor Button
        self.btn_open_controls = ttk.Button(self.scrollable_frame, text="Open Transform Controls", 
                  command=self.open_transform_editor)
        self.btn_open_controls.pack(fill=tk.X, pady=10, padx=5)
        
        # Visibility Controls
        self._create_visibility_controls(self.scrollable_frame)
        
        # Matrix Display
        self._create_matrix_display(self.scrollable_frame)
        
        # --- Settings Tab Content ---
        self._create_settings_controls(self.tab_settings)
        
        # --- Vectors Tab Content ---
        self._create_vector_display(self.tab_vectors)
        

        
        # --- Calculator Tab Content ---
        self._create_calculator_tab(self.tab_calculator)

        # --- Diagram Tab Content ---
        self._create_diagram_tab(self.tab_diagram)
        
        # Start GUI update loop
        self.update_gui()

    def _create_object_selection(self, parent):
        frame = ttk.LabelFrame(parent, text="Select Transform", padding="5")
        frame.pack(fill=tk.X, pady=5)
        
        # Get transform names
        transform_names = [t['name'] for t in self.viz.config['transforms']]
        
        # Use the first transform as default
        default_transform = transform_names[0] if transform_names else ""
        self.transform_var = tk.StringVar(value=default_transform)
        
        # Create Combobox
        self.transform_combo = ttk.Combobox(
            frame, 
            textvariable=self.transform_var, 
            values=transform_names,
            state="readonly"
        )
        self.transform_combo.pack(fill=tk.X, padx=5, pady=5)
        
        # Bind selection event
        self.transform_combo.bind('<<ComboboxSelected>>', lambda e: self.update_gui())
        
        # Kinematic Chain Display
        s = ttk.Style()
        try:
            bg_color = s.lookup('TLabelframe', 'background')
        except:
            bg_color = 'white' # Fallback
            
        self.chain_text = tk.Text(frame, height=1, width=40, font=("TkDefaultFont", 9), relief=tk.FLAT, bg=bg_color)
        self.chain_text.pack(fill=tk.X, padx=5, pady=2)
        self.chain_text.configure(state='disabled')
        
        # Define tags for colors
        self.chain_text.tag_configure("fixed", foreground="black")
        self.chain_text.tag_configure("movable", foreground="green")
        self.chain_text.tag_configure("dynamic", foreground="orange")
        self.chain_text.tag_configure("separator", foreground="black")

    def _create_io_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="I/O & Sync", padding="5")
        frame.pack(fill=tk.X, pady=5)
        
        # File Path
        path_frame = ttk.Frame(frame)
        path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame, text="File:").pack(side=tk.LEFT)
        self.io_path_var = tk.StringVar()
        self.io_path_entry = ttk.Entry(path_frame, textvariable=self.io_path_var)
        self.io_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.btn_select = ttk.Button(path_frame, text="...", width=3, command=self.on_select_file)
        self.btn_select.pack(side=tk.LEFT)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=2)
        
        self.btn_save = ttk.Button(btn_frame, text="Save", command=self.on_save_transform)
        self.btn_save.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.btn_load = ttk.Button(btn_frame, text="Load", command=self.on_load_transform)
        self.btn_load.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        # Subscribe
        self.subscribe_var = tk.BooleanVar(value=False)
        self.chk_subscribe = ttk.Checkbutton(frame, text="Subscribe (2Hz)", 
                                           variable=self.subscribe_var, 
                                           command=self.on_subscribe_toggle)
        self.chk_subscribe.pack(anchor=tk.W, pady=2)

    def open_transform_editor(self):
        if self.editor_window is not None and self.editor_window.winfo_exists():
            self.editor_window.lift()
            return

        self.editor_window = tk.Toplevel(self.root)
        self.editor_window.title("Transform Editor")
        self.editor_window.geometry("450x600") # Slightly wider for matrix
        
        # Handle closing
        def on_close():
            self.editor_window.destroy()
            self.editor_window = None
            # Clear references to avoid memory leaks or update errors
            self.entries = {}
            self.buttons = {}
            self.sliders = {}
            self.slider_widgets = {}
            self.rot_buttons = {}
            self.matrix_entries = []
            
        self.editor_window.protocol("WM_DELETE_WINDOW", on_close)
        
        # Create controls in the new window
        self._create_transform_controls(self.editor_window)
        self._create_matrix_input(self.editor_window)
        self._create_expression_input(self.editor_window)
        
        # Initial update to populate values
        t_config = self.get_active_transform_config()
        if t_config:
            self.update_transform_display(t_config)

    def get_active_transform_config(self):
        name = self.transform_var.get()
        for t in self.viz.config['transforms']:
            if t['name'] == name:
                return t
        return None

    def get_active_object(self):
        t = self.get_active_transform_config()
        if t and t.get('type') != 'dependent':
            return self.viz.object_map.get(t['child'])
        return None

    def _create_transform_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Transform Control (World_from_Object)", padding="5")
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Translation
        t_frame = ttk.Frame(frame)
        t_frame.pack(fill=tk.X, pady=2)
        ttk.Label(t_frame, text="Translation (mm):").pack(anchor=tk.W)
        
        self.entries = {}
        self.buttons = {}
        for i, axis in enumerate(['TX', 'TY', 'TZ']):
            row = ttk.Frame(t_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=axis, width=4).pack(side=tk.LEFT)
            
            entry = ttk.Entry(row, width=10)
            entry.pack(side=tk.LEFT, padx=5)
            entry.bind('<Return>', lambda e, a=i: self.on_entry_return(a))
            self.entries[axis] = entry
            
            btn = ttk.Button(row, text="Set", width=4, command=lambda a=i: self.on_entry_return(a))
            btn.pack(side=tk.LEFT)
            self.buttons[axis] = btn

        # Rotation
        r_frame = ttk.Frame(frame)
        r_frame.pack(fill=tk.X, pady=5)
        ttk.Label(r_frame, text="Rotation (deg):").pack(anchor=tk.W)
        
        self.sliders = {}
        self.slider_widgets = {}
        self.rot_buttons = {}
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            row = ttk.Frame(r_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=axis, width=6).pack(side=tk.LEFT)
            
            var = tk.DoubleVar()
            slider = ttk.Scale(row, from_=-180, to=180, variable=var, orient=tk.HORIZONTAL)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            # Command to update label while dragging (but not object)
            slider.configure(command=lambda v, a=i: self.on_slider_drag(a, float(v)))
            
            val_label = ttk.Label(row, text="0.0", width=6)
            val_label.pack(side=tk.LEFT)
            
            # Add Set Button
            btn = ttk.Button(row, text="Set", width=4, command=lambda a=i: self.on_rotation_set(a))
            btn.pack(side=tk.LEFT)
            
            self.sliders[axis] = (var, val_label)
            self.slider_widgets[axis] = slider
            self.rot_buttons[axis] = btn

        # Reset Button
        self.reset_btn = ttk.Button(frame, text="Reset Transform", command=self.reset_transform)
        self.reset_btn.pack(fill=tk.X, pady=5)

    def _create_visibility_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Visibility", padding="5")
        frame.pack(fill=tk.X, pady=5)
        
        # Checkboxes
        self.vis_vars = {
            'model': tk.BooleanVar(value=True),
            'landmarks': tk.BooleanVar(value=False),
            'vector': tk.BooleanVar(value=True)
        }
        
        self.vis_checkboxes = {}
        
        self.vis_checkboxes['model'] = ttk.Checkbutton(frame, text="Show Model", variable=self.vis_vars['model'], 
                       command=self.on_vis_change)
        self.vis_checkboxes['model'].pack(anchor=tk.W)
        
        self.vis_checkboxes['landmarks'] = ttk.Checkbutton(frame, text="Show Landmarks", variable=self.vis_vars['landmarks'], 
                       command=self.on_vis_change)
        self.vis_checkboxes['landmarks'].pack(anchor=tk.W)
        
        self.vis_checkboxes['vector'] = ttk.Checkbutton(frame, text="Show Transform Vector", variable=self.vis_vars['vector'], 
                       command=self.on_vis_change)
        self.vis_checkboxes['vector'].pack(anchor=tk.W)

    def _create_matrix_input(self, parent):
        frame = ttk.LabelFrame(parent, text="Matrix Input (4x4)", padding="5")
        frame.pack(fill=tk.X, expand=False, pady=5, padx=5)
        
        self.matrix_entries = []
        grid_frame = ttk.Frame(frame)
        grid_frame.pack(fill=tk.X, expand=True)
        
        for i in range(4):
            row_entries = []
            for j in range(4):
                entry = ttk.Entry(grid_frame, width=8)
                entry.grid(row=i, column=j, padx=2, pady=2)
                if i == 3:
                    # Last row is fixed to [0, 0, 0, 1] for rigid transforms
                    if j == 3: entry.insert(0, "1.0")
                    else: entry.insert(0, "0.0")
                    entry.state(['disabled'])
                else:
                    entry.insert(0, "0.0")
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
            
        self.set_matrix_btn = ttk.Button(frame, text="Set Matrix", command=self.on_set_matrix)
        self.set_matrix_btn.pack(pady=5)

    def on_set_matrix(self):
        obj = self.get_active_object()
        if not obj: return
        
        try:
            # Read values
            matrix = np.eye(4)
            for i in range(3):
                for j in range(4):
                    val = float(self.matrix_entries[i][j].get())
                    matrix[i, j] = val
            
            # Orthogonalize Rotation (Top-left 3x3)
            R = matrix[:3, :3]
            
            # Check for zero columns
            norms = np.linalg.norm(R, axis=0)
            if np.any(norms < 1e-9):
                logging.warning("Zero column detected in rotation matrix. Replacing with Identity.")
                R = np.eye(3)
            else:
                # SVD Orthogonalization: R = U @ V.T
                U, S, Vt = np.linalg.svd(R)
                R_ortho = U @ Vt
                
                # Ensure determinant is +1 (Rotation, not Reflection)
                if np.linalg.det(R_ortho) < 0:
                    U[:, -1] *= -1
                    R_ortho = U @ Vt
                
                R = R_ortho
            
            matrix[:3, :3] = R
            
            # Apply to object
            # Note: This overwrites scale with 1.0 (orthonormal)
            obj.local_transform.data = matrix
            
            # Update scale property of object to match new matrix (which is 1.0)
            obj.scale = np.ones(3) 
            
            obj.update_transform(self.viz.transform_map)
            
            logging.info(f"[{obj.name}] Set Matrix (Orthogonalized):\n{matrix}")
            
        except ValueError as e:
            logging.error(f"Invalid matrix input: {e}")
    def _create_expression_input(self, parent):
        frame = ttk.LabelFrame(parent, text="Set Transform (Expression)", padding="5")
        frame.pack(fill=tk.X, expand=False, pady=5, padx=5)
        
        ttk.Label(frame, text="Format: inv(A) @ B").pack(anchor=tk.W)
        
        self.expr_var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.expr_var)
        entry.pack(fill=tk.X, pady=5)
        entry.bind('<Return>', lambda e: self.on_apply_expression())
        
        self.btn_apply_expr = ttk.Button(frame, text="Apply Expression", command=self.on_apply_expression)
        self.btn_apply_expr.pack(pady=2)
        
        self.make_dependent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Constraint-based Transform", variable=self.make_dependent_var).pack(anchor=tk.W)
        
        self.expr_status = ttk.Label(frame, text="", foreground="red", wraplength=400)
        self.expr_status.pack(fill=tk.X, pady=2)

    def on_apply_expression(self):
        obj = self.get_active_object()
        if not obj: return
        
        expr = self.expr_var.get()
        if not expr:
            self.expr_status.config(text="Error: Empty expression", foreground="red")
            return
            
        # Build available transforms map
        available = {}
        for t_config in self.viz.config['transforms']:
            name = t_config['name']
            child_name = t_config['child']
            child_obj = self.viz.object_map.get(child_name)
            
            if child_obj:
                # Prevent self-reference
                if child_obj == obj:
                    continue

                # Determine if editable (movable)
                # Note: We use the object's movable property
                editable = child_obj.movable
                
                # Calculate transform matrix (Parent_from_Child)
                # We need the transform relative to its parent, which is exactly what local_transform is
                # IF the parent matches the config parent.
                # In our system, t_config['name'] usually implies Parent_from_Child.
                # And obj.local_transform IS Parent_from_Child.
                
                matrix = child_obj.local_transform.data
                
                available[name.lower()] = {
                    'matrix': matrix,
                    'editable': editable,
                    'original_name': name
                }
                
        try:
            result_matrix = transform_parser.parse_transform_expression(expr, available)
            
            # Apply to object
            # Decompose and set
            obj.local_transform.data = result_matrix
            
            # Update scale (reset to 1.0 as we assume rigid result)
            obj.scale = np.ones(3)
            
            # Update constraint
            if self.make_dependent_var.get():
                obj.constraint_expression = expr
                logging.info(f"[{obj.name}] Set constraint: {expr}")
                
                # Register dependencies
                new_deps = self.viz.resolve_dependencies(expr, obj)
                
                obj.register_dependencies(new_deps)
                logging.info(f"[{obj.name}] Registered dependencies: {[d.name for d in new_deps]}")
                
            else:
                obj.constraint_expression = None
                obj.register_dependencies([]) # Clear dependencies
            
            obj.update_transform(self.viz.transform_map)
            
            # Update UI
            t_config = self.get_active_transform_config()
            if t_config:
                self.update_transform_display(t_config)
                
            self.expr_status.config(text="Success!", foreground="green")
            logging.info(f"[{obj.name}] Applied expression: {expr}")
            
        except ValueError as e:
            self.expr_status.config(text=f"Error: {str(e)}", foreground="red")
            logging.warning(f"Invalid expression '{expr}': {e}")
        except Exception as e:
            self.expr_status.config(text=f"Error: {str(e)}", foreground="red")
            logging.error(f"Error applying expression: {e}")
    def _create_settings_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Global Settings", padding="5")
        frame.pack(fill=tk.X, pady=5)
        
        # Adjustable Grid
        self.grid_var = tk.BooleanVar(value=self.viz.adjustable_grid)
        ttk.Checkbutton(frame, text="Adjustable Grid (Auto-fit)", variable=self.grid_var,
                       command=self.on_grid_change).pack(anchor=tk.W, pady=5)
                       
        # Axes Length Slider
        ttk.Label(frame, text="Axes Length:").pack(anchor=tk.W)
        
        axes_frame = ttk.Frame(frame)
        axes_frame.pack(fill=tk.X, pady=2)
        
        self.axes_scale_var = tk.DoubleVar(value=30.0) # Default scale
        scale_slider = ttk.Scale(axes_frame, from_=1.0, to=100.0, variable=self.axes_scale_var, orient=tk.HORIZONTAL)
        scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        # Removed direct command
        
        self.axes_scale_label = ttk.Label(axes_frame, text="30.0", width=5)
        self.axes_scale_label.pack(side=tk.LEFT)
        
        # Add Set Button
        ttk.Button(axes_frame, text="Set", width=4, command=self.on_axes_scale_set).pack(side=tk.LEFT)

        # Screenshot Settings
        ss_frame = ttk.LabelFrame(parent, text="Screenshot Settings", padding="5")
        ss_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ss_frame, text="Save Path:").pack(anchor=tk.W)
        
        path_frame = ttk.Frame(ss_frame)
        path_frame.pack(fill=tk.X, pady=2)
        
        self.ss_path_var = tk.StringVar(value=self.viz.screenshot_path)
        ttk.Entry(path_frame, textvariable=self.ss_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(path_frame, text="...", width=3, command=self.on_select_screenshot_folder).pack(side=tk.LEFT)
        
        # Update viz path when entry changes
        self.ss_path_var.trace_add("write", lambda *args: setattr(self.viz, 'screenshot_path', self.ss_path_var.get()))

        # Logging Settings
        log_frame = ttk.LabelFrame(parent, text="Logging Settings", padding="5")
        log_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(log_frame, text="Log Path:").pack(anchor=tk.W)
        
        log_path_frame = ttk.Frame(log_frame)
        log_path_frame.pack(fill=tk.X, pady=2)
        
        self.log_path_var = tk.StringVar(value=self.viz.logging_path)
        ttk.Entry(log_path_frame, textvariable=self.log_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(log_path_frame, text="...", width=3, command=self.on_select_logging_folder).pack(side=tk.LEFT)
        
        # Update viz path when entry changes
        self.log_path_var.trace_add("write", lambda *args: setattr(self.viz, 'logging_path', self.log_path_var.get()))

        # Recording Settings
        rec_frame = ttk.LabelFrame(parent, text="Recording Settings", padding="5")
        rec_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(rec_frame, text="Recording Path:").pack(anchor=tk.W)
        
        rec_path_frame = ttk.Frame(rec_frame)
        rec_path_frame.pack(fill=tk.X, pady=2)
        
        self.rec_path_var = tk.StringVar(value=self.viz.recording_dir)
        ttk.Entry(rec_path_frame, textvariable=self.rec_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(rec_path_frame, text="...", width=3, command=self.on_select_recording_folder).pack(side=tk.LEFT)
        
        # Update viz path when entry changes
        self.rec_path_var.trace_add("write", lambda *args: setattr(self.viz, 'recording_dir', self.rec_path_var.get()))

        # Keyboard Controls Info
        kb_frame = ttk.LabelFrame(parent, text="Keyboard Controls", padding="5")
        kb_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(kb_frame, text="S / s : Take Screenshot").pack(anchor=tk.W)
        ttk.Label(kb_frame, text="L / l : Toggle Logging").pack(anchor=tk.W)
        ttk.Label(kb_frame, text="R / r : Toggle Recording").pack(anchor=tk.W)

    def _create_matrix_display(self, parent):
        frame = ttk.LabelFrame(parent, text="Transform Matrices", padding="5")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.matrix_text = ttk.Label(frame, text="", font=("Courier", 9), justify=tk.LEFT)
        self.matrix_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def get_active_object(self):
        transform_name = self.transform_var.get()
        # Find the transform config
        for t_config in self.viz.config['transforms']:
            if t_config['name'] == transform_name:
                child_name = t_config['child']
                return self.viz.object_map.get(child_name)
        return None

    def _create_vector_display(self, parent):
        frame = ttk.LabelFrame(parent, text="Vector Selection", padding="5")
        frame.pack(fill=tk.X, pady=5)
        
        # Dropdown for Vector Selection
        self.vector_var = tk.StringVar()
        self.vector_combo = ttk.Combobox(frame, textvariable=self.vector_var, state="readonly")
        self.vector_combo.pack(fill=tk.X, padx=5, pady=5)
        self.vector_combo.bind('<<ComboboxSelected>>', lambda e: self.update_gui())
        
        # Info Display
        info_frame = ttk.LabelFrame(parent, text="Vector Details", padding="5")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.vec_info_labels = {}
        fields = ["Name", "Parent", "Length", "Start (World)", "End (World)", "Vector (World)"]
        
        for i, field in enumerate(fields):
            row = ttk.Frame(info_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{field}:", width=15, anchor=tk.W).pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="-", anchor=tk.W)
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.vec_info_labels[field] = lbl



    def _create_calculator_tab(self, parent):
        # Equation Display
        eq_frame = ttk.LabelFrame(parent, text="Equation", padding="5")
        eq_frame.pack(fill=tk.X, pady=5)
        
        self.calc_eq_text = tk.Text(eq_frame, height=3, width=40)
        self.calc_eq_text.pack(fill=tk.X, expand=True)
        self.calc_eq_text.configure(state='disabled')
        
        # Controls
        ctrl_frame = ttk.LabelFrame(parent, text="Controls", padding="5")
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        # Transform Selection
        transform_names = [t['name'] for t in self.viz.config['transforms']]
        self.calc_transform_var = tk.StringVar(value=transform_names[0] if transform_names else "")
        
        combo = ttk.Combobox(ctrl_frame, textvariable=self.calc_transform_var, values=transform_names, state="readonly")
        combo.pack(fill=tk.X, pady=2)
        
        # Inverse Checkbox
        self.calc_inv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="Inverse", variable=self.calc_inv_var).pack(anchor=tk.W)
        
        # Buttons
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add", command=self.on_calc_add).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear", command=self.on_calc_clear).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Calculate", command=self.on_calc_compute).pack(side=tk.RIGHT, padx=2)
        
        # Result Display
        res_frame = ttk.LabelFrame(parent, text="Result", padding="5")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.calc_res_text = tk.Text(res_frame, height=15, width=40, font=("Courier", 9))
        self.calc_res_text.pack(fill=tk.BOTH, expand=True)
        self.calc_res_text.configure(state='disabled')
        
        # Save Controls
        save_frame = ttk.LabelFrame(parent, text="Save Result", padding="5")
        save_frame.pack(fill=tk.X, pady=5)
        
        # Path
        path_frame = ttk.Frame(save_frame)
        path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame, text="File:").pack(side=tk.LEFT)
        
        self.calc_io_path_var = tk.StringVar()
        self.calc_io_path_entry = ttk.Entry(path_frame, textvariable=self.calc_io_path_var)
        self.calc_io_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.calc_btn_select = ttk.Button(path_frame, text="...", width=3, command=self.on_calc_select_file)
        self.calc_btn_select.pack(side=tk.LEFT)
        
        # Save Button
        self.calc_btn_save = ttk.Button(save_frame, text="Save", command=self.on_calc_save)
        self.calc_btn_save.pack(fill=tk.X, pady=2)
        
        # Initially disabled
        self.calc_io_path_entry.configure(state='disabled')
        self.calc_btn_select.configure(state='disabled')
        self.calc_btn_save.configure(state='disabled')
        
        # Internal State
        self.calc_equation = [] # List of dicts: {'name': str, 'inverse': bool}

    def on_calc_add(self):
        name = self.calc_transform_var.get()
        if not name: return
        
        inverse = self.calc_inv_var.get()
        self.calc_equation.append({'name': name, 'inverse': inverse})
        self._update_calc_display()

    def on_calc_clear(self):
        self.calc_equation = []
        self._update_calc_display()
        self.calc_res_text.configure(state='normal')
        self.calc_res_text.delete(1.0, tk.END)
        self.calc_res_text.configure(state='disabled')
        
        self.calc_result = None
        self.calc_io_path_entry.configure(state='disabled')
        self.calc_btn_select.configure(state='disabled')
        self.calc_btn_save.configure(state='disabled')

    def _update_calc_display(self):
        text = "Result = I"
        for item in self.calc_equation:
            name = item['name']
            if item['inverse']:
                text += f" @ inv({name})"
            else:
                text += f" @ {name}"
                
        self.calc_eq_text.configure(state='normal')
        self.calc_eq_text.delete(1.0, tk.END)
        self.calc_eq_text.insert(tk.END, text)
        self.calc_eq_text.configure(state='disabled')

    def on_calc_compute(self):
        # Start with Identity
        result = kg.FrameTransform(np.eye(4))
        
        try:
            for item in self.calc_equation:
                name = item['name']
                # Find object (child of the transform)
                # The transform name is like "W_from_D", which corresponds to object "Device" (child)
                # We need to find the object associated with this transform name
                obj = None
                for t_config in self.viz.config['transforms']:
                    if t_config['name'] == name:
                        child_name = t_config['child']
                        obj = self.viz.object_map.get(child_name)
                        break
                
                if not obj:
                    raise ValueError(f"Object for transform {name} not found")
                    
                T = obj.local_transform
                
                if item['inverse']:
                    T_inv_data = np.linalg.inv(T.data)
                    T_op = kg.FrameTransform(T_inv_data)
                else:
                    T_op = T
                    
                # Multiply
                result = result @ T_op
                
            self.calc_result = result
            
            # Enable Save Controls
            self.calc_io_path_entry.configure(state='normal')
            self.calc_btn_select.configure(state='normal')
            self.calc_btn_save.configure(state='normal')
                
            # Display Result
            res_str = "Result Matrix:\n"
            with np.printoptions(precision=4, suppress=True):
                res_str += str(result.data)
                
            # Euler Angles
            # Extract rotation and normalize
            scale = np.linalg.norm(result.R, axis=0)
            R_norm = result.R / scale
            euler = Rotation.from_matrix(R_norm).as_euler('xyz', degrees=True)
            
            res_str += f"\n\nEuler (XYZ, deg):\n[{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]"
            res_str += f"\n\nTranslation:\n{result.t}"
            
            self.calc_res_text.configure(state='normal')
            self.calc_res_text.delete(1.0, tk.END)
            self.calc_res_text.insert(tk.END, res_str)
            self.calc_res_text.configure(state='disabled')
            

            
        except Exception as e:
            self.calc_result = None
            self.calc_io_path_entry.configure(state='disabled')
            self.calc_btn_select.configure(state='disabled')
            self.calc_btn_save.configure(state='disabled')
            
            self.calc_res_text.configure(state='normal')
            self.calc_res_text.delete(1.0, tk.END)
            self.calc_res_text.insert(tk.END, f"Error: {str(e)}")
            self.calc_res_text.configure(state='disabled')

    def on_calc_select_file(self):
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer_outputs", "transforms")
        if not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)
            
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=default_dir
        )
        
        if path:
            self.calc_io_path_var.set(path)

    def on_calc_save(self):
        if self.calc_result is None:
            return
            
        path = self.calc_io_path_var.get()
        if not path:
            self.on_calc_select_file()
            path = self.calc_io_path_var.get()
            if not path: return
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            matrix = self.calc_result.data
            np.savetxt(path, matrix, fmt='%.8f')
            logging.info(f"[Calculator] Saved result to {path}")
            
        except Exception as e:
            logging.error(f"Error saving calculator result: {e}")

    def _create_diagram_tab(self, parent):
        # Toolbar
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Refresh Diagram", command=self.on_refresh_diagram).pack(side=tk.LEFT)
        
        # Canvas for Image
        self.diagram_canvas = tk.Canvas(parent, bg='white')
        self.diagram_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Load initial image
        self.load_diagram_image()
        
        # Bind resize event
        self.diagram_canvas.bind("<Configure>", self.on_diagram_resize)

    def on_refresh_diagram(self):
        # Generate diagram based on current state
        try:
            dynamic_config = self._get_dynamic_config()
            output_dir = os.path.join(os.path.dirname(__file__), "visualizer_outputs", "diagrams")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Call generation function directly
            generate_diagram.generate_diagram(dynamic_config, output_dir)
            
            # Update image on main thread (in case called from background thread)
            self.root.after(0, self.load_diagram_image)
            
        except Exception as e:
            logging.error(f"Error generating diagram: {e}")

    def _get_dynamic_config(self):
        """Construct a config dictionary reflecting the current visibility state."""
        import copy
        config = copy.deepcopy(self.viz.config)
        
        # 1. Filter Objects
        visible_objects = set()
        filtered_objects = []
        for obj_conf in config.get('objects', []):
            name = obj_conf['name']
            obj = self.viz.object_map.get(name)
            if obj and obj.visible:
                filtered_objects.append(obj_conf)
                visible_objects.add(name)
            elif obj_conf.get('type') == 'virtual':
                # Virtual objects (like World) might not be in object_map or always visible
                filtered_objects.append(obj_conf)
                visible_objects.add(name)
        config['objects'] = filtered_objects
        
        # Always include World if implicit
        visible_objects.add('World')
        
        # 2. Filter Transforms
        filtered_transforms = []
        for t_conf in config.get('transforms', []):
            # Include if child is visible (and parent is visible/known)
            # Usually if child is hidden, the transform is irrelevant visually
            child = t_conf['child']
            parent = t_conf['parent']
            
            # Check if child is a visible object
            # Note: Dependent objects (DynamicEntryPoint) are in object_map
            child_obj = self.viz.object_map.get(child)
            if child_obj and child_obj.visible:
                filtered_transforms.append(t_conf)
            elif child in visible_objects:
                 filtered_transforms.append(t_conf)
                 
        config['transforms'] = filtered_transforms
        
        # 3. Filter Vectors
        filtered_vectors = []
        for vec_conf in config.get('vectors', []):
            name = vec_conf['name']
            # Find CustomVector object
            # They are in self.viz.custom_vectors list, not map
            # We need to find it by name
            found = False
            for vec in self.viz.custom_vectors:
                if vec.name == name:
                    # Check visibility
                    if vec.actor and vec.actor.GetVisibility():
                         filtered_vectors.append(vec_conf)
                    found = True
                    break
            if not found:
                 # If not found in runtime, maybe it's static and valid?
                 # Assume visible if not managed? No, safe to exclude.
                 pass
        config['vectors'] = filtered_vectors

        # 4. Filter Annotations
        filtered_annotations = []
        for ann_conf in config.get('annotations', []):
            name = ann_conf['name']
            
            # Check if dynamic group
            is_dynamic = False
            for group in self.viz.dynamic_groups:
                if group['name'] == name:
                    is_dynamic = True
                    
                    # Create a copy of the config to modify landmarks
                    ann_conf_copy = copy.deepcopy(ann_conf)
                    visible_landmarks = []
                    
                    # Check visibility of each vector in the group
                    if 'landmarks' in ann_conf_copy:
                        for lm in ann_conf_copy['landmarks']:
                            vec_name = lm['name']
                            if vec_name in group['annotations']:
                                ann_obj = group['annotations'][vec_name]
                                if ann_obj.visible:
                                    visible_landmarks.append(lm)
                                    
                        ann_conf_copy['landmarks'] = visible_landmarks
                        
                        # Only add if there are visible landmarks
                        if visible_landmarks:
                            filtered_annotations.append(ann_conf_copy)
                    break
            
            if not is_dynamic:
                # Static annotation
                # Find in self.viz.annotations
                for ann in self.viz.annotations:
                    if ann.name == name:
                        if ann.visible:
                            filtered_annotations.append(ann_conf)
                        break
                        
        config['annotations'] = filtered_annotations
        
        return config
            
    def load_diagram_image(self):
        image_path = os.path.join(os.path.dirname(__file__), "visualizer_outputs", "diagrams", "scene_diagram.png")
        if not os.path.exists(image_path):
            return
            
        try:
            # Open image
            self.original_diagram_image = Image.open(image_path)
            self.display_diagram_image()
        except Exception as e:
            logging.error(f"Error loading diagram image: {e}")

    def on_diagram_resize(self, event):
        if hasattr(self, 'original_diagram_image'):
            self.display_diagram_image()

    def display_diagram_image(self):
        if not hasattr(self, 'original_diagram_image'):
            return
            
        # Get canvas size
        canvas_width = self.diagram_canvas.winfo_width()
        canvas_height = self.diagram_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Calculate scale to fit
        img_width, img_height = self.original_diagram_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        # Resize
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = self.original_diagram_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_diagram_image = ImageTk.PhotoImage(resized_image)
        
        # Display
        self.diagram_canvas.delete("all")
        self.diagram_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.tk_diagram_image)

    def get_active_transform_config(self):
        """Returns the transform configuration dictionary for the currently selected transform."""
        transform_name = self.transform_var.get()
        if not transform_name:
            return None
        for t_config in self.viz.config['transforms']:
            if t_config['name'] == transform_name:
                return t_config
        return None

    def get_active_object(self):
        """Returns the Object3D instance for the currently selected transform, or None if dependent."""
        t_config = self.get_active_transform_config()
        if not t_config:
            return None
        
        if t_config.get('type') == 'dependent':
            return None # Dependent transforms don't have a single active object for direct manipulation
        
        return self.viz.object_map.get(t_config['child'])

    def on_save_transform(self):
        t_config = self.get_active_transform_config()
        if not t_config: return
        
        # Determine matrix to save
        matrix = None
        obj = None
        
        if t_config.get('type') == 'dependent':
            # Calculate dependent transform
            parent = self.viz.object_map.get(t_config['parent'])
            child = self.viz.object_map.get(t_config['child'])
            if parent and child:
                T_parent = parent.global_transform.data
                T_child = child.global_transform.data
                matrix = np.linalg.inv(T_parent) @ T_child
        else:
            obj = self.viz.object_map.get(t_config['child'])
            if obj:
                matrix = obj.local_transform.data
                
        if matrix is None: return
        
        path = self.io_path_var.get()
        if not path: 
            # If path is empty, open a file dialog
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer_outputs", "transforms")
            )
            if not path: return # User cancelled
            self.io_path_var.set(path)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            np.savetxt(path, matrix, fmt='%.8f')
            logging.info(f"[{t_config['name']}] Saved transform to {path}")
            
            # Update object's path if it's a real object
            if obj:
                obj.subscription_file_path = path
                obj.last_file_mtime = os.path.getmtime(path) # Update mtime after saving
            
        except Exception as e:
            logging.error(f"Error saving transform: {e}")

    def on_select_file(self):
        obj = self.get_active_object()
        if not obj: return
        
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer_outputs", "transforms")
        if not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)
            
        path = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=default_dir
        )
        
        if path:
            # Validate
            try:
                matrix = np.loadtxt(path)
                if matrix.shape != (4, 4):
                    logging.error(f"Error: Selected file {os.path.basename(path)} is not a 4x4 matrix.")
                    return
                
                self.io_path_var.set(path)
                obj.subscription_file_path = path
                logging.info(f"[{obj.name}] Selected file: {path}")
                
            except Exception as e:
                logging.error(f"Error reading/validating file: {e}")

    def on_load_transform(self):
        obj = self.get_active_object()
        if not obj: 
            logging.warning("Cannot load transform into a dependent transform.")
            return # Cannot load into dependent transform
        
        path = self.io_path_var.get()
        if not path or not os.path.exists(path):
            # If path is empty or doesn't exist, open a file dialog
            path = filedialog.askopenfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if not path: return # User cancelled
            self.io_path_var.set(path)
            if not os.path.exists(path):
                logging.error(f"File not found: {path}")
                return
            
        try:
            matrix = np.loadtxt(path)
            if matrix.shape != (4, 4):
                raise ValueError("Matrix must be 4x4")
                
            # Orthogonalize Rotation (just in case file has drift/scaling)
            R = matrix[:3, :3]
            # Decompose scale if any (we want rigid)
            scale = np.linalg.norm(R, axis=0)
            scale[scale < 1e-9] = 1.0
            R_norm = R / scale
            
            # Reconstruct Rigid
            matrix[:3, :3] = R_norm
            # Ensure bottom row
            matrix[3, :] = [0, 0, 0, 1]
            
            obj.local_transform.data = matrix
            obj.update_transform(self.viz.transform_map)
            
            logging.info(f"[{obj.name}] Loaded transform from {path}")
            
            # Update object's path and mtime
            obj.subscription_file_path = path
            obj.last_file_mtime = os.path.getmtime(path)
            
        except Exception as e:
            logging.error(f"Error loading transform: {e}")

    def on_subscribe_toggle(self):
        obj = self.get_active_object()
        if not obj: 
            logging.warning("Cannot subscribe a dependent transform.")
            # Reset checkbox if tried to check on dependent
            self.subscribe_var.set(False)
            return
        
        is_sub = self.subscribe_var.get()
        obj.is_subscribed = is_sub
        
        if is_sub:
            # Update path from entry
            obj.subscription_file_path = self.io_path_var.get()
            if not obj.subscription_file_path:
                logging.warning(f"[{obj.name}] Cannot subscribe: No file path specified.")
                self.subscribe_var.set(False)
                obj.is_subscribed = False
                return
            if not os.path.exists(obj.subscription_file_path):
                logging.warning(f"[{obj.name}] Warning: Subscribed file does not exist yet: {obj.subscription_file_path}")
            obj.last_file_mtime = os.path.getmtime(obj.subscription_file_path) if os.path.exists(obj.subscription_file_path) else 0
            obj.last_update_time = time.time() # Initialize last update time
            logging.info(f"[{obj.name}] Subscribed to {obj.subscription_file_path}")
        else:
            logging.info(f"[{obj.name}] Unsubscribed")
            
        self.update_transform_display(obj) # Update UI state based on subscription

    def update_gui(self):
        # Poll subscribed objects
        current_time = time.time()
        for obj in self.viz.objects:
            if getattr(obj, 'is_subscribed', False):
                # Throttle to ~2Hz (0.5s)
                if current_time - getattr(obj, 'last_update_time', 0) > 0.5:
                    obj.last_update_time = current_time
                    
                    path = getattr(obj, 'subscription_file_path', None)
                    if path and os.path.exists(path):
                        try:
                            # Check mtime
                            mtime = os.path.getmtime(path)
                            if mtime > getattr(obj, 'last_file_mtime', 0):
                                obj.last_file_mtime = mtime
                                
                                # Load
                                matrix = np.loadtxt(path)
                                # Orthogonalize/Normalize
                                R = matrix[:3, :3]
                                scale = np.linalg.norm(R, axis=0)
                                scale[scale < 1e-9] = 1.0
                                matrix[:3, :3] = R / scale
                                matrix[3, :] = [0, 0, 0, 1]
                                
                                # Update only if changed significantly
                                if not np.allclose(obj.local_transform.data, matrix, atol=1e-5):
                                    obj.local_transform.data = matrix
                                    obj.update_transform(self.viz.transform_map)
                                    # If this is the active object, the display will update in the next block
                                    
                        except Exception as e:
                            logging.error(f"Error reading subscribed file for {obj.name}: {e}") # Log error but don't stop
                            pass # Ignore read errors during polling
        
        # --- Update Transform List (Dynamic Filtering) ---
        visible_transforms = []
        for t in self.viz.config['transforms']:
            child_name = t['child']
            obj = self.viz.object_map.get(child_name)
            # Include if object exists and is visible
            # For dependent transforms (dynamic frames), they might be hidden by _update_dependent_frames
            if obj and getattr(obj, 'visible', True):
                visible_transforms.append(t['name'])
        
        # Update Combobox values if changed
        if list(self.transform_combo['values']) != visible_transforms:
            self.transform_combo['values'] = visible_transforms
            
        # Handle selection if current became hidden
        current_transform = self.transform_var.get()
        if current_transform and current_transform not in visible_transforms:
            self.transform_var.set("")
            current_transform = ""
            
        if not current_transform and visible_transforms:
             self.transform_var.set(visible_transforms[0])
        
        active_obj = self.get_active_object()
        active_t_config = self.get_active_transform_config()
        
        # Update Open Controls Button State
        is_movable = False
        if active_obj and active_obj.movable:
            is_movable = True
            
        if hasattr(self, 'btn_open_controls'):
            self.btn_open_controls.configure(state='normal' if is_movable else 'disabled')
            
        if hasattr(self, 'btn_load'):
            self.btn_load.configure(state='normal' if is_movable else 'disabled')
            
        if hasattr(self, 'chk_subscribe'):
            self.chk_subscribe.configure(state='normal' if is_movable else 'disabled')
        
        # Update Chain Display
        self.chain_text.configure(state='normal')
        self.chain_text.delete(1.0, tk.END)
        
        chain_obj = None
        if active_obj:
            chain_obj = active_obj
        elif active_t_config and active_t_config.get('type') == 'dependent':
             child_name = active_t_config['child']
             chain_obj = self.viz.object_map.get(child_name)
             
        if chain_obj:
            # Build chain list: [(text, tag), ...]
            chain_parts = []
            curr = chain_obj
            
            while curr and curr.parent:
                parent = curr.parent
                
                # Find transform config
                t_type = "fixed" # Default
                t_name = f"{parent.abbreviation if parent.abbreviation else parent.name}_from_{curr.abbreviation if curr.abbreviation else curr.name}"
                
                # Search config for specific transform properties
                # We need to find the transform where parent=parent.name and child=curr.name
                found_t = None
                for t in self.viz.config['transforms']:
                    if t['child'] == curr.name and t['parent'] == parent.name:
                        found_t = t
                        break
                
                if found_t:
                    if found_t.get('type') == 'dynamic_annotation':
                        t_type = "dynamic"
                    elif curr.movable:
                        t_type = "movable"
                    else:
                        t_type = "fixed"
                else:
                    # Fallback if not found in config (e.g. implicit)
                    if curr.movable:
                        t_type = "movable"
                    else:
                        t_type = "fixed"
                
                chain_parts.append((t_name, t_type))
                curr = parent
            
            # Reverse to show Root -> Leaf
            chain_parts.reverse()
            
            # Insert into Text widget
            for i, (text, tag) in enumerate(chain_parts):
                if i > 0:
                    self.chain_text.insert(tk.END, " @ ", "separator")
                self.chain_text.insert(tk.END, text, tag)
                
        self.chain_text.configure(state='disabled')
        
        # Update Matrix Text - Show ONLY active object
        text = ""
        if active_obj:
            text = active_obj.get_transform_str()
        elif active_t_config and active_t_config.get('type') == 'dependent':
            # Calculate and display for dependent transform
            parent = self.viz.object_map.get(active_t_config['parent'])
            child = self.viz.object_map.get(active_t_config['child'])
            if parent and child:
                T_parent = parent.global_transform.data
                T_child = child.global_transform.data
                matrix = np.linalg.inv(T_parent) @ T_child
                
                # Create a temporary FrameTransform to use its string representation
                temp_transform = kg.FrameTransform(matrix)
                
                # Manual string formatting since FrameTransform doesn't have get_transform_str
                parent_name = active_t_config['parent']
                child_name = active_t_config['child']
                text = f"{parent_name}_from_{child_name}:\n"
                with np.printoptions(precision=2, suppress=True):
                    text += str(matrix)
                    
                    # Add decomposition info
                    R = matrix[:3, :3]
                    scale = np.linalg.norm(R, axis=0)
                    scale[scale < 1e-9] = 1.0
                    R_norm = R / scale
                    
                    text += "\n\nRotation (Orthonormal):\n"
                    text += str(R_norm)
                    text += f"\n\nScale: {scale}"
        
        self.matrix_text.config(text=text)
        
        # --- Update Vector Display ---
        visible_vectors = []
        
        # 1. Custom Vectors
        if hasattr(self.viz, 'custom_vectors'):
            for vec in self.viz.custom_vectors:
                # Check visibility (actor exists and is visible)
                if vec.actor and vec.actor.GetVisibility():
                    visible_vectors.append(vec.name)
                    
        # 2. Annotations
        if hasattr(self.viz, 'annotations'):
            for ann in self.viz.annotations:
                if ann.visible:
                    visible_vectors.append(ann.name)
                    
        visible_vectors.sort()
        
        # Update Combobox values if changed
        if list(self.vector_combo['values']) != visible_vectors:
            self.vector_combo['values'] = visible_vectors
            
        # Handle selection
        current_vec_selection = self.vector_var.get()
        if current_vec_selection and current_vec_selection not in visible_vectors:
            self.vector_var.set("") # Deselect if hidden
            current_vec_selection = ""
            
        # Update Info Display
        if current_vec_selection:
            # Find the object
            vec_obj = None
            is_annotation = False
            
            # Check Custom Vectors
            for vec in self.viz.custom_vectors:
                if vec.name == current_vec_selection:
                    vec_obj = vec
                    break
            
            # Check Annotations
            if not vec_obj:
                for ann in self.viz.annotations:
                    if ann.name == current_vec_selection:
                        vec_obj = ann
                        is_annotation = True
                        break
            
            if vec_obj:
                self.vec_info_labels["Name"].config(text=vec_obj.name)
                self.vec_info_labels["Parent"].config(text=vec_obj.parent_name)
                
                start = None
                end = None
                
                if is_annotation:
                    start = vec_obj.current_start_world
                    end = vec_obj.current_end_world
                else:
                    start = vec_obj.start_pos
                    end = vec_obj.end_pos
                    
                if start is not None and end is not None:
                    vec_val = end - start
                    length = np.linalg.norm(vec_val)
                    
                    self.vec_info_labels["Length"].config(text=f"{length:.2f} mm")
                    self.vec_info_labels["Start (World)"].config(text=f"[{start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}]")
                    self.vec_info_labels["End (World)"].config(text=f"[{end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f}]")
                    self.vec_info_labels["Vector (World)"].config(text=f"[{vec_val[0]:.2f}, {vec_val[1]:.2f}, {vec_val[2]:.2f}]")
                else:
                     for lbl in self.vec_info_labels.values():
                        if lbl != self.vec_info_labels["Name"] and lbl != self.vec_info_labels["Parent"]:
                            lbl.config(text="-")
        else:
            # Clear info
            for lbl in self.vec_info_labels.values():
                lbl.config(text="-")
        
        # Update controls for the active object or transform
        if active_t_config:
            self.update_transform_display(active_t_config)
            
            # Visibility controls only apply to actual objects
            if active_obj:
                self.vis_vars['model'].set(active_obj.show_mesh or active_obj.show_segmentation)
                self.vis_vars['landmarks'].set(active_obj.show_landmarks)
                self.vis_vars['vector'].set(getattr(active_obj, 'show_vector', False))
                
                # Disable visibility controls for virtual objects
                is_virtual = (active_obj.obj_type == 'virtual')
                state = 'disabled' if is_virtual else 'normal'
                
                if 'model' in self.vis_checkboxes:
                    self.vis_checkboxes['model'].configure(state=state)
                if 'landmarks' in self.vis_checkboxes:
                    self.vis_checkboxes['landmarks'].configure(state=state)
            else:
                # Disable visibility controls for dependent transforms
                if 'model' in self.vis_checkboxes:
                    self.vis_checkboxes['model'].configure(state='disabled')
                if 'landmarks' in self.vis_checkboxes:
                    self.vis_checkboxes['landmarks'].configure(state='disabled')
                if 'vector' in self.vis_checkboxes:
                    self.vis_checkboxes['vector'].configure(state='disabled')
        
        # Schedule next update
        self.root.after(100, self.update_gui)

    def update_transform_display(self, obj_or_config):
        if getattr(self, 'updating_controls', False):
            return
            
        self.updating_controls = True
        
        # Determine if we are dealing with a real object or a dependent config
        is_dependent = False
        obj = None
        t_config = None
        
        if isinstance(obj_or_config, dict):
            t_config = obj_or_config
            if t_config.get('type') == 'dependent':
                is_dependent = True
            else:
                obj = self.viz.object_map.get(t_config['child'])
        else:
            obj = obj_or_config
            
        # Calculate Transform Data
        matrix = np.eye(4)
        t = np.zeros(3)
        R = np.eye(3)
        euler = np.zeros(3)
        
        if is_dependent:
            parent = self.viz.object_map.get(t_config['parent'])
            child = self.viz.object_map.get(t_config['child'])
            if parent and child:
                T_parent = parent.global_transform.data
                T_child = child.global_transform.data
                # T_parent * T_rel = T_child => T_rel = inv(T_parent) @ T_child
                matrix = np.linalg.inv(T_parent) @ T_child
                
                # Extract components
                t = matrix[:3, 3]
                R = matrix[:3, :3]
                scale = np.linalg.norm(R, axis=0)
                scale[scale < 1e-9] = 1.0 # Avoid division by zero
                R_norm = R / scale
                euler = Rotation.from_matrix(R_norm).as_euler('xyz', degrees=True)
            else:
                self.updating_controls = False
                return # Cannot calculate
        elif obj:
            t = obj.local_transform.t
            R = obj.local_transform.R
            scale = obj.scale
            scale[scale < 1e-9] = 1.0 # Avoid division by zero
            R_norm = R / scale
            euler = Rotation.from_matrix(R_norm).as_euler('xyz', degrees=True)
            matrix = obj.local_transform.data
            
            # Update I/O Path if not focused
            try:
                if self.root.focus_get() != self.io_path_entry:
                    if obj.subscription_file_path:
                        if self.io_path_var.get() != obj.subscription_file_path:
                            self.io_path_var.set(obj.subscription_file_path)
                    elif not self.io_path_var.get():
                        # Set default path
                        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                  "visualizer_outputs", "transforms", 
                                                  f"{obj.parent.abbreviation if obj.parent else 'W'}_from_{obj.abbreviation}.txt")
                        self.io_path_var.set(default_path)
            except KeyError:
                pass
                
            # Update Subscribe Checkbox
            if self.subscribe_var.get() != obj.is_subscribed:
                self.subscribe_var.set(obj.is_subscribed)
        else:
            self.updating_controls = False
            return

        # --- Update UI State ---
        
        # 1. I/O Controls
        if is_dependent:
            self.btn_load.configure(state='disabled')
            self.chk_subscribe.configure(state='disabled')
            self.io_path_entry.configure(state='normal') # Allow editing path for save
            self.btn_select.configure(state='normal')
            self.btn_save.configure(state='normal')
        elif obj:
            if obj.is_subscribed:
                self.btn_load.configure(state='disabled')
                self.btn_select.configure(state='disabled')
                self.io_path_entry.configure(state='disabled')
            elif obj.constraint_expression:
                self.btn_load.configure(state='disabled')
                self.btn_select.configure(state='normal')
                self.io_path_entry.configure(state='normal')
            else:
                self.btn_load.configure(state='normal')
                self.btn_select.configure(state='normal')
                self.io_path_entry.configure(state='normal')
            self.chk_subscribe.configure(state='normal')
            self.btn_save.configure(state='normal')

        # 2. Transform Controls (Popup)
        if self.editor_window is None or not self.editor_window.winfo_exists():
            self.updating_controls = False
            return

        # Determine if editable
        editable = False
        if not is_dependent and obj and obj.movable and not obj.is_subscribed and not obj.constraint_expression:
            editable = True

        # Update Translation Entries
        if hasattr(self, 'entries'):
            # Check if translation changed
            update_translation = True
            if hasattr(self, 'last_translation_values'):
                 if np.allclose(self.last_translation_values, t, atol=1e-3):
                     update_translation = False
            
            if update_translation:
                self.last_translation_values = t.copy()
                
                for i, axis in enumerate(['TX', 'TY', 'TZ']):
                    if axis in self.entries:
                        entry = self.entries[axis]
                        
                        # State
                        if editable:
                            entry.configure(state='normal')
                            if axis in self.buttons: self.buttons[axis].configure(state='normal')
                        else:
                            entry.configure(state='disabled')
                            if axis in self.buttons: self.buttons[axis].configure(state='disabled')
                            
                        # Value (only update if not focused)
                        try:
                            if self.root.focus_get() != entry:
                                entry.delete(0, tk.END)
                                entry.insert(0, f"{t[i]:.2f}")
                        except KeyError:
                            pass
                        
        # Update Rotation Sliders
        if hasattr(self, 'sliders'):
            # Check if rotation changed
            update_rotation = True
            if hasattr(self, 'last_rotation_values'):
                 if np.allclose(self.last_rotation_values, euler, atol=1e-1):
                     update_rotation = False
            
            if update_rotation:
                self.last_rotation_values = euler.copy()
                
                for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
                    if axis in self.sliders:
                        var, label = self.sliders[axis]
                        btn = self.rot_buttons.get(axis)
                        
                        # State
                        if editable:
                            if axis in self.slider_widgets: self.slider_widgets[axis].configure(state='normal')
                            if btn: btn.configure(state='normal')
                        else:
                            if axis in self.slider_widgets: self.slider_widgets[axis].configure(state='disabled')
                            if btn: btn.configure(state='disabled')
                            
                        # Value
                        current_val = var.get()
                        new_val = euler[i]
                        if abs(current_val - new_val) > 0.1:
                            var.set(new_val)
                            label.configure(text=f"{new_val:.1f}")
                        
        # Update Matrix Input
        if hasattr(self, 'matrix_entries') and self.matrix_entries:
            # If we have a stored "last matrix", compare
            update_matrix = True
            if hasattr(self, 'last_matrix_values'):
                 if np.allclose(self.last_matrix_values, matrix, atol=1e-3):
                     update_matrix = False
            
            if update_matrix:
                self.last_matrix_values = matrix.copy()
                
                for i in range(4):
                    for j in range(4):
                        entry = self.matrix_entries[i][j]
                            
                        # State
                        if i == 3: # Last row always disabled
                            entry.state(['disabled'])
                        elif editable:
                            entry.state(['!disabled'])
                        else:
                            entry.state(['disabled'])
                            
                        # Value
                        try:
                            if self.root.focus_get() != entry:
                                entry.delete(0, tk.END)
                                entry.insert(0, f"{matrix[i, j]:.4f}")
                        except KeyError:
                            pass
                                
            # Update Set Matrix Button
            if hasattr(self, 'set_matrix_btn'):
                if editable:
                    self.set_matrix_btn.configure(state='normal')
                else:
                    self.set_matrix_btn.configure(state='disabled')
                    
            # Update Reset Button
            if hasattr(self, 'reset_btn'):
                if editable:
                    self.reset_btn.configure(state='normal')
                else:
                    self.reset_btn.configure(state='disabled')

        # Update last active object tracking
        self.last_active_object = obj
        self.updating_controls = False

    def on_entry_return(self, axis_idx):
        obj = self.get_active_object()
        if not obj:
            return
            
        axis_name = ['TX', 'TY', 'TZ'][axis_idx]
        try:
            val = float(self.entries[axis_name].get())
            obj.set_translation(axis_idx, val, self.viz.transform_map)
            self.viz.update_custom_vectors()
            self.viz.update_annotations()
            # No need to call update_gui explicitly as it's scheduled
        except ValueError:
            pass

    def on_slider_drag(self, axis_idx, value):
        """Update label while dragging, do not update object."""
        axis_name = ['Roll', 'Pitch', 'Yaw'][axis_idx]
        self.sliders[axis_name][1].configure(text=f"{value:.1f}")

    def on_rotation_set(self, axis_idx):
        axis_name = ['Roll', 'Pitch', 'Yaw'][axis_idx]
        var, label = self.sliders[axis_name]
        value = var.get()
        
        # Update label
        label.configure(text=f"{value:.1f}")
        
        # Call change handler
        self.on_rotation_change(axis_idx, value)
        
        # Update internal state to match new value
        pass

    def on_rotation_change(self, axis_idx, value):
        if getattr(self, 'updating_controls', False):
            return
            
        obj = self.get_active_object()
        if obj:
            obj.set_rotation_euler(axis_idx, value, self.viz.transform_map)
            self.viz.update_custom_vectors()
            self.viz.update_annotations()
            
            # Update label immediately for responsiveness
            axis_name = ['Roll', 'Pitch', 'Yaw'][axis_idx]
            self.sliders[axis_name][1].configure(text=f"{value:.1f}")

    def on_vis_change(self):
        obj = self.get_active_object()
        if obj:
            obj.set_show_model(self.vis_vars['model'].get())
            obj.set_show_landmarks(self.vis_vars['landmarks'].get())
            obj.set_show_vector(self.vis_vars['vector'].get())
            self.viz.plotter.render()

    def on_grid_change(self):
        self.viz.adjustable_grid = self.grid_var.get()
        self.viz.update_grid()
        
    def on_axes_scale_set(self):
        scale = self.axes_scale_var.get()
        self.axes_scale_label.configure(text=f"{scale:.1f}")
        self.viz.update_coordinate_frames(scale)
        
    def on_select_screenshot_folder(self):
        folder = filedialog.askdirectory(initialdir=self.viz.screenshot_path, title="Select Screenshot Folder")
        if folder:
            self.ss_path_var.set(folder)
            
    def on_select_logging_folder(self):
        folder = filedialog.askdirectory(initialdir=self.log_path_var.get())
        if folder:
            self.log_path_var.set(folder)

    def on_select_recording_folder(self):
        folder = filedialog.askdirectory(initialdir=self.rec_path_var.get())
        if folder:
            self.rec_path_var.set(folder)
            self.viz.logging_path = folder

    def on_axes_scale_change(self, value):
        scale = float(value)
        self.axes_scale_label.config(text=f"{scale:.1f}")
        
        # Update all objects
        for obj in self.viz.objects:
            obj.set_axes_scale(scale)
            
        self.viz.plotter.render()

    def reset_transform(self):
        obj = self.get_active_object()
        if obj and obj.movable:
            obj.reset(self.viz.transform_map)
            self.viz.update_custom_vectors()
            self.viz.update_annotations()
            self.update_gui()
