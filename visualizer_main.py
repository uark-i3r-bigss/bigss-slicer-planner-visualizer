import pyvista as pv
import numpy as np
import sys
import os
import yaml
import logging
import re
from datetime import datetime

# Configure logging
# Default to INFO, will be updated after config load
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the path so we can import geo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from transformable_object import TransformableObject
from control_panel import ControlPanel
from data_loaders import load_segmentation, load_landmarks
import geo.core as kg
import time 


class ReferencePlane:
    def __init__(self, name, parent_name, width, length, color, opacity, plotter, object_map):
        self.name = name
        self.parent_name = parent_name
        self.width = width
        self.length = length
        self.color = color
        self.opacity = opacity
        self.plotter = plotter
        self.object_map = object_map
        
        self.actor = None
        self.last_parent_transform = None
        self.local_transform = np.eye(4) # Transform from Parent to Plane
        self.visible = True
        self.last_global_transform = None
        
        self._create_actors()
        
    def _create_actors(self):
        # 1. Plane Mesh
        self.mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=self.width, j_size=self.length)
        self.actor = self.plotter.add_mesh(
            self.mesh, 
            color=self.color, 
            opacity=self.opacity, 
            show_edges=False
        )
        
    def update_dimensions(self, width, height):
        """Update plane dimensions."""
        if abs(self.width - width) > 1e-3 or abs(self.length - height) > 1e-3:
            self.width = width
            self.length = height
            # Recreate mesh
            if self.actor: self.plotter.remove_actor(self.actor)
            self.mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=self.width, j_size=self.length)
            self.actor = self.plotter.add_mesh(self.mesh, color=self.color, opacity=self.opacity, show_edges=False)
            return True
        return False

    def update(self):
        parent = self.object_map.get(self.parent_name)
        if not parent:
            self._set_visibility(False)
            return
            
        if not self.visible:
            self._set_visibility(False)
            return

        # Calculate Global Transform of the Plane
        # W_from_Plane = W_from_P @ P_from_Plane
        global_transform = parent.global_transform.data @ self.local_transform
        
        # Check if transform changed
        if self.last_global_transform is None or not np.allclose(self.last_global_transform, global_transform, atol=1e-3):
            self.last_global_transform = global_transform
            
            # Update Plane Actor
            if self.actor:
                self.actor.user_matrix = global_transform
                self.actor.VisibilityOn()

    def _set_visibility(self, visible):
        if self.actor:
            if visible: self.actor.VisibilityOn()
            else: self.actor.VisibilityOff()

class CustomVector:
    def __init__(self, name, parent_name, landmark_label, landmark_object_name, plotter, visual_settings, object_map):
        self.name = name
        self.parent_name = parent_name
        self.landmark_label = landmark_label
        self.landmark_object_name = landmark_object_name
        self.plotter = plotter
        self.object_map = object_map
        
        # Visual Settings
        self.visual_settings = visual_settings or {}
        self.color = self.visual_settings.get('color', 'yellow')
        self.opacity = self.visual_settings.get('opacity', 0.9)
        self.line_width = self.visual_settings.get('line_width', 5) # Not used for arrow mesh, but good to have
        self.label_size = self.visual_settings.get('label_size', 14)
        self.label_color = self.visual_settings.get('label_color', self.color)
        
        self.actor = None
        self.label_actor = None
        self.current_vector = None
        self.current_length = 0.0
        self.start_pos = None
        self.end_pos = None
        
    def update_transform(self, transform_map):
        """Alias for update to satisfy dependent interface."""
        # logging.info(f"[{self.name}] update_transform called")
        self.update(transform_map)

    def update(self, map_ignored=None):
        # 1. Get Start Position (Parent Origin)
        if self.parent_name == "World":
            start_pos = np.array([0.0, 0.0, 0.0])
        else:
            parent = self.object_map.get(self.parent_name)
            if not parent:
                return # Parent not found
            start_pos = parent.global_transform.t
            
        # 2. Get End Position (Landmark)
        lm_obj = self.object_map.get(self.landmark_object_name)
        if not lm_obj:
            return # Landmark object not found
            
        end_pos = lm_obj.get_landmark_world_position(self.landmark_label)
        
        if end_pos is None:
            # Landmark not found, hide actor if it exists
            if self.actor: self.actor.VisibilityOff()
            if self.label_actor: self.label_actor.VisibilityOff()
            return # Landmark not found
            
        # Check if changed
        if self.start_pos is not None and self.end_pos is not None:
            if np.allclose(start_pos, self.start_pos, atol=1e-5) and \
               np.allclose(end_pos, self.end_pos, atol=1e-5):
                return # No change
            
        # 3. Draw Vector
        vec = end_pos - start_pos
        length = float(np.linalg.norm(vec))
        
        if length < 1e-6:
            if self.actor: self.actor.VisibilityOff()
            if self.label_actor: self.label_actor.VisibilityOff()
            return
            
        direction = vec / length
        
        # Fixed dimensions for visibility
        fixed_shaft_radius = 1.0
        fixed_tip_radius = 3.0
        fixed_tip_length = 10.0
        
        arrow_mesh = pv.Arrow(start=start_pos, direction=direction, scale=length,
                        shaft_radius=fixed_shaft_radius/length,
                        tip_radius=fixed_tip_radius/length,
                        tip_length=fixed_tip_length/length)
                        
        if self.actor:
            # Update existing actor
            self.actor.mapper.dataset.copy_from(arrow_mesh)
            self.actor.VisibilityOn()
        else:
            # Create new
            self.actor = self.plotter.add_mesh(arrow_mesh, color=self.color, opacity=self.opacity, show_scalar_bar=False)
        
        # 4. Draw Label
        midpoint = (start_pos + end_pos) / 2
        
        # Recreate label (2D, less expensive)
        if self.label_actor:
            self.plotter.remove_actor(self.label_actor)
            
        self.label_actor = self.plotter.add_point_labels(
            [midpoint], [self.name],
            font_size=self.label_size, text_color=self.label_color,
            show_points=False, always_visible=True
        )
        
        self.current_vector = vec
        self.current_length = length
        self.start_pos = start_pos
        self.end_pos = end_pos



class Annotation:
    def __init__(self, name, parent_name, start_point, end_point, plotter, visual_settings=None):
        self.name = name
        self.parent_name = parent_name
        self.start_point = start_point # Local coordinates
        self.end_point = end_point     # Local coordinates
        self.plotter = plotter
        
        # Visual Settings
        self.visual_settings = visual_settings or {}
        self.color = self.visual_settings.get('color', 'red')
        self.opacity = self.visual_settings.get('opacity', 0.8)
        self.label_size = self.visual_settings.get('label_size', 12)
        self.label_color = self.visual_settings.get('label_color', self.color)
        
        self.actor = None
        self.label_actor = None
        self.current_midpoint_world = None
        self.current_start_world = None
        self.current_end_world = None
        self.last_matrix = None
        self.visible = True
        
        self._create_actor()
        
    def _create_actor(self):
        # Create Arrow in Local Frame
        vec = self.end_point - self.start_point
        length = float(np.linalg.norm(vec))
        
        if length < 1e-6:
            return
            
        direction = vec / length
        
        # Fixed dimensions
        fixed_shaft_radius = 0.8
        fixed_tip_radius = 2.5
        fixed_tip_length = 8.0
        
        arrow = pv.Arrow(start=self.start_point, direction=direction, scale=length,
                        shaft_radius=fixed_shaft_radius/length,
                        tip_radius=fixed_tip_radius/length,
                        tip_length=fixed_tip_length/length)
                        
        # Pass render=False to prevent immediate rendering at identity
        new_actor = self.plotter.add_mesh(arrow, color=self.color, opacity=self.opacity, show_scalar_bar=False, render=False)
        
        # Apply cached transform immediately to avoid flicker
        if self.last_matrix is not None:
            new_actor.user_matrix = self.last_matrix
            
        # Clean up existing actors
        if self.actor:
            self.plotter.remove_actor(self.actor, render=False)
        if self.label_actor:
            self.plotter.remove_actor(self.label_actor, render=False)
            
        self.actor = new_actor
        
        # Label at midpoint
        midpoint = (self.start_point + self.end_point) / 2
        self.label_actor = self.plotter.add_point_labels(
            [midpoint], [self.name],
            font_size=self.label_size, text_color=self.label_color,
            show_points=False, always_visible=True
        )
        
        # Apply visibility
        self.set_visible(self.visible)

    def update_points(self, start_point, end_point):
        """Update the start and end points and rebuild the actor."""
        self.start_point = start_point
        self.end_point = end_point
        self._create_actor()

    def set_visible(self, visible):
        """Toggle visibility of the annotation."""
        changed = (self.visible != visible)
        if changed:
            logging.info(f"[{self.name}] Setting visibility to {visible}")
            
        self.visible = visible
        if self.actor:
            self.actor.SetVisibility(visible)
        if self.label_actor:
            self.label_actor.SetVisibility(visible)
            
        return changed

    def update(self, object_map):
        parent = object_map.get(self.parent_name)
        if not parent or not self.actor:
            return
            
        # Update transform
        # The geometry is static in local frame, we just move the actor
        self.last_matrix = parent.global_transform.data
        self.actor.user_matrix = self.last_matrix
        
        # Update Label
        if self.visible:
            # Calculate new world position for label
            midpoint_local = (self.start_point + self.end_point) / 2
            midpoint_world = (parent.global_transform.data @ np.append(midpoint_local, 1))[:3]
            
            # Check if label needs update
            if self.label_actor and self.current_midpoint_world is not None and \
               np.allclose(midpoint_world, self.current_midpoint_world, atol=1e-3):
                pass # No change needed
            else:
                # Recreate label
                if self.label_actor:
                    self.plotter.remove_actor(self.label_actor)
                    
                self.label_actor = self.plotter.add_point_labels(
                    [midpoint_world], [self.name],
                    font_size=self.label_size, text_color=self.label_color,
                    show_points=False, always_visible=True
                )
                self.current_midpoint_world = midpoint_world
        else:
            if self.label_actor:
                self.plotter.remove_actor(self.label_actor)
                self.label_actor = None
        
        # Calculate start and end in world frame for logging
        self.current_start_world = (parent.global_transform.data @ np.append(self.start_point, 1))[:3]
        self.current_end_world = (parent.global_transform.data @ np.append(self.end_point, 1))[:3]


class SE3Visualizer:
    def __init__(self, config_path):
        self.plotter = pv.Plotter()
        self.config_path = config_path
        self.plotter.title = "BiGSS Scene Visualizer"
        
        # Grid settings
        self.adjustable_grid = True # Default to auto-fit
        self.dependent_actors = {}
        self.fixed_grid_bounds = None # Will be calculated from objects
        
        # Screenshot settings
        self.screenshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer_outputs", "screenshots")
        if not os.path.exists(self.screenshot_path):
            os.makedirs(self.screenshot_path, exist_ok=True)

        # Logging
        self.logging_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer_outputs", "logs")
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path, exist_ok=True)
        self.is_logging = False
        self.last_log_time = 0
        self.log_file = None
        self.log_start_time = 0
        
        # Threading Lock for Recording
        import threading
        self.recording_lock = threading.Lock()

        # Create World Object (Implicit)
        self.world = TransformableObject("World", "W", self.plotter, color="black")
        self.objects = [self.world]
        self.object_map = {"World": self.world}

        # Load Configuration
        self.config = self.load_config(config_path)
        
        # Update Logging Level
        log_level_str = self.config.get('logging_level', 'INFO').upper()
        numeric_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(numeric_level)
        logging.info(f"Logging level set to {log_level_str}")
        
        base_path = os.path.dirname(os.path.abspath(config_path))
        # project_root is calculated relative to this script

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = script_dir
        
        # Resolve Objects
        # If config doesn't define objects, look them up in libraries based on transforms
        if 'objects' not in self.config:
            self.config['objects'] = self._resolve_objects_from_libraries(project_root)
        
        for obj_config in (self.config.get('objects') or []):
            name = obj_config['name']
            abbr = obj_config['abbreviation']
            if abbr == "W":
                raise ValueError(f"Abbreviation 'W' is reserved for World. Invalid object: {name}")
                
            paths = obj_config.get('paths', {})
            
            # Resolve paths relative to project root
            model_path = os.path.join(project_root, paths['model']) if paths.get('model') else None
            seg_path = os.path.join(project_root, paths['segmentation']) if paths.get('segmentation') else None
            landmarks_path = os.path.join(project_root, paths['landmarks']) if paths.get('landmarks') else None
            
            # Load segmentation to get affine and origin
            affine = None
            origin = None
            if seg_path and os.path.exists(seg_path):
                seg_data = load_segmentation(seg_path)
                affine = seg_data['affine']
                origin = seg_data['origin']
            
            visual_settings = obj_config.get('visual_settings', {})
            
            # Check for Basic Shapes
            # Check Object Type
            obj_type = obj_config.get('type', 'model') # Default to model if not specified
            shape_type = None
            shape_params = None
            
            if obj_type == 'basic_shapes':
                shape_type = obj_config.get('shape')
                shape_params = obj_config.get('parameters')
                # Initial transform is explicit in config for basic shapes
                if 'initial_transform' in obj_config:
                    affine = np.array(obj_config['initial_transform'])
                else:
                    affine = np.eye(4)
                    
            elif obj_type == 'virtual':
                # Virtual object: Just a frame, no geometry
                if 'initial_transform' in obj_config:
                    affine = np.array(obj_config['initial_transform'])
                else:
                    affine = np.eye(4)
                # Ensure no paths are tried to be loaded
                seg_path = None
                landmarks_path = None
                
            elif obj_type == 'model':
                # Existing logic for loading from paths
                pass
            
            obj = TransformableObject(
                name,
                abbr,
                self.plotter,

                mesh_path=None, # Disable loading STL meshes, use segmentation or basic shapes
                color=obj_config.get('color', 'lightgray'),
                ct_origin=origin,
                landmarks_path=landmarks_path,
                segmentation_path=seg_path,
                initial_transform=affine,
                visual_settings=visual_settings,
                shape_type=shape_type,
                shape_params=shape_params,
                obj_type=obj_type,
                movable=obj_config.get('movable', True)
            )
            self.objects.append(obj)
            self.object_map[name] = obj
            
        # Initialize Annotations (Load them first so dynamic transforms can find them)
        self.annotations = []
        self.dynamic_groups = []
        self.dependent_transforms_map = {} # (ann_name, vec_name) -> list of transform configs
        
        # Pre-compute dependent transforms map
        for t in (self.config.get('transforms') or []):
            # Support 'dynamic_annotation' type or dynamic: true flag
            t_type = t.get('type')
            is_dynamic = t.get('dynamic', False)
            
            if t_type == 'dynamic_annotation' or is_dynamic:
                # Support actor_name or vector_name
                vec_name = t.get('actor_name') or t.get('vector_name')
                ann_name = t.get('annotation_name')
                
                if ann_name and vec_name:
                    key = (ann_name, vec_name)
                    if key not in self.dependent_transforms_map:
                        self.dependent_transforms_map[key] = []
                    self.dependent_transforms_map[key].append(t)
        
        # Callbacks
        self.visibility_changed_callbacks = []

        self._load_annotations()
        
        # Initialize Reference Planes
        self.reference_planes = []
        for plane_config in (self.config.get('reference_planes') or []):
            plane = ReferencePlane(
                name=plane_config['name'],
                parent_name=plane_config['parent'],
                width=plane_config.get('width', 300),
                length=plane_config.get('length', 300),
                color=plane_config.get('color', 'blue'),
                opacity=plane_config.get('opacity', 0.2),
                plotter=self.plotter,
                object_map=self.object_map
            )
            self.reference_planes.append(plane)

        # Link Objects and Calculate Local Transforms
        self._link_objects()
        # Calculate Initial Transforms
        self._calculate_initial_local_transforms()
        
        # Build Transform Map (Transform Name -> Object)
        self.transform_map = {}
        for t_conf in (self.config.get('transforms') or []):
            t_name = t_conf['name']
            child_name = t_conf['child']
            if child_name in self.object_map:
                self.transform_map[t_name] = self.object_map[child_name]

        # Recording State
        self.is_recording = False
        default_rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer_outputs", "recordings")
        self.recording_dir = self.config.get('recording_dir', default_rec_dir)
        # Ensure it's absolute
        if not os.path.isabs(self.recording_dir):
             self.recording_dir = os.path.abspath(self.recording_dir)

        self.recording_status_callback = None
        
        # Ensure recording directory exists
        if not os.path.exists(self.recording_dir):
            os.makedirs(self.recording_dir, exist_ok=True)

        # Initialize Custom Vectors
        self.custom_vectors = []
        self._load_custom_vectors()

        # Apply Constraints from Config
        self._apply_constraints()
        
        # Initial Scene Update
        self.update_scene()
        
        # Bind Keys
        self.plotter.add_key_event('r', self.toggle_recording)
        
        # Add Render Callback for Recording
        self.plotter.add_on_render_callback(self._on_render)

    def toggle_recording(self):
        """Toggle screen recording state."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def start_recording(self):
        """Start screen recording."""
        with self.recording_lock:
            if self.is_recording:
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.recording_dir, f"recording_{timestamp}.mp4")
            
            try:
                # Open movie file
                self.plotter.open_movie(filename, framerate=30)
                self.is_recording = True
                logging.info(f"Started recording to {filename}")
                
                if self.recording_status_callback:
                    self.recording_status_callback(True, filename)
                    
            except Exception as e:
                logging.error(f"Failed to start recording: {e}")
            
    def stop_recording(self):
        """Stop screen recording."""
        with self.recording_lock:
            if not self.is_recording:
                return
                
            try:
                # Set flag to False FIRST
                self.is_recording = False
                
                # Close the movie writer directly instead of closing the plotter (which closes the window)
                if hasattr(self.plotter, 'mwriter') and self.plotter.mwriter:
                    self.plotter.mwriter.close()
                elif hasattr(self.plotter, 'm_writer') and self.plotter.m_writer:
                    self.plotter.m_writer.close()
                elif hasattr(self.plotter, 'movie_writer') and self.plotter.movie_writer:
                    self.plotter.movie_writer.close()
                else:
                    logging.warning("Could not find movie writer to close. File might be incomplete.")

                logging.info("Stopped recording")
                
                if self.recording_status_callback:
                    self.recording_status_callback(False, None)
                    
            except Exception as e:
                logging.error(f"Failed to stop recording: {e}")
                self.is_recording = False # Force state reset
            
    def _on_render(self, caller, event=None):
        """Callback triggered on every render."""
        if self.recording_lock.acquire(blocking=False):
            try:
                if self.is_recording:
                    try:
                        if hasattr(self.plotter, 'mwriter') and (self.plotter.mwriter is None or self.plotter.mwriter.closed):
                             return
                             
                        self.plotter.write_frame()
                    except Exception as e:
                        if not hasattr(self, '_logged_render_error'):
                            logging.error(f"Error writing frame: {e}")
                            self._logged_render_error = True
            finally:
                self.recording_lock.release()

    def set_recording_status_callback(self, callback):
        self.recording_status_callback = callback

    def update_scene(self):
        """Update transforms for all objects in the scene."""
        # Update all objects
        for obj in self.objects:
            obj.update_transform(self.transform_map)
            
        # Update Custom Vectors
        self.update_custom_vectors()
        
        # Update Dynamic Annotations (Trajectory Planner)
        self._update_dynamic_annotations()
        
        # Update Dynamic Reference Planes
        self._update_dynamic_transforms()
        
        # Update Annotations
        self.update_annotations()
        
        # Update Reference Planes
        self.update_reference_planes()
        
        # Camera and Grid
        self.plotter.camera_position = 'xy'
        self.plotter.camera.zoom(1.0)
        # Calculate initial grid bounds
        self.calculate_scene_bounds()
        self.update_grid()

    def update_reference_planes(self):
        """Update all reference planes."""
        for plane in self.reference_planes:
            plane.update()
            
    def add_visibility_callback(self, callback):
        """Add a callback function to be called when visibility changes."""
        self.visibility_changed_callbacks.append(callback)
        
    def _trigger_visibility_callbacks(self):
        """Trigger all registered visibility callbacks."""
        for callback in self.visibility_changed_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Error in visibility callback: {e}")

    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _resolve_objects_from_libraries(self, base_path):
        """Load object definitions from libraries if not present in input config."""
        object_library = {}
        
        # 1. Debug Objects
        debug_path = os.path.join(base_path, "configs", "debug_objects.yaml")
        if os.path.exists(debug_path):
            debug_conf = self.load_config(debug_path)
            for obj in (debug_conf.get('objects') or []):
                object_library[obj['name']] = obj
                
        # 2. Standard Config (Model Objects)
        model_path = os.path.join(base_path, "configs", "config.yaml")
        if os.path.exists(model_path):
            model_conf = self.load_config(model_path)
            for obj in (model_conf.get('objects') or []):
                object_library[obj['name']] = obj
                
        # Identify required objects from transforms
        required_objects = set()
        for t in (self.config.get('transforms') or []):
            if t['parent'] != 'World':
                required_objects.add(t['parent'])
            if t['child'] != 'World':
                required_objects.add(t['child'])
                
        resolved_objects = []
        for obj_name in required_objects:
            if obj_name in object_library:
                resolved_objects.append(object_library[obj_name])
            else:
                logging.warning(f"Object {obj_name} required by transforms but not found in libraries.")
                
        return resolved_objects

    def _link_objects(self):
        """Link parent and child objects based on config."""
        for t_config in (self.config.get('transforms') or []):
            parent_name = t_config['parent']
            child_name = t_config['child']
            
            if t_config.get('type') == 'dependent':
                continue
            
            # Allow dynamic_annotation to be linked
            if t_config.get('type') == 'dynamic_annotation':
                pass
                
            parent = self.object_map.get(parent_name)
            child = self.object_map.get(child_name)
            
            if parent and child:
                child.parent = parent
                parent.children.append(child)
                logging.debug(f"Linked {child.name} to parent {parent.name}")

    def _calculate_initial_local_transforms(self):
        """Calculate initial local transforms from global initials or explicit config."""
        
        # 1. Parse explicit local transforms from config
        explicit_transforms = {}
        dynamic_transforms = {}
        for t in (self.config.get('transforms') or []):
            if 'initial_transform' in t:
                explicit_transforms[t['child']] = np.array(t['initial_transform'])
            if t.get('type') == 'dynamic_annotation':
                dynamic_transforms[t['child']] = t

        # 2. Apply transforms
        for obj in self.objects:
            if obj.name == "World":
                continue
                
            # Case A: Explicit Local Transform defined in config
            if obj.name in explicit_transforms:
                local_data = explicit_transforms[obj.name]
                
                # Decompose and Rigidify
                R = local_data[:3, :3]
                scale = np.linalg.norm(R, axis=0)
                scale[scale < 1e-9] = 1.0
                R_norm = R / scale
                
                T_rigid = np.eye(4)
                T_rigid[:3, :3] = R_norm
                T_rigid[:3, 3] = local_data[:3, 3]
                
                obj.local_transform = kg.FrameTransform(T_rigid)
                obj.initial_local_transform = T_rigid
                obj.scale = scale
                
                logging.debug(f"[{obj.name}] Set explicit local transform from config")
                
            # Case B: Dynamic Annotation
            elif obj.name in dynamic_transforms:
                t_config = dynamic_transforms[obj.name]
                ann_name = t_config.get('annotation_name')
                vec_name = t_config.get('vector_name')
                
                # Find the annotation group
                found = False
                for group in self.dynamic_groups:
                    if group.get('name') == ann_name:
                        if vec_name in group['annotations']:
                            ann = group['annotations'][vec_name]
                            # Calculate frame in local space 
                            F = self._calculate_frame_from_vector(ann.start_point, ann.end_point)
                            obj.local_transform = kg.FrameTransform(F)
                            obj.initial_local_transform = F
                            obj.scale = np.ones(3)
                            
                            logging.debug(f"[{obj.name}] Calculated initial dynamic transform from {ann_name}/{vec_name}")
                            found = True
                            break
                
                if not found:
                    logging.warning(f"Could not find dynamic annotation {ann_name}/{vec_name} for {obj.name}")

            # Case C: Calculate from Global (if parent exists and no explicit local)
            elif obj.parent:
                # T_local = T_parent_global_inv * T_global
                parent_global = obj.parent.initial_global_transform
                parent_inv = np.linalg.inv(parent_global)
                local_data = parent_inv @ obj.initial_global_transform
                
                # Decompose and Rigidify
                R = local_data[:3, :3]
                scale = np.linalg.norm(R, axis=0)
                scale[scale < 1e-9] = 1.0
                R_norm = R / scale
                
                T_rigid = np.eye(4)
                T_rigid[:3, :3] = R_norm
                T_rigid[:3, 3] = local_data[:3, 3]
                
                obj.local_transform = kg.FrameTransform(T_rigid)
                obj.initial_local_transform = T_rigid
                obj.scale = scale
                
                logging.debug(f"[{obj.name}] Calculated initial local transform relative to {obj.parent.name}")
            
            # Case C: Parent is World (or no parent), local = global
            else:
                obj.initial_local_transform = obj.local_transform.data
                # local_transform is already set to initial_global_transform in __init__
                pass

    def _load_custom_vectors(self):
        """Load custom vector definitions from config."""
        for vec_config in (self.config.get('vectors') or []):
            name = vec_config['name']
            parent = vec_config['parent']
            lm_label = vec_config['landmark_label']
            lm_obj = vec_config['landmark_object']
            
            # Visual Settings
            visual_settings = vec_config.get('visual_settings', {})

            vec = CustomVector(name, parent, lm_label, lm_obj, self.plotter, visual_settings, self.object_map)
            self.custom_vectors.append(vec)
            
            # Register dependencies
            if parent != "World":
                p_obj = self.object_map.get(parent)
                if p_obj: p_obj.add_dependent(vec)
                
            lm_obj_inst = self.object_map.get(lm_obj)
            if lm_obj_inst: lm_obj_inst.add_dependent(vec)

    def update_custom_vectors(self):
        """Update all custom vectors based on current object transforms."""
        for vec in self.custom_vectors:
            vec.update(self.object_map)

    def resolve_dependencies(self, expression, current_obj=None):
        """
        Parse expression and resolve referenced transforms to objects.
        Returns a list of TransformableObject instances.
        """
        tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', expression)
        deps = []
        for token in tokens:
            # Find transform config with this name (case-insensitive)
            for t_conf in (self.config.get('transforms') or []):
                if t_conf['name'].lower() == token.lower():
                    dep_child_name = t_conf['child']
                    dep_obj = self.object_map.get(dep_child_name)
                    if dep_obj and dep_obj != current_obj:
                        deps.append(dep_obj)
                    break
        return deps

    def _apply_constraints(self):
        """Apply constraints defined in config."""
        for t in (self.config.get('transforms') or []):
            if t.get('type') == 'dependent': continue

            if 'constraint' in t:
                child_name = t['child']
                obj = self.object_map.get(child_name)
                if not obj: continue
                
                expr = t['constraint']
                obj.constraint_expression = expr
                logging.info(f"[{obj.name}] Set constraint from config: {expr}")
                
                # Register dependencies
                deps = self.resolve_dependencies(expr, obj)
                obj.register_dependencies(deps)
                logging.info(f"[{obj.name}] Registered dependencies: {[d.name for d in deps]}")
                
                # Evaluate immediately
                try:
                    obj.update_transform(self.transform_map)
                except Exception as e:
                    logging.error(f"[{obj.name}] Failed to evaluate initial constraint: {e}")

    def _load_annotations(self):
        """Load annotations from config."""
        for ann_config in (self.config.get('annotations') or []):
            # Top-level config for this file/group
            group_name = ann_config.get('name', 'annotation')
            type_ = ann_config.get('type', 'vector')
            
            # Skip reference planes (handled separately)
            if type_ == 'reference_plane':
                continue
                
            parent_name = ann_config.get('parent')
            path = ann_config.get('path')
            
            if not parent_name or not path:
                continue

            is_dynamic = ann_config.get('dynamic', False)
            update_freq = ann_config.get('update_frequency', 1.0)
            
            # Support both 'vector' and 'line' (legacy)
            if type_ not in ['vector', 'line']:
                continue
                
            # Resolve path relative to CWD (Project Root)
            full_path = os.path.join(os.getcwd(), path)

            if not os.path.exists(full_path):
                logging.warning(f"Annotation file not found: {full_path}")
                continue
                
            # Load landmarks (returns World/LPS)
            try:
                lms = load_landmarks(full_path)
            except Exception as e:
                logging.error(f"Error loading landmarks from {full_path}: {e}")
                continue
            
            # Get list of vectors to create
            vectors_to_create = []
            
            if 'landmarks' in ann_config:
                vectors_to_create = ann_config['landmarks']

            group_annotations = {}
            
            for vec_def in vectors_to_create:
                vec_name = vec_def['name']
                start_label = vec_def['start']
                end_label = vec_def['end']
                
                # Check if labels exist
                start_exists = start_label in lms['labels']
                end_exists = end_label in lms['labels']
                
                start_point = np.zeros(3)
                end_point = np.zeros(3)
                is_visible = False
                
                if start_exists and end_exists:
                    try:
                        start_idx = lms['labels'].index(start_label)
                        end_idx = lms['labels'].index(end_label)
                        
                        start_point_world = lms['points'][start_idx]
                        end_point_world = lms['points'][end_idx]
                        
                        # Convert to Local Frame of Parent
                        sp, ep = self._to_local_frame(parent_name, start_point_world, end_point_world)
                        if sp is not None:
                            start_point = sp
                            end_point = ep
                            is_visible = True
                    except Exception as e:
                        logging.warning(f"Error calculating points for {vec_name}: {e}")
                else:
                    # Create hidden annotation
                    logging.warning(f"Labels for {vec_name} ({start_label}, {end_label}) not found in {path}. Creating hidden annotation.")

                visual_settings = ann_config.get('visual_settings', {})
                
                ann = Annotation(vec_name, parent_name, start_point, end_point, self.plotter, visual_settings)
                ann.set_visible(is_visible)
                self.annotations.append(ann)
                group_annotations[vec_name] = ann
            
            if is_dynamic:
                self.dynamic_groups.append({
                    'name': group_name,
                    'path': full_path,
                    'frequency': update_freq,
                    'last_update': time.time(),
                    'parent_name': parent_name,
                    'vectors_def': vectors_to_create,
                    'annotations': group_annotations
                })

    def _to_local_frame(self, parent_name, start_point_world, end_point_world):
        """Convert world points to parent's local frame."""
        parent = self.object_map.get(parent_name)
        if parent:
            # The FCSV/TXT points are in World Space (LPS).
            # The Parent is in a Local Rigid Frame (T_rigid).
            # We need to transform points from World to Local Rigid.
            # P_local = inv(T_rigid) * P_world
            
            if hasattr(parent, 'initial_local_transform'):
                T_rigid_initial = parent.initial_local_transform
            else:
                T_rigid_initial = parent.local_transform.data
            
            T_inv = np.linalg.inv(T_rigid_initial)
            
            start_local = (T_inv @ np.append(start_point_world, 1))[:3]
            end_local = (T_inv @ np.append(end_point_world, 1))[:3]
            return start_local, end_local
        else:
            logging.warning(f"Parent {parent_name} not found")
            return None, None

    def _update_dynamic_annotations(self, force_update_all=False):
        """Check and update dynamic annotations."""
        current_time = time.time()
        visibility_changed = False
        
        for group in self.dynamic_groups:
            # Frequency Check
            if current_time - group['last_update'] < (1.0 / group['frequency']):
                continue
                
            # File MTime Check
            if not os.path.exists(group['path']):
                continue
                
            mtime = os.path.getmtime(group['path'])
            last_mtime = group.get('last_mtime', 0)
            
            if not force_update_all and mtime <= last_mtime:
                continue
                
            group['last_mtime'] = mtime
            
            try:
                # Reload landmarks
                lms = load_landmarks(group['path'])
                logging.debug(f"Loaded labels from {group['path']}: {lms['labels']}")
                
                for vec_def in group['vectors_def']:
                    vec_name = vec_def['name']
                    start_label = vec_def['start']
                    end_label = vec_def['end']
                    
                    if vec_name not in group['annotations']:
                        # Should not happen with new logic, but safe fallback
                        logging.warning(f"Annotation object {vec_name} not found in group {group['name']}. Hiding dependent frames.")
                        if self._update_dependent_frames(group['name'], vec_name, None, None, visible=False):
                            visibility_changed = True
                        continue
                        
                    ann = group['annotations'][vec_name]
                    
                    if start_label in lms['labels'] and end_label in lms['labels']:
                        start_idx = lms['labels'].index(start_label)
                        end_idx = lms['labels'].index(end_label)
                        
                        start_point_world = lms['points'][start_idx]
                        end_point_world = lms['points'][end_idx]
                        
                        # Convert to local frame of parent
                        start_local, end_local = self._to_local_frame(group['parent_name'], start_point_world, end_point_world)
                        
                        if start_local is not None:
                            # Debug diff
                            if logging.getLogger().isEnabledFor(logging.INFO):
                                diff_start = np.linalg.norm(start_local - ann.start_point)
                                diff_end = np.linalg.norm(end_local - ann.end_point)
                                if diff_start > 1e-6 or diff_end > 1e-6:
                                    logging.info(f"[{vec_name}] Check update. Diff S: {diff_start:.6f}, Diff E: {diff_end:.6f}")
                            
                            # Only update if points have changed significantly OR if it was hidden
                            if not ann.visible or \
                               not np.allclose(start_local, ann.start_point, atol=1e-3) or \
                               not np.allclose(end_local, ann.end_point, atol=1e-3):
                                
                                if not ann.visible:
                                    logging.info(f"[{vec_name}] Vector reappeared. Showing annotation.")
                                else:
                                    logging.info(f"[{vec_name}] Updating! Points changed.")
                                    
                                ann.update_points(start_local, end_local)
                                if ann.set_visible(True): visibility_changed = True
                                
                                # Ensure transform is reapplied
                                ann.update(self.object_map)
                                
                                # Update any dependent frames (Visible)
                                logging.info(f"Updating dependent frames for {vec_name} with start {start_local} end {end_local}")
                                if self._update_dependent_frames(group['name'], vec_name, start_local, end_local, visible=True):
                                    visibility_changed = True
                    else:
                        # Vector missing in current file
                        if ann.visible:
                            logging.info(f"[{vec_name}] Vector missing in file. Hiding annotation.")
                        if ann.set_visible(False): visibility_changed = True
                        # Update dependent frames (Hidden)
                        if self._update_dependent_frames(group['name'], vec_name, None, None, visible=False):
                            visibility_changed = True
                            
                group['last_update'] = current_time
                
            except Exception as e:
                # Suppress errors to avoid spamming console
                logging.error(f"Error updating dynamic annotation {group['path']}: {e}")
                    
        if visibility_changed:
            self._trigger_visibility_callbacks()

    def _update_dependent_frames(self, annotation_name, vector_name, start_point, end_point, visible=True):
        """Update transforms that depend on this annotation vector."""
        changed_any = False
        
        key = (annotation_name, vector_name)
        if key not in self.dependent_transforms_map:
            return False
            
        for t_config in self.dependent_transforms_map[key]:
            child_name = t_config['child']
            child_obj = self.object_map.get(child_name)
            
            if child_obj:
                if visible:
                    # Calculate new local transform
                    F = self._calculate_frame_from_vector(start_point, end_point)
                    
                    # Update object
                    child_obj.local_transform = kg.FrameTransform(F)
                    # Also update initial so it persists if needed (though it's dynamic)
                    child_obj.initial_local_transform = F
                    
                    changed = child_obj.set_visible(True)
                    if changed: changed_any = True
                    
                    # Force update of the object's global transform and actor
                    child_obj.update_transform(self.transform_map)
                    
                else:
                    # Hide object
                    changed = child_obj.set_visible(False)
                    if changed: changed_any = True
                    
                    child_obj.update_transform(self.transform_map)
        return changed_any

    def _calculate_frame_from_vector(self, start_point, end_point):
        """Construct a frame where Origin is start_point and Z-axis is (end - start)."""
        z_axis = end_point - start_point
        norm = np.linalg.norm(z_axis)
        if norm < 1e-9:
            z = np.array([0, 0, 1])
        else:
            z = z_axis / norm
            
        # Choose arbitrary axis for X
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z, arbitrary)) > 0.99:
            arbitrary = np.array([0.0, 1.0, 0.0])
            
        x = np.cross(arbitrary, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        R = np.column_stack((x, y, z))
        
        F = np.eye(4)
        F[:3, :3] = R
        F[:3, 3] = start_point
        
        logging.info(f"Calculated Frame:\n{F}")
        return F

    def update_annotations(self):
        """Update all annotations."""
        for ann in self.annotations:
            ann.update(self.object_map)

    def _update_dependent_transforms(self):
        """Update visualization for dependent transforms."""
        for t_config in (self.config.get('transforms') or []):
            # Allow 'dependent', 'dynamic_annotation', or any transform with dynamic: true
            t_type = t_config.get('type')
            is_dynamic = t_config.get('dynamic', False)
            
            if t_type != 'dependent' and t_type != 'dynamic_annotation' and not is_dynamic:
                continue
                
            name = t_config['name']
            parent_name = t_config['parent']
            child_name = t_config['child']
            
            # Get positions
            parent = self.object_map.get(parent_name)
            child = self.object_map.get(child_name)
            
            if not parent or not child:
                continue
            start = parent.global_transform.t
            end = child.global_transform.t
            
            if hasattr(child, 'show_vector'):
                 # Ensure we clean up any old actors from this method
                 if name in self.dependent_actors:
                     cache = self.dependent_actors[name]
                     if cache['arrow']: self.plotter.remove_actor(cache['arrow'])
                     if cache['label']: self.plotter.remove_actor(cache['label'])
                     del self.dependent_actors[name]
                 continue
            
            # Check if update needed
            cache = self.dependent_actors.get(name)
            if cache:
                if np.allclose(cache['last_start'], start, atol=1e-3) and \
                   np.allclose(cache['last_end'], end, atol=1e-3):
                    continue
            
            # Remove old actors
            if cache:
                if cache['arrow']: self.plotter.remove_actor(cache['arrow'])
                if cache['label']: self.plotter.remove_actor(cache['label'])
            
            # Calculate vector
            vec = end - start
            length = float(np.linalg.norm(vec))
            
            if length < 1e-6:
                self.dependent_actors[name] = {'arrow': None, 'label': None, 'last_start': start, 'last_end': end}
                continue
                
            direction = vec / length
            
            # Create Arrow (Light Green to match others)
            fixed_shaft_radius = 1.5
            fixed_tip_radius = 4.0
            fixed_tip_length = 15.0
            
            arrow = pv.Arrow(start=start, direction=direction, scale=length, 
                            shaft_radius=fixed_shaft_radius/length, 
                            tip_radius=fixed_tip_radius/length, 
                            tip_length=fixed_tip_length/length)
                            
            arrow_actor = self.plotter.add_mesh(arrow, color="lightgreen", opacity=0.8, show_scalar_bar=False)
            
            # Create Label
            midpoint = (start + end) / 2
            label_text = f"{name}"
            label_actor = self.plotter.add_point_labels(
                [midpoint], [label_text],
                font_size=12, text_color="lightgreen",
                show_points=False, always_visible=False
            )
            
            self.dependent_actors[name] = {
                'arrow': arrow_actor,
                'label': label_actor,
                'last_start': start,
                'last_end': end
            }

    def _convert_ras_to_lps(self, matrix_ras):
        """Convert a matrix from RAS to LPS coordinate system."""
        # T_lps = Conversion @ T_ras
        # Conversion matrix for RAS to LPS (flip X and Y)
        conversion = np.diag([-1, -1, 1, 1])
        return conversion @ matrix_ras

    def _update_dynamic_transforms(self):
        """Update transforms from dynamic annotation files."""
        if not hasattr(self, 'dynamic_transform_updates'):
            self.dynamic_transform_updates = {} # Stores last update time
        if not hasattr(self, 'dynamic_file_mtimes'):
            self.dynamic_file_mtimes = {} # Stores last file mtime
            
        current_time = time.time()
        
        # Group dynamic transforms by annotation file
        # Key: Annotation Name, Value: List of transform configs
        ann_transforms = {}
        
        for t_config in (self.config.get('transforms') or []):
            if not t_config.get('dynamic', False):
                continue
            
            ann_name = t_config.get('annotation_name')
            if not ann_name: continue
            
            if ann_name not in ann_transforms:
                ann_transforms[ann_name] = []
            ann_transforms[ann_name].append(t_config)
            
        # Process each annotation group
        for ann_name, t_configs in ann_transforms.items():
            # Find annotation config
            ann_config = next((a for a in (self.config.get('annotations') or []) if a.get('name') == ann_name), None)
            if not ann_config: continue
            
            # Only process reference planes here (trajectories are handled by _update_dynamic_annotations)
            if ann_config.get('type') != 'reference_plane':
                continue
            
            file_path = ann_config.get('path')
            if not file_path: continue
            
            # 1. Frequency Check
            freq = ann_config.get('update_frequency', 1.0)
            last_update = self.dynamic_transform_updates.get(ann_name, 0)
            if current_time - last_update < (1.0 / freq):
                continue
            self.dynamic_transform_updates[ann_name] = current_time
            
            # 2. File Existence & MTime Check
            if not os.path.exists(file_path):
                continue
                
            mtime = os.path.getmtime(file_path)
            last_mtime = self.dynamic_file_mtimes.get(file_path, 0)
            
            # If file hasn't changed, skip reading
            if mtime <= last_mtime:
                continue
            self.dynamic_file_mtimes[file_path] = mtime
            
            try:
                # 3. Parse File
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Parse Planes/Transforms from file
                # We assume the file format is CSV: Name, Matrix(16), Width, Height
                file_data = {}
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        continue
                    if line.startswith('PlaneName'): # Header
                        continue
                        
                    parts = line.strip().split(',')
                    if len(parts) < 17: continue # At least Name + 16 matrix
                    
                    name = parts[0].strip()
                    matrix = np.array([float(x) for x in parts[1:17]]).reshape(4, 4)
                    
                    width = 0.0
                    height = 0.0
                    if len(parts) >= 19:
                        width = float(parts[17])
                        height = float(parts[18])
                    
                    # Convert RAS to LPS
                    matrix_lps = self._convert_ras_to_lps(matrix)
                    file_data[name] = {'matrix': matrix_lps, 'width': width, 'height': height}
                
                # 4. Update Transforms
                for t_config in t_configs:
                    # Support actor_name, reference_plane_name, or vector_name
                    actor_name = t_config.get('actor_name') or t_config.get('reference_plane_name') or t_config.get('vector_name')
                    if not actor_name: continue
                    
                    data = file_data.get(actor_name)
                    if not data:
                        # logging.warning(f"Actor {actor_name} not found in {file_path}")
                        continue
                        
                    child_name = t_config.get('child')
                    parent_name = t_config.get('parent')
                    
                    child_obj = self.object_map.get(child_name)
                    parent_obj = self.object_map.get(parent_name)
                    
                    if child_obj and parent_obj:
                        # Calculate Local Transform: P_from_Child = inv(W_from_P) @ W_from_Child
                        w_from_child = data['matrix']
                        w_from_p = parent_obj.global_transform.data
                        p_from_w = np.linalg.inv(w_from_p)
                        p_from_child = p_from_w @ w_from_child
                        
                        # Check for significant change
                        current_local = child_obj.local_transform
                        if isinstance(current_local, kg.FrameTransform):
                            current_local = current_local.data
                            
                        if not np.allclose(current_local, p_from_child, atol=1e-3):
                            # Update Child's Local Transform
                            if isinstance(child_obj.local_transform, kg.FrameTransform):
                                 child_obj.local_transform = kg.FrameTransform(p_from_child)
                            else:
                                 child_obj.local_transform = p_from_child
                            
                            logging.info(f"[{child_name}] Updated dynamic transform from {actor_name}")
                            
                        # 5. Update Dimensions of ReferencePlane children
                        # If the child object (Frame) has children that are ReferencePlanes, update them.
                        # We need to find objects that have 'parent' == child_name and are ReferencePlane
                        for plane in self.reference_planes:
                            if plane.parent_name == child_name:
                                if plane.update_dimensions(data['width'], data['height']):
                                    logging.info(f"[{plane.name}] Updated dimensions: {data['width']}x{data['height']}")

            except Exception as e:
                logging.error(f"Error updating dynamic transforms from {file_path}: {e}")

    def calculate_scene_bounds(self):
        """Calculate bounds based on all loaded objects."""
        bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
        
        has_objects = False
        for obj in self.objects:
            has_objects = True
            
            # 1. Include Mesh Bounds
            if obj.mesh:
                # Create a copy and transform it
                mesh_copy = obj.mesh.copy()
                mesh_copy.transform(obj.global_transform.data, inplace=True)
                obj_bounds = mesh_copy.bounds
                
                bounds[0] = min(bounds[0], obj_bounds[0])
                bounds[1] = max(bounds[1], obj_bounds[1])
                bounds[2] = min(bounds[2], obj_bounds[2])
                bounds[3] = max(bounds[3], obj_bounds[3])
                bounds[4] = min(bounds[4], obj_bounds[4])
                bounds[5] = max(bounds[5], obj_bounds[5])
            
            # 2. Include Object Origin (Crucial for virtual frames)
            origin = obj.global_transform.t
            bounds[0] = min(bounds[0], origin[0])
            bounds[1] = max(bounds[1], origin[0])
            bounds[2] = min(bounds[2], origin[1])
            bounds[3] = max(bounds[3], origin[1])
            bounds[4] = min(bounds[4], origin[2])
            bounds[5] = max(bounds[5], origin[2])
        
        if not has_objects:
            # Default bounds if no objects
            self.fixed_grid_bounds = [-300, 300, -300, 300, -300, 300]
        else:
            # Add some padding (e.g. 20%)
            padding = 0.2
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            
            self.fixed_grid_bounds = [
                bounds[0] - dx*padding, bounds[1] + dx*padding,
                bounds[2] - dy*padding, bounds[3] + dy*padding,
                bounds[4] - dz*padding, bounds[5] + dz*padding
            ]
            logging.info(f"Calculated Scene Bounds: {self.fixed_grid_bounds}")

    def update_grid(self):
        """Update grid display based on settings."""
        if self.adjustable_grid:
            # Auto-adjusting grid
            self.plotter.show_grid(font_size=8)
        else:
            # Fixed grid with integer bounds
            self.plotter.show_grid(
                bounds=self.fixed_grid_bounds,
                font_size=8,
                xtitle='X (mm)',
                ytitle='Y (mm)',
                ztitle='Z (mm)'
            )

    def take_screenshot(self):
        """Capture screenshot and save to file."""
        
        if not os.path.exists(self.screenshot_path):
            os.makedirs(self.screenshot_path, exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(self.screenshot_path, filename)
        
        self.plotter.screenshot(filepath)
        logging.info(f"Screenshot saved to {filepath}")
        
        self.show_temporary_message("Screenshot captured")

    def show_temporary_message(self, message, duration=2.0):
        """Show a temporary message on the plotter."""
        actor = self.plotter.add_text(message, position='upper_left', font_size=12, color='white')
        
        # Schedule removal
        if hasattr(self, 'panel') and self.panel:
            self.panel.root.after(int(duration * 1000), lambda: self.plotter.remove_actor(actor))

    def toggle_logging(self):
        """Toggle data logging on/off."""
        self.is_logging = not self.is_logging
        
        if self.is_logging:
            # Start Logging
            if not os.path.exists(self.logging_path):
                os.makedirs(self.logging_path, exist_ok=True)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"log_{timestamp}.csv"
            filepath = os.path.join(self.logging_path, filename)
            
            try:
                self.log_file = open(filepath, 'w')
                # Write Header
                self.log_file.write("Timestamp,Type,Name,Data\n")
                self.log_start_time = time.time()
                self.show_temporary_message(f"Logging Started: {filename}")
                logging.info(f"Logging started to {filepath}")
            except Exception as e:
                logging.error(f"Failed to start logging: {e}")
                self.is_logging = False
        else:
            # Stop Logging
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            self.show_temporary_message("Logging Stopped")
            logging.info("Logging stopped")
            
        # Update UI Indicator
        if hasattr(self, 'panel') and self.panel:
            self.panel.update_logging_status(self.is_logging)
        else:
            logging.warning("Control Panel not found, cannot update logging status")

    def log_data(self):
        """Log current state of transforms, vectors, and annotations."""
        if not self.is_logging or not self.log_file:
            return

        current_time = time.time()
        # Log at approximately 1Hz
        if current_time - self.last_log_time < 1.0:
            return
            
        self.last_log_time = current_time
        elapsed = current_time - self.log_start_time
        
        try:
            # 1. Log Transforms
            for name, obj in self.object_map.items():
                # Format: R00, R01, R02, Tx, R10, R11, R12, Ty, R20, R21, R22, Tz (Row-major 3x4)
                # Get top 3 rows of the 4x4 matrix
                matrix_3x4 = obj.global_transform.data[:3, :]
                flat_data = matrix_3x4.flatten()
                data_str = ",".join([f"{x:.4f}" for x in flat_data])
                self.log_file.write(f"{elapsed:.3f},Transform,{name},{data_str}\n")
                
            # 2. Log Vectors
            for vec in self.custom_vectors:
                if vec.start_pos is not None and vec.end_pos is not None:
                    data_str = f"{vec.start_pos[0]:.4f},{vec.start_pos[1]:.4f},{vec.start_pos[2]:.4f},{vec.end_pos[0]:.4f},{vec.end_pos[1]:.4f},{vec.end_pos[2]:.4f}"
                    self.log_file.write(f"{elapsed:.3f},Vector,{vec.name},{data_str}\n")
                    
            # 3. Log Annotations (start, end)
            for ann in self.annotations:
                if ann.current_start_world is not None and ann.current_end_world is not None:
                    # Format: StartX, StartY, StartZ, EndX, EndY, EndZ
                    s = ann.current_start_world
                    e = ann.current_end_world
                    data_str = f"{s[0]:.4f},{s[1]:.4f},{s[2]:.4f},{e[0]:.4f},{e[1]:.4f},{e[2]:.4f}"
                    self.log_file.write(f"{elapsed:.3f},Annotation,{ann.name},{data_str}\n")
                    
            self.log_file.flush() # Ensure data is written
            
        except Exception as e:
            logging.error(f"Error during logging: {e}")

    def show(self):

        # Create Control Panel
        self.panel = ControlPanel(self)
        
        # Main Loop
        def update_plotter():
            # 1. Update Dynamic Inputs (Local Transforms)
            self._update_dynamic_transforms() 
            self._update_dynamic_annotations()
            
            # 2. Update All Objects (Global Transforms & Actors)
            for obj in self.objects:
                obj.update_transform(self.transform_map)
                
            # 3. Update Dependent Visuals (Arrows, Planes, etc.)
            self._update_dependent_transforms()
            self.update_reference_planes()
            self.update_custom_vectors()
            self.update_annotations()
            
            # 4. Logging & Render
            self.log_data() 
            self.plotter.update()
            self.panel.root.after(16, update_plotter) # ~60 FPS



        # Add key binding for screenshot
        self.plotter.add_key_event('s', self.take_screenshot)
        self.plotter.add_key_event('S', self.take_screenshot)
        
        # Add key binding for logging
        self.plotter.add_key_event('l', self.toggle_logging)
        self.plotter.add_key_event('L', self.toggle_logging)
        
        self.plotter.show(interactive_update=True)
        
        # Start update loop
        update_plotter()
        
        # Start Tkinter loop (Blocking)
        try:
            self.panel.root.mainloop()
        except KeyboardInterrupt:
            pass
        
        # Close plotter when GUI closes
        self.plotter.close()


if __name__ == "__main__":
    import sys
    # Path to config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        # If relative path, make absolute
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
    else:
        config_path = os.path.join(current_dir, "configs", "config.yaml")
    
    if not os.path.exists(config_path):
        logging.error(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    viz = SE3Visualizer(config_path)
    viz.show()
