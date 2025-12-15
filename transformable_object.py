import pyvista as pv
import numpy as np
import sys
import os
import logging

# Add the project root to the path so we can import geo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import geo.core as kg
from scipy.spatial.transform import Rotation
from data_loaders import load_mesh, load_landmarks, load_segmentation, segmentation_to_mesh
import transform_parser


class TransformableObject:
    def __init__(self, name, abbreviation, plotter, mesh_path=None, color="white", ct_origin=None, landmarks_path=None, segmentation_path=None, initial_transform=None, visual_settings=None,                 shape_type=None,
                 shape_params=None,
                 obj_type='model',
                 movable=True):
        self.name = name
        self.abbreviation = abbreviation
        self.plotter = plotter
        self.obj_type = obj_type
        self.movable = movable
        self.label_color = color
        self.visible = True
        
        # Parse visual settings with defaults
        self.visual_settings = visual_settings or {}
        self.seg_color = self.visual_settings.get('segmentation_color', self.label_color)
        self.seg_opacity = self.visual_settings.get('segmentation_opacity', 0.5)
        self.lm_label_size = self.visual_settings.get('landmark_label_font_size', 8)
        self.lm_label_color = self.visual_settings.get('landmark_label_color', self.label_color)
        self.lm_point_color = self.visual_settings.get('landmark_point_color', self.label_color)
        self.lm_point_opacity = self.visual_settings.get('landmark_point_opacity', 0.8)
        self.ft_color = self.visual_settings.get('frame_transform_color', 'lightgreen')
        self.frame_label_color = self.visual_settings.get('frame_label_color', self.label_color)
        self.frame_label_size = self.visual_settings.get('frame_label_size', 10)
        
        # Hierarchy
        self.parent = None
        self.children = []
        self._cached_global_transform = None
        self.constraint_expression = None
        self.dependents = [] # Objects that depend on this object
        self.dependencies = [] # Objects that this object depends on
        self.is_updating = False # Recursion guard
        self._last_visual_transform = None # Track last transform used for visual update
        
        if initial_transform is not None:
            self.initial_global_transform = initial_transform
            
            # Decompose: T = T_rigid @ S
            # We assume S is diagonal scaling along local axes (columns of R)
            R_full = initial_transform[:3, :3]
            self.initial_scale = np.linalg.norm(R_full, axis=0)
            self.initial_scale[self.initial_scale < 1e-9] = 1.0
            
            # Create Rigid Transform (Orthonormalize R)
            R_norm = R_full / self.initial_scale
            T_rigid_data = np.eye(4)
            T_rigid_data[:3, :3] = R_norm
            T_rigid_data[:3, 3] = initial_transform[:3, 3]
            
            self.local_transform = kg.FrameTransform(T_rigid_data)
            self.initial_local_transform = T_rigid_data # Store rigid initial
        else:
            self.initial_global_transform = np.eye(4)
            self.local_transform = kg.FrameTransform(np.eye(4))
            self.initial_scale = np.ones(3)
            self.initial_local_transform = np.eye(4)
            
        # Current scale is always 1.0 for SE(3) objects
        self.scale = np.ones(3)
            
        # Calculate inverse RIGID transform to move World-space objects to Local Rigid space
        # Objects loaded from files (Mesh, Seg, Landmarks) are in World Coordinates (LPS)
        # We want to store them in Local Rigid Frame (physically scaled but locally oriented)
        # T_rigid * P_local_rigid = P_world
        # P_local_rigid = inv(T_rigid) * P_world
        world_to_local = np.linalg.inv(self.local_transform.data)
        
        self.mesh = None
        self.actor = None
        self.show_mesh = False
        self.show_model = True
        self.show_vector = True
        
        if mesh_path and os.path.exists(mesh_path):
            mesh_data = load_mesh(mesh_path, origin=ct_origin)
            self.mesh = mesh_data['mesh']
            # Transform mesh to Local Frame
            self.mesh.transform(world_to_local, inplace=True)
            
            # Create Mesh Actor
            self.actor = self.plotter.add_mesh(self.mesh, color=color, opacity=0.3, show_edges=False)
            self.actor.VisibilityOff()
            self.actor.user_matrix = self.global_transform.data
            
        elif shape_type == 'box':
            # Create Box
            size = shape_params.get('size', [10, 10, 10])
            # Create Box centered at origin

            sx, sy, sz = size
            self.mesh = pv.Box(bounds=(-sx/2, sx/2, -sy/2, sy/2, -sz/2, sz/2))
            
            # Apply color/opacity from params if available, else use defaults
            shape_color = shape_params.get('color', color)
            shape_opacity = shape_params.get('opacity', 0.8)
            
            self.actor = self.plotter.add_mesh(self.mesh, color=shape_color, opacity=shape_opacity, show_edges=True)
            # Basic shapes are usually shown by default
            self.show_mesh = True
            self.actor.user_matrix = self.global_transform.data

        # Create Frame Actors
        self.frame_actors = []
        self.frame_scale = 30.0
        self.current_frame_scale = None # To track if frame actors need recreation
        
        # Origin Label Actor
        self.origin_label_actor = None
        
        # Transform Vector Actors
        self.vector_actor = None
        self.vector_label_actor = None
        
        # Landmarks
        self.landmarks = None
        self.landmarks_actors = []
        self.landmarks_label_actors = []
        self.show_landmarks = True
        if landmarks_path and os.path.exists(landmarks_path):
            # Load landmarks (World LPS)
            self.landmarks = load_landmarks(landmarks_path)
            logging.info(f"[{self.name}] Loaded landmarks from {os.path.basename(landmarks_path)}")
            logging.debug(f"[{self.name}] Landmarks Coordinate System: {self.landmarks.get('coordinate_system', 'Unknown')} -> Converted to LPS")
            
            # Transform landmarks to Local Frame
            points_homog = np.hstack([self.landmarks['points'], np.ones((len(self.landmarks['points']), 1))])
            points_local = (world_to_local @ points_homog.T).T
            self.landmarks['points'] = points_local[:, :3]

            self._create_landmark_actors()
            self._update_landmark_transforms()
        
        # Segmentation
        self.segmentation_mesh = None
        self.segmentation_actor = None
        self.show_segmentation = True
        if segmentation_path and os.path.exists(segmentation_path):
            seg_data = load_segmentation(segmentation_path)
            logging.info(f"[{self.name}] Loaded segmentation from {os.path.basename(segmentation_path)}")
            logging.debug(f"[{self.name}] Segmentation Coordinate System: RAS (converted to LPS)")
            logging.debug(f"[{self.name}] Loaded segmentation with labels: {seg_data['labels']}")
            
            # Convert to mesh (World Coordinates)
            self.segmentation_mesh = segmentation_to_mesh(
                seg_data['data'], 
                seg_data['affine'], 
                label=1
            )
            
            # Transform to Local Frame
            self.segmentation_mesh.transform(world_to_local, inplace=True)

            self._create_segmentation_actor()
            self._update_segmentation_transform()
            
        # Initial Update
        self.update_transform()
        
        # Subscription State
        self.is_subscribed = False
        self.subscription_file_path = None
        self.last_file_mtime = 0
        self.last_update_time = 0 # For throttling updates

    def _create_segmentation_actor(self):
        """Create or recreate segmentation actor."""
        if self.segmentation_actor:
            self.plotter.remove_actor(self.segmentation_actor)
            self.segmentation_actor = None
            
        if not self.show_segmentation or self.segmentation_mesh is None:
            return

        self.segmentation_actor = self.plotter.add_mesh(
            self.segmentation_mesh,
            color=self.seg_color,
            opacity=self.seg_opacity,
            show_edges=False,
            show_scalar_bar=False
        )
        self.segmentation_actor.user_matrix = self.global_transform.data
        if not self.visible:
            self.segmentation_actor.VisibilityOff()

    def _update_segmentation_transform(self):
        """Update segmentation position based on current transform."""
        if self.segmentation_actor:
            self.segmentation_actor.user_matrix = self.global_transform.data

    def set_show_segmentation(self, visible):
        """Toggle segmentation visibility."""
        self.show_segmentation = visible
        if self.segmentation_actor:
            if visible and self.visible:
                self.segmentation_actor.VisibilityOn()
            else:
                self.segmentation_actor.VisibilityOff()
        elif visible and self.visible:
            self._create_segmentation_actor()

    def set_show_mesh(self, show):
        self.show_mesh = show
        if self.actor:
            if show and self.visible:
                self.actor.VisibilityOn()
            else:
                self.actor.VisibilityOff()

    def set_show_model(self, visible):
        """Toggle visibility of model. Prioritizes segmentation if available."""
        if self.segmentation_mesh:
            self.set_show_segmentation(visible)
            # If we have segmentation, we hide the fallback mesh (STL) to avoid duplicates/z-fighting
            self.set_show_mesh(False)
        else:
            self.set_show_mesh(visible)

    def _update_coordinate_frame(self, scale):
        """Update the coordinate frame actors. Creates them if needed, updates transform."""
        
        # Check if we need to recreate actors (missing or scale changed)
        if not self.frame_actors or self.current_frame_scale != scale:
            # Clear existing actors
            for actor in self.frame_actors:
                self.plotter.remove_actor(actor)
            self.frame_actors = []
            
            # Create XYZ Arrows at Origin
            start = np.array([0.0, 0.0, 0.0])
            directions = [
                np.array([1.0, 0.0, 0.0]), # X
                np.array([0.0, 1.0, 0.0]), # Y
                np.array([0.0, 0.0, 1.0])  # Z
            ]
            colors = ["red", "green", "blue"]
            
            for i in range(3):
                arrow = pv.Arrow(start=start, direction=directions[i], scale=scale,
                               shaft_radius=0.015, tip_radius=0.04, tip_length=0.15)
                actor = self.plotter.add_mesh(arrow, color=colors[i], show_scalar_bar=False)
                self.frame_actors.append(actor)
                
            self.current_frame_scale = scale
            
        # Update Transform of existing actors
        for actor in self.frame_actors:
            actor.user_matrix = self.global_transform.data
            
            if self.visible:
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()

    def set_axes_scale(self, scale):
        """Update the scale of the coordinate frame axes."""
        self.frame_scale = scale
        self._update_coordinate_frame(scale)

    @property
    def global_transform(self):
        """Calculate global transform by chaining parent transforms."""
        if self._cached_global_transform is not None:
            return self._cached_global_transform

        if self.parent:
            # T_global = T_parent_global * T_local
            return self.parent.global_transform @ self.local_transform
        else:
            return self.local_transform

    @property
    def transform(self):
        """Alias for local_transform for backward compatibility with some calls."""
        return self.local_transform

    def calculate_transform(self):
        return self.global_transform

    def update_transform(self, transform_map=None):
        if self.is_updating:
            return
        self.is_updating = True
        
        try:
            # Apply Constraint if exists
            if self.constraint_expression and transform_map:
                try:
                    # Build available transforms map for parser
                    available = {}
                    skipped_self_refs = []
                    
                    for t_name, obj in transform_map.items():
                        if obj == self: 
                            skipped_self_refs.append(t_name)
                            continue # Prevent direct self-reference
                        
                        # We expose all objects in the map
                        # The parser expects { 'matrix': ..., 'editable': ... }
                        available[t_name.lower()] = {
                            'matrix': obj.local_transform.data,
                            'editable': obj.movable,
                            'original_name': t_name
                        }
                    
                    # Evaluate expression
                    result_matrix = transform_parser.parse_transform_expression(self.constraint_expression, available)
                    
                    # Apply
                    try:
                        self.local_transform.data = result_matrix
                        self.scale = np.ones(3) # Reset scale
                    except Exception as e_assign:
                         # Fallback to robust decomposition if direct assignment fails (e.g. non-rigid)
                        logging.debug(f"[{self.name}] Direct assignment failed ({e_assign}), attempting robust decomposition.")
                        
                        # Decompose result to get R and t
                        R = result_matrix[:3, :3]
                        t = result_matrix[:3, 3]
                        
                        # Extract scale from result
                        res_scale = np.linalg.norm(R, axis=0)
                        res_scale[res_scale < 1e-9] = 1.0
                        R_norm = R / res_scale
                        
                        T_rigid = np.eye(4)
                        T_rigid[:3, :3] = R_norm
                        T_rigid[:3, 3] = t
                        
                        self.local_transform = kg.FrameTransform(T_rigid)
                        logging.debug(f"[{self.name}] Updated from constraint (robust): {self.constraint_expression}")

                except Exception as e:
                    # Check for circular dependency
                    is_circular = False
                    error_msg = str(e)
                    for skipped in skipped_self_refs:
                        if f"'{skipped}' not found" in error_msg or f"'{skipped.lower()}' not found" in error_msg:
                            logging.error(f"[{self.name}] CRITICAL: Circular Dependency detected! Constraint uses '{skipped}' which maps to this object itself.")
                            sys.exit(1)
                            is_circular = True
                            
                    if not is_circular:
                        logging.error(f"[{self.name}] Error evaluating constraint '{self.constraint_expression}': {e}")

            # Update Cache
            if self.parent:
                self._cached_global_transform = self.parent.global_transform @ self.local_transform
            else:
                self._cached_global_transform = self.local_transform

            # Update Visuals only if transform changed significantly
            current_global = self.global_transform.data
            visual_update_needed = True
            if self._last_visual_transform is not None:
                if np.allclose(current_global, self._last_visual_transform, atol=1e-5):
                    visual_update_needed = False
            
            if visual_update_needed:
                # Update Mesh (if exists)
                if self.actor:
                    self.actor.user_matrix = current_global
                
                # Update Frame
                # Update coordinate frame (reuses actors)
                self._update_coordinate_frame(self.frame_scale)
                    
                # Update Segmentation (if exists)
                if self.segmentation_actor:
                    self._update_segmentation_transform()
                    
                # Update Labels
                self._update_labels()
                
                # Update Transform Vector
                self._update_vector()
                
                # Update Landmarks
                self._update_landmark_transforms()
                
                self._last_visual_transform = current_global.copy()
            
            for child in self.children:
                child.update_transform(transform_map)
                
            # 5. Notify Dependents
            if self.dependents:
                for dep in self.dependents:
                    # Prevent infinite loops if dependent is also a dependency (shouldn't happen in DAG)
                    if hasattr(dep, 'is_updating') and dep.is_updating:
                        continue
                        
                    # Pass transform_map to dependent
                    if hasattr(dep, 'update_transform'):
                        dep.update_transform(transform_map)
        finally:
            self.is_updating = False

    def get_landmark_world_position(self, label):
        """
        Get the world coordinates of a specific landmark.
        
        Args:
            label: The label/name of the landmark.
            
        Returns:
            np.array (3,) or None if landmark not found.
        """
        if not self.landmarks or 'labels' not in self.landmarks or 'points' not in self.landmarks:
            return None
            
        try:
            # Find index of label
            idx = self.landmarks['labels'].index(label)
            local_pos = self.landmarks['points'][idx]
            
            # Transform to World
            # T_global * P_local = P_world
            # Note: local_pos is (3,), we need (4,) for matmul
            local_pos_homog = np.append(local_pos, 1)
            world_pos = (self.global_transform.data @ local_pos_homog)[:3]
            
            return world_pos
        except ValueError:
            return None

    def _update_labels(self):
        # Remove old label
        if hasattr(self, 'origin_label_actor') and self.origin_label_actor:
            self.plotter.remove_actor(self.origin_label_actor)
            
        if not self.visible:
            return

        # Calculate label position (offset from origin)
        local_pos = np.array([10, 10, 10]) # Simple offset
        local_pos_homog = np.append(local_pos, 1)
        world_pos = (self.global_transform.data @ local_pos_homog)[:3]
        
        self.origin_label_actor = self.plotter.add_point_labels(
            [world_pos], [self.abbreviation],
            font_size=self.frame_label_size, text_color=self.frame_label_color,
            show_points=False, always_visible=True
        )

    def _update_vector(self):
        if not self.show_vector or not self.visible:
            if self.vector_actor: self.vector_actor.VisibilityOff()
            if self.vector_label_actor: self.vector_label_actor.VisibilityOff()
            return
            
        if self.parent:
            start = self.parent.global_transform.t
        else:
            start = np.array([0.0, 0.0, 0.0])
            
        end = self.global_transform.t
        
        # Calculate direction and length
        vec = end - start
        length = float(np.linalg.norm(vec))
        
        if length < 1e-6:
            if self.vector_actor: self.vector_actor.VisibilityOff()
            if self.vector_label_actor: self.vector_label_actor.VisibilityOff()
            return
            
        direction = vec / length
        
        # Create Arrow Mesh
        fixed_shaft_radius = 1.5
        fixed_tip_radius = 4.0
        fixed_tip_length = 15.0
        
        arrow_mesh = pv.Arrow(start=start, direction=direction, scale=length, 
                        shaft_radius=fixed_shaft_radius/length, 
                        tip_radius=fixed_tip_radius/length, 
                        tip_length=fixed_tip_length/length)
                        
        if self.vector_actor:
            # Update existing actor's mesh
            self.vector_actor.mapper.dataset.copy_from(arrow_mesh)
            self.vector_actor.VisibilityOn()
        else:
            # Create new
            self.vector_actor = self.plotter.add_mesh(arrow_mesh, color=self.ft_color, opacity=0.8, show_scalar_bar=False)
        
        # Update Label
        midpoint = (start + end) / 2
        parent_name = self.parent.abbreviation if self.parent else "W"
        label_text = f"{parent_name}_from_{self.abbreviation}"
        
        # So we recreate it, but only if text or position changed significantly? 
        # For now, we recreate as it's 2D and less prone to flicker than 3D meshes
        if self.vector_label_actor:
            self.plotter.remove_actor(self.vector_label_actor)
            
        self.vector_label_actor = self.plotter.add_point_labels(
            [midpoint], [label_text],
            font_size=12, text_color=self.ft_color,
            show_points=False, always_visible=False
        )

    def _create_landmark_actors(self):
        """Create PyVista actors for landmarks."""
        if not self.landmarks:
            return
            
        # Clear existing actors
        for actor in self.landmarks_actors:
            self.plotter.remove_actor(actor)
        for actor in self.landmarks_label_actors:
            self.plotter.remove_actor(actor)
        self.landmarks_actors = []
        self.landmarks_label_actors = []
        
        if not self.show_landmarks or not self.visible:
            return
            
        # Create spheres for each landmark (in Local Frame, transformed by user_matrix)
        for i, point in enumerate(self.landmarks['points']):
            sphere = pv.Sphere(radius=2.0, center=point)
            actor = self.plotter.add_mesh(sphere, color=self.lm_point_color, opacity=self.lm_point_opacity, show_scalar_bar=False)
            self.landmarks_actors.append(actor)

    def _update_landmark_transforms(self):
        """Update landmark positions based on current transform."""
        if not self.landmarks:
            return
            
        # Apply transform to sphere actors
        for actor in self.landmarks_actors:
            actor.user_matrix = self.global_transform.data
            
        # Recreate labels at transformed positions
        for actor in self.landmarks_label_actors:
            self.plotter.remove_actor(actor)
        self.landmarks_label_actors = []
        
        if not self.show_landmarks or not self.visible:
            return
            
        # Calculate world positions for labels
        points_local = self.landmarks['points']
        points_homog = np.hstack([points_local, np.ones((len(points_local), 1))])
        points_world = (self.global_transform.data @ points_homog.T).T[:, :3]
        
        for i, (point, label) in enumerate(zip(points_world, self.landmarks['labels'])):
            label_actor = self.plotter.add_point_labels(
                [point], [label],
                font_size=self.lm_label_size, text_color=self.lm_label_color,
                show_points=False, always_visible=False
            )
            self.landmarks_label_actors.append(label_actor)

    def set_show_landmarks(self, visible):
        """Toggle landmark visibility."""
        self.show_landmarks = visible
        self._create_landmark_actors()
        self._update_landmark_transforms()

    def set_show_vector(self, visible):
        self.show_vector = visible
        self.update_transform()

    def reset(self, transform_map=None):
        """Reset transform to initial state."""
        # Reset local transform to initial RIGID transform
        if hasattr(self, 'initial_local_transform'):
             self.local_transform = kg.FrameTransform(self.initial_local_transform)
        else:
             # Fallback if not stored (shouldn't happen with new init)
             self.local_transform = kg.FrameTransform(np.eye(4))

        # Reset scale to 1.0
        self.scale = np.ones(3)
        self.update_transform(transform_map)

    def get_kinematic_chain_string(self):
        """
        Construct a string representation of the kinematic chain from World to this object.
        Format: W_from_P @ P_from_C ...
        """
        chain = []
        current = self
        
        while current:
            parent = current.parent
            if parent:
                # Format: ParentAbbr_from_ChildAbbr
                # Use abbreviation if available, else name
                p_name = parent.abbreviation if parent.abbreviation else parent.name
                c_name = current.abbreviation if current.abbreviation else current.name
                
                chain.append(f"{p_name}_from_{c_name}")
                current = parent
            else:
                # Reached root (World or detached)
                break
                
        if not chain:
            return "World" # Or Identity
            
        # The chain is built bottom-up (Parent_from_Child, Grandparent_from_Parent...)
        # We want to display it top-down: Grandparent_from_Parent @ Parent_from_Child
        return " @ ".join(reversed(chain))

    def get_transform_str(self):
        F = self.local_transform
        parent_name = self.parent.name if self.parent else "World"
        matrix_str = f"{parent_name}_from_{self.name}:\n"
        with np.printoptions(precision=2, suppress=True):
            matrix_str += str(F.data)
            
            # Add decomposition info
            R_norm = F.R / self.scale
            matrix_str += "\n\nRotation (Orthonormal):\n"
            matrix_str += str(R_norm)
            matrix_str += f"\n\nScale: {self.scale}"
        return matrix_str
    
    def get_rotation_euler(self):
        """Get current rotation in Euler angles (degrees)."""
        # Calculate current scale dynamically
        current_R = self.local_transform.R
        current_scale = np.linalg.norm(current_R, axis=0)
        current_scale[current_scale < 1e-9] = 1.0
        
        # Normalize the rotation matrix by scale to get pure rotation
        R_normalized = current_R / current_scale
        return Rotation.from_matrix(R_normalized).as_euler('xyz', degrees=True)

    def set_translation(self, axis, value, transform_map=None):
        # Update translation in the current transform
        current_t = self.local_transform.t.copy()
        current_t[axis] = value
        self.local_transform.t = current_t
        logging.info(f"[{self.name}] Set Translation {axis}: {value} -> {self.local_transform.t}")
        self.update_transform(transform_map)

    def set_rotation_euler(self, axis, value, transform_map=None):
        # Calculate current scale dynamically
        current_R = self.local_transform.R
        current_scale = np.linalg.norm(current_R, axis=0)
        current_scale[current_scale < 1e-9] = 1.0
        
        # Update rotation
        current_euler = self.get_rotation_euler()
        current_euler[axis] = value
        
        # Reconstruct transform
        new_R_pure = Rotation.from_euler('xyz', current_euler, degrees=True).as_matrix()
        # Re-apply scale
        new_R_scaled = new_R_pure * current_scale
        
        self.local_transform.R = new_R_scaled
        logging.info(f"[{self.name}] Set Rotation {axis}: {value} -> Euler: {current_euler}")
        self.update_transform(transform_map)

    def set_visible(self, visible):
        """Toggle visibility of the entire object (mesh, frame, vector, landmarks)."""
        changed = (self.visible != visible)
        self.visible = visible
        
        # Update components
        self.set_show_model(self.show_model)
        
        # Frame
        for actor in self.frame_actors:
            actor.SetVisibility(visible)
            
        # Label
        if self.origin_label_actor:
            self.origin_label_actor.SetVisibility(visible)
            
        # Vector
        if self.vector_actor:
            self.vector_actor.SetVisibility(visible)
        if self.vector_label_actor:
            self.vector_label_actor.SetVisibility(visible)
            
        # Landmarks
        for actor in self.landmarks_actors:
            actor.SetVisibility(visible)
        for actor in self.landmarks_label_actors:
            actor.SetVisibility(visible)
            
        return changed

    def add_dependent(self, obj):
        """Add an object that depends on this object."""
        if obj not in self.dependents:
            self.dependents.append(obj)
            # logging.info(f"[{self.name}] Added dependent: {obj.name}")
            
    def remove_dependent(self, obj):
        if obj in self.dependents:
            self.dependents.remove(obj)
            
    def register_dependencies(self, new_dependencies):
        """
        Update the list of objects this object depends on.
        Handles subscribing to new dependencies and unsubscribing from old ones.
        """
        old_deps = set(self.dependencies)
        new_deps_set = set(new_dependencies)
        
        # Unsubscribe from removed dependencies
        for dep in old_deps - new_deps_set:
            dep.remove_dependent(self)
            
        # Subscribe to new dependencies
        for dep in new_deps_set - old_deps:
            dep.add_dependent(self)
            
        self.dependencies = list(new_deps_set)

    def set_parent(self, new_parent):
        """
        Change the parent of this object.
        Updates the parent's children list and this object's parent reference.
        Does NOT automatically update the local transform to preserve global position; 
        caller must handle transform updates if needed.
        """
        if self.parent == new_parent:
            return

        # Remove from old parent
        if self.parent:
            if self in self.parent.children:
                self.parent.children.remove(self)
        
        # Set new parent
        self.parent = new_parent
        
        # Add to new parent
        if new_parent:
            if self not in new_parent.children:
                new_parent.children.append(self)
                
        logging.info(f"[{self.name}] Reparented to {new_parent.name if new_parent else 'None'}")
