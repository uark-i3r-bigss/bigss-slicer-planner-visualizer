import sys
import os
import yaml
import numpy as np
import geo.core as kg

# Add script directory to path
sys.path.append(os.getcwd())
from data_loaders import load_segmentation

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python se3_test.py <config_file>")
        return

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"=== SE3 Logic Verification ===")
    print(f"Testing Config: {config_path}\n")
    
    # Load the Input Config (Transforms)
    config = load_config(config_path)
    
    # Load Object Libraries
    # 1. Debug Objects (Basic Shapes)
    debug_objs_path = os.path.join(os.getcwd(), "configs/debug_objects.yaml")
    debug_library = {}
    if os.path.exists(debug_objs_path):
        debug_conf = load_config(debug_objs_path)
        for obj in debug_conf.get('objects', []):
            debug_library[obj['name']] = obj
            
    # 2. Model Objects (Standard Config)
    model_objs_path = os.path.join(os.getcwd(), "configs/config.yaml")
    model_library = {}
    if os.path.exists(model_objs_path):
        model_conf = load_config(model_objs_path)
        for obj in model_conf.get('objects', []):
            model_library[obj['name']] = obj

    # Identify required objects from transforms
    required_objects = set()
    for t in config.get('transforms', []):
        if t['parent'] != 'World':
            required_objects.add(t['parent'])
        if t['child'] != 'World':
            required_objects.add(t['child'])
            
    # Load Global Transforms
    global_transforms = {"World": np.eye(4)}
    
    print("--- Loading Objects ---")
    for obj_name in required_objects:
        affine = None
        
        # Check Debug Library
        if obj_name in debug_library:
            print(f"Loading {obj_name} from Debug Library (Basic Shape)...")
            obj_def = debug_library[obj_name]
            affine = np.array(obj_def['initial_transform'])
            
        # Check Model Library
        elif obj_name in model_library:
            print(f"Loading {obj_name} from Model Library...")
            obj_def = model_library[obj_name]
            paths = obj_def.get('paths', {})
            seg_path = paths.get('segmentation')
            if seg_path:
                full_path = os.path.join(os.getcwd(), seg_path)
                if os.path.exists(full_path):
                    seg_data = load_segmentation(full_path)
                    affine = seg_data['affine']
                else:
                    print(f"Warning: Segmentation file not found: {full_path}")
            else:
                affine = np.eye(4) # Default if no segmentation
                
        else:
            print(f"Error: Object {obj_name} not found in any library.")
            continue
            
        if affine is not None:
            global_transforms[obj_name] = affine
            print(f"Global Transform (World_from_{obj_name}):")
            with np.printoptions(precision=2, suppress=True):
                print(affine)

    print("\n--- Verifying Kinematic Chains ---")
    
    for t_config in config.get('transforms', []):
        parent_name = t_config['parent']
        child_name = t_config['child']
        transform_name = t_config['name']
        
        if parent_name not in global_transforms or child_name not in global_transforms:
            print(f"Skipping {transform_name}: Missing transform data for parent or child.")
            continue
            
        T_W_P = global_transforms[parent_name] # World_from_Parent
        T_W_C = global_transforms[child_name]  # World_from_Child
        
        # T_P_C = inv(T_W_P) @ T_W_C
        T_W_P_inv = np.linalg.inv(T_W_P)
        T_P_C = T_W_P_inv @ T_W_C
        
        print(f"\nTransform {transform_name} ({parent_name} <- {child_name}):")
        print(f"Calculated Local Transform ({parent_name}_from_{child_name}):")
        with np.printoptions(precision=2, suppress=True):
            print(T_P_C)
            
        # Verify reconstruction
        T_W_C_reconstructed = T_W_P @ T_P_C
        error = np.linalg.norm(T_W_C - T_W_C_reconstructed)
        print(f"Reconstruction Error (Global - (Parent @ Local)): {error:.6f}")
        
        if error < 1e-6:
            print("✓ Consistency Check Passed")
        else:
            print("✗ Consistency Check FAILED")

    print("\n=== End Verification ===")

if __name__ == "__main__":
    main()
