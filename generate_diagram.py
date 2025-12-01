import yaml
import os
import sys
import argparse
import logging
try:
    import graphviz
except ImportError:
    print("Error: 'graphviz' python library is not installed.")
    print("Please install it using: pip install graphviz")
    print("Note: You also need the Graphviz system package installed (e.g., 'sudo apt install graphviz').")
    sys.exit(1)

"""
SE(3) Visualizer Scene Diagram Generator

Visual Legend:
- Nodes (Objects):
    - Double Circle (White): World Frame
    - Box (Light Grey): Model Object (Physical)
    - Box (Dashed, White): Virtual Object (Frame only)
    - Ellipse (Light Blue): Static Vector (Custom Vector or Static Annotation)
    - Ellipse (Light Yellow): Dynamic Vector (Dynamic Annotation)

- Edges (Transforms/Relationships):
    - Solid Line: Parent-Child Transform
    - Dotted Line: Vector Definition (Source -> Vector)
"""

def load_config(path):
    if not os.path.exists(path):
        logging.error(f"Config file not found at {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_diagram(config, output_path, format='png'):
    dot = graphviz.Digraph(comment='SE(3) Visualizer Scene', format=format)
    dot.attr(rankdir='TB') # Top to Bottom layout
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    # 1. Add Objects (Nodes)
    # We first collect all objects defined in 'objects' list
    defined_objects = {}
    for obj in config.get('objects', []):
        name = obj['name']
        obj_type = obj.get('type', 'model')
        defined_objects[name] = obj_type
        
        # Style based on type
        abbr = obj.get('abbreviation', '')
        label = f"{name} ({abbr})" if abbr else name
        
        if obj_type == 'virtual':
            dot.node(name, label, shape='box', style='dashed', color='black')
        else:
            # Model or basic_shapes
            dot.node(name, label, shape='box', style='filled', fillcolor='lightgrey', color='black')

    # Also add "World" if not explicitly defined but used
    if 'World' not in defined_objects:
        dot.node('World', 'World', shape='doublecircle', style='filled', fillcolor='white', color='black')
        defined_objects['World'] = 'virtual'

    # 2. Add Transforms (Edges)
    # Track which nodes are connected to ensure we add implicit nodes if missing from 'objects'
    for t in config.get('transforms', []):
        parent = t['parent']
        child = t['child']
        name = t['name']
        
        # Ensure nodes exist (if they weren't in 'objects' list for some reason)
        if parent not in defined_objects:
            dot.node(parent, parent, shape='box', style='dotted')
            defined_objects[parent] = 'unknown'
        if child not in defined_objects:
            dot.node(child, child, shape='box', style='dotted')
            defined_objects[child] = 'unknown'
            
        dot.edge(parent, child, label=name)

    # 3. Add Vectors
    for i, vec in enumerate(config.get('vectors', [])):
        name = vec['name']
        parent = vec['parent']
        
        # Create a node for the vector
        vec_node_name = f"vec_{name}"
        label = f"Vector: {name}\n(-> {vec.get('landmark_label', '?')})"
        dot.node(vec_node_name, label, shape='ellipse', style='filled', fillcolor='lightblue', fontsize='8')
        
        # Edge from Parent to Vector
        if parent in defined_objects:
            dot.edge(parent, vec_node_name, style='dotted', arrowhead='none')

    # 4. Add Annotations
    
    # Pre-process transforms to find dynamic annotation mappings
    # Map (annotation_name, vector_name) -> child_frame_name
    dynamic_vector_map = {}
    for t in config.get('transforms', []):
        if t.get('type') == 'dynamic_annotation':
            ann_name = t.get('annotation_name')
            vec_name = t.get('vector_name')
            child = t.get('child')
            if ann_name and vec_name and child:
                dynamic_vector_map[(ann_name, vec_name)] = child

    for i, ann in enumerate(config.get('annotations', [])):
        parent = ann['parent']
        ann_name = ann['name']
        
        if 'landmarks' in ann:
            # Dynamic Annotation Group: Expand to individual vectors
            for lm in ann['landmarks']:
                vec_name = lm['name']
                # Create a node for the vector
                vec_node_name = f"vec_{vec_name}"
                label = f"Vector: {vec_name}\n(Dynamic)"
                dot.node(vec_node_name, label, shape='ellipse', style='filled', fillcolor='lightyellow', fontsize='8')
                
                # Determine source node
                # If this vector drives a dynamic frame, the vector "belongs" to that frame visually
                source_node = parent
                style = 'dotted'
                
                if (ann_name, vec_name) in dynamic_vector_map:
                    source_node = dynamic_vector_map[(ann_name, vec_name)]
                    # Keep dotted style for consistency, but source is different
                    style = 'dotted' 
                
                # Edge from Source to Vector
                if source_node in defined_objects:
                    dot.edge(source_node, vec_node_name, style=style, arrowhead='none')
        else:
            # Static Annotation
            name = ann['name']
            ann_node_name = f"ann_{name}"
            label = f"Vector: {name}\n(Static)"
            dot.node(ann_node_name, label, shape='ellipse', style='filled', fillcolor='lightblue', fontsize='8')
            
            # Edge from Parent to Annotation
            if parent in defined_objects:
                dot.edge(parent, ann_node_name, style='dotted', arrowhead='none')

    # Render
    output_filename = os.path.join(output_path, 'scene_diagram')
    try:
        # render() saves the file. 'view=False' prevents opening it.
        # It appends the format extension automatically.
        result_path = dot.render(output_filename, view=False)
        logging.info(f"Diagram generated successfully: {result_path}")
    except graphviz.backend.ExecutableNotFound:
        logging.error("Graphviz executable 'dot' not found.")
        logging.error("Please install Graphviz system package (e.g., 'sudo apt install graphviz').")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error generating diagram: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate a diagram of the SE(3) Visualizer scene configuration.")
    parser.add_argument('--config', type=str, default=None, help="Path to config.yaml")
    parser.add_argument('--output', type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Default Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(current_dir, "configs", "config.yaml")
        
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(current_dir, "visualizer_outputs", "diagrams")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    logging.info(f"Generating diagram in: {output_dir}")
    generate_diagram(config, output_dir)

if __name__ == "__main__":
    main()
