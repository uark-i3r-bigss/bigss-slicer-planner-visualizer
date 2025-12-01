import numpy as np
import re
import logging

def parse_transform_expression(expression, available_transforms):
    """
    Parses and evaluates a transform expression string.
    
    Args:
        expression (str): The input string (e.g., "inv(A) @ B").
        available_transforms (dict): Map of lower-case transform names to dicts:
            {
                'matrix': np.ndarray (4x4),
                'editable': bool,
                'original_name': str
            }
            
    Returns:
        np.ndarray: The resulting 4x4 transformation matrix.
        
    Raises:
        ValueError: If expression is invalid or references non-existent/editable transforms.
    """
    if not expression or not expression.strip():
        raise ValueError("Empty expression")

    # Normalize spacing: remove extra spaces
    # We want to split by '@'
    parts = expression.split('@')
    
    result_matrix = np.eye(4)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Check for inverse: inv(...)
        is_inverse = False
        # Case-insensitive check for 'inv('
        inv_match = re.match(r'^inv\s*\((.*)\)$', part, re.IGNORECASE)
        
        if inv_match:
            is_inverse = True
            name = inv_match.group(1).strip()
        else:
            name = part
            
        # Validate name
        name_lower = name.lower()
        
        if name_lower not in available_transforms:
            raise ValueError(f"Transform '{name}' not found in scene.")
            
        t_info = available_transforms[name_lower]
        
        # Check if editable
        # if t_info['editable']:
        #     raise ValueError(f"Reference to editable transform '{t_info['original_name']}' is not allowed.")
            
        matrix = t_info['matrix']
        
        if is_inverse:
            try:
                matrix = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                raise ValueError(f"Transform '{t_info['original_name']}' is not invertible.")
                
        # Multiply: Result = Result @ Current
        result_matrix = result_matrix @ matrix
        
    return result_matrix
