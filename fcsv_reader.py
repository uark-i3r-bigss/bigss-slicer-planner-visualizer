"""
FCSV (Fiducial CSV) file reader for 3D Slicer markup files.

Format: Comma-separated values with header lines starting with '#'
Columns: id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID
"""

import numpy as np
import pandas as pd
import logging

def read_fcsv(filepath):
    """
    Read a .fcsv file and return landmarks as a dictionary.
    
    Args:
        filepath: Path to the .fcsv file
        
    Returns:
        dict: Dictionary with keys:
            - 'points': numpy array of shape (N, 3) with x,y,z coordinates (in LPS)
            - 'labels': list of landmark labels
            - 'coordinate_system': string ('LPS' or 'RAS' - original system)
    """
    # Read header to get coordinate system
    coordinate_system = 'LPS'  # default
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('# CoordinateSystem'):
                value = line.split('=')[1].strip()
                if value == '0':
                    coordinate_system = 'LPS'
                elif value == '1':
                    coordinate_system = 'RAS'
                elif value in ['RAS', 'LPS']:
                    coordinate_system = value
                break
    
    # Read data, skipping comment lines
    df = pd.read_csv(filepath, comment='#', header=None)
    
    # Extract columns: id, x, y, z, ..., label
    # Based on the format: id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID
    points = df.iloc[:, 1:4].values  # columns 1,2,3 are x,y,z
    labels = df.iloc[:, 11].values   # column 11 is label
    
    # Convert to numpy array
    points = points.astype(np.float64)
    
    # Convert RAS to LPS if needed
    # RAS to LPS: negate X and Y coordinates
    if coordinate_system == 'RAS':
        logging.debug(f"Converting landmarks from RAS to LPS")
        points[:, 0] = -points[:, 0]  # negate X
        points[:, 1] = -points[:, 1]  # negate Y
        # Z stays the same
    
    return {
        'points': points,
        'labels': labels.tolist(),
        'coordinate_system': coordinate_system
    }

def read_txt_landmarks(filepath):
    """
    Read a .txt landmark file and return landmarks as a dictionary.
    
    Format example:
    # CoordinateSystem: RAS
    Trajectory,Landmark,R,A,S
    traj_1,Target_1,-13.3218,58.4892,32.3423
    
    Args:
        filepath: Path to the .txt file
        
    Returns:
        dict: Dictionary with keys:
            - 'points': numpy array of shape (N, 3) with x,y,z coordinates (in LPS)
            - 'labels': list of landmark labels
            - 'coordinate_system': string ('LPS' or 'RAS')
    """
    coordinate_system = 'LPS' # default
    
    # Read header to get coordinate system
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('# CoordinateSystem'):
                value = line.split(':')[1].strip()
                if value in ['RAS', 'LPS']:
                    coordinate_system = value
                break
    
    # Read data using pandas
    # Skip comment lines starting with #
    df = pd.read_csv(filepath, comment='#')
    
    # Columns are Trajectory, Landmark, R, A, S (or X, Y, Z)
    # We assume the last 3 columns are coordinates
    points = df.iloc[:, 2:5].values
    labels = df.iloc[:, 1].values # Landmark column
    
    # Convert to numpy array
    points = points.astype(np.float64)
    
    # Convert RAS to LPS if needed
    if coordinate_system == 'RAS':
        logging.debug(f"Converting txt landmarks from RAS to LPS")
        points[:, 0] = -points[:, 0] # negate X
        points[:, 1] = -points[:, 1] # negate Y
        
    return {
        'points': points,
        'labels': labels.tolist(),
        'coordinate_system': coordinate_system
    }

if __name__ == "__main__":
    # Test the reader
    import sys
    if len(sys.argv) > 1:
        result = read_fcsv(sys.argv[1])
        print(f"Coordinate System: {result['coordinate_system']}")
        print(f"Number of landmarks: {len(result['labels'])}")
        print("\nLandmarks:")
        for i, (label, point) in enumerate(zip(result['labels'], result['points'])):
            print(f"  {i+1}. {label}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
