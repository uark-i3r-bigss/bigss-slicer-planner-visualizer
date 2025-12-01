"""
Data loading utilities for the SE(3) visualizer.

This module provides functions to load various medical imaging data formats:
- CT volumes (NIfTI)
- Segmentation masks (NIfTI)
- 3D meshes (STL)
- Landmarks (FCSV)
"""

import numpy as np
import nibabel as nib
import pyvista as pv
from fcsv_reader import read_fcsv
import logging


def load_ct_volume(ct_path):
    """
    Load a CT volume from NIfTI file and convert affine to LPS if needed.
    
    Args:
        ct_path: Path to the NIfTI file
        
    Returns:
        dict with keys:
            - 'data': numpy array of CT values
            - 'affine': 4x4 affine transformation matrix (in LPS)
            - 'origin': image origin (x, y, z)
            - 'spacing': voxel spacing (x, y, z)
    """
    img = nib.load(ct_path)
    affine = img.affine
    
    # Check orientation
    orientation = nib.orientations.aff2axcodes(affine)
    
    # Convert affine to LPS if it is RAS
    if orientation == ('R', 'A', 'S'):
        logging.debug(f"Converting CT affine from RAS to LPS")
        # Pre-multiply by diag([-1, -1, 1, 1]) to flip X and Y in world space
        conversion = np.diag([-1, -1, 1, 1])
        affine = conversion @ affine
    
    origin = affine[:3, 3]
    spacing = np.abs(np.diag(affine[:3, :3]))
    
    return {
        'data': img.get_fdata(),
        'affine': affine,
        'origin': origin,
        'spacing': spacing
    }


def load_segmentation(seg_path):
    """
    Load a segmentation mask from NIfTI file and convert affine to LPS if needed.
    
    Args:
        seg_path: Path to the segmentation NIfTI file
        
    Returns:
        dict with keys:
            - 'data': numpy array of segmentation labels
            - 'affine': 4x4 affine transformation matrix (in LPS)
            - 'origin': image origin (x, y, z)
            - 'spacing': voxel spacing (x, y, z)
            - 'labels': unique label values in the segmentation
    """
    img = nib.load(seg_path)
    affine = img.affine
    
    # Check orientation
    orientation = nib.orientations.aff2axcodes(affine)
    
    # Convert affine to LPS if it is RAS
    if orientation == ('R', 'A', 'S'):
        logging.debug(f"Converting segmentation affine from RAS to LPS (orientation: {orientation})")
        conversion = np.diag([-1, -1, 1, 1])
        affine = conversion @ affine
    
    origin = affine[:3, 3]
    spacing = np.abs(np.diag(affine[:3, :3]))
    data = img.get_fdata()
    
    return {
        'data': data,
        'affine': affine,
        'origin': origin,
        'spacing': spacing,
        'labels': np.unique(data[data > 0])  # Non-zero labels
    }


def segmentation_to_mesh(seg_data, affine, label=1, reduction=0.5):
    """
    Convert a binary segmentation to a 3D mesh using marching cubes.
    
    Args:
        seg_data: 3D numpy array of segmentation
        affine: 4x4 affine transformation matrix (to transform voxel->world)
        label: Label value to extract (default: 1)
        reduction: Mesh reduction factor 0-1 (default: 0.5 for 50% reduction)
        
    Returns:
        PyVista mesh in World Coordinates
    """
    # Create binary mask for the specified label
    binary_mask = (seg_data == label).astype(np.float32)
    
    # Create PyVista ImageData in VOXEL coordinates (origin=0, spacing=1)
    grid = pv.ImageData()
    grid.dimensions = np.array(binary_mask.shape)
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    
    # Add the scalar data as point_data
    grid.point_data['values'] = binary_mask.flatten(order='F')
    
    # Extract surface using contour
    mesh = grid.contour([0.5], scalars='values')
    
    # Reduce mesh complexity
    if reduction > 0 and reduction < 1:
        mesh = mesh.decimate(reduction)
    
    # Transform mesh from Voxel to World space using the affine
    mesh.transform(affine, inplace=True)
    
    return mesh


def load_mesh(mesh_path, ct_path=None, origin=None):
    """
    Load a 3D mesh from STL file.
    
    Args:
        mesh_path: Path to the STL file
        ct_path: Optional path to corresponding CT for origin extraction (deprecated, use origin instead)
        origin: Optional origin to shift mesh (preferred over ct_path)
        
    Returns:
        dict with keys:
            - 'mesh': PyVista mesh
            - 'origin': image origin if provided, else (0,0,0)
    """
    mesh = pv.read(mesh_path)
    
    if origin is not None:
        # Use provided origin directly
        origin = np.array(origin)
        mesh.points -= origin
    elif ct_path:
        # Fallback: load CT to get origin (for backward compatibility)
        ct_data = load_ct_volume(ct_path)
        origin = ct_data['origin']
        mesh.points -= origin
    else:
        origin = np.zeros(3)
    
    return {
        'mesh': mesh,
        'origin': origin
    }


def load_landmarks(landmarks_path, origin=None):
    """
    Load landmarks from FCSV file.
    
    Args:
        landmarks_path: Path to the FCSV file
        origin: Optional image origin to adjust coordinates
        
    Returns:
        dict with keys:
            - 'points': numpy array of shape (N, 3)
            - 'labels': list of landmark names
            - 'coordinate_system': 'LPS' or 'RAS'
    """
    if landmarks_path.endswith('.txt'):
        from fcsv_reader import read_txt_landmarks
        landmarks = read_txt_landmarks(landmarks_path)
    else:
        landmarks = read_fcsv(landmarks_path)
    
    if origin is not None:
        # Adjust for image origin
        landmarks['points'] -= origin
    
    return landmarks
