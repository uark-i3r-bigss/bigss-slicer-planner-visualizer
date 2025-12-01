# SE(3) Visualizer & Kinematic Chain Tools

Tools for visualizing SE(3) kinematic chains, NIfTI models, and verifying transforms.

## Installation

1. Create the conda environment:
   ```bash
   conda create -n bigss-vis python=3.10
   conda activate bigss-vis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Key Scripts

- **`visualizer_main.py`**: Interactive PyVista-based visualizer. Supports hierarchical transforms, NIfTI models, and basic shapes.
- **`se3_test.py`**: Headless verification script for checking transform logic and kinematic chain consistency.

## Configuration Files

- **`configs/config.yaml`**: Main configuration for real models (Phantom, Device).
- **`configs/config_test.yaml`**: Test scenario defining a kinematic chain using mock objects.

## Usage

### 1. Run Visualizer (Test Scenario)
Visualize a mock kinematic chain (World -> Block_01 -> Block_02) with interactive controls.
```bash
python3 visualizer_main.py configs/config_test.yaml
```

### 2. Run Visualizer (Real Models)
Visualize the Phantom and Device setup.
```bash
python3 visualizer_main.py configs/config.yaml
```

### 3. Run Mini Visualizer (Headless)
```bash
python3 mini_visualizer_main.py configs/config_test.yaml
```


### Workflow Overview

Our workflow integrates trajectory planning from 3D Slicer with real-time visualization. Users can create detailed trajectory plans within 3D Slicer (with the `bigss-surgery-planner` repository), as depicted in the planning interface below:

![Slicer Planning Interface](images/slicer-planning.png)

These annotations are dynamically exported, allowing for immediate visualization of the planned `FrameTransforms` in our interactive visualizer. The visualizer also displays the rigid transformation `SE(3)` matrix for the robot/tool

![Visualizer Output](images/visualizer-01.png)


### Contact 
Ping-Cheng Ku (pku1@jh.edu)