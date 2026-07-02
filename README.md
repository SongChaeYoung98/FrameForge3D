# FrameForge3D

A Python-based 3D reconstruction workflow that turns a video or a folder of images into a textured 3D mesh using **OpenCV**, **COLMAP**, and **OpenMVS**.

## What This Repository Does

The main entry point is `init_3d_reconstruction.py`.

It performs the pipeline below:

1. Read input from either a video file or a folder of still images.
2. Extract or normalize images into `test/images`.
3. Run COLMAP feature extraction.
4. Remove images with too few detected features.
5. Run COLMAP matching and sparse mapping.
6. Convert the COLMAP model into OpenMVS format.
7. Run OpenMVS dense reconstruction, meshing, refinement, and texturing.
8. Export a textured mesh such as `scene_dense_mesh_refine_texture.obj`.

## Project Files

- `init_3d_reconstruction.py`: Main 3D reconstruction pipeline.
- `prepare_3d.py`: Older helper script for frame extraction and sparse COLMAP setup.
- `extract_video_frames.py`: Standalone utility that extracts frames from videos under `video/` into `extracted_frames/`.
- `requirements.txt`: Python package dependencies used by the scripts.

## Recommended Directory Layout

The current scripts assume this layout under the repository root:

```text
FrameForge3D/
|-- init_3d_reconstruction.py
|-- prepare_3d.py
|-- extract_video_frames.py
|-- requirements.txt
|-- test.mp4                    # optional video input for the main pipeline
|-- image/                      # optional image input folder for the main pipeline
|-- test/                       # generated working directory
|-- colmap/                     # local COLMAP installation
|   `-- COLMAP.bat
`-- vcpkg/
    `-- installed/x64-windows/tools/openmvs/
        |-- InterfaceCOLMAP.exe
        |-- DensifyPointCloud.exe
        |-- ReconstructMesh.exe
        |-- RefineMesh.exe
        `-- TextureMesh.exe
```

## Prerequisites

### 1. Operating system

- Windows is the primary target in the current script.
- Linux code paths exist, but they assume system packages and executable locations that you must verify manually.

### 2. Python

Install one of the following:

- Python 3.10
- Python 3.11
- Python 3.12

Python 3.12 is recommended because this workspace already contains `.cpython-312.pyc` cache files.

### 3. Native tools

You must have all of the following available before the main pipeline will run:

- COLMAP
- OpenMVS
- OpenCV Python bindings
- A C++ toolchain suitable for building OpenMVS dependencies when using `vcpkg`

For Windows, the practical setup is:

- Visual Studio 2022 Build Tools or full Visual Studio with C++ workload
- CMake
- Git
- vcpkg
- COLMAP for Windows

### 4. Optional tools

These are not required for the main reconstruction pipeline, but are used by the secondary scripts:

- `ffmpeg.exe` for `prepare_3d.py`
- No additional Python packages beyond `requirements.txt` for the current secondary scripts.

## Installation Guide

Follow the steps in this order.

### Step 1. Clone the repository

```bash
git clone https://github.com/SongChaeYoung98/FrameForge3D.git
cd FrameForge3D
```

### Step 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS shell:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4. Install COLMAP

Install COLMAP and place it so that this file exists:

```text
colmap/COLMAP.bat
```

If your installation lives elsewhere, update the Windows path block near the top of `init_3d_reconstruction.py`.

### Step 5. Install vcpkg

Example:

```bash
git clone https://github.com/microsoft/vcpkg.git
```

Place the cloned folder at:

```text
FrameForge3D/vcpkg
```

Then bootstrap it.

Windows:

```powershell
.\vcpkg\bootstrap-vcpkg.bat
```

Linux/macOS:

```bash
./vcpkg/bootstrap-vcpkg.sh
```

### Step 6. Install OpenMVS through vcpkg

The script expects the OpenMVS executables under:

```text
vcpkg/installed/x64-windows/tools/openmvs/
```

A typical Windows installation command is:

```powershell
.\vcpkg\vcpkg install openmvs:x64-windows
```

After installation, verify these files exist:

- `InterfaceCOLMAP.exe`
- `DensifyPointCloud.exe`
- `ReconstructMesh.exe`
- `RefineMesh.exe`
- `TextureMesh.exe`

### Step 7. Prepare your input data

Choose one input mode.

#### Option A. Video input

- Put a source video in the repository root, for example `test.mp4`.
- The script default is `DEFAULT_INPUT_MODE = "video"`.

#### Option B. Image folder input

- Put source images under `image/`.
- Change `DEFAULT_INPUT_MODE` to `"images"`, or call `run_pipeline(input_mode="images")` manually.

### Step 8. Verify script paths before the first run

This matters because some paths are hardcoded.

Open `init_3d_reconstruction.py` and confirm these values match your machine:

- `default_video_path`
- `default_image_path`
- `work_dir`
- `colmap_bat`
- `openmvs_bin_dir`

Also note:

- `extract_video_frames.py` looks for videos under `video/` and writes frames under `extracted_frames/`.
- `prepare_3d.py` expects `ffmpeg.exe` next to the script.

## How to Run

### Run the main 3D pipeline

```bash
python init_3d_reconstruction.py
```

### Run from Python with explicit options

```python
from pathlib import Path
from init_3d_reconstruction import run_pipeline

run_pipeline(
    input_mode="video",
    input_path=Path("test.mp4"),
    n_frames=100,
    rotate_180=True,
)
```

For image input:

```python
from pathlib import Path
from init_3d_reconstruction import run_pipeline

run_pipeline(
    input_mode="images",
    input_path=Path("image"),
    rotate_180=False,
)
```

## How the Main Pipeline Works

### 1. Workspace creation

`create_dirs()` creates the working folders such as:

- `test/images`
- `test/sparse`

### 2. Input preparation

If `input_mode == "video"`:

- `extract_frames()` samples frames uniformly from the source video.
- `rename_frames_with_timestamps()` renames them to filenames such as `frame_0001_000.000s.jpg`.

If `input_mode == "images"`:

- `prepare_captured_images()` copies images into the working directory and optionally rotates them.

### 3. Feature extraction

`colmap_feature_extraction()` creates the COLMAP database and runs feature extraction with a `PINHOLE` camera model.

### 4. Low-feature filtering

`remove_low_feature_images()` opens the COLMAP SQLite database and removes images whose keypoint count is below the configured threshold.

### 5. Sparse reconstruction

The script then runs:

- `colmap_exhaustive_matching()`
- `colmap_mapper()`

This produces a sparse COLMAP model under `test/sparse/0`.

### 6. Conversion to OpenMVS

- `colmap_model_converter()` converts COLMAP binary output to text.
- `colmap_to_openmvs()` uses `InterfaceCOLMAP` to create `scene.mvs`.

### 7. Dense reconstruction and mesh generation

`openmvs_dense_reconstruction()` runs the OpenMVS tools in sequence:

1. `DensifyPointCloud`
2. `ReconstructMesh`
3. `RefineMesh`
4. `TextureMesh`

## Expected Output

After a successful run, look inside `test/` for files such as:

- `scene.mvs`
- `scene_dense.mvs`
- `scene_dense_mesh.mvs`
- `scene_dense_mesh_refine.mvs`
- `scene_dense_mesh_refine_texture.obj`
- `scene_dense_mesh_refine_texture.png`

## Notes and Limitations

- The current implementation uses hardcoded paths and local folder assumptions.
- `DEFAULT_ROTATE_180 = True`, so source frames are rotated unless you change that option.
- `organize_files()` exists but is not currently called because it is commented out in `run_pipeline()`.
- The repository does not currently track the heavy native dependencies such as `colmap/`, `vcpkg/`, or media files.
- `prepare_3d.py` appears to be an older prototype and is not the recommended entry point.

## Troubleshooting

### COLMAP command fails immediately

Check:

- `colmap/COLMAP.bat` exists.
- The path in `colmap_bat` is correct.
- Any required COLMAP DLLs are present beside the executable.

### OpenMVS executable not found

Check that OpenMVS was installed to:

```text
vcpkg/installed/x64-windows/tools/openmvs/
```

If not, update `openmvs_bin_dir` in `init_3d_reconstruction.py`.

### Video cannot be opened

Check:

- The video path exists.
- OpenCV can decode the file format.
- The path in `default_video_path` is correct.

### No output mesh is generated

Possible causes:

- Too few source images.
- Low-texture or blurry frames.
- Wrong image orientation.
- COLMAP features were filtered too aggressively.
- Sparse mapping failed before OpenMVS started.

## Suggested Next Improvement

The most valuable cleanup would be replacing the hardcoded path block with command-line arguments or environment-based configuration, so the repository works without local file edits.


