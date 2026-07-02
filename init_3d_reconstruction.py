"""
__author__      = "Song Chae Young"
__date__        = "Sep 22, 2025"
__email__       = "0.0yeriel@gmail.com"
__fileName__    = "init_3d_reconstruction.py"
__github__      = "SongChaeYoung98"
__status__      = "Development"
__description__ = "Prepare essential data and inputs for check-lab-python-3d-script reconstruction"
"""

import os
import platform
import shutil
import sqlite3
import subprocess
from pathlib import Path

import cv2

project_root = Path(__file__).resolve().parent
system_os = platform.system()
xvfb_available = False
if system_os != "Windows":
    try:
        from xvfbwrapper import Xvfb  # noqa: F401
        xvfb_available = True
    except ImportError:
        xvfb_available = False

if system_os == "Windows":
    default_video_path = project_root / "test.mp4"
    default_image_path = project_root / "image"
    work_dir = project_root / "test"
    colmap_bat = project_root / "colmap" / "COLMAP.bat"
    colmap_bin_dir = colmap_bat.parent
    openmvs_bin_dir = project_root / "vcpkg" / "installed" / "x64-windows" / "tools" / "openmvs"
else:
    default_video_path = project_root / "boat modeling.mp4"
    default_image_path = project_root / "image"
    work_dir = project_root / "test"
    colmap_bin_dir = Path("/usr/local/src/colmap/build/src/colmap/exe")
    colmap_bat = colmap_bin_dir / "colmap"
    openmvs_bin_dir = Path("/usr/local/src/vcpkg/installed/x64-linux/tools/openmvs")


DEFAULT_INPUT_MODE = "images"
# DEFAULT_INPUT_MODE = "video"
DEFAULT_FRAME_COUNT = 100
DEFAULT_ROTATE_180 = True

images_dir = work_dir / "images"
database_path = work_dir / "database.db"
sparse_dir = work_dir / "sparse"
supported_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def has_supported_images(directory: Path) -> bool:
    return directory.exists() and directory.is_dir() and any(
        path.is_file() and path.suffix.lower() in supported_image_extensions
        for path in directory.iterdir()
    )


def resolve_default_input_path(input_mode: str) -> Path:
    if input_mode == "video":
        return default_video_path

    image_candidates = [
        work_dir / "images",
        Path(__file__).resolve().parent / "image",
        default_image_path,
    ]
    for candidate in image_candidates:
        if has_supported_images(candidate):
            return candidate
    return default_image_path

if system_os == "Windows":
    interface_colmap_exe = openmvs_bin_dir / "InterfaceCOLMAP.exe"
    densify_pointcloud_exe = openmvs_bin_dir / "DensifyPointCloud.exe"
    reconstruct_mesh_exe = openmvs_bin_dir / "ReconstructMesh.exe"
    refine_mesh_exe = openmvs_bin_dir / "RefineMesh.exe"
    texture_mesh_exe = openmvs_bin_dir / "TextureMesh.exe"
else:
    interface_colmap_exe = openmvs_bin_dir / "InterfaceCOLMAP"
    densify_pointcloud_exe = openmvs_bin_dir / "DensifyPointCloud"
    reconstruct_mesh_exe = openmvs_bin_dir / "ReconstructMesh"
    refine_mesh_exe = openmvs_bin_dir / "RefineMesh"
    texture_mesh_exe = openmvs_bin_dir / "TextureMesh"


def run_colmap_cmd(cmd_list, cwd=None):
    """
    Run COLMAP commands with OS-specific display handling.
    """
    env = os.environ.copy()
    if platform.system() != "Windows":
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":/usr/local/lib:/usr/lib"
        cmd_list = ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24"] + cmd_list

    print(f"Running COLMAP command: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True, env=env)
    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd_list)}")


def create_dirs():
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)


def reset_working_images_dir(target_images_dir: Path):
    if not target_images_dir.exists():
        target_images_dir.mkdir(parents=True, exist_ok=True)
        return

    for item in target_images_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink()


def extract_frames(video_path: Path, target_images_dir: Path, n_frames: int = 100, rotate_180: bool = True):
    """
    Extract n_frames images uniformly from a video.
    """
    print(f"==> Extracting {n_frames} frames uniformly from video...")

    reset_working_images_dir(target_images_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Cannot read frame count from video: {video_path}")

    step = max(total_frames // n_frames, 1)
    frame_indices = [i * step for i in range(n_frames)]
    saved_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        frame_name = target_images_dir / f"temp_{saved_count + 1:04d}.jpg"
        cv2.imwrite(str(frame_name), frame)
        saved_count += 1

    cap.release()
    print(f"==> Extracted {saved_count} frames to {target_images_dir}")


def rename_frames_with_timestamps(video_path: Path, target_images_dir: Path):
    print("==> Renaming frames with timestamps...")
    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    if fps <= 0:
        raise RuntimeError(f"Cannot read FPS from video: {video_path}")

    image_files = sorted(target_images_dir.glob("temp_*.jpg"))
    for i, img_path in enumerate(image_files, start=1):
        seconds = (i - 1) / fps
        new_name = target_images_dir / f"frame_{i:04d}_{seconds:07.3f}s.jpg"
        img_path.replace(new_name)
    print(f"==> Renamed {len(image_files)} frames with timestamps!")


def prepare_captured_images(source_images_dir: Path, target_images_dir: Path, rotate_180: bool = True):
    """
    Copy pre-captured images into the working images directory with normalized names.
    If source and target are the same directory, keep the images in place.
    """
    print(f"==> Preparing captured images from {source_images_dir}...")

    if not source_images_dir.exists() or not source_images_dir.is_dir():
        raise RuntimeError(f"Input image directory does not exist: {source_images_dir}")

    source_images = sorted(
        path for path in source_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_image_extensions
    )
    if not source_images:
        raise RuntimeError(f"No supported image files found in: {source_images_dir}")

    same_directory = source_images_dir.resolve() == target_images_dir.resolve()
    if same_directory:
        print(f"==> Using existing images in place: {target_images_dir}")
        return

    reset_working_images_dir(target_images_dir)

    saved_count = 0
    for source_path in source_images:
        image = cv2.imread(str(source_path))
        if image is None:
            print(f"Skipping unreadable image: {source_path}")
            continue

        if rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)

        target_path = target_images_dir / f"image_{saved_count + 1:04d}.jpg"
        cv2.imwrite(str(target_path), image)
        saved_count += 1

    if saved_count == 0:
        raise RuntimeError(f"No readable images found in: {source_images_dir}")

    print(f"==> Prepared {saved_count} captured images in {target_images_dir}")


def colmap_feature_extraction(database_path: Path, target_images_dir: Path):
    print("==> Running COLMAP feature extraction...")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    if not database_path.exists():
        db_cmd = [str(colmap_bat), "database_creator", "--database_path", str(database_path)]
        run_colmap_cmd(db_cmd, cwd=str(colmap_bin_dir))

    feat_cmd = [
        str(colmap_bat), "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(target_images_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--SiftExtraction.peak_threshold", "0.004",
        "--SiftExtraction.edge_threshold", "5"
    ]
    run_colmap_cmd(feat_cmd, cwd=str(colmap_bin_dir))


def remove_low_feature_images(database_path: Path, target_images_dir: Path, min_features: int = 20):
    print(f"==> Removing images with fewer than {min_features} features...")
    conn = sqlite3.connect(database_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT images.name, keypoints.rows as feature_count
        FROM images
        LEFT JOIN keypoints ON images.image_id = keypoints.image_id
    """)
    image_feature_counts = cur.fetchall()
    remove_images = [name for name, count in image_feature_counts if count is None or count < min_features]

    for img_name in remove_images:
        img_path = target_images_dir / img_name
        if img_path.exists():
            feature_count = [c for n, c in image_feature_counts if n == img_name][0]
            print(f"Removing {img_name} ({feature_count if feature_count else 0} features)")
            img_path.unlink()
        cur.execute("DELETE FROM images WHERE name=?", (img_name,))

    conn.commit()
    conn.close()
    print(f"==> Removed {len(remove_images)} images due to insufficient features.")


def colmap_exhaustive_matching(database_path: Path):
    print("==> Running COLMAP exhaustive matcher...")
    cmd = [str(colmap_bat), "exhaustive_matcher", "--database_path", str(database_path)]
    run_colmap_cmd(cmd, cwd=str(colmap_bin_dir))


def colmap_mapper(database_path: Path, target_images_dir: Path, target_sparse_dir: Path):
    print("==> Running COLMAP mapper...")

    env = os.environ.copy()
    if platform.system() != "Windows":
        env["LD_PRELOAD"] = "/lib64/libopenblas.so.0"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["QT_QPA_PLATFORM"] = "offscreen"
        cmd_list = [
            "xvfb-run", "-a", "-s", "-screen 0 1024x768x24",
            str(colmap_bat), "mapper",
            "--database_path", str(database_path),
            "--image_path", str(target_images_dir),
            "--output_path", str(target_sparse_dir),
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0"
        ]
    else:
        cmd_list = [
            str(colmap_bat), "mapper",
            "--database_path", str(database_path),
            "--image_path", str(target_images_dir),
            "--output_path", str(target_sparse_dir),
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0"
        ]

    print(f"Running COLMAP mapper command: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, cwd=colmap_bin_dir, env=env, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"COLMAP mapper failed with return code {result.returncode}")


def select_best_sparse_model_dir(target_sparse_dir: Path) -> Path:
    candidate_dirs = [
        path for path in target_sparse_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "points3D.bin").exists()
    ]
    if not candidate_dirs:
        raise RuntimeError(f"No COLMAP sparse model directories found in: {target_sparse_dir}")

    best_dir = max(candidate_dirs, key=lambda path: (path / "points3D.bin").stat().st_size)
    print(f"==> Selected COLMAP sparse model: {best_dir.name}")
    return best_dir


def colmap_model_converter(target_sparse_dir: Path):
    sparse_bin_dir = select_best_sparse_model_dir(target_sparse_dir)
    sparse_txt_dir = target_sparse_dir / f"{sparse_bin_dir.name}_txt"
    sparse_txt_dir.mkdir(exist_ok=True)

    print("==> Converting COLMAP binary model to TXT format...")
    subprocess.run([
        str(colmap_bat), "model_converter",
        "--input_path", str(sparse_bin_dir),
        "--output_path", str(sparse_txt_dir),
        "--output_type", "TXT"
    ], check=True, cwd=colmap_bin_dir)

    sparse_subdir = sparse_txt_dir / "sparse"
    sparse_subdir.mkdir(exist_ok=True)

    txt_files = ["cameras.txt", "images.txt", "points3D.txt", "frames.txt", "rigs.txt"]
    for file_name in txt_files:
        src = sparse_txt_dir / file_name
        dst = sparse_subdir / file_name
        if src.exists():
            shutil.copy(src, dst)

    print(f"==> Created 'sparse' folder inside {sparse_txt_dir.name} and copied TXT files for InterfaceCOLMAP.exe")
    return sparse_txt_dir

def colmap_to_openmvs(sparse_txt_dir: Path, scene_mvs: Path):
    print("==> Converting COLMAP TXT model to OpenMVS scene.mvs...")
    subprocess.run([
        str(interface_colmap_exe),
        "-i", str(sparse_txt_dir),
        "-o", str(scene_mvs)
    ], check=True, cwd=work_dir)


def openmvs_dense_reconstruction(scene_mvs: Path, target_work_dir: Path):
    print("==> Running OpenMVS Dense Reconstruction...")

    scene_dense_mvs = target_work_dir / "scene_dense.mvs"
    scene_dense_mesh_mvs = target_work_dir / "scene_dense_mesh.mvs"
    scene_dense_mesh_refine_mvs = target_work_dir / "scene_dense_mesh_refine.mvs"
    scene_dense_mesh_refine_texture_mvs = target_work_dir / "scene_dense_mesh_refine_texture.mvs"
    scene_obj = target_work_dir / "scene_dense_mesh_refine_texture.obj"
    scene_texture = target_work_dir / "scene_dense_mesh_refine_texture.png"

    subprocess.run([str(densify_pointcloud_exe), str(scene_mvs), "-o", str(scene_dense_mvs), "--resolution-level", "1"], check=True, cwd=target_work_dir)
    subprocess.run([str(reconstruct_mesh_exe), str(scene_dense_mvs), "-o", str(scene_dense_mesh_mvs)], check=True, cwd=target_work_dir)
    subprocess.run([str(refine_mesh_exe), str(scene_dense_mesh_mvs), "-o", str(scene_dense_mesh_refine_mvs)], check=True, cwd=target_work_dir)
    # subprocess.run([str(texture_mesh_exe), str(scene_dense_mesh_refine_mvs), "-o", str(scene_dense_mesh_refine_texture_mvs), "--resolution-level", "1", "--export-type", "all"], check=True, cwd=target_work_dir)
    subprocess.run([str(texture_mesh_exe), str(scene_dense_mesh_refine_mvs), "-o", str(scene_dense_mesh_refine_texture_mvs), "--resolution-level", "1", "--export-type", "obj"], check=True, cwd=target_work_dir)     

    return scene_obj, scene_texture


def organize_files(target_work_dir: Path):
    logs_dir = target_work_dir / "logs"
    depth_dir = target_work_dir / "depth_maps"

    logs_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    for log_file in target_work_dir.glob("*.log"):
        shutil.move(str(log_file), logs_dir / log_file.name)

    for dmap_file in target_work_dir.glob("depth*.dmap"):
        dmap_file.replace(depth_dir / dmap_file.name)
    print(f"==> Moved {len(list(logs_dir.glob('*.log')))} log files to '{logs_dir}'")

    sparse_root = target_work_dir / "sparse"
    sparse_0_txt = sparse_root / "0_txt"

    if sparse_0_txt.exists():
        for txt_file in sparse_0_txt.glob("*.txt"):
            target = sparse_root / txt_file.name
            shutil.move(str(txt_file), target)
            print(f"Moved {txt_file} -> {target}")

        shutil.rmtree(sparse_0_txt, ignore_errors=True)
        print(f"Removed directory: {sparse_0_txt}")

    sparse_0 = sparse_root / "0"
    if sparse_0.exists():
        shutil.rmtree(sparse_0, ignore_errors=True)
        print(f"Removed directory: {sparse_0}")

    keep_items = {
        "sparse",
        "images",
        "scene.mvs",
        "scene_dense_mesh_refine_texture.ply",
        "scene_dense_mesh_refine_texture.png"
    }

    for item in target_work_dir.iterdir():
        if item.name in keep_items:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
            print(f"Removed directory: {item}")
        else:
            try:
                item.unlink()
                print(f"Removed file: {item}")
            except Exception as exc:
                print(f"Failed to remove {item}: {exc}")


def run_pipeline(input_mode: str = DEFAULT_INPUT_MODE, input_path: Path | None = None, n_frames: int = DEFAULT_FRAME_COUNT, rotate_180: bool = DEFAULT_ROTATE_180):
    if input_path is None:
        input_path = resolve_default_input_path(input_mode)
    create_dirs()

    if input_mode == "video":
        extract_frames(input_path, images_dir, n_frames=n_frames, rotate_180=rotate_180)
        rename_frames_with_timestamps(input_path, images_dir)
    else:
        prepare_captured_images(input_path, images_dir, rotate_180=rotate_180)

    colmap_feature_extraction(database_path, images_dir)
    remove_low_feature_images(database_path, images_dir)
    colmap_exhaustive_matching(database_path)
    colmap_mapper(database_path, images_dir, sparse_dir)
    sparse_txt_dir = colmap_model_converter(sparse_dir)
    scene_mvs = work_dir / "scene.mvs"
    colmap_to_openmvs(sparse_txt_dir, scene_mvs)
    scene_obj, scene_texture = openmvs_dense_reconstruction(scene_mvs, work_dir)
    # organize_files(work_dir)

    print("==> Pipeline finished successfully!")
    print(f"3D model with texture: {scene_obj}")
    print(f"Texture image: {scene_texture}")


if __name__ == "__main__":
    run_pipeline()
