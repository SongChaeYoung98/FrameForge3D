"""
__author__      = "Song Chae Young"
__date__        = "Sep 05, 2025"
__email__       = "0.0yeriel@gmail.com"
__fileName__    = "prepare_3d.py"
__github__      = "SongChaeYoung98"
__status__      = "Development"
__description__ = "Prepare essential data and inputs for check-lab-python-3d-script reconstruction"
"""

import os
import subprocess
from pathlib import Path


# TODO ─────────── PIPELINE ────────────
def run_3d_pipeline(video_file):
    """
    1. 프로젝트 초기화
    2. 동영상 -> 이미지 프레임
    3. COLMAP CPU 모드 처리
    """
    # STEP 1. Init project folders + ffmpeg
    config = init_ffmpeg(folders=["images", "frames", "sparse", "dense"])

    # STEP 2. FFMPEG 프레임 추출
    extract_frames(video_file, config["folders"][1])  # frames 폴더

    # STEP 3. COLMAP 초기화
    colmap_config = init_colmap(sparse_folder=config["folders"][2])

    # STEP 4. COLMAP feature extraction (CPU only)
    subprocess.run([
        colmap_config["colmap"],
        "feature_extractor",
        "--database_path", str(colmap_config["database"]),
        "--image_path", str(config["folders"][0]),  # images 폴더
        "--ImageReader.single_camera", "1",
        "--use_gpu", "0"  # CPU 모드
    ], check=True)

    # STEP 5. COLMAP exhaustive matcher (CPU)
    subprocess.run([
        colmap_config["colmap"],
        "exhaustive_matcher",
        "--database_path", str(colmap_config["database"]),
        "--use_gpu", "0"
    ], check=True)

    print("COLMAP sparse reconstruction ready.")

    # STEP 6. OpenMVS 연계 가능 (추가)
    # subprocess.run([...])

    print("check-lab-python-3d-script pipeline finished.")


# STEP 1. ─────────── INIT FFMPEG ────────────
def init_ffmpeg(folders=None):
    """
    프로젝트 초기화: 필요한 폴더 생성, ffmpeg 경로 확인 등.
    folders: 생성할 폴더 리스트 (상대 경로)
    반환: dict {'ffmpeg': ffmpeg_path, 'folders': [Path,...]}
    """
    if folders is None:
        folders = ["images", "frames", "sparse", "dense"]

    # ffmpeg 경로 확인
    ffmpeg_path = get_ffmpeg_path()

    # 폴더 생성
    project_dir = Path(__file__).parent
    created_folders = []
    for folder in folders:
        folder_path = project_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        created_folders.append(folder_path)

    print("Project initialized.")
    print(f"ffmpeg path: {ffmpeg_path}")
    print("Folders:", [str(f) for f in created_folders])

    return {"ffmpeg": ffmpeg_path, "folders": created_folders}


# STEP 2. ─────────── FFMPEG 프레임 추출 ────────────
def extract_frames(video_file, output_folder, framerate=30):
    """
    동영상 -> 이미지 프레임 추출
    """
    ffmpeg_path = get_ffmpeg_path()
    output_pattern = Path(output_folder) / "frame_%04d.png"

    subprocess.run([
        ffmpeg_path,
        "-i", str(video_file),
        "-framerate", str(framerate),
        str(output_pattern)
    ], check=True)

    print(f"Frames extracted to {output_folder}")


# STEP 3. ─────────── INIT COLMAP ────────────
def init_colmap(database_path=None, sparse_folder=None):
    """
    COLMAP 초기화: database.db 및 sparse 폴더 준비
    반환: dict {'colmap': colmap_path, 'database': Path, 'sparse': Path}
    """
    project_dir = Path(__file__).parent

    # COLMAP 경로 확인
    colmap_path = get_colmap_path()

    # database.db 준비
    if database_path is None:
        database_path = project_dir / "database.db"
    database_path.touch(exist_ok=True)  # 파일 없으면 생성

    # sparse 폴더 준비
    if sparse_folder is None:
        sparse_folder = project_dir / "sparse"
    sparse_folder.mkdir(parents=True, exist_ok=True)

    print("COLMAP initialized.")
    print(f"COLMAP path: {colmap_path}")
    print(f"Database path: {database_path}")
    print(f"Sparse folder: {sparse_folder}")

    return {"colmap": colmap_path, "database": database_path, "sparse": sparse_folder}


# ─────────── ffmpeg 경로 가져오기 ────────────
def get_ffmpeg_path():
    """
    현재 스크립트와 같은 폴더에 있는 ffmpeg.exe 경로를 반환.
    존재하지 않으면 예외 발생.
    """
    script_dir = Path(__file__).parent
    ffmpeg_path = script_dir / "ffmpeg.exe"
    if not ffmpeg_path.exists():
        raise FileNotFoundError(f"ffmpeg.exe not found in {script_dir}")
    return str(ffmpeg_path)


# ─────────── COLMAP 경로 가져오기 ────────────
def get_colmap_path():
    """
    현재 스크립트와 같은 폴더에 있는 COLMAP.bat 경로를 반환.
    존재하지 않으면 예외 발생.
    """
    script_dir = Path(__file__).parent
    colmap_path = script_dir / "colmap" / "COLMAP.bat"
    if not colmap_path.exists():
        raise FileNotFoundError(f"COLMAP.bat not found in {colmap_path.parent}")
    return str(colmap_path)


# TODO ─────────── 실행 ────────────
if __name__ == "__main__":
    video_path = "input.mp4"  # 같은 폴더에 존재
    run_3d_pipeline(video_path)
