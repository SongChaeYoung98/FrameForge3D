"""
Standalone frame extraction utility for one or more videos.
"""

from __future__ import annotations

from pathlib import Path

import cv2


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
    ".webm",
}

project_root = Path(__file__).resolve().parent
base_root = project_root / "video"
output_root = project_root / "extracted_frames"
extract_all_frames = False
frame_interval = 10
rotate_frames_180 = False
rotate_frames_90_ccw = False # Set to True to rotate frames 90 degrees counter-clockwise
rotate_frames_90_cw = True # Set to True to rotate frames 90 degrees clockwise
keep_existing_frames = False


def discover_video_paths(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    video_paths = sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise FileNotFoundError(f"No supported video files found in: {video_dir}")
    return video_paths


def validate_video_paths(video_paths: list[Path]) -> None:
    missing_paths = [path for path in video_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Video files not found: {missing}")

    invalid_paths = [
        path for path in video_paths if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS
    ]
    if invalid_paths:
        invalid = ", ".join(str(path) for path in invalid_paths)
        raise ValueError(f"Unsupported video file extensions: {invalid}")


def prepare_output_dir(output_dir: Path, keep_existing: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if keep_existing:
        return

    for item in output_dir.iterdir():
        if item.is_file():
            item.unlink()


def build_frame_indices(total_frames: int, interval: int, extract_all: bool) -> list[int]:
    if extract_all:
        return list(range(total_frames))

    return list(range(0, total_frames, interval))


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval: int,
    rotate_180: bool,
    rotate_90_ccw: bool,
    rotate_90_cw: bool,
    keep_existing: bool,
    extract_all: bool,
) -> int:
    prepare_output_dir(output_dir, keep_existing)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Cannot read frame count from video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_indices = build_frame_indices(total_frames, interval, extract_all)
    saved_count = 0

    for index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue

        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if rotate_90_ccw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rotate_90_cw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        timestamp_seconds = index / fps if fps > 0 else 0.0
        output_path = output_dir / f"frame_{saved_count + 1:04d}_{timestamp_seconds:07.3f}s.jpg"
        cv2.imwrite(str(output_path), frame)
        saved_count += 1

    capture.release()
    return saved_count


def build_output_dir(base_output_root: Path, video_path: Path) -> Path:
    return base_output_root / video_path.stem


def main() -> None:
    video_paths = discover_video_paths(base_root)
    validate_video_paths(video_paths)

    if not extract_all_frames and frame_interval <= 0:
        raise ValueError("frame_interval must be greater than 0 when extract_all_frames is False")

    mode_label = "all frames" if extract_all_frames else f"1 frame every {frame_interval} frames"
    rotation_flags = []
    if rotate_frames_180:
        rotation_flags.append("180")
    if rotate_frames_90_ccw:
        rotation_flags.append("90ccw")
    if rotate_frames_90_cw:
        rotation_flags.append("90cw")
    rotation_label = "none" if not rotation_flags else " + ".join(rotation_flags)

    for video_path in video_paths:
        output_dir = build_output_dir(output_root, video_path)
        print(f"==> Processing: {video_path}")
        print(f"==> Output: {output_dir}")
        print(f"==> Mode: {mode_label}")
        print(f"==> Rotation: {rotation_label}")
        saved_count = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            interval=frame_interval,
            rotate_180=rotate_frames_180,
            rotate_90_ccw=rotate_frames_90_ccw,
            rotate_90_cw=rotate_frames_90_cw,
            keep_existing=keep_existing_frames,
            extract_all=extract_all_frames,
        )
        print(f"==> Extracted {saved_count} frames")


if __name__ == "__main__":
    main()
