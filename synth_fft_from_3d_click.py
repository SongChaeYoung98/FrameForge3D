"""
__author__      = "Song Chae Young"
__date__        = "Sep 11, 2025"
__email__       = "0.0yeriel@gmail.com"
__fileName__    = "temp_synth_fft_librosa.py"
__github__      = "SongChaeYoung98"
__status__      = "Development"
__description__ = "Combine weighted FFT segments from multiple times and save as spectrogram PNG and WAV"
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import re
import os
from PIL import Image
import soundfile as sf


# TODO.─────────────────────────── 핸들러 ───────────────────────────
def handler(wav_file, image_coords, sample_rate=48000, window_duration=2.0,
            n_fft=4096, hop_length=512, volume_boost=9.0):
    """핸들러 함수: 전체 합성 및 저장"""

    # 1️⃣ 오디오 로드
    audio, sr = load_audio(wav_file, sample_rate)

    # 2️⃣ 이미지 중심 계산
    img = Image.open(image_coords[0][0])
    img_cx, img_cy = img.width / 2, img.height / 2

    stft_segments = []
    weights = []

    # 3️⃣ 각 이미지별 STFT + 가중치
    for img_path, x, y in image_coords:
        timestamp = float(re.search(r'_(\d+\.\d+)s', os.path.basename(img_path)).group(1))
        segment = extract_segment(audio, sr, timestamp, window_duration)
        weight = calc_weight(x, y, img_cx, img_cy)
        weights.append(weight)
        stft_segments.append(compute_weighted_stft(segment, weight, n_fft, hop_length))

    # 4️⃣ STFT 합성
    combined_stft = combine_stfts(stft_segments, weights)

    # 5️⃣ 시간 신호 복원
    combined_audio = stft_to_audio(combined_stft, hop_length, volume_boost=volume_boost)

    # 6️⃣ 저장 경로
    base_dir = os.path.dirname(wav_file)
    output_wav = os.path.join(base_dir, "weighted_combined.wav")
    output_png = os.path.join(base_dir, "weighted_spectrogram.png")

    # 7️⃣ WAV 및 스펙트로그램 저장
    save_wav(combined_audio, sr, output_wav)
    save_spectrogram(combined_stft, sr, hop_length, output_png)


# TODO.─────────────────────────── 함수 정의 ───────────────────────────
# STEP 1. ───────────────────────── 오디오 로드 ─────────────────────────
# ==================== 함수 정의 ====================
def load_audio(wav_file: str, sample_rate: int):
    """WAV 파일 로드 및 리샘플링"""
    audio, sr = librosa.load(wav_file, sr=sample_rate)
    print(f"샘플링 레이트: {sr}, 오디오 길이: {len(audio)/sr:.2f}초")
    return audio, sr


# STEP 2. ─────────────────────────── 거리 기반 가중치 계산 ───────────────────────────
def calc_weight(x, y, cx, cy):
    """이미지 중심으로부터 거리 기반 가중치 계산"""
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return 1 / (1 + d)


# STEP 3. ────────────────────── 타임스탬프 오디오 추출 ──────────────────────
def extract_segment(audio, sr, timestamp, window_duration):
    """주어진 타임스탬프 주변 오디오 ±window_duration/2 추출"""
    start_sample = max(int((timestamp - window_duration / 2) * sr), 0)
    end_sample = min(int((timestamp + window_duration / 2) * sr), len(audio))
    return audio[start_sample:end_sample]


# STEP 4. ─────────────────────── 가중치 적용 STFT 계산 ───────────────────────
def compute_weighted_stft(segment, weight, n_fft=4096, hop_length=512):
    """STFT 계산 후 가중치 적용"""
    stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length, window='hann')
    return stft * weight


# STEP 5. ─────────────────────────── STFT 합성 ───────────────────────────
def combine_stfts(stft_segments, weights):
    """STFT 세그먼트 복소수 가중합 및 시간 bin 맞추기"""
    max_time_bins = max(stft.shape[1] for stft in stft_segments)
    padded_segments = [
        np.pad(stft, ((0, 0), (0, max_time_bins - stft.shape[1])), mode="constant")
        for stft in stft_segments
    ]
    combined_stft = np.sum(padded_segments, axis=0) / (np.sum(weights) + 1e-8)
    return combined_stft


# STEP 6. ─────────────────────────── 사운드 크기 증폭 ───────────────────────────
def stft_to_audio(stft, hop_length=512, window='hann', volume_boost=1.0):
    """STFT → 시간 신호 복원 후 볼륨 증폭"""
    audio = librosa.istft(stft, hop_length=hop_length, window=window)
    audio *= volume_boost
    return audio


# STEP 7. ─────────────────────────── WAV 파일 저장 ───────────────────────────
def save_wav(audio, sr, output_path):
    """WAV 파일 저장"""
    sf.write(output_path, audio, sr)
    print(f"✅ 합성 오디오 WAV 저장 완료: {output_path}")


# STEP 8. ────────────────────── dB 스펙트로그램 PNG 저장 ──────────────────────
def save_spectrogram(stft, sr, hop_length, output_path, cmap="magma"):
    """dB 스펙트로그램 PNG 저장"""
    db_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(db_spec, sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis="hz", cmap=cmap)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Weighted Combined Spectrogram")
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ 스펙트로그램 PNG 저장 완료: {output_path}")


# TODO.─────────────────────────── 스크립트 실행 ───────────────────────────
if __name__ == "__main__":
    # ====== 사용자 설정 ======
    wav_file = r"C:\Users\sooji\PycharmProjects\check-lab-python-3d-script\20250626_141039.wav"
    image_coords = [
        (r"C:\Users\sooji\PycharmProjects\check-lab-python-3d-script\test\images\frame_0028_005.339s.jpg", 436, 37),
        (r"C:\Users\sooji\PycharmProjects\check-lab-python-3d-script\test\images\frame_0031_005.932s.jpg", 309, 148)
    ]

    handler(wav_file, image_coords)
