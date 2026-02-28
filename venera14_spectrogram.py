"""
Venera 14 Sound - Spectrogram Analyzer (FIXED)
===============================================
Fixes applied:
  1. Raw string r"..." on Windows path to avoid escape sequence warning
  2. Replaced shading="gouraud" with shading="auto" to prevent memory error
  3. Downsampled spectrogram for large/long audio files (240 sec stereo)
  4. Handles int32 audio normalization correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# ── 1. LOAD the .wav file ──────────────────────────────────────────────────────
wav_path = r"C:\python3_13\Venera14 Sound.wav"   # FIX 1: raw string (r"...")

sample_rate, audio = wavfile.read(wav_path)
print(f"Sample rate : {sample_rate} Hz")
print(f"Duration    : {len(audio) / sample_rate:.2f} seconds")
print(f"Channels    : {'Stereo' if audio.ndim == 2 else 'Mono'}")
print(f"Data type   : {audio.dtype}")

# ── 2. CONVERT to mono if stereo ──────────────────────────────────────────────
if audio.ndim == 2:
    audio = audio.mean(axis=1)
    print("Converted stereo to mono.")

# ── 3. NORMALIZE int32 to float32 (-1.0 to 1.0) ──────────────────────────────
# FIX 3: int32 max is 2,147,483,647 — must use correct dtype max
audio = audio.astype(np.float32) / np.iinfo(np.int32).max
print(f"Normalized to float32. Max amplitude: {np.abs(audio).max():.4f}")

# ── 4. COMPUTE spectrogram ────────────────────────────────────────────────────
frequencies, times, spec = signal.spectrogram(
    audio,
    fs=sample_rate,
    nperseg=2048,
    noverlap=1024
)

# FIX 2: Downsample time axis to max 2000 columns to avoid memory error
MAX_TIME_COLS = 2000
if spec.shape[1] > MAX_TIME_COLS:
    step = spec.shape[1] // MAX_TIME_COLS
    spec  = spec[:, ::step]
    times = times[::step]
    print(f"Downsampled spectrogram to {spec.shape[1]} time columns.")

# Convert to dB scale
spec_db = 10 * np.log10(spec + 1e-10)

# ── 5. PLOT ───────────────────────────────────────────────────────────────────
duration = len(audio) / sample_rate
t = np.linspace(0, duration, len(audio))

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Venera 14 — Surface Sound Spectrogram", fontsize=15, fontweight="bold")

# Top: Waveform
axes[0].plot(t, audio, color="steelblue", linewidth=0.3)
axes[0].set_title("Waveform")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, duration)

# Bottom: Spectrogram — FIX 2: shading="auto" instead of "gouraud"
img = axes[1].pcolormesh(
    times, frequencies, spec_db,
    shading="auto",          # prevents 3.79 GiB allocation error
    cmap="inferno"
)
axes[1].set_title("Spectrogram (Frequency over Time)")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_ylim(0, 8000)   # focus on 0-8 kHz
fig.colorbar(img, ax=axes[1], label="Intensity (dB)")

plt.tight_layout()
output_path = r"C:\python3_13\venera14_spectrogram.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")
