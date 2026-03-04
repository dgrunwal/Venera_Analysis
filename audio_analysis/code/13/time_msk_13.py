"""
Venera 13 Sound Recording - Audio Analysis
Dependencies: numpy, scipy, matplotlib (no librosa needed)
Run: python time_msk_13.py
"""

import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram as scipy_spec
from scipy.fft import rfft, rfftfreq

WAV = "Venera13 Sound.wav"

# ── Load first 30 seconds (memory-safe) ──────────────────────────────────────
with wave.open(WAV) as wf:
    sr_orig        = wf.getframerate()       # 48000
    n_frames_total = wf.getnframes()
    n_read         = sr_orig * 30 * 2        # 30s, stereo
    raw            = wf.readframes(n_read)

duration_total = n_frames_total / sr_orig

# Decode stereo int16 → mono float32
audio = (np.frombuffer(raw, dtype=np.int16)
           .reshape(-1, 2)
           .mean(axis=1)
           .astype(np.float32) / 32768.0)

# Decimate 48 kHz → 16 kHz (every 3rd sample — fast, no resample_poly RAM spike)
audio = audio[::3]
sr    = sr_orig // 3   # 16000 Hz
dur   = len(audio) / sr
t     = np.linspace(0, dur, len(audio))

# ── Statistics ────────────────────────────────────────────────────────────────
rms   = float(np.sqrt(np.mean(audio ** 2)))
peak  = float(np.max(np.abs(audio)))
crest = peak / rms

print(f"File duration : {duration_total:.1f} s  (analysing first {dur:.0f} s)")
print(f"Sample rate   : {sr} Hz (downsampled from {sr_orig})")
print(f"RMS           : {rms:.4f}")
print(f"Peak          : {peak:.4f}")
print(f"Crest factor  : {crest:.2f}  ({20*np.log10(crest):.1f} dB)")

# ── FFT ───────────────────────────────────────────────────────────────────────
yf = np.abs(rfft(audio))
xf = rfftfreq(len(audio), 1 / sr)

top   = np.argsort(yf)[::-1][:10]
tf, tm = xf[top], yf[top]

print("\nTop 10 Dominant Frequencies:")
for i, (f, m) in enumerate(zip(tf, tm), 1):
    print(f"  {i:>2}. {f:>8.2f} Hz   magnitude={m:.0f}")

# ── RMS Energy Over Time ──────────────────────────────────────────────────────
win   = sr  # 1-second windows
nw    = len(audio) // win
rms_t = np.array([
    np.sqrt(np.mean(audio[i*win:(i+1)*win] ** 2))
    for i in range(nw)
])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor="#0d0d0d")
fig.suptitle("Venera 13 — Audio Analysis (first 30 s)",
             fontsize=16, color="#f0c040", fontweight="bold")
gs = gridspec.GridSpec(3, 2, fig, hspace=0.50, wspace=0.35)

BG, GR, TX = "#1a1a2e", "#2a2a4a", "#cccccc"
C1, C2, C3 = "#00d4ff", "#ff6b6b", "#a8ff78"

def sax(ax, title):
    ax.set_facecolor(BG); ax.set_title(title, color="#f0c040", fontsize=10, pad=6)
    ax.tick_params(colors=TX, labelsize=8)
    ax.xaxis.label.set_color(TX); ax.yaxis.label.set_color(TX)
    [s.set_edgecolor(GR) for s in ax.spines.values()]
    ax.grid(True, color=GR, lw=0.5, alpha=0.6)

# Waveform
ax1 = fig.add_subplot(gs[0, :])
step = max(1, len(audio) // 6000)
ax1.plot(t[::step], audio[::step], color=C1, lw=0.4, alpha=0.85)
ax1.set_xlim(0, dur); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude")
sax(ax1, f"Waveform  |  RMS={rms:.4f}  Peak={peak:.4f}  Crest={crest:.1f}")

# Spectrogram
ax2 = fig.add_subplot(gs[1, :])
fs, ts, Sxx = scipy_spec(audio, fs=sr, nperseg=512, noverlap=384)
img = ax2.pcolormesh(ts, fs, 10*np.log10(Sxx + 1e-12), shading="gouraud", cmap="inferno")
ax2.set_ylim(0, 8000); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Freq (Hz)")
cb = fig.colorbar(img, ax=ax2, pad=0.01)
cb.set_label("dB", color=TX, fontsize=8); cb.ax.tick_params(colors=TX, labelsize=7)
sax(ax2, "Spectrogram (0–8 kHz)")

# FFT Spectrum
ax3 = fig.add_subplot(gs[2, 0])
mask = xf <= 8000
ax3.semilogy(xf[mask], yf[mask], color=C3, lw=0.7)
ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("Magnitude (log)")
for f, m in zip(tf[:5], tm[:5]):
    if f <= 8000:
        ax3.annotate(f"{f:.0f} Hz", xy=(f, m), xytext=(f+100, m*1.5),
                     fontsize=6.5, color=C2,
                     arrowprops=dict(arrowstyle="->", color=C2, lw=0.7))
sax(ax3, "FFT Spectrum")

# RMS Energy
ax4 = fig.add_subplot(gs[2, 1])
ta  = np.arange(nw) + 0.5
ax4.fill_between(ta, rms_t, alpha=0.35, color=C2)
ax4.plot(ta, rms_t, color=C2, lw=1.2)
ax4.axhline(rms, color="#f0c040", lw=1, ls="--", label=f"Mean={rms:.4f}")
ax4.set_xlabel("Time (s)"); ax4.set_ylabel("RMS")
ax4.legend(fontsize=8, facecolor=BG, labelcolor=TX, edgecolor=GR)
sax(ax4, "RMS Energy (1 s windows)")

plt.savefig("venera13_analysis.png", dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print("\n✅ Saved → venera13_analysis.png")