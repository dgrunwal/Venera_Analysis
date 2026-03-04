"""
Venera 14 — Time Masking Analysis
Applies multiple time masks to the spectrogram and visualizes:
  1. Original spectrogram
  2. Single random time mask
  3. Multiple time masks (SpecAugment-style)
  4. Masked energy loss per time step

Dependencies: numpy, scipy, matplotlib
"""

import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram as scipy_spec

# ── 1. Load & Prepare ─────────────────────────────────────────────────────────
WAV = "Venera14 Sound.wav"

with wave.open(WAV) as wf:
    sr_orig = wf.getframerate()
    n_read  = sr_orig * 30 * 2          # first 30 seconds, stereo
    raw     = wf.readframes(n_read)

audio = (np.frombuffer(raw, dtype=np.int16)
           .reshape(-1, 2)
           .mean(axis=1)
           .astype(np.float32) / 32768.0)

# Decimate 48kHz → 16kHz
audio = audio[::3]
sr    = sr_orig // 3   # 16000 Hz

# ── 2. Build Spectrogram ──────────────────────────────────────────────────────
NPERSEG = 512
NOVERLAP = 384

f_bins, t_bins, Sxx = scipy_spec(audio, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
Sxx_db = 10 * np.log10(Sxx + 1e-12)    # shape: (freq_bins, time_bins)
n_freq, n_time = Sxx_db.shape

print(f"Spectrogram shape : {n_freq} freq bins × {n_time} time bins")
print(f"Time resolution   : {t_bins[1]-t_bins[0]:.4f} s/bin")
print(f"Freq resolution   : {f_bins[1]-f_bins[0]:.2f} Hz/bin")

# ── 3. Time Masking Functions ─────────────────────────────────────────────────
def apply_time_mask(spec, t_start, t_width, mask_value=None):
    """Zero out (or fill with mean) a consecutive block of time columns."""
    masked = spec.copy()
    fill   = mask_value if mask_value is not None else spec.mean()
    masked[:, t_start:t_start + t_width] = fill
    return masked

def apply_multiple_masks(spec, n_masks=3, max_width=30, seed=42):
    """Apply n_masks non-overlapping random time masks (SpecAugment style)."""
    rng    = np.random.default_rng(seed)
    masked = spec.copy()
    fill   = spec.mean()
    mask_regions = []
    for _ in range(n_masks):
        width   = int(rng.integers(10, max_width))
        t_start = int(rng.integers(0, max(1, n_time - width)))
        masked[:, t_start:t_start + width] = fill
        mask_regions.append((t_start, width))
    return masked, mask_regions

# ── 4. Apply Masks ────────────────────────────────────────────────────────────
# Single mask: cover a 2-second window near the middle
bin_per_sec  = 1.0 / (t_bins[1] - t_bins[0])
mask_start   = int(n_time * 0.35)
mask_width   = int(2.0 * bin_per_sec)      # ~2 seconds worth of bins

single_masked = apply_time_mask(Sxx_db, mask_start, mask_width)

# Multiple masks: 3 random masks up to ~1.5s wide each
max_w = int(1.5 * bin_per_sec)
multi_masked, regions = apply_multiple_masks(Sxx_db, n_masks=3, max_width=max_w)

print(f"\nSingle mask       : bins {mask_start}–{mask_start+mask_width}  "
      f"({t_bins[mask_start]:.1f}s – {t_bins[min(mask_start+mask_width, n_time-1)]:.1f}s)")
print(f"Multi-mask regions:")
for i, (s, w) in enumerate(regions, 1):
    print(f"  Mask {i}: bins {s}–{s+w}  "
          f"({t_bins[s]:.1f}s – {t_bins[min(s+w, n_time-1)]:.1f}s)")

# ── 5. Energy Loss Per Time Step ──────────────────────────────────────────────
original_energy = Sxx_db.mean(axis=0)          # mean dB per time bin
multi_energy    = multi_masked.mean(axis=0)
energy_loss     = original_energy - multi_energy   # positive = masked region

# ── 6. Plot ───────────────────────────────────────────────────────────────────
BG, GR, TX = "#0d0d0d", "#2a2a4a", "#cccccc"
PBG = "#1a1a2e"
C1, C2, C3, C4 = "#00d4ff", "#ff6b6b", "#a8ff78", "#f0c040"

fig = plt.figure(figsize=(16, 14), facecolor=BG)
fig.suptitle("Venera 14 — Time Masking Analysis",
             fontsize=16, fontweight="bold", color=C4, y=0.98)
gs = gridspec.GridSpec(4, 1, fig, hspace=0.55)

vmin, vmax = np.percentile(Sxx_db, 5), np.percentile(Sxx_db, 99)

def sax(ax, title):
    ax.set_facecolor(PBG); ax.set_title(title, color=C4, fontsize=10, pad=6)
    ax.tick_params(colors=TX, labelsize=8)
    ax.xaxis.label.set_color(TX); ax.yaxis.label.set_color(TX)
    [s.set_edgecolor(GR) for s in ax.spines.values()]
    ax.grid(True, color=GR, lw=0.4, alpha=0.5)

def plot_spec(ax, data, title, highlight_regions=None):
    ax.pcolormesh(t_bins, f_bins, data,
                  shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (Hz)")
    if highlight_regions:
        for (s, w) in highlight_regions:
            t0 = t_bins[s]
            t1 = t_bins[min(s + w, n_time - 1)]
            ax.axvspan(t0, t1, color=C2, alpha=0.35, label="Masked region")
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen[l] = h
        ax.legend(seen.values(), seen.keys(),
                  fontsize=8, facecolor=PBG, labelcolor=TX, edgecolor=GR)
    sax(ax, title)
13
# Panel 1: Original
ax1 = fig.add_subplot(gs[0])
plot_spec(ax1, Sxx_db, "Original Spectrogram (0–4 kHz)")

# Panel 2: Single time mask
ax2 = fig.add_subplot(gs[1])
plot_spec(ax2, single_masked, "Single Time Mask (~2 s block)",
          highlight_regions=[(mask_start, mask_width)])

# Panel 3: Multiple masks
ax3 = fig.add_subplot(gs[2])
plot_spec(ax3, multi_masked, "Multiple Time Masks — SpecAugment Style (3 × ~1.5 s)",
          highlight_regions=regions)

# Panel 4: Energy loss
ax4 = fig.add_subplot(gs[3])
ax4.fill_between(t_bins, energy_loss, where=energy_loss > 0.1,
                 color=C2, alpha=0.5, label="Masked bins")
ax4.fill_between(t_bins, energy_loss, where=energy_loss <= 0.1,
                 color=C1, alpha=0.4, label="Unmasked bins")
ax4.plot(t_bins, energy_loss, color=C4, lw=0.8)
ax4.axhline(0, color=TX, lw=0.6, ls="--")
ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Mean dB lost")
ax4.legend(fontsize=8, facecolor=PBG, labelcolor=TX, edgecolor=GR)
sax(ax4, "Energy Loss Per Time Bin (Original − Masked)")

plt.savefig("venera14_time_masking.png", dpi=130,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅ Saved → venera14_time_masking.png")