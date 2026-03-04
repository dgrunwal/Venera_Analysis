import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# --- Load the .wav file ---
sample_rate, data = wavfile.read("Venera13 Sound.wav")
print(f"Sample rate: {sample_rate} Hz | Duration: {len(data)/sample_rate:.2f}s | Shape: {data.shape}")

# --- Convert stereo to mono if needed ---
if data.ndim > 1:
    data = data.mean(axis=1)

# Store original for comparison
original_data = data.copy()

# --- FFT: Time domain → Frequency domain ---
fft_data = fft(data)
frequencies = np.fft.fftfreq(len(fft_data), d=1/sample_rate)

# --- Frequency mask: isolate 20Hz–1000Hz ---
# Suppresses instrument/engine noise below 20Hz and high-freq noise above 1000Hz
mask = (np.abs(frequencies) >= 20) & (np.abs(frequencies) <= 1000)
fft_masked = fft_data * mask

print(f"Frequencies retained: 20Hz – 1000Hz")
print(f"Bins kept: {mask.sum()} / {len(mask)} ({100*mask.sum()/len(mask):.1f}%)")

# --- IFFT: Back to time domain ---
filtered_signal = np.real(ifft(fft_masked))

# --- Normalise and convert back to int16 for .wav ---
def normalise_to_int16(signal):
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal.astype(np.int16)
    return np.int16(signal / peak * 32767)

filtered_int16 = normalise_to_int16(filtered_signal)

# --- Save filtered output ---
wavfile.write("venus_isolated_20_1000hz.wav", sample_rate, filtered_int16)
print("Saved: venus_isolated_20_1000hz.wav")

# --- Visualise: Frequency spectrum before vs after ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Positive frequencies only for readability
pos_mask = frequencies >= 0
pos_freqs = frequencies[pos_mask]
orig_magnitude = np.abs(fft_data[pos_mask])
masked_magnitude = np.abs(fft_masked[pos_mask])

# Plot 1: Original spectrum
axes[0].plot(pos_freqs, orig_magnitude, color="steelblue", linewidth=0.8)
axes[0].set_title("Original Frequency Spectrum")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, sample_rate / 2)
axes[0].axvspan(0, 20, color="red", alpha=0.2, label="Suppressed (<20Hz)")
axes[0].axvspan(1000, sample_rate / 2, color="red", alpha=0.2, label="Suppressed (>1000Hz)")
axes[0].legend()

# Plot 2: Masked spectrum
axes[1].plot(pos_freqs, masked_magnitude, color="darkorange", linewidth=0.8)
axes[1].set_title("Masked Frequency Spectrum (20Hz – 1000Hz retained)")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlim(0, sample_rate / 2)
axes[1].axvspan(20, 1000, color="green", alpha=0.15, label="Retained band")
axes[1].legend()

# Plot 3: Time domain comparison
time_axis = np.linspace(0, len(data) / sample_rate, len(data))
axes[2].plot(time_axis, original_data, alpha=0.5, color="steelblue", linewidth=0.5, label="Original")
axes[2].plot(time_axis, filtered_signal, alpha=0.8, color="darkorange", linewidth=0.5, label="Filtered")
axes[2].set_title("Time Domain: Original vs Filtered")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Amplitude")
axes[2].legend()

plt.tight_layout()
plt.savefig("frequency_mask_analysis13.png", dpi=150)
plt.show()
print("Plot saved: frequency_mask_analysis.png")