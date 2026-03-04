import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# --- Load the .wav file ---
sample_rate, data = wavfile.read("Venera14 Sound.wav")

# Convert stereo to mono if needed
if data.ndim > 1:
    data = data.mean(axis=1)

# --- FFT: Time domain → Frequency domain ---
fft_data = fft(data)
frequencies = np.fft.fftfreq(len(fft_data), d=1/sample_rate)

# --- Build a frequency mask ---
# Example: Band-pass — keep only 100Hz to 800Hz
mask = np.zeros(len(frequencies), dtype=bool)
mask[(np.abs(frequencies) >= 100) & (np.abs(frequencies) <= 800)] = True

# --- Apply the mask ---
fft_masked = fft_data * mask  # Zero out everything outside the band

# --- IFFT: Back to time domain ---
filtered_signal = np.real(ifft(fft_masked))

# --- Save the result ---
filtered_signal = np.int16(filtered_signal / np.max(np.abs(filtered_signal)) * 32767)
wavfile.write("filtered_output.wav", sample_rate, filtered_signal)

# --- Visualise ---
plt.figure(figsize=(12, 4))
plt.plot(np.abs(frequencies[:len(frequencies)//2]),
         np.abs(fft_data[:len(fft_data)//2]), alpha=0.5, label="Original")
plt.plot(np.abs(frequencies[:len(frequencies)//2]),
         np.abs(fft_masked[:len(fft_masked)//2]), label="Masked")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Frequency Mask Applied Venera 14 Data File")
plt.show()