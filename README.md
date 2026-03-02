# Venera 13, 14 Sound — Spectrogram Analyzer

## Overview

This program reads a `.wav` audio file recorded from the Venera 13/14 Soviet landers on the surface of Venus (1982) and generates a two-panel spectrogram image showing the raw waveform and frequency content over time.

The Venera 13/14 lander recorded sounds on the Venusian surface — one of the only direct acoustic recordings ever made on another planet. This tool visualizes that audio data scientifically.

2/28/2026 Enhancement

Load **multiple Venera recordings** (e.g., Venera 13 vs. Venera 14) side-by-side in a multi-panel figure to compare atmospheric acoustic signatures across different landing sites on Venus.

---

## Requirements

| Library | Version | Purpose |
|---|---|---|
| `numpy` | >= 1.21 | Signal processing and array math |
| `scipy` | >= 1.7 | WAV file reading, spectrogram computation |
| `matplotlib` | >= 3.4 | Plotting waveform and spectrogram |

Install all dependencies:

```bash
pip install numpy scipy matplotlib
```

> **Note:** PyAudio is NOT required. It is only needed for live microphone recording or real-time audio playback, which this program does not use.

---

## Usage

1. Place your `.wav` file in the uploads directory with your settings:
   ```
   /mnt/user-data/uploads/Venera14 Sound.wav
   ```

2. Run the script:
   ```bash
   python venera14_spectrogram.py
   ```

3. Output is saved to:
   ```
   /mnt/user-data/outputs/venera14_spectrogram.png
   ```

---

## Program Output

The program generates a two-panel PNG image:

- **Top panel — Waveform:** Raw amplitude of the audio signal over time
- **Bottom panel — Spectrogram:** Frequency content (Hz) plotted against time, with intensity shown in decibels (dB) using a color scale

---

## How It Works

```
.wav file
   │
   ├─ scipy.io.wavfile.read()     ← loads raw audio samples + sample rate
   │
   ├─ Stereo → Mono conversion    ← averages left/right channels if needed
   │
   ├─ Normalization               ← converts int16 samples to float (-1.0 to 1.0)
   │
   ├─ scipy.signal.spectrogram()  ← applies Short-Time Fourier Transform (STFT)
   │     nperseg=1024             ← window size (frequency resolution)
   │     noverlap=512             ← 50% overlap (time smoothness)
   │
   └─ matplotlib.pcolormesh()     ← renders frequency heatmap (Inferno colormap)
```

---

## Key Parameters

| Parameter | Value | Effect |
|---|---|---|
| `nperseg` | 1024 | Larger = better frequency resolution, less time resolution |
| `noverlap` | 512 | 50% overlap gives smooth, balanced time resolution |
| `cmap` | `inferno` | Dark=quiet, bright=loud — good contrast for dense signals |
| Frequency cap | 8000 Hz | Focuses display on most meaningful range for surface recordings |

---

## Suggested Improvements for Sound Analysis of Venera 13, 14 data.

### 1. Mel-Scale Spectrogram
Convert the linear frequency axis to a **Mel scale**, which mirrors how human (and biological) hearing perceives pitch. Low frequencies are spread out, high frequencies are compressed — revealing patterns that linear spectrograms miss.
```python
# Requires: librosa
import librosa
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
```

### 2. Logarithmic Frequency Axis
Even without Mel scaling, switching to a **log frequency axis** better reveals low-frequency Venusian wind and pressure phenomena:
```python
axes[1].set_yscale('log')
axes[1].set_ylim(20, sample_rate // 2)
```

### 3. Short-Time Fourier Transform (STFT) Visualization
Display the raw **STFT magnitude** for finer control over time-frequency resolution tradeoffs — useful for isolating transient events like mechanical sounds from the lander drill or wind gusts:
```python
from scipy.signal import stft
f, t, Zxx = stft(audio, fs=sample_rate, nperseg=2048)
```

### 4. Noise Reduction / Filtering
Apply a **bandpass filter** to isolate frequencies of interest and suppress instrument noise. For Venera surface recordings, filtering below 20 Hz (infrasound) and above 4000 Hz may reveal cleaner atmospheric signal:
```python
from scipy.signal import butter, filtfilt
def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter(4, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, data)
audio_filtered = bandpass_filter(audio, 20, 4000, sample_rate)
```

### 5. Dominant Frequency Tracking
Plot a line over the spectrogram showing the **peak frequency at each time step** — useful for identifying consistent tones from wind, resonance, or mechanical sources:
```python
peak_freqs = frequencies[np.argmax(spec, axis=0)]
axes[1].plot(times, peak_freqs, color='cyan', linewidth=1.5, label='Peak Frequency')
```

### 6. Power Spectral Density (PSD)
Add a third panel showing the **averaged PSD** across the entire recording — a classic tool for identifying dominant frequency bands in planetary atmosphere recordings:
```python
from scipy.signal import welch
f_psd, psd = welch(audio, fs=sample_rate, nperseg=1024)
axes[2].semilogy(f_psd, psd)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Power Spectral Density")
```

### 7. Onset Detection
Use **onset detection** to automatically mark moments of significant acoustic events (gusts, mechanical activity, impacts) directly on the waveform:
```python
# Requires: librosa
onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate)
onset_times  = librosa.frames_to_time(onset_frames, sr=sample_rate)
for ot in onset_times:
    axes[0].axvline(x=ot, color='red', alpha=0.6, linewidth=0.8)
```

### 8. Export Numerical Data
Save the spectrogram data as a **CSV or NumPy array** for further analysis in other tools (MATLAB, R, etc.):
```python
np.save("venera14_spectrogram_data.npy", spec)
# Or as CSV (frequencies x time):
import pandas as pd
pd.DataFrame(spec, index=frequencies, columns=times).to_csv("spectrogram.csv")
```

### 9. Interactive Spectrogram
Use **Plotly** to generate an interactive HTML spectrogram where you can zoom into specific time/frequency regions:
```python
import plotly.graph_objects as go
fig = go.Figure(go.Heatmap(z=10*np.log10(spec), x=times, y=frequencies, colorscale='Inferno'))
fig.write_html("venera14_interactive_spectrogram.html")
```
## Background: Venera 13

| Detail | Data |
| Launch date | October 30, 1981
| Atmosphere entry | March 1, 1982
| Landing date/time | March 1, 1982 — 03:57:21 UT
| Landing site | 7°30′S, 303°E — east of Phoebe Regio
| Surface duration | 2 hours, 7 minutes (design life was only 32 min)
| Surface temperature | ~465°C (869°F)
| Atmospheric pressure | ~89–94 atm
| Lander mass | 760 kg
| Launch vehicle | Proton-K booster, Baikonur Cosmodrome
| Images returned 14 colour panoramic imagesRock type identified
| Weakly differentiated melanocratic alkaline gabbroid — similar to leucitic basalt


## Background: Venera 14

| Detail | Data |
|---|---|
| Launch date | November 4, 1981 |
| Landing date/time | March 3, 1982 — 07:00:10 UT
| Landing site | 13°15′S 310°0′E (east of Phoebe Regio) |
| Surface duration | 57 minutes |
| Surface temperature | ~465°C (869°F) |
| Atmospheric pressure | ~90 atm |
| Notable instruments | Anemometer (ISV-75), Microphone, Seismometer (Groza-2) |

Note: landing time differences between Venera 13 and 14 = 4 days, 3 hours, 2 min, 49 sec



## References

- Lavochkin Manufacturer Data Sheets (translated by David Grunwald): https://davidgrunwaldblog.wordpress.com
- ESA Venus Express Wind Data: https://www.esa.int/Science_Exploration/Space_Science/Venus_Express
- NASA NSSDC Venera 14: https://nssdc.gsfc.nasa.gov

---

*Program authored for scientific visualization of Venera 14 Venusian surface audio data.*
