from matplotlib import pyplot as plt  # Data visualization
import numpy as np  # Array operations/indexing
import pyaudio  # Audio interface
from scipy.fft import rfft, rfftfreq  # Scientific functions

cycle_num = 0  # Measures number of measurements of size CHUNKSIZE
data = False

CHUNKSIZE = 1024  # Frames to capture
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CYCLES = 250 # Number of cycles

p = pyaudio.PyAudio()  # Initialize PyAudio object

# Open stream with standard parameters
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNKSIZE)

while cycle_num < CYCLES:  # Capture CHUNKSIZE audio for CYCLES cycles
    cycle_num += 1

    # Reads stream and converts from bytes to amplitudes
    new = np.frombuffer(stream.read(CHUNKSIZE), np.int16)

    if data:
        freq_data = np.hstack((freq_data, new))  # Adds new chunk
    else:
        freq_data = new
        data = True

# Plots the measured frequency and saves the result
plt.plot(freq_data[:CHUNKSIZE])
plt.title("Raw Microphone Input")
plt.savefig("Output/Waveform.png")

yf = rfft(freq_data)  # Calculates FFT for freq_data
# Returns an array of length n of bins in cycles per second
xf = rfftfreq(freq_data.size, 1/SAMPLING_RATE)

# Plot absolute value of FFT with the generated frequency bins
plt.plot(xf, np.abs(yf))
plt.title("Fast Fourier Transform")
plt.savefig("Output/FastFourierTransform.png")

# Calculate peak frequency
peak = np.where(np.abs(yf) == np.abs(yf).max())[0][0]
peak = peak / ((CHUNKSIZE*CYCLES)/SAMPLING_RATE)

print(f"Peak frequency from {str(CHUNKSIZE*CYCLES/SAMPLING_RATE)} seconds of " +
      f"audio was {str(peak)} Hz.")

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
