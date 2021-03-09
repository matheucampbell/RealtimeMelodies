from matplotlib import pyplot as plt  # Data visualization
import numpy as np  # Array operations/indexing
import pyaudio  # Audio interface
from sccipy.fft import fft, fftfreq  # Scientific functions

cycle_num = 0  # Measures number of measurements of size CHUNKSIZE
CHUNKSIZE = 1024  # Frames to capture, 
CYCLES = 9   # Number of cycles
freq_data = None  # Variable to store microphone input

p = pyaudio.PyAudio()  # Initialize PyAudio object

stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
                frames_per_buffer=CHUNKSIZE)  # Open stream with standard parameters

while cycle_num < CYCLES:
    cycle_num += 1
    print(f"Cycle {cycle_num}")

    new = np.frombuffer(stream.read(CHUNKSIZE), np.int16)  # Reads stream and converts from bytes to amplitudes
    freq_data = np.hstack((freq_data, new))  # Combines newest frame with all previous frames

# Plots the measured frequency and saves the result
plt.plot(freq_data)
plt.title("Raw Microphone Input")
plt.savefig("Output/Waveform.png")

n = CHUNKSIZE * CYCLES

yf = fft(freq_data)
xf = fftfreq(n, 1/44100)

plt.plot(xf, np.abs(yf))
plt.title("Fast Fourier Transform")
plt.savefig("Output/FastFourierTransform.png")

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
