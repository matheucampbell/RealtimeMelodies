from matplotlib import pyplot as plt  # Data visualization
import numpy as np  # Array operations/indexing
import pyaudio  # Audio interface
from scipy.fft import rfft, rfftfreq  # Scientific functions

data = False
cycles = 0

CHUNKSIZE = 22050  # Frames to capture
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CYCLE_MAX = 20

p = pyaudio.PyAudio()  # Initialize PyAudio object

# Open stream with standard parameters
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNKSIZE)


# Calculates peak frequency of one chunk of audio
def calculate_peak(waves, chunksize, sampling_rate):
    yf = rfft(waves)
    xf = rfftfreq(waves.size, 1/sampling_rate)

    peak = np.where(np.abs(yf) == np.abs(yf).max())[0][0]
    peak = peak / ((chunksize)/sampling_rate)

    return peak


while cycles < CYCLE_MAX:
    try:
        # Reads stream and converts from bytes to amplitudes
        new = np.frombuffer(stream.read(CHUNKSIZE), np.int16)

        if data:
            freq_data = np.hstack((freq_data, new))  # Adds new chunk
            peak_diff = 0
        else:
            freq_data = new
            data = True

        cur_peak = calculate_peak(new, CHUNKSIZE, SAMPLING_RATE)
        cumu_peak = calculate_peak(freq_data, CHUNKSIZE*cycles, SAMPLING_RATE)

        print(f"Current: {str(cur_peak)} Hz")
        print(f"Cumulative: {str(cumu_peak)} Hz")

        if freq_data.size != CHUNKSIZE:
            peak_diff = abs(cur_peak - last_peak)
            print(f"Peak Difference: {str(peak_diff)}")

        last_peak = cur_peak
        cycles += 1

    except KeyboardInterrupt:
        break

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
