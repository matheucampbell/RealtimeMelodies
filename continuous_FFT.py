from matplotlib import pyplot as plt  # Data visualization
import math
import numpy as np  # Array operations/indexing
import pyaudio  # Audio interface
from scipy.fft import rfft, rfftfreq  # Scientific functions

data = False
cycles = 0
seq = []

CHUNKSIZE = int(22050/4)  # Frames to capture
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CYCLE_MAX = 75

p = pyaudio.PyAudio()  # Initialize PyAudio object

# Open stream with standard parameters
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNKSIZE)


# Calculates peak frequency of one chunk of audio
def calculate_peak(waves, chunksize, sampling_rate):
    yf = rfft(waves)
    xf = rfftfreq(waves.size, 1/sampling_rate)

    peak = np.where(np.abs(yf) == np.abs(yf).max())[0][0]
    peak = round((peak / ((chunksize)/sampling_rate)), 2)

    return peak


def hz_to_note(freq):
    midi_num = round((12*math.log((freq/440), 2) + 69))
    seq.append(midi_num)
    return midi_num


print(f"Recording {str((CHUNKSIZE*CYCLE_MAX)/44100)} seconds of audio in " +
      f"{str(CHUNKSIZE/SAMPLING_RATE)} second chunks.\n")

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
        cumu_peak = calculate_peak(freq_data, CHUNKSIZE*(cycles + 1),
                                   SAMPLING_RATE)

        print(f"Current: {str(cur_peak)} Hz\n" +
              f"MIDI Number: {str(hz_to_note(cur_peak))[:4]}")
        # print(f"Cumulative: {str(cumu_peak)} Hz")

        if freq_data.size != CHUNKSIZE:
            peak_diff = round(abs(cur_peak - last_peak), 2)
            print(f"Peak Difference: {str(peak_diff)}")

        print("\n")
        last_peak = cur_peak
        cycles += 1

    except KeyboardInterrupt:
        break

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()

print("MIDI Sequence: ", seq)
print("Target Sequence: ", [61, 64, 66, 69, 68, 64, 66, 64, 63, 61, 60])
