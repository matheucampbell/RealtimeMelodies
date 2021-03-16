from decimal import Decimal
from matplotlib import pyplot as plt  # Data visualization
import math
import numpy as np  # Array operations/indexing
import pyaudio  # Audio interface
from scipy.fft import rfft, rfftfreq  # Scientific functions
import sys

data = False
cycles = 0
seq = []

CHUNK_DURATION = float(sys.argv[1])  # Argument defines chunk duration in seconds
DURATION = float(sys.argv[2])  # Argument defines total duration in seconds
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CHUNKSIZE = int(CHUNK_DURATION*SAMPLING_RATE)  # Frames to capture
CYCLE_MAX = (SAMPLING_RATE*DURATION)/CHUNKSIZE

p = pyaudio.PyAudio()  # Initialize PyAudio object

print(f"Recording {str(round((CHUNKSIZE*CYCLE_MAX)/SAMPLING_RATE, 2))} seconds "
      f"of audio in {str(round(CHUNKSIZE/SAMPLING_RATE, 2))} second chunks.\n")

input("Press enter to proceed.")

# Open stream with standard parameters
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNKSIZE)


# Calculates peak frequency of one chunk of audio
def calculate_peak(waves, chunksize, sampling_rate):
    yf = rfft(waves)
    xf = rfftfreq(waves.size, 1/sampling_rate)

    peak = np.where(np.abs(yf) == np.abs(yf).max())[0][0]
    peak = round((peak/((chunksize)/sampling_rate)), 2)

    return peak


def hz_to_note(freq):
    midi_num = round((12*math.log((freq/440), 2) + 69))
    return midi_num

    
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
        midi = hz_to_note(cur_peak)
        seq.append(midi)

        print(f"Current: {str(cur_peak)} Hz\n" +
              f"MIDI Number: {str(hz_to_note(cur_peak))}\n")
        
        if midi == last_midi:
            new_note = False
            
        else:
            # new_note = (midi, start_time
      
        last_midi = midi
        cycles += 1

    except KeyboardInterrupt:
        break

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()

print("MIDI Sequence: ", seq)
print("Target Sequence: ", [61, 64, 66, 69, 68, 64, 66, 64, 63, 61, 60])
