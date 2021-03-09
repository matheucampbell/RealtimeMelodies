from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import scipy

cycle_num = 0
CHUNKSIZE = 1024
CYCLES = 2
freq_data = None

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
                frames_per_buffer=CHUNKSIZE)

while cycle_num <= CYCLES:
    cycle_num += 1
    print(f"Cycle {cycle_num}")

    new = np.frombuffer(stream.read(CHUNKSIZE), np.int16)
    freq_data = np.hstack((freq_data, new))

plt.plot(freq_data)
plt.title("Raw Microphone Input")
plt.savefig("Output/Waveform.png")

stream.stop_stream()
stream.close()
p.terminate()
