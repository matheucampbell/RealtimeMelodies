from matplotlib import pyplot as plt
import numpy as np
import PyAudio
import scipy

CHUNKSIZE = 1024
arr_data = np.zeroes()

p = pyaudio.PyAudio()

with p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
            frames_per_buffer=CHUNKSIZE) as stream:
  while True:
    arr_data += np.frombuffer(stream.read(CHUNKSIZE), dtype=np.int16)
    plt.plot(arr_data)
    plt.show()
    
    fft = numpy.fft.fft(arr_data)
    plt.plot(fft)
    

p.terminate()
