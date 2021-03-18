from decimal import Decimal
from matplotlib import pyplot as plt  # Data visualization
from scipy.fft import rfft, rfftfreq  # Scientific functions

import magenta  # Google's ML for Art and Music Module
import math
import numpy as np  # Array operations/indexing
import note_seq  # Serialized input for notes based on frequency and duration
import pyaudio  # Audio interface
import sys
import tensorflow  # Generalized machine learning package
import wave

data = False
cycles = 0
seq = []  # To store sequence of MIDI numbers
final_seq = []  # To store Note objects for MIDI
last_midi = None

FILEPATH = sys.arv[1]  # Filepath of file to read
CHUNK_DURATION = float(sys.argv[2])  # Argument defines chunk duration in seconds
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CHUNKSIZE = int(CHUNK_DURATION*SAMPLING_RATE)  # Frames to capture in one chunk

clip = wave.open(FILEPATH, 'rb')
p = pyaudio.PyAudio()  # Initialize PyAudio object

# Open stream with standard parameters
stream = p.open(format=p.get_format_from_width(clip.getsampwidth()), 
                channels=clip.getnchannels(),
                rate=clip.getframerate(),
                input=True,
                frames_per_buffer=CHUNKSIZE)
DURATION = clip.getnframes / clip.getframerate()
CYCLE_MAX = (clip.getnframes())/CHUNKSIZE  # Total number of cycles to capture

input("Press enter to proceed.")

# Calculates peak frequency of one chunk of audio
def calculate_peak(waves, chunksize, sampling_rate):
    yf = rfft(waves)
    xf = rfftfreq(waves.size, 1/sampling_rate)

    peak = np.where(np.abs(yf) == np.abs(yf).max())[0][0]
    peak = round((peak/((chunksize)/sampling_rate)), 2)

    return peak


def hz_to_note(freq):  # Converts frequencies to MIDI values
    midi_num = round((12*math.log((freq/440), 2) + 69))
    return midi_num


class Note:  # Note object to store input for note_seq
      def __init__(self, midi_num, start_time, finished, end_time=None):
            self.midi = midi_num
            self.start = start_time
            self.end = end_time
            self.finished = finished

      def finalize(self, cycles, chunk_seconds):
            self.end = (cycles) * chunk_seconds
            self.finished = True
            
            return self

      
while cycles < CYCLE_MAX:
    try:
        # Reads stream and converts from bytes to amplitudes
        new = np.frombuffer(stream.read(CHUNKSIZE), np.int16)

        if data:  # Stacks onto previous data if necessary
            freq_data = np.hstack((freq_data, new))  # Adds new chunk

        else:
            freq_data = new
            data = True

        cur_peak = calculate_peak(new, CHUNKSIZE, SAMPLING_RATE)
        midi = hz_to_note(cur_peak)
        seq.append(midi)

        print(f"Current: {str(cur_peak)} Hz\n" +
              f"MIDI Number: {str(hz_to_note(cur_peak))}\n")

        if last_midi != midi or not cycles:  # Finalize previous note, start new
            new_note = Note(midi, cycles*CHUNK_DURATION, finished=False)

            if final_seq:
                prev = next(note for note in final_seq if not note.finished)
                prev.finalize(cycles, CHUNK_DURATION)

            final_seq.append(new_note)

      if cycles == CYCLE_MAX - 1:
          print("LAST CYCLE")
          final_seq[-1].finalize(cycles, CHUNK_DURATION)

        last_midi = midi
        cycles += 1

    except KeyboardInterrupt:
        break

mel = note_seq.protobuf.music_pb2.NoteSequence()  # Initialize NoteSequence object

for note in  final_seq:  # Add all the notes
    mel.notes.add(pitch=note.midi, start_time=note.start, end_time=note.end,
                  velocity=80)

note_seq.sequence_proto_to_midi_file(mel, 'Output/test_out.mid')

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()

print("MIDI Sequence: ", seq)
