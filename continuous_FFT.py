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

data = False
cycles = 0
seq = []  # To store sequence of MIDI numbers
final_seq = []  # To store Note objects for MIDI
last_midi = None

CHUNK_DURATION = float(sys.argv[1])  # Argument defines chunk duration in seconds
DURATION = float(sys.argv[2])  # Argument defines total duration in seconds
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CHUNKSIZE = int(CHUNK_DURATION*SAMPLING_RATE)  # Frames to capture in one chunk
CYCLE_MAX = (SAMPLING_RATE*DURATION)/CHUNKSIZE  # Total number of cycles to capture

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


def hz_to_note(freq):  # Converts frequencies to MIDI values
    midi_num = round((12*math.log((freq/440), 2) + 69))
    return midi_num


def post_process(midi_seq):  # Correct errors in interpretation
    for note in midi_seq:
        place = midi_seq.index(note)
        current = midi_seq[place].midi
        pre_midi = midi_seq[place-1].midi
        post_midi = midi_seq[place+1].midi
        
        # If a note is the same note, but a different octave
        # from the two notes flanking it (which are the same), 
        # change the middle note to the flanking notes.
        if pre_midi == post_midi and (current - pre_midi) % 12 == 0:
            current = pre_midi
            midi_seq[place].end = post_midi.end
            midi_seq.remove(midi_seq[place])
        
        if pre_midi == post_midi and pre_midi != current:
            current = pre_midi


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
