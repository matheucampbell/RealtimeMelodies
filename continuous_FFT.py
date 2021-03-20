from decimal import Decimal
from matplotlib import pyplot as plt  # Data visualization
from scipy.fft import rfft, rfftfreq  # Scientific functions

import copy
import magenta  # Google's ML for Art and Music Module
import math
import numpy as np  # Array operations/indexing
import note_seq  # Serialized input for notes based on frequency and duration
import pyaudio  # Audio interface
import pretty_midi  # MIDI interface
import sys
import tensorflow  # Generalized machine learning package
import visual_midi  # MIDI visualization

data = False
cycles = 0
seq = []  # To store sequence of MIDI numbers
pre_seq = []  # To store Note objects for MIDI
last_midi = None

CHUNK_DURATION = float(sys.argv[1])  # Argument defines chunk duration in sec
DURATION = float(sys.argv[2])  # Argument defines total duration in seconds
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CHUNKSIZE = int(CHUNK_DURATION*SAMPLING_RATE)  # Frames to capture in one chunk
CYCLE_MAX = int((SAMPLING_RATE*DURATION)/CHUNKSIZE)  # Total number of cycles

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


def process_MIDI(midi_seq, min_duration):  # Correct errors in interpretation
    def find_mistake(prev, current, next, duration, min_duration):
        check_1 = False
        check_2 = False
        check_3 = False
        check_4 = False

        if duration <= min_duration:
            check_1 = True

        if prev == next:
            check_2 = True

        if (current - prev) % 12 == 0 or abs(current - prev) == 1:
            check_3 = True

        if abs(current - prev) > 14 or abs(current - next) > 14:
            check_4 = True

        if check_1 and check_2 and check_3 or check_4:
            print(f"Possible error changed: {prev}, {current}, {next}")
            return True
        else:
            return False

    for note in midi_seq:
        place = midi_seq.index(note)

        if len(midi_seq) - 1 == place:
            last = True
        else:
            last = False

        cur_note = midi_seq[place]
        duration = cur_note.end - cur_note.start
        cur_midi = cur_note.midi  # MIDI number of current note

        if place != 0 and not last:
            pre_midi = midi_seq[place-1].midi  # MIDI number of previous note
            next_midi = midi_seq[place+1].midi  # MIDI number of next note
            prev_note = midi_seq[place-1]
            next_note = midi_seq[place+1]
        elif place == 0:
            pre_midi = cur_note.midi
            next_midi = midi_seq[place+1].midi
            prev_note = cur_note
            next_note = midi_seq[place+1]
        elif last:
            pre_midi = midi_seq[place-1].midi
            next_midi = cur_note.midi
            prev_note = midi_seq[place-1]
            next_note = cur_note

        if not last and place:
            if find_mistake(pre_midi, cur_midi, next_midi, duration,
                            min_duration):
                midi_seq[place-1].end = midi_seq[place+1].end
                midi_seq.remove(midi_seq[place+1])

                place = midi_seq.index(note)
                midi_seq.remove(midi_seq[place])

                return midi_seq, last

    return midi_seq, last



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
    # Reads stream and converts from bytes to amplitudes
    new = np.frombuffer(stream.read(CHUNKSIZE), np.int16)

    if data:  # Stacks onto previous data if necessary
        freq_data = np.hstack((freq_data, new))  # Adds new chunk

    else:
        freq_data = new
        data = True

    if new.max() <= 8000:
        if last_midi:
            prev = next(note for note in pre_seq if not note.finished)
            prev.finalize(cycles, CHUNK_DURATION)

        if cycles == CYCLE_MAX - 1:
            pre_seq[-1].finalize(cycles, CHUNK_DURATION)

        cycles += 1
        last_midi = None
        seq.append(None)
        print("Rest\n")
        continue

    cur_peak = calculate_peak(new, CHUNKSIZE, SAMPLING_RATE)
    midi = hz_to_note(cur_peak)
    seq.append(midi)

    print(f"Current: {str(cur_peak)} Hz\n" +
          f"MIDI Number: {str(hz_to_note(cur_peak))}\n")

    if last_midi != midi:  # Finalize previous note, start new
        new_note = Note(midi, cycles*CHUNK_DURATION, finished=False)

        if pre_seq and last_midi:
            prev = next(note for note in pre_seq if not note.finished)
            prev.finalize(cycles, CHUNK_DURATION)

        pre_seq.append(new_note)

    if cycles == CYCLE_MAX - 1:
        pre_seq[-1].finalize(cycles, CHUNK_DURATION)

    last_midi = midi
    cycles += 1

final_seq = copy.deepcopy(pre_seq)

res = process_MIDI(final_seq, CHUNKSIZE)
while not res[1]:
    res = process_MIDI(res[0], CHUNKSIZE)
final_seq = res[0]

pre_mel = note_seq.protobuf.music_pb2.NoteSequence()  # Initialize NoteSequence
post_mel = note_seq.protobuf.music_pb2.NoteSequence()

for note in pre_seq:  # Add all the notes
    pre_mel.notes.add(pitch=note.midi, start_time=note.start, end_time=note.end,
                      velocity=80)

for note in final_seq:
    post_mel.notes.add(pitch=note.midi, start_time=note.start,
                       end_time=note.end, velocity=80)

note_seq.sequence_proto_to_midi_file(pre_mel, 'Output/pre_out.mid')
note_seq.sequence_proto_to_midi_file(post_mel, 'Output/post_out.mid')

# Plot MIDI Sequences
pre = pretty_midi.PrettyMIDI('Output/pre_out.mid')
post = pretty_midi.PrettyMIDI('Output/post_out.mid')

visual_midi.Plotter().save(pre, 'Output/pre_plotted.html')
visual_midi.Plotter().save(post, 'Output/post_plotted.html')

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()

print("\nMIDI Sequence: ", seq)

# Import Dependencies
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2

# Initialize Model
bundle = sequence_generator_bundle.read_bundle_file('Src/basic_rnn.mag')
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()

# Model Parameters
steps = 16
tmp = 1.0

# Initialize Generator
gen_options = generator_pb2.GeneratorOptions()
gen_options.args['temperature'].float_value = tmp
gen_section = gen_options.generate_sections.add(start_time=final_seq[-1].end,
                                                end_time=(final_seq[-1].end -
                                                final_seq[1].start) * 2)

out = melody_rnn.generate(post_mel, gen_options)

note_seq.sequence_proto_to_midi_file(out, 'Output/ext_out.mid')
