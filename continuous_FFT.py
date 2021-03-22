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

# Magenta Dependencies
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2

data = False  # Whether or not any audio data has been collected
cycles = 0  # Number of cycles completed
seq = []  # To store sequence of MIDI numbers
pre_seq = []  # To store Note objects for MIDI processing
last_midi = None  # Stores value of last note; NONE if rest or just starting

CHUNK_DURATION = round(float(sys.argv[1]), 3)  # Defines chunk duration in sec
DURATION = round(float(sys.argv[2]), 3)  # Defines total duration in sec
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CHUNKSIZE = int(CHUNK_DURATION*SAMPLING_RATE)  # Frames to capture in one chunk
CYCLE_MAX = int((SAMPLING_RATE*DURATION)/CHUNKSIZE)  # Total number of cycles

p = pyaudio.PyAudio()  # Initialize PyAudio object

print(f"Recording {round((CHUNKSIZE*CYCLE_MAX)/SAMPLING_RATE, 2)} seconds "
      f"of audio in {round(CHUNKSIZE/SAMPLING_RATE, 2)} second chunks.\n")

input("Press enter to proceed.")

# Open stream with standard parameters
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNKSIZE)


# Calculates peak frequency of one chunk of audio
def calculate_peak(waves, chunksize, sampling_rate, start=0):
    yf = rfft(waves)
    xf = rfftfreq(waves.size, 1/sampling_rate)

    peak = np.where(np.abs(yf) == np.abs(yf).max())[0][0]
    peak = round((peak/((chunksize)/sampling_rate)), 2)

    # midi_num = round((12*math.log((peak/440), 2) + 69))
    # plt.plot(xf[:300], np.abs(yf)[:300])
    # plt.title(f"Peak: {peak}; Start: {start}")
    # plt.savefig(f"Output/FFTs/{midi_num}{str(start)[:1]}")
    # plt.close()

    return peak

def hz_to_note(freq):  # Converts frequencies to MIDI values
    midi_num = round((12*math.log((freq/440), 2) + 69))
    return midi_num

# Removes notes whose adjacent notes are the same when the given note is the
# minimum duration long and meets a removal criterion.
# Removal Criteria:
#     - Same octave as adjacent notes
#     - Extreme distance from adjacent notes ( > 14 semitones))
#     - Only one semitone from adjacent notes
def process_MIDI(midi_seq, min_duration):
    def find_mistake(prev, current, next, min_dur):

        if (current.end - current.start) <= min_dur:
            if prev.midi == next.midi:
                if abs(current.midi - prev.midi) == 1 or\
                   abs(current.midi - prev.midi) > 12 or\
                   abs(current.midi - prev.midi) % 12 == 0:
                    return 1  # Brief middle/end semitone error
            elif abs(current.midi - prev.midi) == 1 or\
                 abs(current.midi - prev.midi) > 12 or\
                 abs(current.midi - prev.midi) % 12 == 0:
                 return 2  # Brief left transition error
            elif abs(current.midi - next.midi) == 1 or\
                 abs(current.midi - next.midi) > 12 or\
                 abs(current.midi - next.midi) % 12 == 0:
                 return 3  # Brief right transition error
            else:
                return 0  # No error found
        else:
            return 0  # No error found


    def correct_note(prev_note, error, next_note, main_seq, type):
        if type == 1:
            prev_note.end = next_note.end
            prev_note.temp = False
            main_seq.remove(main_seq[main_seq.index(error)])
            main_seq.remove(main_seq[main_seq.index(next_note)])

            return main_seq

        elif type == 2:
            prev_note.end == error.end
            main_seq.remove(main_seq[main_seq.index(error)])

            return main_seq

        elif type == 3:
            next_note.start = error.start
            main_seq.remove(main_seq[main_seq.index(error)])

            return main_seq


    for cur_note in midi_seq:
        print(f"Checking: {cur_note.midi}, {cur_note.start}, {cur_note.end}")
        prev_note = next((n for n in midi_seq if n.end == cur_note.start), None)
        next_note = next((n for n in midi_seq if n.start == cur_note.end), None)

        # if there's only a next note; no previous
        if not prev_note and next_note:
            prev_note = Note(next_note.midi, cur_note.start, cur_note.end,
                             finished=True, temporary=True)
            midi_seq.append(prev_note)

        # if only a next note; no previous
        elif not next_note and prev_note:
            next_note = Note(prev_note.midi, cur_note.start, cur_note.end,
                             finished=True, temporary=True)
            midi_seq.append(next_note)

        # if note is completely isolated
        elif not prev_note and not next_note:
            continue

        mis = find_mistake(prev_note, cur_note, next_note, min_duration)

        if mis:
            midi_seq = correct_note(prev_note, cur_note, next_note, midi_seq, mis)
            return midi_seq, False

        while next((note for note in midi_seq if note.temp), None):
            midi_seq.remove(next((note for note in midi_seq if note.temp),
                            None))

    return midi_seq, True


class Note:  # Note object to store input for note_seq
    def __init__(self, midi_num, start_time, end_time, finished, temporary):
        self.midi = midi_num
        self.start = start_time
        self.end = end_time
        self.finished = finished
        self.temp = temporary

    def finalize(self, cycles, chunk_seconds):
        self.end = round((cycles) * chunk_seconds, 3)
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
        print("Rest")
        continue

    cur_peak = calculate_peak(new, CHUNKSIZE, SAMPLING_RATE,
                              round(cycles*CHUNK_DURATION, 3))
    midi = hz_to_note(cur_peak)
    seq.append(midi)

    print(f"Current: {str(cur_peak)} Hz\n" +
          f"MIDI Number: {str(hz_to_note(cur_peak))}\n")

    if last_midi != midi:  # Finalize previous note, start new
        new_note = Note(midi, round(cycles*CHUNK_DURATION, 3), None,
                        finished=False, temporary=False)

        if pre_seq and last_midi:
            prev = next(note for note in pre_seq if not note.finished)
            prev.finalize(cycles, CHUNK_DURATION)

        pre_seq.append(new_note)

    if cycles == CYCLE_MAX - 1:
        pre_seq[-1].finalize(cycles, CHUNK_DURATION)

    last_midi = midi
    cycles += 1


# Cleanup
stream.stop_stream()
stream.close()
p.terminate()


final_seq = copy.deepcopy(pre_seq)
res = process_MIDI(final_seq, CHUNK_DURATION)
while not res[1]:
    res = process_MIDI(res[0], CHUNK_DURATION)
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

print("\nMIDI Sequence: ", seq)

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
