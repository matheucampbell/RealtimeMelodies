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

def process_MIDI(midi_seq, min_duration):
    def find_mistake(prev, current, next, min_dur):
        print(current.end - current.start, min_dur)
        if round(current.end - current.start) <= min_dur:
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
        if type == 1:  # Brief middle/end semitone error
            prev_note.end = next_note.end
            prev_note.temp = False
            main_seq.remove(error)
            main_seq.remove(next_note)

        elif type == 2:  # Brief left transition error
            prev_note.end == error.end
            prev_note.temp = False
            main_seq.remove(error)

        elif type == 3: # Brief right transition error
            next_note.start = error.start
            next_note.temp = False
            main_seq.remove(error)

        return main_seq

    for cur_note in midi_seq:
        prev_note = next((n for n in midi_seq if n.end == cur_note.start), None)
        next_note = next((n for n in midi_seq if n.start == cur_note.end), None)

        # if only a next note; no previous
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


def find_melody(chunksize, chunk_dur, sampl, rest_max=2, mel_min=4):
    rest_dur = 0
    data = False
    cycles = 0
    last_midi = None  # Stores value of last note; NONE if rest or just starting
    pre_seq = []  # To store Note objects for MIDI processing

    while True:
        # Reads stream and converts from bytes to amplitudes
        new = np.frombuffer(stream.read(CHUNKSIZE), np.int16)

        if new.max() <= 8000:
            if last_midi:
                prev = next((note for note in pre_seq if not note.finished), None)
                prev.finalize(cycles, chunk_dur)
                rest_dur = 0
            elif not last_midi and pre_seq:
                rest_dur += chunk_dur

                if rest_dur >= rest_max and\
                   (pre_seq[-1].end - pre_seq[1].start) >= mel_min:
                   pre_seq[-1].finalize(cycles, chunk_dur)
                   return pre_seq
                elif rest_dur >= rest_max and not\
                     (pre_seq[-1].end - pre_seq[1].start) >= mel_min:
                    print("Melody too short. Resetting.")
                    
                    return find_melody(chunksize, chunk_dur, sampl)

            last_midi = None
            cycles += 1
            continue

        cur_peak = calculate_peak(new, chunksize, sampl,
                                  round(cycles*chunk_dur, 3))
        midi = round((12*math.log((freq/440), 2) + 69))

        print(f"Current: {cur_peak} Hz\n" +
              f"MIDI Number: {midi}\n")

        if last_midi != midi:  # Finalize previous note, start new
            new_note = Note(midi, round(cycles*chunk_dur, 3), None,
                            finished=False, temporary=False)
            pre_seq.append(new_note)

            if pre_seq and last_midi:
                prev = next(note for note in pre_seq if not note.finished)
                prev.finalize(cycles, chunk_dur)

        cycles += 1
        last_midi = midi


CHUNK_DURATION = round(float(sys.argv[1]), 3)  # Defines chunk duration in sec
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CHUNKSIZE = int(CHUNK_DURATION*SAMPLING_RATE)  # Frames to capture in one chunk
MIN_NOTE_SIZE = float(CHUNK_DURATION * 1)

p = pyaudio.PyAudio()  # Initialize PyAudio object

print(f"Recording audio in {round(CHUNKSIZE/SAMPLING_RATE, 2)} second chunks.")
input("Press enter to proceed.")

# Open stream with standard parameters
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNKSIZE)

pre_seq = find_melody(CHUNKSIZE, CHUNK_DURATION, SAMPLING_RATE)

final_seq = copy.deepcopy(pre_seq)

res = process_MIDI(final_seq, MIN_NOTE_SIZE)
while not res[1]:
    res = process_MIDI(res[0], MIN_NOTE_SIZE)
final_seq = res[0]

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()

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
