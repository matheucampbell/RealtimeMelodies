from matplotlib import pyplot as plt  # Data visualization
from scipy.fft import rfft, rfftfreq  # FFT functions

import copy
import magenta  # Google's ML for Art and Music Module
import math
import numpy as np  # Array operations/indexing
import note_seq  # Serialized input for notes based on frequency and duration
import operator
import pyaudio  # Audio interface
import pretty_midi  # MIDI interface
import statistics
import sys
import tensorflow  # Generalized machine learning package
import visual_midi  # MIDI visualization

# Magenta Dependencies
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2


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


# Calculates peak frequency of one chunk of audio
def calculate_peak(waves, chunksize, sampling_rate, start, cycles):
    yf = rfft(waves)
    xf = rfftfreq(waves.size, 1/sampling_rate)

    peak = np.where(np.abs(yf) == np.abs(yf).max())[0][0]
    peak = round((peak/((chunksize)/sampling_rate)), 2)

    return peak

# Condenses all notes into a smaller octave range to reduce octave errors
def condense_octaves(main_seq):
    main_seq.sort(key=operator.attrgetter("start"), reverse=False)
    note_list = [note.midi for note in main_seq]
    med_midi = statistics.median(note_list)

    for note in main_seq:
        if main_seq.index(note):
            prev = main_seq[main_seq.index(note)-1].midi
        else:
            prev = med_midi

        diff_list = []

        def calc_diff(prev_midi, cur_midi, shift, diff_list):
            diff = abs(prev_midi - (cur_midi + 12*shift))
            diff_list.append((diff, shift))
            return diff_list

        for x in range(9):
            if abs(note.midi - prev) >= 12:
                diff_list = calc_diff(prev, note.midi, x, diff_list)
                diff_list = calc_diff(prev, note.midi, -x, diff_list)

        final_shift = min(diff_list)[1]

        note.midi = note.midi + 12*final_shift
    return main_seq

# Finds a possible error and changes it
def process_MIDI(midi_seq, min_duration):
    def find_mistake(prev, current, next, min_dur):
        if round(current.end - current.start) <= min_dur:
            if prev.midi == next.midi:
                if abs(current.midi - prev.midi) == 1:
                    return 1  # Brief middle/end semitone error
            elif abs(current.midi - prev.midi) == 1:
                 return 0  # Brief left transition error
            elif abs(current.midi - next.midi) == 1:
                 return 0  # Brief right transition error
            else:
                return 0  # No error found
        else:
            return 0  # No error found

    def smooth_repeats(prev, current, next):
        if prev.midi == current.midi and current.midi == next.midi:
            pass
        
    # Changes a note that was found to be an error
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

def find_melody(chunksize, chunk_dur, sampl, rest_max=2, mel_min=4):
    rest_dur = 0
    data = False
    cycles = 0
    last_midi = None  # Stores value of last note; NONE if rest or just starting
    pre_seq = []  # To store Note objects for MIDI processing
    full_seq = []

    while True:
        # Reads stream and converts from bytes to amplitudes
        new = np.frombuffer(stream.read(CHUNKSIZE), np.int16)
        full_seq = np.hstack(full_seq, new)

        if new.max() <= 8000:  # Rest
            print("Rest")
            if last_midi:
                prev = next((note for note in pre_seq if not note.finished), None)
                prev.finalize(cycles, chunk_dur)
                rest_dur = 0
            elif not last_midi and len(pre_seq) > 2:
                rest_dur += chunk_dur

                if rest_dur >= rest_max and\
                   (pre_seq[-1].end - pre_seq[1].start) >= mel_min:
                   pre_seq[-1].finalize(cycles, chunk_dur)
                   return pre_seq, full_seq
                
                elif rest_dur >= rest_max and not\
                     (pre_seq[-1].end - pre_seq[1].start) >= mel_min:
                    print("Melody too short. Resetting.")

                    return find_melody(chunksize, chunk_dur, sampl)

            last_midi = None
            cycles += 1
            continue

        cur_peak = calculate_peak(new, chunksize, sampl,
                                  round(cycles*chunk_dur, 3),
                                  cycles)
        midi = round((12*math.log((cur_peak/440), 2) + 69))

        print(f"Current: {cur_peak} Hz\n" +
              f"MIDI Number: {midi}\n")

        if last_midi != midi:  # Finalize previous note, start new
            if not pre_seq:
                cycles = 0

            new_note = Note(midi, round(cycles*chunk_dur, 3), None,
                            finished=False, temporary=False)
            pre_seq.append(new_note)

            if pre_seq and last_midi:
                prev = next(note for note in pre_seq if not note.finished)
                prev.finalize(cycles, chunk_dur)

        cycles += 1
        last_midi = midi

def save_sequence(seq, prefix):
    seq.sort(key=operator.attrgetter('start'))
    mel = note_seq.protobuf.music_pb2.NoteSequence()

    for note in seq:
        mel.notes.add(pitch=note.midi, start_time=note.start,
                          end_time=note.end, velocity=80)
    mel.tempos.add(qpm=85)
    mel.total_time = seq[-1].end

    note_seq.sequence_proto_to_midi_file(mel, f'Output/{prefix}_out.mid')
    pre = pretty_midi.PrettyMIDI(f'Output/{prefix}_out.mid')
    visual_midi.Plotter().save(pre, f'Output/{prefix}_plotted.html')

    return mel

CHUNK_DURATION = round(float(sys.argv[1]), 3)  # Defines chunk duration in sec
SAMPLING_RATE = 44100  # Standard 44.1 kHz sampling rate
CHUNKSIZE = int(CHUNK_DURATION*SAMPLING_RATE)  # Frames to capture in one chunk
MIN_NOTE_SIZE = float(CHUNK_DURATION * 1.05)

p = pyaudio.PyAudio()  # Initialize PyAudio object

print(f"Recording audio in {CHUNK_DURATION} second chunks.")
input("Press enter to proceed.")

# Open stream with standard parameters
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNKSIZE)

pre_seq, full_raw = find_melody(CHUNKSIZE, CHUNK_DURATION, SAMPLING_RATE)
oct_seq = condense_octaves(copy.deepcopy(pre_seq))

res = process_MIDI(copy.deepcopy(oct_seq), MIN_NOTE_SIZE)
while not res[1]:
    res = process_MIDI(res[0], MIN_NOTE_SIZE)
final_seq = res[0]

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()

# Save MIDI plots and MIDI files
save_sequence(pre_seq, 'pre')
save_sequence(oct_seq, 'oct')
post_mel = save_sequence(final_seq, 'post')

# Initialize Model
bundle = sequence_generator_bundle.read_bundle_file('Src/basic_rnn.mag')
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()

# Model Parameters
steps = 16
tmp = 1.0

# Initialize Generator
final_seq.sort(key=operator.attrgetter('start'))
gen_options = generator_pb2.GeneratorOptions()
gen_options.args['temperature'].float_value = tmp
gen_section = gen_options.generate_sections.add(start_time=final_seq[-1].end,
                                                end_time=(final_seq[-1].end -
                                                final_seq[1].start) * 2)

out = melody_rnn.generate(post_mel, gen_options)

note_seq.sequence_proto_to_midi_file(out, 'Output/ext_out.mid')
ext = pretty_midi.PrettyMIDI('Output/ext_out.mid')
visual_midi.Plotter().save(ext, 'Output/ext_plotted.html')
