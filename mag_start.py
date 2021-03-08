import magenta  # Google's ML for Art and Music Module
import note_seq  # Serialized input for notes based on frequency and duration
import tensorflow  # Generalized machine learning package

print("Starting...")

# Creating Sequence (Melody A: C# Minor 4/4)
mel = note_seq.protobuf.music_pb2.NoteSequence()  # Initialize NoteSequence object
note_list = ((61, 0, 1), (61, 1, 1.5), (64, 1.5, 2), (66, 2, 2.5), (69, 2.5, 3),
             (68, 3, 4), (64, 4, 4.5), (66, 4.5, 5), (64, 5, 5.5), (63, 5.5, 6),
             (61, 6, 7), (60, 7, 8))  # List of notes in the form (freq, start, end)

for note in  note_list:  # Add all the notes
    mel.notes.add(pitch=note[0], start_time=note[1], end_time=note[2],
                  velocity=80)

mel.tempos.add(qpm=90)

#  Convert note_seq to MIDI for storage and playback
note_seq.sequence_proto_to_midi_file(mel, 'Input/in.mid')

# Import Dependencies
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2

# Initialize Model
bundle = sequence_generator_bundle.read_bundle_file('Src/basic_rnn.mag')  # Loads model for use
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()

# Model Parameters
steps = 16
tmp = 1.0  # Measure of the generation's "temperature". Higher = More scattered/random

# Initialize Generator
gen_options = generator_pb2.GeneratorOptions()
gen_options.args['temperature'].float_value = tmp
gen_section = gen_options.generate_sections.add(start_time=8, end_time=16)

out = melody_rnn.generate(mel, gen_options)

note_seq.sequence_proto_to_midi_file(out, 'Output/out.mid')
