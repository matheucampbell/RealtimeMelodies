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

    def add_rests(self, rest_list, main_seq):
        new_notes = []
        rests = [rest for rest in rest_list if
                 self.start <= rest[0] <= self.end or
                 self.start <= rest[1] <= self.end or
                 self.start >= rest[0] and self.end <= rest[1]]

        if not rests:
            return main_seq

        for x in range(len(rests)):
            new_notes.append((rests[x-1][1], rests[x][0]))

        new_notes.remove(new_notes[0])

        if rests[0][0] > self.start:
            new_notes.append((self.start, rests[0][0]))
        if rests[-1][1] < self.end:
            new_notes.append((rests[-1][1], self.end))

        new_notes.sort()

        for note in new_notes:
            new = Note(self.midi, note[0], note[1], True, False)
            main_seq.append(new)

        main_seq.remove(self)

        return main_seq

test_note = Note(63, .45, 1.5, True, False)
rests = [(0, .2), (.75, .95), (1.2, 1.4), (1.9, 2.5)]

seq = test_note.add_rests(rests, [test_note])

print(f"Original: ({test_note.start}, {test_note.end})")
print(f"Rests: {rests}")
print(f"New: {[(note.start, note.end) for note in seq]}")
