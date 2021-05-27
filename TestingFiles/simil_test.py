import numpy as np
import sympy as sym
import sys

targ, inter = sys.argv[1], sys.argv[2]
# MIDI to note sequence function
# Note sequence to list of Notes function
final_mu = compare_sequences(targ, inter)


def compare_sequences(target, interpreted):  # Calculate average difference in semitones
    x, y= sym.symbols('x y')
    f, g = sym.symbols('target interpreted', cls=sym.Function)

    # Target/Interpreted are lists of Note objects
    trg = [(note.midi, note.start, note.end) for note in target]
    terp = [(note.midi note.start, note.end) for note in interpreted]

    trg_funcs = [(note[0], sym.And(note[1] <= x, x <= note[2])) for note in trg]
    terp_funcs = [(note[0], sym.And(note[1] <= x, x <= note[2])) for note in terp]

    f = sym.Piecewise(*trg_funcs)
    g = sym.Piecewise(*terp_funcs)


    def check_cont(f1, f2, y):
        # XOR for two given functions for a given x
        if bool(f1.subs(x, y) == sym.nan) != bool(f2.subs(x, y) == sym.nan):
            return y
        elif f1.subs(x, y) == sym.nan and f2.subs(x, y) == sym.nan:
            return y
        else:
            return None

        
    def find_disconts(f1, f2, end):
        values = []
        for x_val in np.arange(0, end + .01, .01):
            if check_cont(f1, f2, x_val):
                values.append(round(x_val, 3))
        return values


    def convert_to_intervals(points):
        end = 0
        ret = []
        while len(points):
            if round(points[end] - points[0], 2) == end/100 and \
               end != len(points) - 1:
                end += 1
            elif end == len(points) - 1:
                ret.append(points[0:end + 1])
                del points[0:end + 1]
            else:
                ret.append(points[0:end])
                del points[0:end]
                end = 0

        ret = [(round((ls[0] - .01), 2), round((ls[-1] + .01), 2)) for ls in ret]

        return ret
        
    vals = find_disconts(f, g, trg[-1][2])  # List of all points of discontinuity
    int_vals = convert_to_intervals(vals)  # List form intervals of discontinuity

    # Interval form discontinuities
    val_form = sym.Union(*[sym.Interval(spc[0], spc[1]) for spc in int_vals])
    full_domain = sym.Interval(0, trg[-1][2])
    final_domain = full_domain - val_form

    if int_vals:
        discont_domain = sum([spc[1] - spc[0] for spc in int_vals])
        total_length = trg[-1][2]
        cont_domain = total_length - discont_domain
        
        total = 0
        for sub in final_domain.args:
            total += sym.integrate(np.abs(f - g), (x, sub.left, sub.right))

        mu = total/cont_domain
    else:
        mu = sym.integrate(np.abs(f - g), (x, 0, trg[-1][1])) / trg[-1][1]

    print(f"μ = {mu}")
    return mu
