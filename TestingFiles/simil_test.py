import sympy as sym
import numpy as np

from sympy import solveset, S

x, y, z = sym.symbols('x y z')
f, g = sym.symbols('target interpreted', cls=sym.Function)

trg = ((65, 0, 1.2), (63, 1.2, 3.1), (60, 3.1, 3.9))
terp = ((64, 0, 1.7), (65, 1.7, 2.3), (63, 2.3, 2.8), (60, 2.8, 3.9))

trg_funcs = [(note[0], sym.And(note[1] <= x, x <= note[2])) for note in trg]
terp_funcs = [(note[0], sym.And(note[1] <= x, x <= note[2])) for note in terp]

f = sym.Piecewise(*trg_funcs)
g = sym.Piecewise(*terp_funcs)


def check_cont(f1, f2, y):  # XOR for two given functions for a given x
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

    ret = [(ls[0], ls[-1]) for ls in ret]

    return ret
    
vals = find_disconts(f, g, trg[-1][2])
int_vals = convert_to_intervals(vals)

total = 0
for spc in int_vals:
    total += sym.integrate(np.abs(f - g), (x, spc[0], spc[1]))

domain_total = sum([spc[1] - spc[0] for spc in int_vals])
avg = total / domain_total

print(avg)
res = sym.integrate(np.abs(f - g), (x, 0, 3.9)) / 3.9
print(res)
