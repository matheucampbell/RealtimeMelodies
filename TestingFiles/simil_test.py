import sympy as sym
import numpy as np

from sympy.calculus.singularities import singularities

x, y, z = sym.symbols('x y z')
f, g = sym.symbols('target interpreted', cls=sym.Function)

trg = ((65, 0, 2.3), (63, 2.3, 3.1), (60, 3.1, 3.9))
terp = ((64, 0, 1.7), (65, 1.7, 2.3), (63, 2.3, 2.8), (60, 3.2, 3.9))

trg_funcs = [(note[0], sym.And(note[1] <= x, x <= note[2])) for note in trg]
terp_funcs = [(note[0], sym.And(note[1] <= x, x <= note[2])) for note in terp]

f = sym.Piecewise(*trg_funcs)
g = sym.Piecewise(*terp_funcs)

res = sym.integrate(np.abs(f - g), (x, 0, 3.9)) / 3.9
print(res)

def check_cont(f1, f2, y):  # XOR for two given functions for a given x
    if bool(f1.subs(x, y) == sym.nan) != bool(f2.subs(x, y) == sym.nan):
        return True, y
    else:
        return False, -1

def find_disconts(f1, f2, end):
    values = []
    for x_val in range(0, end, .001):
        if check_cont(f1, f2, x_val):
            values.append(x_val)
    return values

print(find_disconts(f, g, trg[-1][2]))

