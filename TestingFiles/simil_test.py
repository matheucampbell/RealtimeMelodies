import sympy as sym

x, y, z = sym.symbols('x y z')

trg = ((65, 0, 2.3), (63, 2.3, 3.1), (60, 3.1, 3.9))
terp = ((64, 0, 1.7), (65, 1.7, 2.3), (63, 2.3, 2.8), (60, 3.1, 3.9))

funcs = ((note[0], sym.And(note[1]<=x, x<note[2])) for note in trg)

(val, cond) = next(funcs)
trg_pce = sym.Piecewise((val, cond), (y, z))

while (interval := next(funcs, None)):
    trg_pce = trg_pce.subs([(y, interval[0]), (z, interval[1])])
