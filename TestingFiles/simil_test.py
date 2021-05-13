import sympy as sym


sym.init_printing(use_unicode=True)

x, y, z = sym.symbols('x y z')
vals = []

trg = ((65, 0, 2.3), (63, 2.3, 3.1), (60, 3.1, 3.9))
terp = ((64, 0, 1.7), (65, 1.7, 2.3), (63, 2.3, 2.8), (60, 3.1, 3.9))

funcs = ((note[0], sym.And(note[1]<=x, x<note[2])) for note in trg)

while (new_val, new_cond) = next(funcs)
    trg_pce = sym.Piecewise(next(funcs), (y, z))
    trg_pce = trg_pce.subs([(y, new_val), (z, new_cond)])

print(trg_pce)
                       
