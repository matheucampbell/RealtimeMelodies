import melody_extension

print("Starting realtime melody generation. First, calibrate the program.")

active = True
final_rest, min_volume = melody_extension.calibrate()

def extend(min_volume, final_rest):
    melody_extension.listen_and_extend(.15, min_volume, .1, final_rest)

    
def run_calibrate(): return melody_extension.calibrate()

    
while active:
    com = input("Choose an option.\n1. Calibrate\n2. Listen and extend." +
                "\n3. Exit")
    
    if com == '1':
        final_rest, min_volume = run_calibrate()
    elif com == '2':
        extend(min_volume, final_rest)
    elif com == '3':
        active = False
        print("Exiting")
    else:
        print("Choose 1, 2, or 3.")
