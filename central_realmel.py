import melody_extension

FINAL_REST, MIN_VOLUME = melody_extension.calibrate()

print(FINAL_REST, FINAL_REST)

melody_extension.listen_and_extend(.15, MIN_VOLUME, .1, FINAL_REST)
