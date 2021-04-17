import melody_extension
import sys

CHUNK_DURATION = round(float(sys.argv[1]), 3)  # Defines chunk duration in sec
MIN_VOLUME = round(float(sys.argv[2]))  # Minimum max volume to qualify as note
MIN_REST = float(sys.argv[3])  # Minimum length of a rest in seconds
MEL_MIN = round(float(sys.argv[4]))
REST_MAX = round(float(sys.argv[5]))

melody_extension.listen_and_extend(CHUNK_DURATION, MIN_VOLUME, MIN_REST,
                                   MEL_MIN, REST_MAX)
