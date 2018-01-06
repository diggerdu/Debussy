import os
import numpy as np

import matplotlib.pyplot as plt
import sys
import soundfile as sf

import webrtcvad
import struct


Fname = sys.argv[1]
y, sr = sf.read(Fname)
vad = webrtcvad.Vad()
vad.set_mode(3)

y = struct.pack("%dh" % len(y), *y)

window_duration = 0.03
samples_per_window = int(window_duration * sr + 0.5)
bytes_per_sample = 2







