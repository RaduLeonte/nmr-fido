import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import matplotlib.pyplot as plt
import time


data = nf.read_nmrpipe("tests/test2d.fid")


start_time = time.perf_counter()

data = nf.SP(data, off=0.35, end=0.98, pow=1, c=1.0) # 2.344 ms
data = nf.ZF(data, size=4096) # 1.516 ms
data = nf.FT(data) # 30.795 ms
data = nf.PS(data, p0=-29.0, p1=0.0) # 10.278 ms
data = nf.DI(data) # 3.497 m
data = nf.EXT(data, x1="70ppm", xn="40ppm") # 3.532 ms

data = nf.TP(data) # 1.278 ms

data = nf.SP(data, off=0.35, end=0.9, pow=1, c=0.5) # 1.008 ms
data = nf.ZF(data, size=2048) # 0.596 ms
data = nf.FT(data) # 20.867 ms
data = nf.PS(data, p0=0.0, p1=0.0) # 5.942 ms
data = nf.DI(data) # 1.673 ms
data = nf.EXT(data, x1="135ppm", xn="100ppm") # 2.200 ms

data = nf.TP(data) # 0.261 ms

for entry in data.processing_history: print(f'{entry["Function"]} -> {entry["time_elapsed_s"]*1000:.3f} ms')
print(f"\n--- Done! Elapsed: {time.perf_counter() - start_time:.3f} s", "\n")

print(data.summary())
nf.phasing_gui(data)


fig, ax = plt.subplots(1,1, figsize=(14,10), layout="constrained")

ax.contour(
    data,
    levels = 20_000 * 1.2 ** np.arange(16),
    extent = data.extent(),
    colors = "crimson",
)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
ax.set_title("NMR Fido")
ax.set_xlabel(f'{data.axes[-1]["label"]} [{data.axes[-1]["unit"]}]')
ax.set_ylabel(f'{data.axes[-2]["label"]}\n[{data.axes[-1]["unit"]}]', rotation=0, ha="right")
#ax.set_xlim(70, 40)
#ax.set_ylim(135, 100)
plt.show()