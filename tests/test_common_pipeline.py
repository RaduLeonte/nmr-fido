import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import matplotlib.pyplot as plt
import time


dic, data = ng.pipe.read("tests/test2d.fid")

data = nf.NMRData(
    data,
    axes= [
        {
            "label": "15N",
            "SW": 5555.55615234375,
            "ORI": 3333.448974609375,
            "OBS": 50.64799880981445,
            "interleaved_data": True,
        },
        {
            "label": "13C",
            'SW': 50000.0,
            'ORI': -18053.66015625,
            'OBS': 125.69100189208984,
        },
    ]
)


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
    

fig, ax = plt.subplots(1,1, figsize=(14,10), layout="constrained")

limits_x = (data.axes[-1]["scale"][0], data.axes[-1]["scale"][-1])
limits_y = (data.axes[-2]["scale"][0], data.axes[-2]["scale"][-1])
ax.contour(
    data,
    levels = 30_000 * 1.1 ** np.arange(16),
    extent = (*limits_x, *limits_y),
    colors = "crimson",
)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
ax.set_title("NMR Fido")
ax.set_xlabel(data.axes[-1]["label"])
ax.set_ylabel(data.axes[-2]["label"])
#ax.set_xlim(70, 40)
#ax.set_ylim(135, 100)
plt.show()