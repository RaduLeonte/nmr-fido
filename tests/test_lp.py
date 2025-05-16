import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import time
from scipy import signal


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



data = nf.ZF(data)
data = nf.FT(data)
data = nf.EXT(data, start="70ppm", end="40ppm")
data = nf.DI(data)

data = nf.TP(data)

print(data)

num_fids = 5
fid_indices = np.linspace(0, data.shape[-1], num_fids, dtype=int)
fig, axs = plt.subplots(num_fids,1, figsize=(14,10), layout="constrained")

fid = data[0]
truncate = 30
for ax, fid_index in zip(axs, fid_indices):
    fid = data[fid_index]
    truncated_fid = fid[:30]
    print(fid)
    
    ax.plot(fid, label="Raw")

    lp_fid = nf.LP(truncated_fid)
    ax.plot(lp_fid, label="LP")

    ax.plot(truncated_fid, label="Truncated")
    
    ax.legend()
    ax.set_xlim(0, 50)


""" fig, ax = plt.subplots(1,1, figsize=(14,10), layout="constrained")
ax.set_title("NMR Fido") """

""" data = nf.ZF(data)
data = nf.FT(data)
data = nf.EXT(data, start="70ppm", end="40ppm")
data = nf.DI(data)

data = nf.TP(data)

data = nf.ZF(data)
data = nf.DI(data)

data = nf.TP(data)

vmax = np.max(np.abs(data))
vmin = -vmax
colors = [
    (0.0, "blue"),
    (0.4, "blue"),
    (0.5, "black"),
    (0.6, "orange"),
    (1.0, "orange")
]
cmap = LinearSegmentedColormap.from_list("blue_black_orange", colors)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
ax.imshow(
    data,
    extent=data.extent(),
    cmap=cmap,
    norm=norm,
    aspect="auto",
    interpolation="none",
    origin='lower'
) """

#ax.set_xlim(650, 850)
plt.show()