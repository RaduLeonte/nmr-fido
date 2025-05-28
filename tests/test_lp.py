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

num_fids = 6
#fid_indices = np.linspace(0, data.shape[-1], num_fids, dtype=int)
fid_indices = np.random.choice(data.shape[0], size=num_fids, replace=False)
fig, axs = plt.subplots(3, 2, figsize=(14,10), layout="constrained")
flattened_axs = axs.ravel()

fid = data[0]
print(fid.size)
truncate = 166
pred_size = truncate
for ax, fid_index in zip(flattened_axs, fid_indices):
    fid = data[fid_index]
    truncated_fid = fid[:truncate]
    
    #ax.plot(fid, label="Raw")
    ax.plot(truncated_fid, label="Model fid", zorder=99)

    lp_fid = nf.LP(truncated_fid, prediction_size=pred_size)
    ax.plot(lp_fid, label="LP after", zorder=1)

    
    lp_fid_no_root_fix = nf.LP(truncated_fid, prediction_size=pred_size, nofix=True)
    ax.plot(lp_fid_no_root_fix, label="LP no root fixing")


    
    ax.legend()
    #ax.set_xlim(0, lp_fid.size)
    ax.set_title(f"{fid_index=}")


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