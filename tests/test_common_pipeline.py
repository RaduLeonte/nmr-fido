import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import matplotlib.pyplot as plt

dic, data = ng.pipe.read("tests/test.fid")

data = nf.NMRData(
    data,
    labels=[
        "15N",
        "13C"
    ],
    axis_info=[
        {
            'SW': 5555.55615234375,
            'ORI': 3333.448974609375,
            'OBS': 50.64799880981445
        },
        {
            'SW': 50000.0,
            'ORI': -18053.66015625,
            'OBS': 125.69100189208984
        },
    ]
)

#pprint(dic)

print("\n")
print(data.shape)

#data = nf.SP(data, off=0.35, end=0.98, pow=1, c=1.0)
#data = nf.ZF(data)
data = nf.FT(data)
#data = nf.PS(data, p0=29.0, p1=0.0)
#data = nf.DI(data)
#print(data, "\n")
#data = nf.EXT(data, x1="70ppm", xn="40ppm")
#print(data, "\n")

#data = nf.TP(data)

#data = nf.SP(data, off=0.35, end=0.9, pow=1, c=0.5)
#data = nf.ZF(data)
#data = nf.FT(data, neg=True)
#data = nf.PS(data)
#data = nf.DI(data)

#data = nf.TP(data)
print("--- done ---")
print(data)

data.scale_to_ppm(-2)

#pprint(data.processing_history)

for entry in data.processing_history:
    print(f'{entry["Function"]} -> {entry["time_elapsed_str"]}')
    

fig, ax = plt.subplots(1,1, figsize=(14,10), layout="constrained")


contour_start = 2_000           # contour level start value
contour_num = 16                # number of contour levels
contour_factor = 1.10          # scaling factor between contour levels
cl = contour_start * contour_factor ** np.arange(contour_num)
limits_x = (data.scales[-1][0], data.scales[-1][-1])
limits_y = (data.scales[-2][0], data.scales[-2][-1])
print(limits_x)
print(limits_y)
ax.contour(
    data,
    cl,
    extent=(*limits_x, *limits_y),
    colors="crimson",
)

plt.gca().invert_xaxis()
ax.set_xlabel(data.labels[-1])
ax.set_ylabel(data.labels[-2])
ax.set_xlim(70, 40)
#ax.set_ylim(135, 100)
plt.show()