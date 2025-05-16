import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import matplotlib.pyplot as plt
import time
from scipy import signal


dic, data = ng.pipe.read("tests/test2d.fid")

pprint(dic)

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


fig, ax = plt.subplots(1,1, figsize=(14,10), layout="constrained")
ax.set_title("NMR Fido")
ax.set_ylabel("Signal")

""" data = data

ax.plot(nf.FT(data)[0], label="Raw")

lowpass_size: int = 16
filter_width = lowpass_size*2 + 1
filter = np.ones(filter_width, dtype=np.float32)
filtered_fid = signal.convolve(data[0], filter, mode="same") / filter_width
ax.plot(nf.FT(filtered_fid), label="Filtered")

subtracted = np.array(data[0] - filtered_fid)
ax.plot(nf.FT(subtracted), label="Subtracted") """



""" data = nf.FT(data)
data = nf.PS(data, p0=-17.7, p1=-36.0)
data = nf.DI(data)

x_scale = data.axes[-1]["scale"]
x_scale = np.arange(len(data))
data = nf.POLY(data, sub_end=-1, order=4)
ax.plot(x_scale, data, label="POLY") """

#plt.gca().invert_xaxis()
#ax.set_xlim(70, 40)


no_SOL = nf.FT(data)
boxcar = nf.FT(nf.SOL(data))
sine = nf.FT(nf.SOL(data, lowpass_shape="Sine"))
sine2 = nf.FT(nf.SOL(data, lowpass_shape="Sine^2"))
gauss = nf.FT(nf.SOL(data, lowpass_shape="Gaussian"))
butter = nf.FT(nf.SOL(data, lowpass_shape="Butterworth"))

no_SOL.scale_to_hz()
x_scale = no_SOL.axes[-1]["scale"]

ax.plot(x_scale, no_SOL[0], label="No SOL")
ax.plot(x_scale, boxcar[0], label="SOL -> Boxcar")
ax.plot(x_scale, sine[0], label="SOL -> Sine")
#ax.plot(x_scale, sine2[0], label="SOL -> Sine^2")
ax.plot(x_scale, gauss[0], label="SOL -> Gaussian")
#ax.plot(x_scale, butter[0], label="SOL -> Butterworth")
ax.set_xlabel(no_SOL.axes[-1]["unit"])

ax.legend()


#ax.set_xlim(650, 850)
plt.show()