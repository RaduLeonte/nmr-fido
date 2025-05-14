import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import matplotlib.pyplot as plt
import time


dic, data = ng.pipe.read("tests/test1d.fid")

pprint(dic)

data = nf.NMRData(
    data,
    axes= [
        {
            "label": "13C",
            'SW': 50000.0,
            'ORI': -12524.2470703125,
            'OBS': 125.68099975585938,
        },
    ]
)


fig, ax = plt.subplots(1,1, figsize=(14,10), layout="constrained")
ax.set_title("NMR Fido")
ax.set_ylabel("Signal")


""" ax.plot(data[0], label="Boxcar filter")

lowpass_size: int = 16
filter_width = lowpass_size*2 + 1
filter = np.ones(filter_width, dtype=np.float32)
ax.plot(filter, label="Boxcar filter") """


data = nf.FT(data)
data = nf.PS(data, p0=-17.7, p1=-36.0)
data = nf.DI(data)

x_scale = data.axes[-1]["scale"]
x_scale = np.arange(len(data))
ax.plot(x_scale, data, label="PS")

data = nf.POLY(data, sub_end=-1, order=4)
x_subtract = np.arange(0, 1500)
coeffs = [-3.198594474413598e-13, 1.7376258141020913e-09, -3.563573333216777e-06, 0.0033490547027675245, -1.586491747089806, 455.90436351363957, -56041.89104446261]
baseline = np.polyval(coeffs, x_subtract)

threshold = 293318.2234069654
ax.axhline(y=threshold, label="Threshold")
ax.plot(x_scale, baseline, label="Baseline")
ax.plot(x_scale, data, label="POLY")

#plt.gca().invert_xaxis()
#ax.set_xlim(70, 40)


""" no_SOL = nf.FT(data)
boxcar = nf.FT(nf.SOL(data))
sine = nf.FT(nf.SOL(data, lowpass_shape="Sine"))
sine2 = nf.FT(nf.SOL(data, lowpass_shape="Sine^2"))
butter = nf.FT(nf.SOL(data, lowpass_shape="Butterworth"))

no_SOL.scale_to_hz()
x_scale = no_SOL.axes[-1]["scale"]

ax.plot(x_scale, no_SOL[0], label="No SOL")
ax.plot(x_scale, boxcar[0], label="SOL -> Boxcar")
ax.plot(x_scale, sine[0], label="SOL -> Sine")
ax.plot(x_scale, sine2[0], label="SOL -> Sine^2")
ax.plot(x_scale, butter[0], label="SOL -> Butterworth")
ax.set_xlabel(no_SOL.axes[-1]["unit"]) """

ax.legend()


#ax.set_xlim(650, 850)
plt.show()