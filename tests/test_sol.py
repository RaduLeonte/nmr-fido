import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import matplotlib.pyplot as plt
import time


dic, data = ng.pipe.read("tests/test.fid")

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


no_SOL = nf.FT(data)

boxcar = nf.FT(nf.SOL(data))
sine = nf.FT(nf.SOL(data, lowpass_shape="Sine"))
sine2 = nf.FT(nf.SOL(data, lowpass_shape="Sine^2"))
butter = nf.FT(nf.SOL(data, lowpass_shape="Butterworth"))


fig, ax = plt.subplots(1,1, figsize=(14,10), layout="constrained")

ax.plot(no_SOL[0], label="No SOL")
ax.plot(boxcar[0], label="SOL -> Boxcar")
ax.plot(sine[0], label="SOL -> Sine")
ax.plot(sine2[0], label="SOL -> Sine^2")
ax.plot(butter[0], label="SOL -> Butterworth")

ax.legend()

ax.set_title("NMR Fido")
ax.set_xlabel("pts")
ax.set_ylabel("Signal")

#ax.set_xlim(650, 850)
plt.show()