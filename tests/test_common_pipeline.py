import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint


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

data = nf.SP(data, off=0.35, end=0.98, pow=1, c=1.0)
data = nf.ZF(data)
data = nf.FT(data)
data = nf.PS(data, p0=29.0, p1=0.0)
#data = data.DI()

data = nf.TP(data)

data = nf.SP(data, off=0.35, end=0.9, pow=1, c=0.5)
data = nf.ZF(data, size=2048)
data = nf.FT(data)
data = nf.PS(data)
#data = data.DI()

data = nf.TP(data)
print(data)

#pprint(data.processing_history)

for entry in data.processing_history:
    print(f'{entry["Function"]} -> {entry["time_elapsed_str"]}')
