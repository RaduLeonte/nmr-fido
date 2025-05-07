import nmrglue as ng
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import nmr_fido as nf

ng_dic, ng_data = ng.pipe.read("tests/test.fid")


nf_data = nf.NMRData(
    ng_data,
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

print(ng_data[0][0:3])
print(np.array(nf_data[0][0:3]))


fig, ax = plt.subplots(1,1, figsize=(16,9), layout="constrained")

ng_dic, ng_data = ng.pipe_proc.ft(ng_dic, ng_data, auto=True, debug=True)
nf_data = nf.FT(nf_data)

print(ng_data[0][0:3])
print(np.array(nf_data[0][0:3]))


ax.plot(ng_data[0], color="crimson",)
ax.plot(nf_data[0], color="dodgerblue",)

ax.set_xlabel("15N")
ax.set_ylabel("13C")
plt.show()