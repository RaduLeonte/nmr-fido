import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint


dic, data = ng.pipe.read("tests/test.fid")

data = nf.NMRData(
    data,
    labels=["15N", "13C"],
)
print("Processing history:", data.processing_history)

data = nf.FT(data)

print(data)

pprint(data.processing_history[-1])
