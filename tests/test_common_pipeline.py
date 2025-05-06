import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import sys


dic, data = ng.pipe.read("tests/test.fid")

data = nf.NMRData(
    data,
    labels=["15N", "13C"],
)

print("\n")

data = nf.SP(data, off=0.35, end=0.98, pow=1, c=1.0)
data = nf.ZF(data)
data = nf.FT(data)
data = nf.PS(data, p0=29.0, p1=0.0)
#data = data.DI()

print(data, "\n")
data = nf.TP(data)
print(data)
sys.exit()

data = data.SP(off=0.35, end=0.9, pow=1, c=0.5)
data = data.ZF(size=2048)
data = data.FT()
data = data.PS()
#data = data.DI()

data = data.TP()

pprint(data.processing_history)
