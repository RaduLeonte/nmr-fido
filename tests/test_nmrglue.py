import nmrglue as ng
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import time


dic, data = ng.pipe.read("tests/test.fid")



keys = ["FDF1", "FDF2"]
filt_dic = dic.copy()
filt_dic = {k:v for k,v in filt_dic.items() if any(key in k for key in keys)}
#pprint(filt_dic)

start_time = time.perf_counter()
# process the direct dimension
dic, data = ng.pipe_proc.sp(dic, data, off=0.35, end=0.98, pow=1, c=1.0) # 3.18250001873821 ms
dic, data = ng.pipe_proc.zf(dic, data, auto=True) # 9.909600019454956 ms
dic, data = ng.pipe_proc.ft(dic, data, auto=True) # 29.63499998440966 ms
dic, data = ng.pipe_proc.ps(dic, data, p0=-29.0, p1=0.0) # 8.634899975731969 ms
dic, data = ng.pipe_proc.di(dic, data) # 0.05540001438930631 ms

dic, data = ng.pipe_proc.tp(dic, data) # 11.297399993054569 ms

dic, data = ng.pipe_proc.sp(dic, data, off=0.35, end=0.9, pow=1, c=0.5) # 4.255299980286509 ms
dic, data = ng.pipe_proc.zf(dic, data, size=2048) # 67.57319997996092 ms
dic, data = ng.pipe_proc.ft(dic, data, auto=True, debug=False) # 179.58789999829605 ms
dic, data = ng.pipe_proc.ps(dic, data, p0=0.0, p1=0.0) # 51.987099985126406 ms
dic, data = ng.pipe_proc.di(dic, data) # 0.022499996703118086 ms

start_time_func = time.perf_counter() 
dic, data = ng.pipe_proc.tp(dic, data) # 0.18179998733103275 ms
print(f" # {(time.perf_counter() - start_time_func)*1000} ms")
print(f"--- Done! Elapsed: {time.perf_counter() - start_time:.3f} s", "\n")

# write out processed data
#ng.pipe.write("2d_pipe.ft2", dic, data, overwrite=True)


fig, ax = plt.subplots(1,1, figsize=(14,10), layout="constrained")


uc_13c = ng.pipe.make_uc(dic, data, dim=1)
ppm_13c = uc_13c.ppm_scale()
ppm_13c_0, ppm_13c_1 = uc_13c.ppm_limits()
uc_15n = ng.pipe.make_uc(dic, data, dim=0)
ppm_15n = uc_15n.ppm_scale()
ppm_15n_0, ppm_15n_1 = uc_15n.ppm_limits()

contour_start = 30_000           # contour level start value
contour_num = 16                # number of contour levels
contour_factor = 1.10          # scaling factor between contour levels
cl = contour_start * contour_factor ** np.arange(contour_num)
limits_x = (ppm_13c_0, ppm_13c_1)
limits_y = (ppm_15n_0, ppm_15n_1)
ax.contour(
    data,
    cl,
    extent=(*limits_x, *limits_y),
    colors="crimson",
)

plt.gca().invert_xaxis()
ax.set_title("nmrglue")
ax.set_xlabel("13C")
ax.set_ylabel("15N")
ax.set_xlim(70, 40)
ax.set_ylim(135, 100)
plt.show()