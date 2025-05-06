import nmrglue as ng
from pprint import pprint

dic, data = ng.pipe.read("tests/test.fid")


keys = ["LABEL", "SW", "CAR", "OBS", "ORIG"]
filt_dic = dic.copy()
filt_dic = {k:v for k,v in filt_dic.items() if any(key in k for key in keys)}
#pprint(filt_dic)
print(data.shape)


# process the direct dimension
dic, data = ng.pipe_proc.sp(dic, data, off=0.35, end=0.98, pow=1, c=1.0)
dic, data = ng.pipe_proc.zf(dic, data, auto=True)
dic, data = ng.pipe_proc.ft(dic, data, auto=True)

print(ng.pipe.make_uc(dic, data, dim=0).ppm_limits())

dic, data = ng.pipe_proc.ps(dic, data, p0=-29.0, p1=0.0)
dic, data = ng.pipe_proc.di(dic, data)

# process the indirect dimension
dic, data = ng.pipe_proc.tp(dic, data)
dic, data = ng.pipe_proc.sp(dic, data, off=0.35, end=0.9, pow=1, c=0.5)
dic, data = ng.pipe_proc.zf(dic, data, size=2048)
dic, data = ng.pipe_proc.ft(dic, data, auto=True)
dic, data = ng.pipe_proc.ps(dic, data, p0=0.0, p1=0.0)
dic, data = ng.pipe_proc.di(dic, data)
dic, data = ng.pipe_proc.tp(dic, data)

# write out processed data
ng.pipe.write("2d_pipe.ft2", dic, data, overwrite=True)