'''
import numpy as np

x = np.ones((6000,6000))
x = x + x.T
x = x*x
x = x.T * sum(x) * x
y = x + x.T


print(y)
'''


import dask.array as da
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler, visualize
from cachey import nbytes

x = da.ones((15,15),chunks=(5,5))
y = x.T
pbar = ProgressBar()
with pbar, Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler(metric=nbytes) as cprof:	
	y.compute()

visualize([prof, rprof, cprof])

