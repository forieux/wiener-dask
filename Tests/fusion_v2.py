import time
from dask.distributed import Client
import dask.array as da
from dask.distributed import progress


def main(x):
    time.sleep(1)
    return x+1

client = Client()

temps = []

for i in range(1):
	start = time.time()
	futures = client.map(main, range(10))
        progress(futures)
	end = time.time()
	temps.append(end-start)

print('time: {}'.format(round(sum(temps)/len(temps), 4)) + '\n')

