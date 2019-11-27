import dask.array as da
import time

def inc(x):
    return x + 1

def double(x):
    return x + 2

def add(x, y):
    return x + y

def main():
	data = da.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],chunks=(10,))
	print(data.compute)

	output = []
	for x in data:
	    a = inc(x)
	    b = double(x)
	    c = add(a, b)
	    output.append(c)

	total = sum(output)

for i in range(1):
	start = time.time()
	main()
	end = time.time()
	print(end-start)
