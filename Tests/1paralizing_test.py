import time
#import dask
import numpy as np
#import dask.array as da

#@dask.delayed
def inc(x):
    return x + 1

#@dask.delayed
def double(x):
    return x + 2

#@dask.delayed
def add(x, y):
    return x + y

def mul(x, y):
    return x*y

def exp(x, y):
	return x**y

def main():
	data = np.random.randint(10, size=(100, 100))

	output = []
	for y in data:
		for x in y: 
			a = inc(x)
			b = double(x)
			c = mul(a, b)
			d = exp(a, b)
			e = add(c, d)
			output.append(e)

	total = sum(output)
	#total.visualize()
	#total = total.compute()

for i in range(3):
	start = time.time()
	main()
	end = time.time()
	print(end-start)

'''
import dask.array as da
import time
from dask import delayed
import numpy as np

def gradient(image):
    return da.where(
            da.fabs(image) <= huber['threshold'],
            2 * image,
            2 * huber['threshold'] * da.sign(image))

def min_gy(image):
    return image - gradient(image)

def main():

	global huber

	i = 0

	huber = {'threshold': 0.01, 'inf': 1}

	data_npy = [[1, 2, 3, 4],
				[4, 3, 2, 1],
				[5, 6, 7, 8],
				[8, 7, 6, 5]]
	data = da.from_array(data_npy)
	#data = [1,2,3,4,5,6,7,8]
#	print(data)
	#print((da.fabs(data)).compute())
	#print((da.where(da.fabs(data) <= '5', 2+data, 4*data)).compute())
	#output = []
	#for x in data:
		#for z in x:
	y = min_gy(data)
	y.visualize()
		#output.append(y)

#	print(output)    
#	print(total)
	#array_final = da.from_array(output) 
	#print(array_final)
#	array_final.visualize()
	#output.compute()
	#total.visualize()

for i in range(1):
	start = time.time()
	main()
	end = time.time()
	print(end-start)
'''
