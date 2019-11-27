import time
import dask.array as da

def inc(x):
    return x + 1

def double(x):
    return x + 2

def add(x, y):
    return x + y

def mul(x, y):
    return x*y

def exp(x, y):
	return x**y

def main():
	data = da.random.randint(10, size=(100, 100))

	output = []
	for x in data: 
		a = inc(x)
		b = double(x)
		c = inc(x)
		d = double(x)
		e = inc(x)
		f = double(x)
		g = mul(a, b)
		h = add(c, d)
		i = mul(e, f)
		j = add(a, d)
		k = add(b, e)
		l = mul(c, f)
		m = add(g, h)
		n = mul(i, j)
		o = add(k, l)
		p = mul(m, n)
		q = add(o, p)
		output.append(q)

	total = sum(output)
	#total.visualize()
	total = total.compute()

temps = []
for i in range(1):
	start = time.time()
	main()
	end = time.time()
	temps.append(end-start)

print('time: {}'.format(round(sum(temps)/len(temps), 4)))

