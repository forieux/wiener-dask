#! /usr/bin/env ipython2
# -*- coding: utf-8 -*- 

import sys

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import functools

from edwin import optim
from edwin import udft
from edwin.udft import uirdft2 as idft
from edwin.udft import urdft2 as dft

#from edwin.criterions import Huber

from edwin import improcessing

import time
#import cProfile
'''
#%% Load
sky = np.load("sky.npy").astype(np.double)
dirty = np.load("dirty.npy").astype(np.double)
psf = np.load("psf.npy").astype(np.double)

#reg_lapl = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 8
#reg_laplf = udft.ir2fr(reg_lapl, sky.shape)

#reg_indep = 1
#reg_indepf = np.ones_like(reg_laplf)

fr = udft.ir2fr(psf, sky.shape)

#%% Deconv
# im, chains = improcessing.udeconv(dirty, freq_resp=fr,
#                                   regf=reg_indepf,
#                                   user_params={'min_iter': 300,
#                                                'max_iter': 500,
#                                                'σ': 1e-6,
#                                                'burnin': 200})

# plt.figure(1)
# plt.clf()
# plt.subplot(2, 2, 1)
# plt.imshow(im)
# plt.title('reg : {}'.format(np.mean(chains['prior'][200:]) /
#                             np.mean(chains['noise'][200:])))
# plt.colorbar()
# plt.subplot(2, 2, 2)
# plt.plot(im[800], label='im')
# plt.plot(sky[800], label='true')
# plt.plot(dirty[800], label='dirty')
# plt.legend()
# plt.subplot(2, 2, 3)
# plt.plot(chains['noise'])
# plt.subplot(2, 2, 4)
# plt.plot(chains['prior'])

#%% CvxDiff deconv
#n_iter = 50
#huber = Huber(0.01)
L = 0.0005

'''
def wiener(data, aux, fr, L):
	return idft((np.conj(fr) * dft(data) + L * dft(aux)) / (np.abs(fr)**2 + L))

'''
def deconv_huber(data, fr, L):
	aux = np.zeros_like(data)
	for it in range(n_iter):
		im = wiener(dirty, aux, fr, L)
		aux = huber.min_gy(im)

	return im, aux


def prox_l1(obj, alpha):
	out = obj.copy()
	out[(obj >= -alpha) * (obj <= alpha)] = 0
	out[obj > alpha] -= alpha
	out[obj < -alpha] += alpha
	return out


def deconv_l1(data, fr, init, L=0.9, step=1, n_iter=200):
	obj = init.copy()
	obj_prev = np.zeros_like(dirty)
	z = np.zeros_like(dirty)

	dataf = dft(data)
	dataft = fr.conj() * dataf
	fr2 = np.abs(fr)**2
	beta = 0.5

	def crit(obj):
		return np.sum(np.abs((
			dataf - fr * dft(obj)))**2) + L * np.sum(np.abs(obj))
	crit_val = []

	fig = plt.figure(1)
	for it in range(n_iter):
		previous = obj.copy()
		grad = idft(fr2 * dft(obj) - dataft)
		while True:
			z = prox_l1(obj - step * grad, step * L)
			if crit(z) <= (crit(obj) + np.sum(grad * (z - obj)) +
				(1 / (2 * step)) * np.sum(np.abs(z - obj)**2)):
				break
			step = beta * step
		obj = z + (it + 1) / (it + 5) * (obj - previous)
		crit_val.append(crit(obj))

		plt.clf()
		plt.subplot(1, 2, 1)
		plt.plot(crit_val)
		plt.subplot(1, 2, 2)
		plt.imshow(obj)
		plt.colorbar()
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.show()

	return obj


def warm_restart_deconv_l1(data, fr, lrange, step=1, n_iter=200):
	obj = data
	for L in lrange:
		obj = deconv_l1(data, fr, obj, L, step=1, n_iter=n_iter // len(lrange))
	return obj

'''

def load_data(data):
	data_npy = np.load(data).astype(np.double)
	#data_npy = np.concatenate([data_npy, data_npy])
	#data_npy = np.concatenate([data_npy, data_npy], axis=1)
	#data_npy = np.concatenate([data_npy, data_npy])
	#data_npy = np.concatenate([data_npy, data_npy], axis=1)	


	return data_npy

def main():

	sky = load_data("sky.npy")
	dirty = load_data("dirty.npy")
	psf = load_data("psf.npy")

	L = 0.0005

	fr = udft.ir2fr(psf, sky.shape)
	
	#%% Run
	b = idft(fr.conj() * dft(dirty))
	quad = wiener(dirty, np.zeros_like(sky), fr, L)
	#hub, aux_hub = deconv_huber(dirty, fr, L)


	# l1 = warm_restart_deconv_l1(dirty, fr,
	#                             np.logspace(np.log10(np.max(b)),
	#                                         np.log10(1e-5 * np.max(b)),
	#                                         10),
	#                             n_iter=200)


	# plt.figure(2)
	# plt.imshow(np.log(l1))

	#%% Hub
	'''
	plt.close('all')

	plt.figure(1)
	plt.clf()
	plt.plot(dirty[800], label='dirty')
	plt.plot(quad[800], label='quad')
	#plt.plot(hub[800], label='hub')
	plt.plot(sky[800], label='true')
	plt.legend()

	plt.figure(2)
	plt.clf()
	plt.imshow(quad, vmin = np.min(quad), vmax = np.max(quad))
	plt.title('quad')
	'''

for i in range (5):
	start_time = time.time()
	main()
	end_time = time.time()
	print(end_time - start_time)
	time.sleep(1)
	#cProfile.run('re.compile("foo|bar")')
	#cProfile.run('main()')

#plt.show()

#plt.savefig('sky.png')

