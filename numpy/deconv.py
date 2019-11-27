#! /usr/bin/env ipython2
from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import scipy.signal as sig
import functools

sky   = np.load("sky.npy")
dirty = np.load("dirty.npy")
psf   = np.load("psf.npy")


def l2_reg(dirty, psf, iterations, mu=1):
    x = np.zeros(dirty.shape)
    psf_t = np.fliplr(np.flipud(psf))
    psf_t_psf = sig.fftconvolve(psf_t, psf)

    for i in range(iterations):
        try:
            print("Iteration #{}".format(i + 1))
            residual = sig.fftconvolve(x, psf, mode="same") - dirty
            print("\tresidual = {}".format(lin.norm(residual)))
            gradient = sig.fftconvolve(residual, psf_t, mode="same") + mu * x

            alpha = lin.norm(gradient) / (
                lin.norm(sig.fftconvolve(gradient, psf_t_psf, mode="same")) + mu * lin.norm(gradient))
            print("\talpha = {}".format(alpha))
            print("\tcrit = {}".format(lin.norm(residual) + mu * lin.norm(x)))

        except KeyboardInterrupt:
            return x
        x -= alpha * gradient
    return x


def prox_l1(x, alpha):
    x[(x <= alpha) * (x >= alpha)] = 0
    x[x > alpha] -= alpha
    x[x < alpha] += alpha
    return x


def l1_reg_proximal(dirty, psf, iterations, gamma=1):
    x = np.zeros(dirty.shape)
    x_prev = np.zeros(dirty.shape)
    z = np.zeros(dirty.shape)

    psf_t = psf.transpose()
    psf_t_psf = sig.fftconvolve(psf_t, psf, mode="same")
    alpha = 0.1
    beta = .5
    residual = lambda x: sig.fftconvolve(x, psf, mode="same") - dirty

    for i in range(iterations):
        print("Iteration #{}".format(i + 1))
        residual_x = residual(x)
        residual_x_L2 = lin.norm(residual_x)
        criterion = residual_x_L2 + lin.norm(x, 1)
        print("\tresidual = {}, criterion = {}".format(residual_x_L2, criterion))

        gradient = sig.fftconvolve(residual_x, psf_t, mode="same")
        while True:
            z = prox_l1(x - alpha * gradient, alpha * gamma)
            zx = (z - x)
            if lin.norm(residual(z)) <= (residual_x_L2
                                         + 2 * np.sum(residual_x * sig.fftconvolve(zx, psf_t, mode="same"))
                                         + (1/alpha) * lin.norm(zx)):
                print(alpha)
                break
            alpha *= beta
        x_prev = x
        x = z
    return x

#x_reg = l2_reg(dirty, psf, 50, 0.000001)

import optim


psf_t = np.fliplr(np.flipud(psf))
psf_t_psf = sig.fftconvolve(psf_t, psf)
mu = 0.001
reg = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 8
reg_t_reg = sig.fftconvolve(reg, reg)

def hessian_indep(x, mu):
    return sig.fftconvolve(x, psf_t_psf, mode="same") + mu * x

def hessian_lapl(x, mu):
    return sig.fftconvolve(x, psf_t_psf, mode="same") + mu * sig.fftconvolve(x, reg_t_reg, mode="same")

def crit(x):
    return np.sum((sig.fftconvolve(x, psf, mode="same") - dirty)**2) + mu * np.sum(sig.fftconvolve(x, reg, mode="same")**2)

ht_y = sig.fftconvolve(dirty, psf_t, mode="same")
#%% MCR
tests = np.logspace(-2, 1, 100)
results = {'reg': [], 'info': [], 'e': []}
for i in tests:
    print(i)
    x_reg, info = optim.conj_grad(functools.partial(hessian_indep, mu=i), dirty, ht_y, user_settings={'cg max iter': 100, 'cg min iter': 100, 'f crit': crit})
    results['e'].append(lin.norm(x_reg-sky))

#x_reg, info = optim.conj_grad(hessian_lapl, dirty, ht_y, user_settings={'cg max iter': 10, 'cg min iter': 10, 'f crit': crit})

#%%
plt.figure()
plt.plot(tests, results['e'])
best = tests[np.argmin(results['e'])]
best_x_indep, _ = optim.conj_grad(functools.partial(hessian_indep, mu=best), dirty, ht_y, user_settings={'cg max iter': 50, 'cg min iter': 50, 'f crit': crit})

#%% plot best
plt.close('all')
plt.figure()
plt.imshow(np.fft.fftshift(np.log(np.abs(np.fft.fft2(best_x_indep)))))
plt.jet()

#%% plot
plt.close("all")
plt.figure(1)
plt.imshow(x_reg)
plt.colorbar()
plt.figure(2)
plt.imshow(dirty)
plt.colorbar()
plt.show()

plt.figure(3)
plt.subplot(2, 2, 1)
plt.plot(dirty[800], label='dirty')
plt.plot(x_reg[800], label='reg')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(dirty[800], label='dirty')
plt.plot(sky[800], label='true')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(sky[800], label='true')
plt.plot(x_reg[800], label='reg')
plt.legend()
plt.show()
