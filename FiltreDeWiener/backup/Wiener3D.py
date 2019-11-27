import os
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
import time

try:
    from dask.array.fft import rfftn as dask_rfftn
    from dask.array.fft import irfftn as dask_irfftn
    from pyfftw.interfaces.numpy_fft import fftn
    from pyfftw.interfaces.numpy_fft import rfftn
    from pyfftw.interfaces.numpy_fft import irfftn
except ImportError:
    print("Installation of the dask.array and/or pyfftw package improve preformance"
           " by using fft/fftw library.")
    from dask.array.fft import rfftn as dask_rfftn
    from dask.array.fft import irfftn as dask_irfftn
    from numpy.fft import fftn as fftn
    from numpy.fft import rfftn as rfftn
    from numpy.fft import irfftn as irfftn

def ir2fr(imp_resp, shape, center=None, real=True):
    """Return the frequency response from impulsionnal responses

    This function make the necessary correct zero-padding, zero
    convention, correct DFT etc. to compute the frequency response
    from impulsionnal responses (IR).

    The IR array is supposed to have the origin in the middle of the
    array.

    The Fourier transform is performed on the last `len(shape)`
    dimensions.

    Parameters
    ----------
    imp_resp : ndarray
    The impulsionnal responses.

    shape : tuple of int
    A tuple of integer corresponding to the target shape of the
    frequency responses, without hermitian property.

    center : tuple of int, optional
    The origin index of the impulsionnal response. The middle by
    default.

    real : boolean (optionnal, default True)
    If True, imp_resp is supposed real, the hermissian property is
    used with rfftn DFT and the output has `shape[-1] / 2 + 1`
    elements on the last axis.

    Returns
    -------
    y : ndarray
    The frequency responses of shape `shape` on the last
    `len(shape)` dimensions.

    Notes
    -----
    - For convolution, the result have to be used with unitary
    discrete Fourier transform for the signal (udftn or equivalent).
    - DFT are always peformed on last axis for efficiency.
    - Results is always C-contiguous.

    See Also
    --------
    udftn, uidftn, urdftn, uirdftn
    """
    if len(shape) > imp_resp.ndim:
        raise ValueError("length of shape must be inferior to imp_resp.ndim")

    if not center:
        center = [int(np.floor(length / 2))
                  for length in imp_resp.shape[-len(shape):]]

    if len(center) != len(shape):
        raise ValueError("center and shape must have the same length")

    # Place the provided IR at the beginning of the array
    irpadded = np.zeros(imp_resp.shape[:len(shape) - 1] + shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp

    # Roll, or circshift to place the origin at index 0, the
    # hypothesis of the DFT
    for axe, shift in enumerate(center):
        irpadded = np.roll(irpadded, -shift,
                           imp_resp.ndim - len(shape) + axe)

    # Perform the DFT on the last axes
    if real:
        return np.ascontiguousarray(rfftn(
            irpadded, axes=list(range(imp_resp.ndim - len(shape),
                                      imp_resp.ndim))))
    else:
        return np.ascontiguousarray(fftn(
            irpadded, axes=list(range(imp_resp.ndim - len(shape),
                                      imp_resp.ndim))))

def urdftn(inarray, ndim=None, *args, **kwargs):
    """N-dim real unitary discrete Fourier transform

    This transform consider the Hermitian property of the transform on
    real input

    Parameters
    ----------
    inarray : ndarray
    The array to transform.

    ndim : int, optional
    The `ndim` last axis along wich to compute the transform. All
    axes by default.

    Returns
    -------
    outarray : array-like (the last ndim as  N / 2 + 1 lenght)
    """
    if not ndim:
        ndim = inarray.ndim

    return dask_rfftn(inarray, axes=range(-ndim, 0), *args, **kwargs) / da.sqrt(
    da.prod(da.asarray(inarray.shape[-ndim:])))

def urdft2(inarray, *args, **kwargs):
    """2-dim real unitary discrete Fourier transform

    Compute the real discrete Fourier transform on the last 2 axes. This
    transform consider the Hermitian property of the transform from
    complex to real real input.

    Parameters
    ----------
    inarray : ndarray
    The array to transform.

    Returns
    -------
    outarray : array-like (the last dim as (N - 1) *2 lenght)

    See Also
    --------
    udft2, udftn, urdftn
    """
    return urdftn(inarray, 2, *args, **kwargs)

def uirdftn(inarray, ndim=None, *args, **kwargs):
    """N-dim real unitary discrete Fourier transform

    This transform consider the Hermitian property of the transform
    from complex to real real input.

    Parameters
    ----------
    inarray : ndarray
    The array to transform.

    ndim : int, optional
    The `ndim` last axis along wich to compute the transform. All
    axes by default.

    Returns
    -------
    outarray : array-like (the last ndim as (N - 1) * 2 lenght)
    """
    if not ndim:
        ndim = inarray.ndim

    return dask_irfftn(inarray, axes=range(-ndim, 0), *args, **kwargs) * da.sqrt(
    da.prod(da.asarray(inarray.shape[-ndim:-1])) * (inarray.shape[-1] - 1) * 2)

def uirdft2(inarray, *args, **kwargs):
    """2-dim real unitary discrete Fourier transform

    Compute the real inverse discrete Fourier transform on the last 2 axes.
    This transform consider the Hermitian property of the transform
    from complex to real real input.

    Parameters
    ----------
    inarray : ndarray
    The array to transform.

    Returns
    -------
    outarray : array-like (the last ndim as (N - 1) *2 lenght)

    See Also
    --------
    urdft2, uidftn, uirdftn
    """
    return uirdftn(inarray, 2, *args, **kwargs)

def wiener(data, aux, fr, L):
    print((da.conj(fr) * urdft2(data) + lamb * urdft2(aux)))
    return uirdft2((da.conj(fr) * urdft2(data) + lamb * urdft2(aux)) / (da.absolute(fr_npy)**2 + lamb))

def load_data(data):
    data_npy = np.load(data).astype(np.double)
    #data_dask = da.from_array(data_npy, chunks = data_npy.shape) 
    nr, nc = data_npy.shape

    data = da.from_array(np.tile(data_npy[np.newaxis], (n, 1, 1)),
                         chunks=(1, nr, nc))
    return data_npy, data

def main():


    global lamb
    global fr_npy
    global n

    #Le client est une classe du module DASK qui permet faire une conexion du user avec une Cluster.
    #client = Client('tcp://160.228.203.193:8786')

    #Parametres du code
    lamb = 0.0005
    n = 3

    sky_data = os.path.split(os.getcwd())[0] + '/sky.npy'
    dirty_data = os.path.split(os.getcwd())[0] + '/dirty.npy'
    psf_data = os.path.split(os.getcwd())[0] + '/psf.npy'

    sky_npy, sky = load_data(sky_data)
    dirty_npy, dirty = load_data(dirty_data)
    psf_npy, psf = load_data(psf_data)

    #print(sky)
    #print(dirty)
    #print(psf)

    quad_freq = []

    #Boucle pour faire le calcul de toutes les frequences du cube.
    fr_npy = ir2fr(np.tile(psf_npy[np.newaxis], (n, 1, 1)),
                   shape=sky.shape[1:])
    fr = da.from_array(fr_npy, chunks = ((1, ) + fr_npy.shape[1:]))  # Attention ici au chunk de "fr"
    
    # b = uirdft2(fr.conj() * urdft2(dirty[i]))

    quad = wiener(dirty, da.zeros_like(sky), fr, lamb)

    quad = quad.compute()
    #b.visualize()
    #quad.visualize()
    
    '''
    plt.figure(1)
    plt.clf()
    plt.plot((dirty[0])[800], label='dirty')
    plt.plot((quad[0])[800], label='quad')
    plt.plot((sky[0])[800], label='true')
    plt.legend()
    plt.figure(2)
    plt.clf()
    plt.plot((dirty[1])[800], label='dirty')
    plt.plot((quad[1])[800], label='quad')
    plt.plot((sky[1])[800], label='true')
    plt.legend()
    plt.figure(3)
    plt.clf()
    plt.plot((dirty[2])[800], label='dirty')
    plt.plot((quad[2])[800], label='quad')
    plt.plot((sky[2])[800], label='true')
    plt.legend()
    plt.show()
    '''
 



    """
    plt.imshow(quad[2], vmin = da.min(quad), vmax = da.max(quad))
    plt.title('quad')
    plt.show()
    """    
    """
    for i in range(len(quad_freq)):
        quad_freq[i].compute()
    """
for i in range (1):

    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)
