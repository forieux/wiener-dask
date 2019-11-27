import os
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
import time
import cProfile
import timeit


try:
    from dask.array.fft import fftn as dask_fftn 
    from dask.array.fft import ifftn as dask_ifftn
    from dask.array.fft import rfftn as dask_rfftn
    from dask.array.fft import irfftn as dask_irfftn
    from pyfftw.interfaces.numpy_fft import fftn
    from pyfftw.interfaces.numpy_fft import ifftn
    from pyfftw.interfaces.numpy_fft import rfftn
    from pyfftw.interfaces.numpy_fft import irfftn    
except ImportError:
    print("Installation of the dask.array and/or pyfftw package improve preformance"
               " by using fft/fftw library.")
    from dask.array.fft import fftn as dask_fftn
    from dask.array.fft import ifftn as dask_ifftn
    from dask.array.fft import rfftn as dask_rfftn
    from dask.array.fft import irfftn as dask_irfftn
    from numpy.fft import fftn as fftn
    from numpy.fft import ifftn as ifftn
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
        raise ValueError("length of shape must inferior to imp_resp.ndim")

    if not center:
        center = [int(np.floor(length / 2)) for length in imp_resp.shape]

    if len(center) != len(shape):
        raise ValueError("center and shape must have the same length")

    # Place the provided IR at the beginning of the array
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp

    # Roll, or circshift to place the origin at 0 index, the
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
    return uirdft2((da.conj(fr) * urdft2(data) + lamb * urdft2(aux)) / (da.absolute(fr_npy)**2 + lamb))

def load_data(data):
    data_npy = np.load(data).astype(np.double)
    data_npy1=data_npy[0:len(data_npy)/2, 0:len(data_npy)/2]
    data_npy2=data_npy[0:len(data_npy)/2, len(data_npy)/2:len(data_npy)]
    data_npy3=data_npy[len(data_npy)/2:len(data_npy), 0:len(data_npy)/2]
    data_npy4=data_npy[len(data_npy)/2:len(data_npy), len(data_npy)/2:len(data_npy)]
    data1 = da.from_array(data_npy1,chunks=data_npy1.size)
    data2 = da.from_array(data_npy2,chunks=data_npy2.size)
    data3 = da.from_array(data_npy3,chunks=data_npy3.size)
    data4 = da.from_array(data_npy4,chunks=data_npy4.size)
    #data_npy = np.concatenate([data_npy,data_npy])
    #data_npy = np.concatenate([data_npy,data_npy], axis=1)
    data = da.from_array(data_npy,chunks=data_npy.size)
    #data = da.concatenate([data,data])
    #data = da.concatenate([data,data], axis=1)
    #data = da.concatenate([data,data])
    #data = da.concatenate([data,data], axis=1)
    #data = data.rechunk(data.shape)
    return data_npy, data, data_npy1, data1 , data_npy2, data2, data_npy3, data3, data_npy4, data4

def main():


    global lamb
    global fr_npy

    #client = Client('tcp://192.168.43.199:8786')
    lamb = 0.0005

    sky_data = os.getcwd() + '/data/sky.npy'
    dirty_data = os.getcwd() + '/data/dirty.npy'
    psf_data = os.getcwd() + '/data/psf.npy'


    sky_npy, sky, sky_npy1, sky1 , sky_npy2, sky2 , sky_npy3, sky3, sky_npy4, sky4 = load_data(sky_data)
    dirty_npy, dirty, dirty_npy1, dirty1 , dirty_npy2, dirty2 , dirty_npy3, dirty3, dirty_npy4, dirty4 = load_data(dirty_data)
    psf_npy, psf , psf_npy1, psf1 , psf_npy2, psf2 , psf_npy3, psf3, psf_npy4, psf4 = load_data(psf_data)

    #psf_npy = np.concatenate([psf_npy,psf_npy])
    #psf_npy = np.concatenate([psf_npy,psf_npy], axis=1)
    #psf_npy = np.concatenate([psf_npy,psf_npy])
    #psf_npy = np.concatenate([psf_npy,psf_npy], axis=1)
    
    fr_npy = ir2fr(psf_npy, sky.shape)
    fr = da.from_array(fr_npy, psf.shape)

    fr_npy1 = ir2fr(psf_npy1, sky1.shape)
    fr1 = da.from_array(fr_npy1, psf1.shape)
    fr_npy2 = ir2fr(psf_npy2, sky2.shape)
    fr2 = da.from_array(fr_npy2, psf2.shape)
    fr_npy3 = ir2fr(psf_npy3, sky3.shape)
    fr3 = da.from_array(fr_npy3, psf3.shape)
    fr_npy4 = ir2fr(psf_npy4, sky4.shape)
    fr4 = da.from_array(fr_npy4, psf4.shape)

    b = uirdft2(fr.conj() * urdft2(dirty))
    quad = wiener(dirty, da.zeros_like(sky), fr, lamb)

    quad1 = wiener(dirty1, da.zeros_like(sky1), fr1, lamb)
    quad2 = wiener(dirty2, da.zeros_like(sky2), fr2, lamb)
    quad3 = wiener(dirty3, da.zeros_like(sky3), fr3, lamb)
    quad4 = wiener(dirty4, da.zeros_like(sky4), fr4, lamb)

    print(quad)
    print(quad1)
    print(quad2)
    print(quad3)
    print(quad4)
    
    #b.visualize()
    #quad.visualize()
    
    plt.figure(1)
    plt.clf()
    plt.plot(dirty[800], label='dirty')
    plt.plot(quad[800], label='quad')
    plt.plot(sky[800], label='true')
    plt.legend()
    plt.figure(2)
    plt.clf()
    plt.imshow(quad, vmin = da.min(quad), vmax = da.max(quad))
    plt.title('quad')
    
    quad.compute()

start_time = time.time()
main()
end_time = time.time()
print(end_time - start_time)
#print('next')
#n = 5
#print((timeit.timeit(main, number=n))/n)
#cProfile.run('re.compile("foo|bar")')
#cProfile.run('main()')


plt.show()
