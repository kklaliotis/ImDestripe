import os
import glob
import time
import ctypes
import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.signal import convolve2d
from utils import compareutils
import re
import sys
import copy
import pyimcom_croutines
from destripe import  sca_img

def interpolate_image_bilinear(image_B, image_A, interpolated_image, mask=None):
    """
    Interpolate values from a "reference" SCA image onto a "target" SCA coordinate grid
    Uses pyimcom_croutines.bilinear_interpolation(float* image, float* g_eff, int rows, int cols, float* coords,
                                                    int num_coords, float* interpolated_image)
    :param image_B : an SCA object of the image to be interpolated
    :param image_A : an SCA object of the image whose grid you are interpolating B onto
    :param interpolated_image : an ndarray of zeros with shape of Image A.
    interpolated_image_B is updated in-place.
    """

    x_target, y_target, is_in_ref = compareutils.map_sca2sca(image_A.w, image_B.w, pad=0)
    coords = np.column_stack((x_target.ravel(), y_target.ravel())).flatten().astype(np.float32)

    # Verify data just before C call
    rows = int(image_B.shape[0])
    cols = int(image_B.shape[1])
    num_coords = coords.shape[0] // 2

    print('fwd: rows, cols, ncoords ', rows, cols, num_coords)
    print('fwd: coords 0-10', coords[0:10])
    print('fwd: coords min, mean, max', np.min(coords), np.max(coords), np.mean(coords))

    sys.stdout.flush()

    if mask is not None and isinstance(mask, np.ndarray):
        mask_geff = np.ones_like(image_A.image)
        pyimcom_croutines.bilinear_interpolation(mask,
                                                 mask_geff,
                                                 rows, cols,
                                                 coords,
                                                 num_coords,
                                                 interpolated_image)
    else:
        pyimcom_croutines.bilinear_interpolation(image_B.image,
                                             image_B.g_eff,
                                             rows, cols,
                                             coords,
                                             num_coords,
                                             interpolated_image)

    sys.stdout.flush()


def transpose_interpolate( image_A, wcs_A, image_B, original_image):
     """
     Interpolate backwards from image_A to image B space.
     :param image_A : a 2D numpy array (will be the interpolated gradient image)
     :param wcs_A : the WCS (a wcs.WCS object) that goes with image A
     :param image_B : An SCA object, the image we're interpolating back onto
     :param original_image: 2D numpy array, the gradient image re-interpolated into image B space
     note: bilinear_transpose(float* image, int rows, int cols, float* coords, int num_coords, float* original_image)
     :return:
     """
     x_target, y_target, is_in_ref = compareutils.map_sca2sca(image_B.w, wcs_A, pad=0)
     coords = np.column_stack((x_target.ravel(), y_target.ravel())).flatten().astype(np.float32)

     rows = int(image_B.shape[0])
     cols = int(image_B.shape[1])
     num_coords = coords.shape[0] // 2
     print('transpose: rows, cols, ncoords ', rows, cols, num_coords)
     print('transpose: coords 0-10', coords[0:10])
     print('transpose: coords min, mean, max', np.min(coords), np.max(coords), np.mean(coords))
     sys.stderr.flush()

     pyimcom_croutines.bilinear_transpose(image_A,
                                            rows, cols,
                                            coords,
                                            num_coords,
                                            original_image)

def interp_test(I_A, I_B, I_B_to_A, I_A_to_B):
    u = I_A.image
    v = I_B.image
    Mv = I_B_to_A
    Mu=I_A_to_B
    lhs = np.sum(u*Mv)
    rhs = np.sum(v*Mu)
    # Detailed diagnostics
    print("\nDetailed Interpolation Test:")
    print(f"A: {I_A.obsid}_{I_A.scaid}, B: {I_B.obsid}_{I_B.scaid}")

    # Print array statistics
    print("\nArray Statistics:")
    print(f"u (A image):   mean={np.mean(u):.4f}, std={np.std(u):.4f}, min={np.min(u):.4f}, max={np.max(u):.4f}")
    print(f"v (B image):   mean={np.mean(v):.4f}, std={np.std(v):.4f}, min={np.min(v):.4f}, max={np.max(v):.4f}")
    print(f"Mv (B to A):   mean={np.mean(Mv):.4f}, std={np.std(Mv):.4f}, min={np.min(Mv):.4f}, max={np.max(Mv):.4f}")
    print(f"Mu (A to B):   mean={np.mean(Mu):.4f}, std={np.std(Mu):.4f}, min={np.min(Mu):.4f}, max={np.max(Mu):.4f}")

    # Compute dot products
    lhs = np.sum(u * Mv)
    rhs = np.sum(v * Mu)

    print("\nDot Product Test:")
    print(f"u·Mv = {lhs:.4f}")
    print(f"v·Mu = {rhs:.4f}")
    print(f"Difference:   {lhs - rhs:.4f}")


def test_interp():
    I_A=sca_img("670","10")
    I_B= sca_img("668","3")
    B_interp = np.zeros_like(I_B.image)
    A_interp = np.zeros_like(I_A.image)
    interpolate_image_bilinear(I_B, I_A, B_interp)
    B_interp/=I_A.g_eff
    transpose_interpolate(I_A.image, I_A.w, I_B, A_interp)
    A_interp/=I_B.g_eff

    interp_test(I_A, I_B, B_interp, A_interp)
    stuff=[I_B.image,A_interp, B_interp]

    phdu = fits.PrimaryHDU(I_A.image)
    hdul = fits.HDUList([phdu])
    for i in stuff:
        hdu = fits.ImageHDU(i)
        hdul.append(hdu)
    hdul.writeto('INTERP_TESTS.fits', overwrite=True)


# Run the test
test_interp()
