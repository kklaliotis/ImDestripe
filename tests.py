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

    sys.stdout.flush()
    sys.stderr.flush()
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
    sys.stderr.flush()


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
    print(f'Interpolation test:\n A={I_A.obsid}_{I_A.scaid}, B={I_B.obsid}_{I_B.scaid}\n Diff={lhs-rhs}')

def test_interp():
    I_A=sca_img("670","10")
    I_B= sca_img("668","3")
    B_interp = np.zeros_like(I_B.image)
    A_interp = np.zeros_like(I_A.image)
    interpolate_image_bilinear(I_B, I_A, B_interp)
    interpolate_image_bilinear(I_A, I_B, A_interp)

    interp_test(I_A, I_B, B_interp, A_interp)

# Run the test
test_interp()
