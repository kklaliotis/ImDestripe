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


def transpose_interpolate(image_A, wcs_A, image_B, original_image):
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


def test_transpose_interpolate(verbose=True):
    # Create synthetic test cases
    test_cases = [
        # Simple case: uniform gradient
        {
            'gradient_interpolated': np.full((4088, 4088), 1.0),
            'wcs_A': create_test_wcs(),
            'I_B': create_test_image(),
            'description': 'Uniform gradient'
        },
        # Random gradient
        {
            'gradient_interpolated': np.random.randn(4088, 4088),
            'wcs_A': create_test_wcs(),
            'I_B': create_test_image(),
            'description': 'Random gradient'
        }
    ]

    for case in test_cases:
        gradient_interpolated = case['gradient_interpolated']
        wcs_A = case['wcs_A']
        I_B = case['I_B']
        gradient_original = np.zeros_like(I_B.image)

        # Perform transformation
        transpose_interpolate(gradient_interpolated, wcs_A, I_B, gradient_original)

        # Detailed analysis
        if verbose:
            print(f"\nTest case: {case['description']}")
            print("Gradient Interpolated:")
            print(f"  Shape: {gradient_interpolated.shape}")
            print(
                f"  Min/Max/Mean: {gradient_interpolated.min()}, {gradient_interpolated.max()}, {gradient_interpolated.mean()}")

            print("\nGradient Original:")
            print(f"  Shape: {gradient_original.shape}")
            print(f"  Min/Max/Mean: {gradient_original.min()}, {gradient_original.max()}, {gradient_original.mean()}")

            # Check for potential numerical instability
            print("\nNumerical Stability Checks:")
            print(f"  Max absolute value: {np.max(np.abs(gradient_original))}")
            print(f"  Variance: {np.var(gradient_original)}")

            # Coordinate transformation sanity checks
            print("\nCoordinate Transformation:")
            test_coords = [(0, 0), (2000, 2000), (4087, 4087)]
            for x, y in test_coords:
                pixel_A = [x, y]
                world_coord = wcs_A.pixel_to_world(x, y)
                pixel_B = I_B.wcs.world_to_pixel(world_coord)
                print(f"  Pixel A {pixel_A} -> World -> Pixel B {pixel_B}")

    return test_cases


def create_test_wcs():
    # Create a mock WCS object with known properties
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 4088
    header['NAXIS2'] = 4088
    header['CRPIX1'] = 2044  # Center of image
    header['CRPIX2'] = 2044
    header['CRVAL1'] = 0  # Reference coordinates
    header['CRVAL2'] = 0
    header['CDELT1'] = 1 / 3600  # 1 arcsecond pixel scale
    header['CDELT2'] = 1 / 3600
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    return wcs.WCS(header)


def create_test_image():
    # Create a mock SCA image object
    class MockImage:
        def __init__(self):
            self.image = np.zeros((4088, 4088))
            self.w = create_test_wcs()
            self.wcs=self.w
            self.g_eff = 1.0
            self.shape = np.shape(self.image)

    return MockImage()


# Run the test
test_cases = test_transpose_interpolate()