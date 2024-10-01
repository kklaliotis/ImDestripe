"""
Program to remove correlated noise stripes from Roman ST images.
"""
import os
import numpy as np
from astropy.io import fits
from astropy import wcs
import re

# KL: Placeholders, some of these could be input arguments or in a config or something
input_dir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/simple/'
image_prefix = 'Roman_WAS_simple_model_'
labnoise_prefix = 'fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise/slope_'
filter = 'H158'
model_params = {'constant': 1, 'linear': '2'} # KL this is probably sufficient


class sca_img:
    """
    Class defining an SCA image object.
    Attributes:
        id: the SCA id
        ra_ctr: RA coordinate of the SCA image center
        dec_ctr: dec coordinate of the SCA image center
        image: the SCA image (4088x4088) (KL: np array or pd dataframe?_
        shape: shape of the image
        wcs: the astropy.wcs object associated with this SCA
    Functions:
    KL to do: idk about this whole thing
    """
    def __init__(self):
        file = fits.open(input_dir+image_prefix+str(self))
        self.image = np.copy(file[1].data())
        self.shape = np.shape(self.image)
        self.w = wcs.WCS(file[1].header)
        self.ra_ctr = self.w.wcs.crval[0]
        self.dec_ctr = self.w.wcs.crval[1]

    def apply_noise(self):
        noiseframe = np.copy(fits.open(labnoise_prefix+self.obsid+'_'+self.scaid)['PRIMARY'].data)
        self.image += noiseframe
        return self.image

    def get_overlap(self):
        # KL: use chris utilities to get overlapping scas

def get_scas(observation):
    """
    Function to get an array of SCA images for an observation
    :param observation: an observation object
    :return: numpy array with all the SCA images
    """
    n_scas = len(observation.sca_list)
    sca_images = np.zeros((n_scas, 4088, 4088))
    for i in range(0,n_scas-1):
        sca_id = observation.sca_list[i]
        filename = image_prefix+sca_id
        sca = sca_img(filename)
        this_sca_image = sca.image
        sca_images[i, :, :] = this_sca_image


class ds_parameters:
    """
    Class holding the destriping parameters for a given image.
    Attributes:
        model: which destriping model to use, which specifies the number of parameters per row based on the
         model_params dict
        params_per_row: number of parameters per row, given by the model
        params: the actual array of parameters.
    Functions:
        params_2_images: reshape params into the 2D array
        flatten_params: reshape params into 1D vector
    KL to do:
    """
    def __init__(self, model):
        self.model = model
        self.params_per_row = model_params[str(self.model)]
        self.params = np.zeros((len(all_scas), n_rows*self.params_per_row))

    def params_2_images(self):
        self.params = np.reshape(self.params, ((len(all_scas), n_rows*self.params_per_row)))
        return self.params

    def flatten_params(self):
        self.params = np.ravel(self.params)
        return self.params

# KL: Adapted from Naim CiC algorithm
@njit("(f8[:, :], f8[:, :], f8, i8)")
def T_interpolate(

        interpolated_image, imageB_grid, d_xy, n_grid

):
    """Reverse interpolate pixels from Image B onto `interpolated_image`, the pixel grid from Image A.

    Arguments

    ---------

    interpolated_image (np.ndarray): 2D grid of shape

        (n_grid, n_grid). This array is updated in place.

    imageB_grid (np.ndarray): 2D array of shape (n_grid, n_grid). needs to have coordinates of pixels in a form
        that will be useful to compare with image A

    d_xy (float): Grid size in x and y directions (image A pixel spacing in arcsec I think)

    n_grid (int): Number of grid points for x and y directions. =4088

    KL: interpolation weights and effective gain in here??

    """

    dis_r = imageB_grid / np.array([d_xy, d_xy])

    idx_r = dis_r.astype(np.int_)

    dis_r -= idx_r

    # Periodic boundary conditions.

    idx_r %= np.array([n_grid, n_grid], dtype=np.int_)

    for (x, y), (dx, dy) in zip(idx_r, dis_r):
        # Next grid points (e.g., x + 1). The following function applies
        # periodic boundary conditions.
        # KL: I think we said we would just not use boundary pixels for this project ??

        xp = clip_grid_idx(x, n_grid)

        yp = clip_grid_idx(y, n_grid)

        interpolated_image[x, y] += (1 - dx) * (1 - dy) # KL: multiply these by image B pixel value??

        interpolated_image[x, yp] += (1 - dx) * dy

        interpolated_image[xp, y] += dx * (1 - dy)

        interpolated_image[xp, yp] += dx * dy

    return interpolated_image

#for loop: check if image B overlaps image A; if yes, make interpolated B->A;
# add to J_A w/ some interp. weight and effective gain; divide by A effective gain and number in the interpolation

