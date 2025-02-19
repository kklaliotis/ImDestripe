"""
Program to remove correlated noise stripes from Roman ST images.
KL To do list:
- Link with config file
- Write outputs to file instead of print
"""

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

tempfile_Katherine_dir = True

# KL: Placeholders, some of these should be input arguments or in a config or something
#input_dir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/simple/'
input_dir = '/fs/scratch/PCON0003/klaliotis/destripe/inputs/'
image_prefix = 'Roman_WAS_simple_model_'
labnoise_prefix = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise/slope_'
filter = 'H158'
model_params = {'constant': 1, 'linear': 2}
permanent_mask = '/users/PCON0003/cond0007/imcom/coadd-test-fall2022/permanent_mask_220730.fits'
if tempfile_Katherine_dir:
    tempfile = '/fs/scratch/PCON0003/klaliotis/destripe/test_out/'  # temporary temp file so that i can check these things
else:
    tempfile = '/fs/scratch/PCON0003/cond0007/test_out/'
outfile = tempfile + 'destripe_' + filter + '_out.txt'
# tempfile = '/tmp/klaliotis-tmp/'
s_in = 0.11  # arcsec^2
t_exp = 154  # sec
A_eff = 7340  # cm ^2; for H
t0 = time.time()
test = True


def write_to_file(text):
    """
    Function to write some text to an output file
    :param text: a string to print
    :return: nothing
    """
    global outfile
    with open(outfile, "w") as f:
        f.write(text + '\n')
    with open(outfile, "r") as f:
        print(f.readlines())


# C.H. wanted to define this before any use of sca_img so moved it up.
def apply_object_mask(image, mask=None):
    """
    Apply a bright object mask to an image.
    :param image: 2D numpy array, the image to be masked.
    :param mask: optional: 2D numpy array, the pre-existing object mask you wish to use
    :return: the image with bright objects (flux>1.5*median; could modify later) masked out
    """
    if mask is not None and isinstance(mask, np.ndarray):
        neighbor_mask = mask
    else:
        # Create a binary mask for high-value pixels (KL: could modify later)
        high_value_mask = image >= 1.5 * np.median(image)

        # Convolve the binary mask with a 5x5 kernel to include neighbors
        kernel = np.ones((5, 5), dtype=int)
        neighbor_mask = convolve2d(high_value_mask, kernel, mode='same') > 0

    # Set the target pixels and their neighbors to zero
    image = np.where(neighbor_mask, 0, image)
    return image, neighbor_mask

class sca_img:
    """
    Class defining an SCA image object.
    Arguments:
        scaid: the SCA id (str)
        obsid: the observation id (str)
        interpolated: True if you want the interpolated version of this SCA and not the original. Default:False
    Attributes:
        image: the SCA image (4088x4088)
        shape: shape of the image
        w: the astropy.wcs object associated with this SCA
        mask: the full pixel mask that is used on this image. Is correct only after calling apply_permanent_mask
        g_eff : effective gain in each pixel of the image
        params_subtracted : Check whether parameters have been subtracted from this image.
    Functions:
    apply_noise: apply the appropriate lab noise frame to the SCA image
    apply_permanent_mask: apply the SCA permanent pixel mask to the image
    apply_all_mask: apply the full SCA mask to the image
    """

    def __init__(self, obsid, scaid, interpolated=False, add_noise=True, add_objmask=True):
        """

        :param obsid:
        :param scaid:
        :param interpolated:
        :param add_noise:
        :param add_objmask:
        """
        if interpolated:
            file = fits.open(tempfile +'interpolations/' + obsid + '_' + scaid + '_interp.fits')
            image_hdu = 'PRIMARY'
        else:
            file = fits.open(input_dir + image_prefix + filter + '_' + obsid + '_' + scaid + '.fits')
            image_hdu = 'SCI'
        self.image = np.copy(file[image_hdu].data).astype(np.float32)
        self.shape = np.shape(self.image)
        self.w = wcs.WCS(file[image_hdu].header)
        file.close()

        self.obsid = obsid
        self.scaid = scaid
        self.mask = np.ones(self.shape, dtype=bool)
        self.params_subtracted = False

        # Calculate effecive gain
        if not os.path.isfile(tempfile + obsid+'_'+scaid+'_geff.dat'):
            g_eff = np.memmap(tempfile + obsid+'_'+scaid+'_geff.dat', dtype='float32', mode='w+', shape=self.shape)
            ra, dec = self.get_coordinates(pad=2.)
            ra = ra.reshape((4090, 4090))
            dec = dec.reshape((4090, 4090))
            derivs = np.array(((ra[1:-1,2:] - ra[1:-1,:-2])/2, (ra[2:, 1:-1] - ra[:-2, 1:-1])/2,
                              (dec[1:-1,2:] - dec[1:-1,:-2])/2, (dec[2:, 1:-1] - dec[:-2, 1:-1])/2))
            derivs_px = np.reshape(np.transpose(derivs), (4088**2, 2, 2))
            det_mat = np.reshape(np.linalg.det(derivs_px), (4088,4088))
            g_eff[:,:] = 1 / (np.abs(det_mat) * np.cos(np.deg2rad(dec[1:4089,1:4089])) * t_exp * A_eff )
            g_eff.flush()
            del g_eff

        self.g_eff = np.memmap(tempfile + obsid+'_'+scaid+'_geff.dat', dtype='float32', mode='r', shape=self.shape)

        # Add a noise frame, if requested
        if add_noise: self.apply_noise()
        if add_objmask:
            _, object_mask = apply_object_mask(self.image)
            self.apply_permanent_mask()
            self.mask *= np.logical_not(object_mask) # self.mask = True for good pixels, so set object_mask'ed pixels to False

    def apply_noise(self):
        """
        Add detector noise to self.image
        KL to do: make an option to write out image of image + noise (for comparison fig)
        :return:
        """
        noiseframe = np.copy(fits.open(labnoise_prefix + self.obsid + '_' + self.scaid + '.fits')['PRIMARY'].data)
        self.image += noiseframe[4:4092, 4:4092]

    def apply_permanent_mask(self):
        """
        Apply permanent pixel mask. updates self.image and self.mask
        :return:
        """
        pm = fits.open(permanent_mask)[0].data[int(self.scaid) - 1].astype(bool)
        self.image *= pm
        self.mask *= pm

    def apply_all_mask(self):
        """
        Apply permanent pixel mask. updates self.image and self.mask
        :return:
        """
        self.image *= self.mask

    def subtract_parameters(self, p, j):
        if self.params_subtracted == True:
            print('\n WARNING: PARAMS HAVE ALREADY BEEN SUBTRACTED. ABORTING NOW')
            sys.exit()

        params_image = p.forward_par(j)  # Make destriping params into an image
        self.image = self.image - params_image  # Update I_A.image to have the params image subtracted off
        self.params_subtracted = True


    def get_coordinates(self, pad=0.):
        """
        create an array of ra, dec coords for the image
        :param pad: add padding to the array. default is zero. float64
        :return: ra, dec flattened arrays of length(height*width)
        """
        wcs = self.w
        h = self.shape[0] + pad
        w = self.shape[1] + pad
        x_i, y_i = np.meshgrid(np.arange(h), np.arange(w), indexing='xy')
        x_i -= pad / 2.
        y_i -= pad / 2.
        x_flat = x_i.flatten()
        y_flat = y_i.flatten()
        ra, dec = wcs.all_pix2world(x_flat, y_flat, 0)  # 0 is for the first frame (1-indexed)
        return ra, dec

    def make_interpolated(self, ind, params=None, N_eff_min=0.5):
        """
        Construct a version of this SCA interpolated from other, overlapping ones.
        The N_eff_min parameter requires some minimum effective coverage, otherwise masks that pixel.
        :return:
        interpolated image, mask
        """
        if self.obsid=='670' and self.scaid=='10':
            print(f"Check the index: {all_scas[ind]} =? {self.scaid}_{self.obsid}")

        this_interp = np.zeros(self.shape)

        if not os.path.isfile(tempfile + self.obsid+'_'+self.scaid+'_Neff.dat'):
            N_eff = np.memmap(tempfile + self.obsid + '_' + self.scaid + '_Neff.dat', dtype='float32', mode='w+',
                              shape=self.shape)
            make_Neff=True
        else:
            N_eff = np.memmap(tempfile + self.obsid + '_' + self.scaid + '_Neff.dat', dtype='float32', mode='r',
                              shape=self.shape)
            make_Neff=False

        t_a_start = time.time()
        print('Starting interpolation for SCA' + self.obsid + '_' + self.scaid + ' (index '+ str(ind)+')')
        sys.stdout.flush()

        N_BinA = 0

        for k, sca_b in enumerate(all_scas):
            obsid_B, scaid_B = get_ids(sca_b)

            if obsid_B != self.obsid and ov_mat[ind, k] != 0:  # Check if this sca_b overlaps sca_a
                N_BinA += 1
                I_B = sca_img(obsid_B, scaid_B) # Initialize image B
                # I_B.apply_noise() <-- redundant

                if self.obsid=='670' and self.scaid=='10':
                    print('Image B index:' + str(k))
                    print('\nI_B: ', obsid_B, scaid_B, 'Pre-Param-Subtraction mean:', np.mean(I_B.image))

                if params:
                    I_B.subtract_parameters(params, k)

                I_B.apply_all_mask() # now I_B is masked
                B_interp = np.zeros_like(self.image)
                interpolate_image_bilinear(I_B, self, B_interp)

                if make_Neff:
                    B_mask_interp = np.zeros_like(self.image)
                    interpolate_image_bilinear(I_B, self, B_mask_interp, mask=I_B.mask) # interpolate B pixel mask onto A grid

                if obsid_B=='670' and scaid_B=='10' and make_Neff: #only do this once
                    hdu=fits.PrimaryHDU(B_interp)
                    hdu.writeto(test_image_dir+'670_10_B'+self.obsid+'_'+self.scaid+'_interp.fits', overwrite=True)
                if self.obsid=='670' and self.scaid=='10' and make_Neff:
                    hdu=fits.PrimaryHDU(B_interp)
                    hdu.writeto(test_image_dir+'670_10_A'+obsid_B+'_'+scaid_B+'_interp.fits', overwrite=True)

                this_interp += B_interp

                if make_Neff:
                    N_eff += B_mask_interp


        print('Interpolation done. Number of contributing SCAs: ', N_BinA)
        new_mask = N_eff > N_eff_min
        this_interp = np.where(new_mask, this_interp/np.where(new_mask, N_eff, N_eff_min), 0) # only do the division where N_eff nonzero
        header =self.w.to_header(relax=True)
        this_interp = np.divide(this_interp, self.g_eff)
        hdu = fits.PrimaryHDU(this_interp, header=header)
        hdu.writeto(tempfile + 'interpolations/' + self.obsid + '_' + self.scaid + '_interp.fits', overwrite=True)
        t_elapsed_a = time.time() - t_a_start

        if make_Neff: N_eff.flush()
        del N_eff
        return this_interp, new_mask

class parameters:
    """
    Class holding the parameters for a given mosaic. This can be the destriping parameters, or a slew of other
    parameters that need to be the same shape and have the same abilities...
    Attributes:
        model: which destriping model to use, which specifies the number of parameters per row based on the
         model_params dict
        n_rows: number of rows in the image
        params_per_row: number of parameters per row, given by the model
        params: the actual array of parameters.
    Functions:
        params_2_images: reshape params into the 2D array
        flatten_params: reshape params into 1D vector
    To do:
        add option for additional parameters
    """

    def __init__(self, model, n_rows):
        self.model = model
        self.n_rows = n_rows
        self.params_per_row = model_params[str(self.model)]
        self.params = np.zeros((len(all_scas), self.n_rows * self.params_per_row))
        self.current_shape = '2D'

    def params_2_images(self):
        """
        Reshape flattened parameters into 2D array with 1 row per sca and n_rows (in image) * params_per_row entries
        :return:
        """
        self.params = np.reshape(self.params, (len(all_scas), self.n_rows * self.params_per_row))
        self.current_shape = '2D'

    def flatten_params(self):
        """
        Reshape 2D params array into flat
        :return:
        """
        self.params = np.ravel(self.params)
        self.current_shape = '1D'

    def forward_par(self, sca_i):
        """
        Takes one SCA row (n_rows) from the params and casts it into 2D (n_rows x n_rows)
        :param sca_i: index of which SCA
        :return:
        """
        if not self.current_shape == '2D':
            self.params_2_images()
        return np.array(self.params[sca_i, :])[:, np.newaxis] * np.ones((self.n_rows, self.n_rows))


def get_scas(filter, prefix):
    """
    Function to get an array of SCA images for this mosaic
    :param : None
    :return: numpy array with all the SCA images
    """
    n_scas = 0
    all_scas = []
    all_wcs = []
    for f in glob.glob(input_dir + prefix + filter + '_*'):
        n_scas += 1
        m = re.search(r'(\w\d+)_(\d+)_(\d+)', f)
        if m:
            this_obsfile = str(m.group(0))
            all_scas.append(this_obsfile)
            this_file = fits.open(f)
            this_wcs = wcs.WCS(this_file['SCI'].header)
            all_wcs.append(this_wcs)
            this_file.close()
    write_to_file('N SCA images in this mosaic: ' + str(n_scas))
    print('\nSCA List:')
    for i,s in enumerate(all_scas):
        print(f"SCA {i}: {s}\n")
    return all_scas, all_wcs

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
    coords = np.column_stack(( y_target.ravel(), x_target.ravel()))
    if image_A.obsid=='670' and image_A.scaid=='10':
        print(f"670_10 first 3 coord pairs: {coords[0:3]}")

    # Verify data just before C call
    rows = int(image_B.shape[0])
    cols = int(image_B.shape[1])
    num_coords = coords.shape[0]

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
     x_target, y_target, is_in_ref = compareutils.map_sca2sca(wcs_A, image_B.w, pad=0)
     coords = np.column_stack(( y_target.ravel(), x_target.ravel()))
     if image_B.obsid == '670' and image_B.scaid == '10':
         print(f"670_10 first 3 coord pairs: {coords[0:3]}")

     rows = int(image_B.shape[0])
     cols = int(image_B.shape[1])
     num_coords = coords.shape[0]

     pyimcom_croutines.bilinear_transpose(image_A,
                                            rows, cols,
                                            coords,
                                            num_coords,
                                            original_image)

def transpose_par(I):
    """
    transpose interpolates an image (sums across rows)
    :param array: input array
    :return:  1D vec of N_params
    """
    return np.sum(I, axis=1)


def get_effective_gain(sca):
    """
    retrieve the effective gain and n_eff of the image. valid only for already-interpolated images
    :param sca: whose info you want
    :return: g_eff: memmap array of the effective gain in each pixel
    :return: N_eff: memmap array of how many image "B"s contributed to that interpolated image
    """
    m = re.search(r'_(\d+)_(\d+)', sca)
    obsid = m.group(1)
    scaid = m.group(2)
    g_eff = np.memmap(tempfile + obsid+'_'+scaid+'_geff.dat', dtype='float32', mode='r', shape=(4088,4088) )
    N_eff = np.memmap(tempfile + obsid + '_' + scaid + '_Neff.dat', dtype='float32', mode='r', shape=(4088,4088))
    return g_eff, N_eff


def get_ids(sca):
    """
    Take an SCA label and parse it out to get the Obsid and SCA id strings.
    :param sca: sca name from all_scas
    :return: obsid (str), scaid (str)
    """
    m = re.search(r'_(\d+)_(\d+)', sca)
    obsid = m.group(1)
    scaid = m.group(2)
    return obsid, scaid

############################ Main Sequence ############################

all_scas, all_wcs = get_scas(filter, image_prefix)
print(len(all_scas), " SCAs in this mosaic")

if test:
    if os.path.isfile(tempfile + 'ovmat.npy'):
        ov_mat = np.load(tempfile + 'ovmat.npy')
    else:
        ovmat_t0 = time.time()
        print('Overlap matrix computing start')
        ov_mat = compareutils.get_overlap_matrix(all_wcs, verbose=True)  # an N_wcs x N_wcs matrix containing fractional overlap
        np.save(tempfile+'ovmat.npy', ov_mat)
        print("Overlap matrix complete. Duration: ", (time.time()-ovmat_t0)/60, 'Minutes' )
        print("Overlap matrix saved to: "+tempfile+"ovmat.npy")
else:
    ovmat_t0 = time.time()
    print('Overlap matrix computing start')
    ov_mat = compareutils.get_overlap_matrix(all_wcs,
                                             verbose=True)  # an N_wcs x N_wcs matrix containing fractional overlap
    print("Overlap matrix complete. Duration: ", (time.time() - ovmat_t0) / 60, 'Minutes')


# Function options. KL: Could move these to another .py file and call them as modules?
# import functions and then function.quadratic etc.
# Each of these will have the input x be a 2D array of a sca image.

def quadratic(x):
    return x**2

def absolute_value(x):
    return np.abs(x)

def quadratic_loss(x, x0, b):
    if (x-x0)<=b:
        return quadratic(x-x0)
    else:
        return absolute_value(x-x0)

# Derivatives
def quad_prime(x):
    return 2*x

def absval_prime(x):
    return np.sign(x)

def quadloss_prime(x, x0, b):
    if (x-x0)<=b:
        return quad_prime(x-x0)
    else:
        return absval_prime(x-x0)

# function_dictionary = {"quad": quadratic, "abs": absolute_value, "quadloss": quadratic_loss}

# Optimization Functions

def main():

    def cost_function(p, f):
        """
        p: parameters object
        f: keyword for function dictionary options; should also set an f_prime
        """
        print('Initializing cost function')
        t0_cost = time.time()
        psi = np.zeros((len(all_scas), 4088, 4088))
        epsilon = 0

        for j, sca_a in enumerate(all_scas):
            m = re.search(r'_(\d+)_(\d+)', sca_a)
            obsid_A = m.group(1)
            scaid_A = m.group(2)
            I_A = sca_img(obsid_A, scaid_A)  # Inititalize image A

            # this is now redundant
            #I_A.apply_noise()
            #I_A.image,object_mask = apply_object_mask(I_A.image)
            I_A.subtract_parameters(p, j)
            #I_A.image = apply_object_mask(I_A.image, mask=object_mask)[0]  # re-apply mask to make mask pxls 0 again
            I_A.apply_all_mask()

            if obsid_A=='670' and scaid_A=='10':
                print(f'670_10 is image A with index {j}')
                hdu = fits.PrimaryHDU(I_A.image)
                hdu.writeto(test_image_dir+'670_10_I_A_sub_masked.fits', overwrite=True)

            J_A_image, J_A_mask = I_A.make_interpolated(j, params=p)
            J_A_mask *= I_A.mask # apply permanent mask from A
            # J_A_image = apply_object_mask(J_A_image, mask=object_mask)[0] # <-- inputs are already masked

            if obsid_A=='670' and scaid_A=='10':
                hdu = fits.PrimaryHDU(J_A_image*J_A_mask)
                hdu.writeto(test_image_dir+'670_10_J_A_masked.fits', overwrite=True)

            psi[j, :, :] = np.where(J_A_mask, I_A.image - J_A_image, 0)

            if obsid_A=='670' and scaid_A=='10':
                hdu = fits.PrimaryHDU(psi[j,:,:])
                hdu.writeto(test_image_dir+'670_10_Psi.fits', overwrite=True)

            # Compute local epsilon
            local_epsilon = np.sum(f(psi[j, :, :]))

            if obsid_A=='670' and scaid_A=='10':
                print('Image A mean, std: ', np.mean(I_A.image), np.std(I_A.image))
                print('Image B mean, std: ', np.mean(J_A_image), np.std(J_A_image))
                print ('Psi mean, std: ', np.mean(psi[j, :, :]), np.std(psi[j, :, :]) )
                print('f(Psi) mean, std:', np.mean(f(psi[j, :, :])), np.std(f(psi[j, :, :])))
                print(f"Local epsilon for SCA {j}: {local_epsilon}")

            epsilon += local_epsilon

        print('Ending cost function. Minutes elapsed: ', (time.time()-t0_cost)/60)
        return epsilon, psi


    def residual_function(psi, f_prime, extrareturn=False):
        """
        Calculate the residuals.
        :param psi: the image difference array (I_A - J_A) (N_SCA, 4088, 4088)
        :param f_prime: the function to be used to calculate the gradient.
                in the future this should be set by default based on what you pass for f
        :return: resids, a 2D array with one row per SCA and one col per image-row-parameter
        """
        resids = (parameters('constant', 4088).params)
        if extrareturn:
            resids1 = np.zeros_like(resids)
            resids2 = np.zeros_like(resids)
        print('\nResidual calculation started')
        for k, sca_a in enumerate(all_scas):

            # Go and get the WCS object for image A
            obsid_A, scaid_A = get_ids(sca_a)
            file = fits.open(tempfile + 'interpolations/' + obsid_A + '_' + scaid_A + '_interp.fits')
            wcs_A = wcs.WCS(file[0].header)
            file.close()

            # Calculate and then transpose the gradient of I_A-J_A
            gradient_interpolated = f_prime(psi[k, :, :])
            if obsid_A == '670' and scaid_A == '10':
                # hdu = fits.PrimaryHDU(gradient_interpolated)
                # hdu.writeto(test_image_dir+'Fp_Psi_670_10.fits', overwrite=True)
                print(f"check index of 670_10 (in resids): k={k} and all_scas[k]={all_scas[k]}")

            term_1 = transpose_par(gradient_interpolated)

            # Retrieve the effective gain and N_eff to normalize the gradient before transposing back
            g_eff_A, n_eff_A = get_effective_gain(sca_a)

            # Avoid dividing by zero
            valid_mask = n_eff_A != 0
            gradient_interpolated[valid_mask] = gradient_interpolated[valid_mask] / (
                        g_eff_A[valid_mask] * n_eff_A[valid_mask])
            gradient_interpolated[~valid_mask] = 0

            if obsid_A == '670' and scaid_A == '10':
                hdu = fits.PrimaryHDU(gradient_interpolated)
                hdu.writeto(test_image_dir+'Fp_norm_Psi_670_10.fits', overwrite=True)

            for j, sca_b in enumerate(all_scas):
                obsid_B, scaid_B = get_ids(sca_b)

                if obsid_B != obsid_A and ov_mat[k, j] != 0:
                    I_B = sca_img(obsid_B, scaid_B)
                    gradient_original = np.zeros(I_B.shape)

                    transpose_interpolate(gradient_interpolated, wcs_A, I_B, gradient_original)

                    gradient_original *= I_B.g_eff

                    if obsid_A == '670' and scaid_A == '10':
                        hdu = fits.PrimaryHDU(gradient_original)
                        hdu.writeto(test_image_dir+'Fp_norm_Psi_B_'+obsid_B+scaid_B+'.fits', overwrite=True)

                    term_2 = transpose_par(gradient_original)
                    if obsid_A == '670' and scaid_A == '10':
                        print('Terms 1 and 2 means: ', np.mean(term_1), np.mean(term_2))
                        print('G_eff_a, G_eff_b means: ', np.mean(g_eff_A), np.mean(I_B.g_eff))
                        print(f"B image: {obsid_B}_{scaid_B} match index {j} in all_scas: {all_scas[j]}")

                    resids[j,:] += term_2
                    if extrareturn: resids2[j,:] += term_2

            resids[k, :] -= term_1
            if extrareturn: resids1[k, :] -= term_1

        print('Residuals calculation finished\n')
        if extrareturn: return resids,resids1,resids2
        return resids


    def linear_search(p, direction, f, f_prime, n_iter=50, tol=10**-5):

        # KL: first version of LS using constant direction depth alpha
        # alpha = 0.1  # Step size
        # p.params = p.params + alpha * direction

        best_epsilon, best_psi = cost_function(p, f)
        best_p = copy.deepcopy(p)

        # Simple linear search
        working_p = copy.deepcopy(p)

        if not np.any(p.params):
            alpha_max = 4
        else:
            alpha_max = 4 / np.max(p.params)

        alpha_min = -alpha_max

        for k in range(1, n_iter):
            t0_ls_iter = time.time()

            if k==1:
                print('\n!!!! Inside linear search function now.')
                print("Direction:", direction)
                hdu = fits.PrimaryHDU(direction)
                hdu.writeto(test_image_dir+'LSdirection.fits', overwrite=True)
                print("Initial params:", p.params)
                print("Initial epsilon:", best_epsilon)

            if k == n_iter - 1:
                print('WARNING: Linear search did not converge!! This is going to break because best_p is not assigned.')

            # KL: previous version of LS, using adaptive direction depth alpha
            # current_step = alpha * k / (1 + k)
            # clipped_direction = np.clip(direction,
            #                          -np.abs(working_p.params),
            #                          np.abs(working_p.params))

            # candidate_params = working_p.params + current_step * direction
            # candidate_params = np.clip(candidate_params,
            #                            working_p.params - 10*np.std(working_p.params),
            #                            working_p.params + 10*np.std(working_p.params))

            alpha_test = .5 * (alpha_min + alpha_max)

            working_params = p.params + alpha_test * direction
            working_p.params = working_params

            working_epsilon, working_psi = cost_function(working_p, f)
            working_resids = residual_function(working_psi, f_prime)

            d_cost = np.sum(working_resids * direction)
            convergence_crit = (alpha_max-alpha_min)/2

            print('\nEnding LS iteration', k)
            print('Current d_cost = ', d_cost, 'epsilon = ', working_epsilon)
            print("Working resids:", working_resids)
            print("Working params:", working_p.params)
            print('Current alpha range (min, test, max): ', (alpha_min, alpha_test, alpha_max))
            print('Current convergence criterion (alpha_max-alpha_min)/2: ', convergence_crit)
            print('Time spent in this LS iteration:', (time.time()-t0_ls_iter)/60, "Minutes."'\n')

            hdu = fits.PrimaryHDU(working_resids)
            hdu.writeto(test_image_dir+'LS_Residuals_'+str(k)+'.fits', overwrite=True)

            if working_epsilon < best_epsilon:
                best_epsilon = working_epsilon
                best_p = copy.deepcopy(working_p)
                best_psi=working_psi

            # if convergence_crit < tol: # KL this was an arbitrary choice
            #     print("Linear search convergence via crit<", tol, " in ", k, " iterations and ",
            #           (time.time() - t0_ls_iter) / 60, "Minutes.")
            #     return best_p, best_psi

            if d_cost > 0:
                alpha_max = alpha_test
                continue  # go to next iteration
            elif d_cost < 0:
                alpha_min = alpha_test
                continue
            else:
                print("Linear search convergence via d_cost=0 in ", k, " iterations and ",
                      (time.time()-t0_ls_iter)/60, "Minutes.")
                return best_p, best_psi



                # new_p = copy.deepcopy(working_p)
            # new_p.params = working_params
            #
            # new_epsilon = working_epsilon
            # new_psi = working_psi

            # if working_epsilon < best_epsilon:
                # best_p = copy.deepcopy(working_p)
            #     best_epsilon = working_epsilon
                # best_psi = working_psi
            # else:
            #     break
            # print(f'Linear search iteration {k}: Δε = {best_epsilon:.4e}, '
            #       f'Time = {(time.time() - t0_ls_iter) / 60:.4f} Minutes')

        return best_p, best_psi


    # Conjugate Gradient Descent
    def conjugate_gradient(p, f, f_prime, tol=1e-5, max_iter=100, alpha=0.1):
        """
        :param p: p is a parameters object
        :param tol:
        :param max_iter:
        :param f: function to use for cost function
        :param f_prime: derivative of f. KL: eventually f should dictate f prime
        :param alpha: magnitude of direction steps to take in linear search function
        :return:
        """
        print('Starting conjugate gradient optimization\n')

        # Initialize variables
        grad_prev = None  # No previous gradient initially
        direction = None  # No initial direction

        t_start_cost = time.time()
        print('Starting initial cost function')
        global test_image_dir
        test_image_dir = 'test_images/'+str(0)+'/'
        psi = cost_function(p, f)[1]
        print('Minutes in initial cost function: ', (time.time() - t_start_cost)/60, '\n')
        sys.stdout.flush()

        for i in range(max_iter):
            print("\nCG Iteration:", i+1)
            if not os.path.exists('test_images/'+str(i+1)):
                os.makedirs('test_images/'+str(i+1))
            test_image_dir = 'test_images/' + str(i+1) + '/'
            t_start_CG_iter = time.time()

            # Compute the gradient
            grad, gr_term1, gr_term2 = residual_function(psi, f_prime, extrareturn=True)
            #if i==0:
            #    hdu_ = fits.PrimaryHDU(np.stack((grad,gr_term1,gr_term2)))
            #    hdu_.writeto('grterms.fits', overwrite=True)
            #    del hdu_
            del gr_term1, gr_term2
            print('Minutes spent in residual function:', (time.time() - t_start_CG_iter) / 60)
            sys.stdout.flush()

            # Compute the norm of the gradient
            current_norm = np.linalg.norm(grad)

            if i == 0:
                print('Initial gradient: ', grad)
                norm_0 = np.linalg.norm(grad)
                print('Initial norm: ', norm_0)
                tol = tol * norm_0
                direction = -grad
                #direction = -grad/(np.linalg.norm(grad) +1e-10) # First direction is negative grad
                beta = 0  # First beta is zero
            else:
                beta = np.sum(np.square(grad)) / np.sum(np.square(grad_prev))  # Calculate beta (direction scaling)
                print('Current Beta: ', beta)
                # direction_prev = direction  # set previous direction
                direction = -grad + beta * direction_prev

            if current_norm < tol:
                print('\nConvergence reached at iteration:', i + 1)
                break

            # Perform linear search
            t_start_LS = time.time()
            print('\nInitiating linear search in direction: ', direction)
            p_new, psi_new = linear_search(p, direction, f, f_prime)
            print('Minutes spent in linear search: ', (time.time() - t_start_LS) / 60)
            print('Current norm: ', current_norm, 'Tol * Norm_0: ', tol, 'Difference (CN-TOL): ', current_norm - tol)
            print('Current best params: ', p_new.params)

            # Update to current values
            p = p_new
            psi = psi_new
            grad_prev = grad
            direction_prev = direction

            print('\nMinutes spent in this CG iteration: ', (time.time()-t_start_CG_iter)/60)
            sys.stdout.flush()

            if i==max_iter-1:
                print('\nCG reached max iterations and did not converge.')

        print('\nConjugate gradient complete. Finished in ', i+1, '/', max_iter, ' iterations with tolerance', tol)
        print('Final parameters:', p.params)
        print('Final norm:', current_norm)
        return p


    # Initialize parameters
    p0 = parameters('constant', 4088)

    # Do it
    p = conjugate_gradient(p0, quadratic, quad_prime)
    hdu = fits.PrimaryHDU(p.params)
    hdu.writeto(tempfile + filter + '/' + 'final_params.fits', overwrite=True)
    print(tempfile + filter + '/' + 'final_params.fits created \n')

    for i,sca in enumerate(all_scas):
        obsid, scaid = get_ids(sca)
        this_sca = sca_img(obsid, scaid)
        this_param_set = p.forward_par(i)
        ds_image = this_sca.image - this_param_set

        header = this_sca.w
        hdu = fits.PrimaryHDU(ds_image, header=header)
        hdu.writeto(tempfile + filter + '/DS_' + obsid + scaid + '.fits', overwrite=True)

    print('Destriped images saved to' + tempfile + filter + '/DS_*.fits \n')
    print('Total hours elapsed: ', (time.time() - t0)/3600)

if __name__=='__main__':
    main()
