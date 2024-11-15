"""
Program to remove correlated noise stripes from Roman ST images.
KL To do list:
- Link with config file
- Write outputs to file instead of print
"""

#import os
import glob
import time
import ctypes
import numpy as np
from astropy.io import fits
from astropy import wcs
from utils import compareutils
import re
import sys
import pyimcom_croutines

# KL: Placeholders, some of these should be input arguments or in a config or something
#input_dir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/simple/'
input_dir = '/fs/scratch/PCON0003/klaliotis/destripe/inputs/'
image_prefix = 'Roman_WAS_simple_model_'
labnoise_prefix = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise/slope_'
filter = 'H158'
model_params = {'constant': 1, 'linear': 2}
permanent_mask = '/users/PCON0003/cond0007/imcom/coadd-test-fall2022/permanent_mask_220730.fits'
outfile = '/fs/scratch/PCON0003/klaliotis/destripe/test_out/destripe_' + filter + '_out.txt'
tempfile = '/fs/scratch/PCON0003/klaliotis/destripe/test_out'  # temporary temp file so that i can check these things
# tempfile = '/tmp/klaliotis-tmp/'
s_in = 0.11  # arcsec^2
t_exp = 154  # sec
A_eff = 7340  # cm ^2
t0 = time.time()


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
        mask: the full pixel mask that is used on this image. Is correct only after running BOTH apply mask methods
        g_eff : effective gain in each pixel of the image
    Functions:
    apply_noise: apply the appropriate lab noise frame to the SCA image
    apply_permanent_mask: apply the SCA permanent pixel mask to the image
    apply_object_mask: mask out bright objects from the image
    """

    def __init__(self, obsid, scaid, interpolated=False):
        if interpolated:
            file = fits.open(tempfile + filter + '/' + obsid + '_' + scaid + '_interp.fits')
        else:
            file = fits.open(input_dir + image_prefix + filter + '_' + obsid + '_' + scaid + '.fits')
        self.image = np.copy(file['SCI'].data)
        self.shape = np.shape(self.image)
        self.w = wcs.WCS(file['SCI'].header)
        file.close()

        self.obsid = obsid
        self.scaid = scaid
        self.mask = np.ones(self.shape)

        # Calculate effecive gain
        self.g_eff = np.memmap(tempfile + obsid+'_'+scaid+'_geff.dat', dtype='float16', mode='w+', shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                derivative_matrix = wcs.utils.local_partial_pixel_derivatives(self.w, i, j)
                det = np.linalg.det(derivative_matrix)
                dec = self.w.pixel_to_world(i,j).dec.value
                self.g_eff[i,j] = det * np.cos(np.deg2rad(dec))

    def apply_noise(self):
        """
        Add detector noise to self.image
        :return:
        """
        noiseframe = np.copy(fits.open(labnoise_prefix + self.obsid + '_' + self.scaid + '.fits')['PRIMARY'].data)
        self.image += noiseframe[4:4092, 4:4092]
        return self.image

    def apply_permanent_mask(self):
        """
        Apply permanent pixel mask. updates self.image and self.mask
        :return:
        """
        pm = np.copy(fits.open(permanent_mask)[0].data[int(self.scaid) - 1])
        self.image = self.image * pm
        self.mask = self.mask * pm
        return self.image

    def apply_object_mask(self):
        """
        Apply bright object mask. updates self.image and self.mask
        :return:
        """
        median = np.median(self.image)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.image[i, j] >= 1.5 * median:
                    self.image[i - 2:i + 2, j - 2:j + 2] = 0
                    self.mask[i - 2:i + 2, j - 2:j + 2] = 0
        return self.image

    def get_coordinates(self):
        """
        create an array of ra, dec coords for the image
        :return: coords, an array of (ra, dec) pairs
        """
        wcs = self.w
        h = self.shape[0]
        w = self.shape[1]
        x_i, y_i = np.meshgrid(np.arange(h), np.arange(w))
        x_flat = x_i.flatten()
        y_flat = y_i.flatten()
        ra, dec = wcs.all_pix2world(x_flat, y_flat, 0)  # 0 is for the first frame (1-indexed)
        coords = np.column_stack((ra, dec))
        return coords


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
        self.params = np.reshape(self.params, ((len(all_scas), self.n_rows * self.params_per_row)))
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
    return all_scas, all_wcs

def interpolate_image_bilinear(image_B, image_A, interpolated_image_B):
    """
    Interpolate values from a "reference" SCA image onto a "target" SCA coordinate grid
    Uses pyimcom_croutines.bilinear_interpolation(float* image, float* g_eff, int rows, int cols, float* coords,
                                                    int num_coords, float* interpolated_image)
    :param image_B : an SCA object of the image to be interpolated
    :param image_A : an SCA object of the image whose grid you are interpolating B onto
    :param interpolated_image_B : an ndarray of zeros with shape of Image A.
    interpolated_image_B is updated in-place.
    """
    x_target, y_target, is_in_ref = compareutils.map_sca2sca(image_A.w, image_B.w, pad=0)
    coords = np.column_stack((x_target, y_target)).flatten()
    pyimcom_croutines.bilinear_interpolation(image_B.image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                             image_B.g_eff.ctypes.data_as(
                                                 ctypes.POINTER(ctypes.c_float)),
                                             image_B.shape[0], image_B.shape[1],
                                             coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                             coords.shape[0],
                                             interpolated_image_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))


def transpose_interpolate( image_A, wcs_A, image_B, interpolated_image):
     """
     Interpolate backwards from image_A to image B space
     bilinear_transpose(float* image, int rows, int cols, float* coords, int num_coords, float* original_image)
     :return:
     """
     x_target, y_target, is_in_ref = compareutils.map_sca2sca(image_B.w, wcs_A, pad=0)
     coords = np.column_stack((x_target, y_target)).flatten()
     pyimcom_croutines.bilinear_transpose(image_A.image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            image_B.shape[0], image_B.shape[1],
                                            coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            coords.shape[0],
                                            interpolated_image.ctypes.data_as(
                                                               ctypes.POINTER(ctypes.c_float)))


def transpose_par(array):
    """
    transpose interpolates a params image (sums across rows)
    :param array: input array
    :return:  1D vec of N_params
    """
    return np.sum(array, axis=1)


def get_effective_gain(sca):
    """
    retrieve the effective gain and n_eff of the image. valid only for interpolated images
    :param sca: whose info you want
    :return: g_eff: memmap array of the effective gain in each pixel
    :return: N_eff: memmap array of how many image "B"s contributed to that interpolated image
    """
    m = re.search(r'_(\d+)_(\d+)', sca)
    obsid = m.group(1)
    scaid = m.group(2)
    g_eff = np.memmap(tempfile + obsid+'_'+scaid+'_geff.dat', dtype='float16', mode='r', shape=(4088,4088) )
    N_eff = np.memmap(tempfile + obsid + '_' + scaid + '_Neff.dat', dtype='float16', mode='r', shape=(4088,4088))
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
print(len(all_wcs), "WCS in the list (if not same as above, we have a problem)")

ovmat_t0 = time.time()
print('Overlap matrix computing start')
ov_mat = compareutils.get_overlap_matrix(all_wcs, verbose=True)  # an N_wcs x N_wcs matrix containing fractional overlap
print("Overlap matrix complete. Duration: ", (ovmat_t0-time.time())/3600, 'hours' )
# hdu = fits.PrimaryHDU(ov_mat)
# hdu.writeto(tempfile + filter + '/' + 'overlap_matrix.fits', overwrite=True)
# print(tempfile + filter + '/' + 'overlap_matrix.fits saved to tempfile \n')

# Iterate through all the SCAs and create interpolated
# versions of them from all the other SCAs that overlap them
def make_interpolated_images():
    for i, sca_a in enumerate(all_scas):
        m = re.search(r'_(\d+)_(\d+)', sca_a)
        obsid_A = m.group(1)
        scaid_A = m.group(2)
        print('Img A: ' + obsid_A + '_' + scaid_A)
        I_A = sca_img(obsid_A, scaid_A)
        I_A.apply_noise()
        I_A.apply_permanent_mask()
        I_A.apply_object_mask()

        I_A_interp = np.zeros(I_A.shape)
        N_eff = np.memmap(tempfile + obsid_A + '_' + scaid_A + '_Neff.dat', dtype='float16', mode='w+', shape=I_A.shape)
        t_a_start = time.time()
        print('Starting interpolation for SCA A: ', t_a_start)
        sys.stdout.flush()

        for j, sca_b in enumerate(all_scas):
            m = re.search(r'_(\d+)_(\d+)', sca_b)
            obsid_B = m.group(1)
            scaid_B = m.group(2)

            if obsid_B != obsid_A and ov_mat[i, j] != 0: # Check if this sca_b overlaps sca_a
                I_B = sca_img(obsid_B, scaid_B)
                I_B.apply_noise()
                I_B.apply_permanent_mask()
                I_B.apply_object_mask()
                interpolated_image = np.zeros_like(I_A.image)
                interpolate_image_bilinear(I_B, I_A, interpolated_image)
                I_A_interp += interpolated_image
                N_eff += I_B.mask

        header = I_A.w.to_header()
        hdu = fits.PrimaryHDU(np.divide(np.divide(I_A_interp, N_eff), I_A.g_eff()), header=header)
        hdu.writeto(tempfile + filter + '/' + obsid_A + '_' + scaid_A + '_interp.fits', overwrite=True)
        print(tempfile + filter + '/' + obsid_A + '_' + scaid_A + '_interp.fits created \n')
        t_elapsed_a = time.time() - t_a_start
        print('Time to generate this SCA: ', t_elapsed_a)
        print('Remaining SCAs: ' + str(len(all_scas) - 1 - i) + '\n')


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

#function_dictionary = {"quad": quadratic, "abs": absolute_value, "quadloss": quadratic_loss}

# Optimization Functions

def cost_function(p, f):
    """
    p: params vector; shape is n_rows * n_scas * n_params_per_row.
        p should be a parameters object
    f: keyword for function dictionary options; should also set an f_prime
    """
    psi = np.zeros((len(all_scas), 4088, 4088))
    epsilon = np.copy(psi)

    for i, sca_a in enumerate(all_scas):
        m = re.search(r'_(\d+)_(\d+)', sca_a)
        obsid_A = m.group(1)
        scaid_A = m.group(2)
        I_A = sca_img(obsid_A, scaid_A)
        I_A.apply_noise()
        I_A.apply_permanent_mask()
        I_A.apply_object_mask()

        params_mat_A = p.forward_par(i, I_A.shape)
        I_current = I_A.image - params_mat_A

        if i == 0:
            make_interpolated_images()

        J_A = sca_img(obsid_A, scaid_A, interpolated=True)
        J_current = J_A.image - params_mat_A
        psi[i, :, :] = I_current - J_current
        epsilon[i, :, :] = f(psi[i])
    return epsilon, psi


def residual_function(psi, f_prime):
    """
    Calculate the residuals.
    :param psi: the image difference array (I_A - J_A) (N_SCA, 4088, 4088)
    :param f_prime:
    :return:
    """
    resids = parameters('constant', 4088).params
    for i, sca_a in enumerate(all_scas):
        obsid_A , scaid_A = get_ids(sca_a)
        file = fits.open(tempfile + filter + '/' + obsid_A + '_' + scaid_A + '_interp.fits')
        wcs_A = wcs.WCS(file[0].header)
        file.close()

        gradient_interpolated = f_prime(psi[i])
        term_1 = transpose_par(gradient_interpolated)

        g_eff_A, n_eff_A = get_effective_gain(sca_a)
        gradient_interpolated = gradient_interpolated / g_eff_A / n_eff_A

        for j, sca_b in enumerate(all_scas):
            obsid_B, scaid_B = get_ids(sca_b)
            g_eff_B = get_effective_gain(sca_a)[0]

            if obsid_B != obsid_A and ov_mat[i, j] != 0:
                I_B = sca_img(obsid_B, scaid_B)
                gradient_original = np.zeros(I_B.shape)
                transpose_interpolate(gradient_interpolated, wcs_A, I_B, gradient_original)
                gradient_original *= g_eff_B
                term_2 = transpose_par(gradient_original)
                resids[j,:] += term_2

        resids[i, :] -= term_1

    return resids


def linear_search(p, direction, f):
    alpha = 0.1  # Step size
    p.params = p.params + alpha * direction
    best_epsilon, best_psi= cost_function(p, f)

    # Simple linear search
    new_p = parameters('constant', 4088)
    for i in range(1, 11):
        new_p.params = new_p.params + i * alpha * direction
        new_epsilon, new_psi = cost_function(new_p, f)
        if new_epsilon < best_epsilon:
            p = new_p
            best_epsilon = new_epsilon
            best_psi = new_psi
    return p, best_psi


# Conjugate Gradient Descent
def conjugate_gradient(p, f, f_prime, tol=1e-5, max_iter=100):
    """
    :param p: p is a parameters object
    :param tol:
    :param max_iter:
    :param f: function to use for cost function
    :param f_prime: derivative of f. KL: eventually f should dictate f prime
    :return:
    """
    direction = np.copy(p.params)
    grad_prev = np.copy(p.params)
    t_start = time.time()
    psi = cost_function(p, f)[1]
    print('Hours in initial cost function: ', (time.time() - t_start)/3600)
    sys.stdout.flush()

    for _ in range(max_iter):
        t_start = time.time()
        grad = residual_function(psi, f_prime)
        print('Hours spent in residual function: ', (time.time() - t_start) / 3600)
        sys.stdout.flush()

        if _ == 0:
            norm_0 = np.linalg.norm(grad)

        if np.linalg.norm(grad) < tol * norm_0:
            break

        beta = np.square(grad) / np.square(grad_prev)
        direction = -grad + beta * direction

        # Perform linear search
        t_start = time.time()
        p_new, psi_new= linear_search(p, direction, f)
        print('Hours spent in residual function: ', (time.time() - t_start) / 3600)
        sys.stdout.flush()

        p = p_new
        psi = psi_new
        grad_prev = grad

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
    this_param_set =  p.forward_par(i)
    ds_image = this_sca.image - this_param_set

    header = this_sca.w
    hdu = fits.PrimaryHDU(ds_image, header=header)
    hdu.writeto(tempfile + filter + '/DS_' + obsid + scaid + '.fits', overwrite=True)

print('Destriped images saved to' + tempfile + filter + '/DS_*.fits \n')
print('Total hours elapsed: ', (time.time() - t0)/3600)
