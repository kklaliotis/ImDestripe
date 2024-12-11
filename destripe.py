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

# KL: Placeholders, some of these should be input arguments or in a config or something
#input_dir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/simple/'
input_dir = '/fs/scratch/PCON0003/klaliotis/destripe/inputs/'
image_prefix = 'Roman_WAS_simple_model_'
labnoise_prefix = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise/slope_'
filter = 'H158'
model_params = {'constant': 1, 'linear': 2}
permanent_mask = '/users/PCON0003/cond0007/imcom/coadd-test-fall2022/permanent_mask_220730.fits'
outfile = '/fs/scratch/PCON0003/klaliotis/destripe/test_out/destripe_' + filter + '_out.txt'
tempfile = '/fs/scratch/PCON0003/klaliotis/destripe/test_out/'  # temporary temp file so that i can check these things
# tempfile = '/tmp/klaliotis-tmp/'
s_in = 0.11  # arcsec^2
t_exp = 154  # sec
A_eff = 7340  # cm ^2
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
    Functions:
    apply_noise: apply the appropriate lab noise frame to the SCA image
    apply_permanent_mask: apply the SCA permanent pixel mask to the image
    """

    def __init__(self, obsid, scaid, interpolated=False):
        """

        :param obsid:
        :param scaid:
        :param interpolated:
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
        self.mask = np.ones(self.shape)

        # Calculate effecive gain
        if not os.path.isfile(tempfile + obsid+'_'+scaid+'_geff.dat'):
            g_eff = np.memmap(tempfile + obsid+'_'+scaid+'_geff.dat', dtype='float32', mode='w+', shape=self.shape)
            ra, dec = self.get_coordinates(pad=2)
            ra = ra.reshape((4090, 4090))
            dec = dec.reshape((4090, 4090))
            derivs = np.array(((ra[1:-1,2:] - ra[1:-1,:-2])/2, (ra[2:, 1:-1] - ra[:-2, 1:-1])/2,
                              (dec[1:-1,2:] - dec[1:-1,:-2])/2, (dec[2:, 1:-1] - dec[:-2, 1:-1])/2))
            derivs_px = np.reshape(np.transpose(derivs), (4088**2, 2, 2))
            det_mat = np.reshape(np.linalg.det(derivs_px), (4088,4088))
            g_eff[:,:] = det_mat * np.cos(np.deg2rad(dec[1:4089,1:4089]))
            del g_eff

        self.g_eff = np.memmap(tempfile + obsid+'_'+scaid+'_geff.dat', dtype='float32', mode='r', shape=self.shape)

    def apply_noise(self):
        """
        Add detector noise to self.image
        KL to do: make an option to write out image of image + noise (for comparison fig)
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
        self.image *= pm
        self.mask *= pm

    def get_coordinates(self, pad=0):
        """
        create an array of ra, dec coords for the image
        :param pad: add padding to the array. default is zero
        :return: coords, an array of (ra, dec) pairs
        """
        wcs = self.w
        h = self.shape[0] + pad
        w = self.shape[1] + pad
        x_i, y_i = np.meshgrid(np.arange(h), np.arange(w))
        x_i -= pad / 2.
        y_i -= pad / 2.
        x_flat = x_i.flatten()
        y_flat = y_i.flatten()
        ra, dec = wcs.all_pix2world(x_flat, y_flat, 0)  # 0 is for the first frame (1-indexed)
        return ra, dec

    def make_interpolated(self, ind):
        """
        Construct an version of this SCA interpolated from other, overlapping ones.
        :return:
        """
        this_interp = np.zeros(self.shape)
        N_eff = np.memmap(tempfile + self.obsid + '_' + self.scaid + '_Neff.dat', dtype='float32', mode='w+', shape=self.shape)
        t_a_start = time.time()
        print('Starting interpolation for SCA' + self.obsid + '_' + self.scaid)
        # print('Time: ', t_a_start)
        # print('Image A Checks: Image nonzero, G_eff nonzero', np.nonzero(self.image), np.nonzero(self.g_eff))
        sys.stdout.flush()

        N_BinA = 0

        for j, sca_b in enumerate(all_scas):
            obsid_B, scaid_B = get_ids(sca_b)

            if obsid_B != self.obsid and ov_mat[ind, j] != 0:  # Check if this sca_b overlaps sca_a
                #print('Image B: ' + obsid_B + '_' + scaid_B)
                this_j = str(j)
                N_BinA += 1
                I_B = sca_img(obsid_B, scaid_B)
                I_B.apply_noise()
                I_B.apply_permanent_mask() # now I_B.mask is the permanent mask
                # I_B.apply_object_mask() # KL took this out bc I think we want to do this after interpolation actually
                B_interp = np.zeros_like(self.image)
                B_mask_interp = np.zeros_like(self.image)
                # print('Image B Checks: Image nonzero, G_eff nonzero', np.nonzero(I_B.image)[0],
                  #    np.nonzero(I_B.g_eff)[0])
                interpolate_image_bilinear(I_B, self, B_interp)
                interpolate_image_bilinear(I_B, self, B_mask_interp, mask=I_B.mask) # interpolate B pixel mask onto A grid
                this_interp += B_interp
                N_eff += B_mask_interp


        print('Interpolation done. Number of contributing SCAs: ', N_BinA)
        new_mask = N_eff > 10**-12
        this_interp = np.where(new_mask, this_interp/np.where(new_mask, N_eff, 10**-12), 0) # only do the division where N_eff nonzero
        header =self.w.to_header()
        this_interp = np.divide(this_interp, self.g_eff)
        hdu = fits.PrimaryHDU(this_interp, header=header)
        hdu.writeto(tempfile + 'interpolations/' + self.obsid + '_' + self.scaid + '_interp.fits', overwrite=True)
        #print(tempfile + 'interpolations/' + self.obsid + '_' + self.scaid + '_interp.fits created \n')
        t_elapsed_a = time.time() - t_a_start
        #print('Minutes to generate this interpolation: ', t_elapsed_a / 60)

        del N_eff
        return this_interp

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
        self.params = np.ones((len(all_scas), self.n_rows * self.params_per_row))
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
    return neighbor_mask

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
     coords = np.column_stack((x_target, y_target)).flatten()

     rows = int(image_B.shape[0])
     cols = int(image_B.shape[1])
     num_coords = coords.shape[0] // 2

     pyimcom_croutines.bilinear_transpose(image_A,
                                            rows, cols,
                                            coords,
                                            num_coords,
                                            original_image)


def transpose_par(array):
    """
    transpose interpolates a params image (sums across rows)
    :param array: input array
    :return:  1D vec of N_params
    """
    return np.sum(array, axis=1)


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


def make_interpolated_images():
    """
    Iterate through all the SCAs and create interpolated
    versions of them from all the other SCAs that overlap them
    :return: None
    """
    for i, sca_a in enumerate(all_scas):
        m = re.search(r'_(\d+)_(\d+)', sca_a)
        obsid_A = m.group(1)
        scaid_A = m.group(2)
        print('Img A: ' + obsid_A + '_' + scaid_A)
        I_A = sca_img(obsid_A, scaid_A)
        I_A.apply_noise()
        I_A.apply_permanent_mask()
        # I_A.apply_object_mask() # KL: I think I don't want to apply the object mask until I compute psi_A
        # KL sidenote I also dont think it matters at this stage whether I_A has been masked ? or whether noise is there?
        # like I think I could take out apply noise and apply permanent mask here.

        I_A.make_interpolated(i)

        #print('SCAs remaining: ', len(all_scas)-i)


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

def cost_function(p, f):
    """
    p: params vector; shape is n_rows * n_scas * n_params_per_row.
        p should be a parameters object
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
        I_A = sca_img(obsid_A, scaid_A)
        I_A.apply_noise()

        params_mat_A = p.forward_par(j)  # Make destriping params into an image
        if obsid_A=='670' and scaid_A=='10':
            print('Info for j: ', j, 'sca: ', sca_a)
            print('Image A: ', I_A.image)
            print('Params vector: ', p.params[j, :])
            print('Params matrix: ', params_mat_A)
        I_A.image = I_A.image - params_mat_A  # Update I_A.image to have the params image subtracted off
        if obsid_A=='670' and scaid_A=='10':
            print('Subtracted Image A: ', I_A.image)
        I_A.apply_permanent_mask()  # Apply permanent mask; Now I_A.mask is the permanent mask
        object_mask = apply_object_mask(I_A.image)

        J_A_image = I_A.make_interpolated(j)  # make_interpolated uses I_A.image so I think this I_A has the params off
        J_A_image *= I_A.mask # apply permanent mask from A
        apply_object_mask(J_A_image, mask=object_mask)

        psi[j, :, :] = I_A.image - J_A_image

        # Compute local epsilon
        local_epsilon = np.sum(f(psi[j, :, :]))

        if j%5==0:
            print('Image A mean, std: ', np.mean(I_A.image), np.std(I_A.image))
            print('Image B mean, std: ', np.mean(J_A_image), np.std(J_A_image))
            print ('Psi mean, std: ', np.mean(psi[j, :, :]), np.std(psi[j, :, :]), )
            print(f"Local epsilon for SCA {j}: {local_epsilon}")

        epsilon += local_epsilon

    print('Ending cost function. Minutes elapsed: ', (time.time()-t0_cost)/60)
    return epsilon, psi


def residual_function(psi, f_prime):
    """
    Calculate the residuals.
    :param psi: the image difference array (I_A - J_A) (N_SCA, 4088, 4088)
    :param f_prime: the function to be used to calculate the gradient.
            in the future this should be set by default based on what you pass for f
    :return: resids, a 2D array with one row per SCA and one col per image-row-parameter
    """
    resids = (parameters('constant', 4088).params) * 0  # params initialize as 1, I think I want this to start at zero
    print('\nResidual calculation started')
    for k, sca_a in enumerate(all_scas):

        # Go and get the WCS object for image A
        obsid_A, scaid_A = get_ids(sca_a)
        file = fits.open(tempfile + 'interpolations/' + obsid_A + '_' + scaid_A + '_interp.fits')
        wcs_A = wcs.WCS(file[0].header)
        file.close()

        # Calculate and then transpose the gradient of I_A-J_A
        gradient_interpolated = f_prime(psi[k, :, :])

        if k==1:  # check
            print('Gradient in A space: ', gradient_interpolated)

        term_1 = transpose_par(gradient_interpolated)

        # Retrieve the effective gain and N_eff to normalize the gradient before transposing back
        g_eff_A, n_eff_A = get_effective_gain(sca_a)

        # Avoid dividing by zero
        valid_mask = n_eff_A != 0
        gradient_interpolated[valid_mask] = gradient_interpolated[valid_mask] / (
                    g_eff_A[valid_mask] * n_eff_A[valid_mask])
        gradient_interpolated[~valid_mask] = 0

        if k==1:  #check
            print('Gradient in A space post-normalization: ', gradient_interpolated)

        for j, sca_b in enumerate(all_scas):
            obsid_B, scaid_B = get_ids(sca_b)

            if obsid_B != obsid_A and ov_mat[k, j] != 0:
                I_B = sca_img(obsid_B, scaid_B)
                gradient_original = np.zeros(I_B.shape)

                transpose_interpolate(gradient_interpolated, wcs_A, I_B, gradient_original)

                gradient_original *= I_B.g_eff
                print('J: ', j, 'gradient in B coordinates: ', gradient_original)

                term_2 = transpose_par(gradient_original)

                resids[j,:] += term_2

                if k==1 :
                    print('K: ', k, 'j: ', j)
                    print('Term 1: ', term_1)
                    print('Term 2: ', term_2)

        resids[k, :] -= term_1
    print('Residuals calculation finished\n')
    return resids


def linear_search(p, direction, f, f_prime, n_iter=50, alpha=0.1):

    # KL: first version of LS using constant direction depth alpha
    # alpha = 0.1  # Step size
    # p.params = p.params + alpha * direction

    best_epsilon, best_psi = cost_function(p, f)

    # Simple linear search
    working_p = copy.deepcopy(p)

    alpha_max = 2**8 / np.max(p.params)
    alpha_min = -alpha_max

    for k in range(1, n_iter):
        t0_ls_iter = time.time()

        if k==1:
            print('\n!!!! Inside linear search function now.')
            print("Direction:", direction)
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

        alpha_test = .5 * (np.abs(alpha_min) + np.abs(alpha_max))

        working_params = working_p.params + alpha_test * direction
        working_p.params = working_params

        working_epsilon, working_psi = cost_function(working_p, f)
        working_resids = residual_function(working_psi, f_prime)

        d_cost = np.sum(working_resids * direction)

        if k==1 or k%5==0:
            print('LS iteration ', k, ': d_cost = ', d_cost, 'epsilon = ', working_epsilon)
            print("Working resids:", working_resids)
            print("Working params:", working_p.params)
            print('Current alpha range (min, test, max): ', (alpha_min, alpha_test, alpha_max))

        if d_cost > 0:
            alpha_max = alpha_test
            continue  # go to next iteration

        elif d_cost < 0:
            alpha_min = alpha_test
            continue

        else:
            print("Linear search convergence in ", k, " iterations and ", (time.time()-t0_ls_iter)/60, "Minutes.")

        # new_p = copy.deepcopy(working_p)
        # new_p.params = working_params
        #
        # new_epsilon = working_epsilon
        # new_psi = working_psi

        # if working_epsilon < best_epsilon:
        best_p = copy.deepcopy(working_p)  # this was indented before
        #     best_epsilon = working_epsilon
        best_psi = working_psi  # this was indented before
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
    psi = cost_function(p, f)[1]
    final_iter = 0.
    print('Minutes in initial cost function: ', (time.time() - t_start_cost)/60, '\n')
    sys.stdout.flush()

    for i in range(max_iter):
        print("CG Iteration: ", i+1)
        t_start_CG_iter = time.time()

        # Compute the gradient
        grad = residual_function(psi, f_prime)
        print('Minutes spent in residual function: ', (time.time() - t_start_CG_iter) / 60)
        sys.stdout.flush()

        # Compute the norm of the gradient
        current_norm = np.linalg.norm(grad)

        # Things for CG Iteration 0 only

        if i == 0:
            print('Initial gradient: ', grad)
            norm_0 = np.linalg.norm(grad)
            print('Initial norm: ', norm_0)
            tol = tol * norm_0
            direction = -grad # First direction is negative grad
            beta = 0 # First beta is zero

        if current_norm < tol:
            print('Convergence reached at iteration:', i + 1)
            break

        if i > 0:
            beta = np.sum(np.square(grad)) / np.sum(np.square(grad_prev))  # Calculate beta (direction scaling)
            print('Beta: ', beta, '\n')
            direction_prev = direction  # set previous direction
            direction = -grad + beta * direction_prev  # update direction

        # Perform linear search
        t_start_LS = time.time()
        print('Initiating linear search in direction: ', direction)
        p_new, psi_new= linear_search(p, direction, f, f_prime)
        print('Minutes spent in linear search: ', (time.time() - t_start_LS) / 60)
        print('Current norm: ', current_norm, 'Tol * Norm_0: ', tol, 'Difference (CN-TOL): ', current_norm - tol)

        # Update to current values
        p = p_new
        psi = psi_new
        grad_prev = grad

        print('Minutes spent in this CG iteration: ', (time.time()-t_start_CG_iter)/60)
        sys.stdout.flush()

    print('Conjugate gradient complete. Finished in ', i+1, '/', max_iter, ' iterations')
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
