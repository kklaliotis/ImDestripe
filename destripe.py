"""
Program to remove correlated noise stripes from Roman ST images.
KL To do list:
- ?? Finish the transpose interpolation function
- Link with config file
- Implement conjugate gradient descent solution
- Write outputs

"""
# import os
import glob
import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy import ndimage
import compareutils
import re
import sys

# KL: Placeholders, some of these should be input arguments or in a config or something
input_dir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/simple/'
image_prefix = 'Roman_WAS_simple_model_'
labnoise_prefix = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise/slope_'
filter = 'H158'
model_params = {'constant': 1, 'linear': 2}
permanent_mask = '/users/PCON0003/cond0007/imcom/coadd-test-fall2022/permanent_mask_220730.fits'
outfile = '/fs/scratch/PCON0003/klaliotis/destripe/destripe_'+filter+'_out.txt'
tempfile = '/fs/scratch/PCON0003/klaliotis/destripe/' #temporary temp file so that i can check these things
#tempfile = '/tmp/klaliotis-tmp/'
s_in  = 0.11


def write_to_file(text):
    """
    Function to write some text to an output file
    :param text: a string to print
    :return: nothing
    """
    global outfile
    with open(outfile,"w") as f:
        f.write(text + '\n')
    with open(outfile, "r") as f:
        print(f.readlines())


class sca_img:
    """
    Class defining an SCA image object.
    Attributes:
        scaid: the SCA id (str)
        obsid: the observation id (str)
        ra_ctr: RA coordinate of the SCA image center
        dec_ctr: dec coordinate of the SCA image center
        image: the SCA image (4088x4088) (KL: np array or pd dataframe?_
        shape: shape of the image
        wcs: the astropy.wcs object associated with this SCA
        mask: the full pixel mask that is used on this image. Is correct only after running both apply mask methods
    Functions:
    apply_noise: apply the appropriate lab noise frame to the SCA image
    get_overlap: figure out which other SCA images overlap this one
    apply_permanent_mask: apply the SCA permanent pixel mask to the image
    apply_object_mask: mask out bright objects from the image
    """
    def __init__(self, obsid, scaid):
        file = fits.open(input_dir+image_prefix+filter+'_'+obsid+'_'+scaid+'.fits')
        self.image = np.copy(file['SCI'].data)
        self.shape = np.shape(self.image)
        self.w = wcs.WCS(file['SCI'].header)
        self.ra_ctr = self.w.wcs.crval[0]
        self.dec_ctr = self.w.wcs.crval[1]
        self.obsid = obsid
        self.scaid = scaid
        file.close()
        self.mask = np.ones(self.shape)

    def apply_noise(self):
        noiseframe = np.copy(fits.open(labnoise_prefix+self.obsid+'_'+self.scaid+'.fits')['PRIMARY'].data)
        self.image += noiseframe[4:4092,4:4092]
        return self.image

    def apply_permanent_mask(self):
        pm = np.copy(fits.open(permanent_mask)[0].data[int(self.scaid)-1])
        self.image = self.image * pm
        self.mask = self.mask * pm
        return self.image

    def apply_object_mask(self):
        median = np.median(self.image)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.image[i,j]>=1.5*median:
                    self.image[i-2:i+2,j-2:j+2]=0
                    self.mask[i - 2:i + 2, j - 2:j + 2] = 0
        return self.image


class ds_parameters:
    """
    Class holding the destriping parameters for a given mosaic.
    Attributes:
        model: which destriping model to use, which specifies the number of parameters per row based on the
         model_params dict
        n_rows: number of rows in the image
        params_per_row: number of parameters per row, given by the model
        params: the actual array of parameters.
    Functions:
        params_2_images: reshape params into the 2D array
        flatten_params: reshape params into 1D vector
    """
    def __init__(self, model, n_rows):
        self.model = model
        self.n_rows = n_rows
        self.params_per_row = model_params[str(self.model)]
        self.params = np.zeros((len(all_scas), self.n_rows*self.params_per_row))

    def params_2_images(self):
        self.params = np.reshape(self.params, ((len(all_scas), self.n_rows*self.params_per_row)))
        return self.params

    def flatten_params(self):
        self.params = np.ravel(self.params)
        return self.params

# KL: Adapted from Naim CiC algorithm
# @njit("(f8[:, :], f8[:, :], f8, i8)")
# def Cic_interpolate(
#
#         interpolated_image, imageB_grid, d_xy, n_grid,
#
# ):
#     """Reverse interpolate pixels from Image B onto `interpolated_image`,
#     the pixel grid from Image A (plus some padding).
#
#     Arguments
#
#     ---------
#
#     interpolated_image (np.ndarray): 2D grid of shape
#
#         (n_grid+pad, n_grid+pad). This array is updated in place.
#
#     imageB_grid (np.ndarray): 2D array of shape (n_grid, n_grid). needs to have coordinates of pixels in a form
#         that will be useful to compare with image A
#
#     d_xy (float): Grid size in x and y directions (image A pixel spacing (in arcsec?) I think)
#
#     n_grid (int): Number of grid points for x and y directions. =4088
#
#     """
#
#     dis_r = imageB_grid / np.array([d_xy, d_xy])
#
#     idx_r = dis_r.astype(np.int_)
#
#     dis_r -= idx_r
#
#     idx_r %= np.array([n_grid, n_grid], dtype=np.int_)
#
#     for (x, y), (dx, dy) in zip(idx_r, dis_r):
#         # Next grid points (e.g., x + 1).
#
#         interpolated_image[x, y] += (1 - dx) * (1 - dy) # KL: multiply these by image B pixel value??
#
#         interpolated_image[x, yp] += (1 - dx) * dy
#
#         interpolated_image[xp, y] += dx * (1 - dy)
#
#         interpolated_image[xp, yp] += dx * dy
#
#     return interpolated_image

def get_scas(filter, prefix):
    """
    Function to get an array of SCA images for this mosaic
    :param : None
    :return: numpy array with all the SCA images
    """
    n_scas = 0
    all_scas = []
    all_wcs = []
    for f in glob.glob(input_dir+prefix+filter+'_*'):
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

def interpolate_image(target_wcs, ref_wcs, ref_image):
    """
    Interpolate values from a "reference" SCA image onto a "target" SCA coordinate grid
    :param target_wcs: WCS of the image whose grid you want to interpolate onto
    :param ref_wcs: WCS of the image whose values you want to use
    :param ref_image: the image whose values you want to use
    :return: an image of ref_image interpolated onto target_image grid
    """
    x_target, y_target, is_in_ref = compareutils.map_sca2sca(target_wcs, ref_wcs, pad=0)
    interp_image = ndimage.map_coordinates(ref_image, [[x_target], [y_target]])[0,:,:]
    return interp_image



############################ Main Sequence ############################

all_scas, all_wcs = get_scas(filter, image_prefix)
print(len(all_scas), " SCAs in this mosaic")
print(len(all_wcs), "WCS in the list (if not same as above, we have a problem)")

ov_mat = compareutils.get_overlap_matrix(all_wcs, verbose=True) #an N_wcs x N_wcs matrix containing fractional overlap
print("Overlap matrix complete")
# In this chunk of code, we iterate through all the SCAs and create interpolated
# versions of them from all the other SCAs that overlap them
for i,sca_a in enumerate(all_scas):
    m = re.search(r'_(\d+)_(\d+)', sca_a)
    obsid_A = m.group(1)
    scaid_A = m.group(2)
    print('Img A: ' + obsid_A + '_' + scaid_A)
    I_A = sca_img(obsid_A, scaid_A)
    I_A.apply_noise()
    I_A.apply_permanent_mask()
    I_A.apply_permanent_mask()

    I_A_interp = np.zeros(I_A.shape)
    N_eff = np.zeros(I_A.shape)

    print('Starting interpolation for SCA A')
    sys.stdout.flush()

    for j,sca_b in enumerate(all_scas):
        m = re.search(r'_(\d+)_(\d+)', sca_b)
        obsid_B = m.group(1)
        scaid_B = m.group(2)

        if obsid_B != obsid_A and ov_mat[i,j] != 0:
            I_B = sca_img(obsid_B, scaid_B)
            I_B.apply_noise()
            I_B.apply_permanent_mask()
            I_B.apply_object_mask()
            I_A_interp += interpolate_image(I_A.w, I_B.w, I_B.image)
            N_eff += I_B.mask

    hdu = fits.PrimaryHDU(np.divide(I_A_interp, N_eff))
    hdu.writeto(tempfile+filter+'/'+obsid_A+'_'+scaid_A+'_interp.fits', overwrite=True)
    print(tempfile+filter+'/'+obsid_A+'_'+scaid_A+'_interp.fits created \n')
    print('Remaining SCAs: ' + str(len(all_scas)-1-i) + '\n')




