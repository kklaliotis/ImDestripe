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
img_prefix = 'Roman_WAS_simple_model_'
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
    KL to do: is this all this thing needs?
    """
    def __init__(self):
        self.image = np.copy(fits.open(img_prefix+filter+str(self))[1].data())
        self.shape = np.shape(self.image)
        self.w = wcs.WCS(self.image[1].header)
        self.ra_ctr = self.w.wcs.crval[0]
        self.dec_ctr = self.w.wcs.crval[1]

class observation:
    """
    Observation object specifies a particular Roman WAS observation.
    Attributes:
        ra_ctr: RA coordinate of the observation center
        dec_ctr: dec coordinate of the observation center
        scas: list of the SCAs in this image
        ds_params: destriping parameters for this observation
    Functions:
        which_scas: generate the list of scas that overlap this observation. saved as .scas attr
    KL to do: incorporate ds_params; incorporate SCAs
    """
    def __init__(self, id):
        self.id = id
        self.scas = []

    def which_scas(self):
        for file in os.listdir(input_dir+img_prefix+self.id+'_*'):
            m = re.search(r'(?P<filter>\d+)_(?P<obsid>\d+)_(?P<sca>\d+)', file)
            this_sca = int(m.groupdict()['sca'])
            self.scas.append(this_sca)
            return self.scas

    sca_images = np.zeros((len(self.scas), 4088, 4088)) # KL: initialize the array of SCA images. Idk whats up w this

    def get_sca_imgs(self):
        for i in len(self.scas):
            this_sca = sca_img()




class ds_parameters:
    """
    Class holding the destriping parameters for a given mosaic/image.
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
    def __init__(self, model, observation, scas):
        """

        :param model: destriping model choice
        :param observation: an observation object
        :param scas: a dataframe of sca objects for the input observation
        """
        self.model = model
        self.params = len(observation.which_scas())*scas.shape[0]*model_params[model]
