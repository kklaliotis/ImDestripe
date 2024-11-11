"""This is a helper module with functions for comparing different SCAs. The SCAs are assumed to be square.

The SCAs are described in terms of astropy WCS objects.

getfootprint: Extracts from the SCA a Mangle cap (Cartesian coordinates of the center and a bounding circle)
map_sca2sca: Gets the (x,y) in one SCA for each pixel in another SCA
overlap_matrix: Computes the Boolean overlap matrix of a list of SCAs
str2dirstem: utility to separate the directory and file name from a stem

"""

import numpy as np
from astropy import wcs
import re

def getfootprint(mywcs, pad):
   """Gets the Cartesian coordinates of the corners.
   This is a numpy array of the form [x,y,z,p], where:
      x,y,z = Cartesian coordinates of center
      p = 1 - cos(theta_max), where theta_max is the maximum distance from the center.
   The corners are padded by the indicated number of native pixels on each axis. The SCA is assumed square.
   """
   nside = mywcs.array_shape[-1]

   hw = nside/2.+pad
   xi = np.array([0, -hw, -hw, hw, hw]) + (nside-1.)/2.
   yi = np.array([0, -hw, hw, -hw, hw]) + (nside-1.)/2.
   ra,dec = mywcs.all_pix2world(xi, yi, 0)
   deg = np.pi/180.
   M = np.stack((np.cos(dec*deg)*np.cos(ra*deg), np.cos(dec*deg)*np.sin(ra*deg), np.sin(dec*deg)), axis=1)
   this_p = np.sum((M-M[0,:][None,:])**2, axis=1)/2. # 1 - cos(angle from center)

   return np.array([M[0,0], M[0,1], M[0,2], np.amax(this_p)])

def map_sca2sca(target_wcs, ref_wcs, pad=0, dtype=np.float64):
   """Finds the pixel mappings from a 'reference' WCS to a 'target' WCS

   Inputs:
      target_wcs: WCS that we want to 'map to' (we will make a map of the full nside x nside region)
      ref_wcs: WCS of the reference exposure that we want to 'map from'
      pad: number of pixels by which to pad the input *and* output exposures (integer)
      dtype: ouput data type for xf and yf (note is_in_ref is always Boolean)

   Returns:
      xpix, ypix: the x and y values corresponding to the reference WCS
      is_in_ref: (boolean) whether that pixel is in the reference exposure
   """
   nside = target_wcs.array_shape[-1]
   xi,yi = np.meshgrid(np.linspace(-pad,nside-1+pad,nside+2*pad), np.linspace(-pad,nside-1+pad,nside+2*pad))
   ra,dec = target_wcs.all_pix2world(xi, yi, 0)
   del xi,yi
   xf,yf = ref_wcs.all_world2pix(ra,dec,0)
   del ra,dec
   is_in_ref = np.logical_and((xf+.5+pad)*(nside-.5-xf+pad)>0, (yf+.5+pad)*(nside-.5-yf+pad)>=0)
   return xf.astype(dtype), yf.astype(dtype), is_in_ref

def get_overlap_matrix(list_of_wcs, pad=0, verbose=False):
   """ Computes the fractional overlap matrix of a list of WCSs.
   Extends the boundaries by pad pixels on each side.
   For N WCSs, this is an NxN symmetric matrix.

   """
   N = len(list_of_wcs)
   if verbose: print('get_overlap_matrix:',N,'chips')

   # extract the caps
   caps = np.zeros((N,4))
   for i in range(N): caps[i,:] = getfootprint(list_of_wcs[i], float(pad))
   p = caps[:,-1]
   sep2max = 2*( p[:,None] + p[None,:] - p[:,None]*p[None,:] + np.sqrt(p[:,None]*p[None,:]*(2.-p[:,None])*(2.-p[None,:])) )
      # note: if p1 = 1-cos theta1 and p2 = 1-cos theta2 then
      # 4 sin^2 {(theta1+theta2)/2} = 2 [ p1 + p2 - p1*p2 + SQRT{p1*p2*(2-p1)*(2-p2)} ]
   x = caps[:,:-1]
   sep2 = np.sum((x[:,None,:]-x[None,:,:])**2, axis=2)
   ov = np.where(sep2<sep2max, 1., 0.).astype(np.float64)

   # check candidate overlaps
   for i in range(1,N):
      for j in range(i):
         if ov[i,j]:
            x_,y_,m_ = map_sca2sca(list_of_wcs[i], list_of_wcs[j], pad=pad, dtype=np.float32)
            del x_,y_
            ov[i,j] = np.count_nonzero(m_)/np.size(m_)
            ov[j,i] = ov[i,j]
            if verbose: print('get_overlap_matrix: ->',i,j,ov[i,j])

   return ov

def str2dirstem(st):
   """Splits a string st into a directory and a file stem:
   e.g. if input is 'A/c24/B_' then output is ('A/c24', 'B_')
   """
   if st is None: raise('called str2dirstem with None')
   parts = re.split('/', st)
   N = len(parts)
   if N==1: return('./', st)
   stdir = ''
   for k in range(N-1): stdir += parts[k] + '/'
   return(stdir,parts[-1])

