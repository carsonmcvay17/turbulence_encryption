import jax_cfd
import matplotlib.image as mpi
import matplotlib.pyplot as plt
import numpy as np
import os 
import math

import functools
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax_cfd.base import equations
from jax_cfd.base import filter_utils
from jax_cfd.base import grids
from jax_cfd.base import validation_problems

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]


import matplotlib.image as mpi
import matplotlib.pyplot as plt
import numpy as np
import os 
import math

class FourierTransform:
    """
    A class for fourier transforming images and band pass filtering 
    the spectra in order to get forcing functions
    """
    def _init_(self, alpha=0.2):
        self.alpha = alpha

    def circle_filter(self, input):
        """
        Takes in an image, ft and filters out high and low spectra in a circle
        Inverse transforms the filtered image
        Returns the filtered image in real space
        """

        ft_img = np.fft.fft2(input)
        # remove high frequency info
        # for some reason it throws a fit when I have the keep_frac=self.alpha 
        # this is the temporary solution I guess
        keep_frac = .2
        ft_img2 = ft_img.copy()
        [row,col] = ft_img2.shape

        # now we gotta make circles :(

        # find the midpoint
        mid = [row // 2, col //2]
        # find radius
        area = row*col
        r = round(np.sqrt(area // math.pi))

        # delete everything within the radius
        for i in range(row):
            for j in range(col):
                if (i-mid[0])**2 + (j-mid[1])**2 <= r**2:
                    ft_img2[i,j] = 0


        # Get the new matrix by subtracting the cut out 
        ft_img = ft_img-ft_img2
        img = np.fft.ifft2(ft_img)
        return img
    
    def amp_filter(self, input):
        """
        Takes in an image, ft and filters out the highest and lowest 20% 
        of amplitudes. 
        Returns the filtered image in real space
        """
        ft_img = np.fft.fft2(input)
        # create an empty array of the amplitudes
        # array size is the same as ft_img
        amp_array = np.zeros([ft_img.shape[0], ft_img.shape[1]])
        # look at rows and cols
        for i in range(ft_img.shape[0]):
            for j in range(ft_img.shape[1]):
                real = ft_img[i,j].real
                imaginary = ft_img[i,j].imag
                amplitude = np.arctan(abs(imaginary/real))
                amp_array[i,j] = amplitude
        #find the threshold values
        lower = np.percentile(amp_array, self.alpha*100)
        upper = np.percentile(amp_array,100-(self.alpha*100))
        # look at rows and cols
        for i in range(amp_array.shape[0]):
            for j in range(amp_array.shape[1]):
                if amp_array[i,j] < lower or amp_array[i,j] > upper:
                    # if the amplitude isn't in the middle, delete from fourier space
                    ft_img[i,j] = 0
        
        # inverse transform
        img = np.fft.ifft2(ft_img)
        return img
    
    def prep_function(self):
        img = mpi.imread('/Users/carsonmcvay/Desktop/GradSchool/Research/turbulence_encryption/test_images/dsc_2.jpg')
        img = img[:,:,:3].mean(axis=2)
        img2 = self.circle_filter(img)
        # img2 = self.amp_filter(img)
        
        return img2 

    
    


    





class Forcings(FourierTransform):
    """
    A class for creating a forcing function and applying it in a grid
    """
    def __init__(self, alpha=0):
        # idk what we need here yet
        self.alpha = alpha

    def kolmogorov_focing(
            grid: grids.Grid, 
            scale: float = 1, 
            k: int = 2, 
            swap_xy: bool = False, 
            offsets: Optional[Tuple[Tuple[float, ...], ...]] = None, 
    )   -> ForcingFn:
        """Returns the Kolmogorov focing function for turbulence in 2D"""

        # probably not the cleanest way to do this
        model = FourierTransform()
        forfun = model.prep_function()


        if offsets is None:
            offsets = grid.cell_faces

        if swap_xy:
            x = grid.mesh(offsets[1])[0]
            # v = scale * grids.GridArray(jnp.sin(k*x), offsets[1], grid)
            v = scale * grids.GridArray(forfun, offsets[1], grid)

            if grid.ndim == 2:
                u = grids.GridArray(jnp.zeros_like(v.data), (1, 1/2), grid)
                f = (u, v)
            elif grid.ndim == 3:
                u = grids. GridArray(jnp.zeros_like(v.data), (1, 1/2, 1/2), grid)
                w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
                f = (u, v, w)
            else:
                raise NotImplementedError
        else:
            y = grid.mesh(offsets[0])[1]
            # u = scale * grids.GridArray(jnp.sin(k*y), offsets[0], grid)
            u = scale * grids.GridArray(forfun, offsets[0], grid)

            if grid.ndim == 2:
                v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1), grid)
                f = (u,v)
            elif grid.ndim == 3:
                v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1, 1/2), grid)
                w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
                f = (u, v, w)
            else: 
                raise NotImplementedError
            
        
        def forcing(v):
            del v
            return f 
        return forcing
              
                


 
    
        

