import matplotlib.image as mpi
import matplotlib.pyplot as plt
import numpy as np

class FourierTransform:
    """
    A class for fourier transforming images and band pass filtering 
    the spectra in order to get forcing functions
    """
    def _init_(self, alpha=.2):
        self.alpha = alpha

    def calculate_2dft(self, input):
        """
        Calculates the fourier transform of an image and returns the image in fourier space
        """
        ft = np.fft.ifftshift(input)
        ft = np.fft.fft2(ft)
        return np.fft.fftshift(ft)
    
    def filter(self, input):
        """
        Takes in the result of calculate_2dft and filters high and low frequency spectra
        Returns the filtered image in fourier space
        """
        ft_img = input
        # remove high frequency info
        keep_frac = 0.2
        ft_img2 = ft_img.copy()
        [row,col] = ft_img2.shape

        # cut out high frequency
        # rows
        ft_img2[int(row*keep_frac):int(row*(1-keep_frac)),:] = 0
        # cols
        ft_img2[:,int(col*keep_frac):int(col*(1-keep_frac))] = 0 

        # Get the new matrix by subtracting the cut out 
        ft_img = ft_img-ft_img2
        return ft_img
    
    def amp_filter(self, input):
        ft_img = input
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
        # find the threshold values
        lower = np.percentile(amp_array,self.alpha)
        upper = np.percentile(amp_array,)
        # look at rows and cols
        for i in range(amp_array.shape[0]):
            for j in range(amp_array.shape[1]):
                if amp_array[i,j] < lower or amp_array[i,j] > upper:
                # if the amplitude isn't in the middle, delete from fourier space
                    ft_img[i,j] = 0
        return ft_img
    
    def inv_fft(self, result):
        """
        Takes in the result of filter and then performs the 
        inverse fft 
        Returns the image in normal space
        """
        ft_img = result
        img = np.fft.ifft2(ft_img)
        return img
    
    


