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

    def trying_something(self):
        test = 4
        return test 
    
img = FourierTransform()
test = img.prep_function()
print(test[0,3])
# also throwing a fit about plotting complex numbers but whatever
# plt.imshow(np.log(test), cmap='gray')

    


