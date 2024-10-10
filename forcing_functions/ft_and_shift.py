import matplotlib.image as mpi
import matplotlib.pyplot as plt
import numpy as np
import os 

class FourierTransform:
    """
    A class for fourier transforming images and band pass filtering 
    the spectra in order to get forcing functions
    """
    def _init_(self, alpha=20):
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
        upper = np.percentile(amp_array,100-self.alpha)
        # look at rows and cols
        for i in range(amp_array.shape[0]):
            for j in range(amp_array.shape[1]):
                if amp_array[i,j] < lower or amp_array[i,j] > upper:
                # if the amplitude isn't in the middle, delete from fourier space
                    ft_img[i,j] = 0
        return ft_img
    
    
    def amp_filter_circle(self, input):
        """
        Takes in a FT image and band pass filters the spectra
        Returns the filtered image

        """
        ft_img = input
        # create an empty array of the amplitudes
        # array size is the same as ft_img
        amp_array = np.zeros([ft_img.shape[0], ft_img.shape[1]])
        # look at rows and cols
        for i in range(ft_img.shape[0]):
            for j in range(ft_img.shape[1]):
                i_comp = ft_img[i,j].real
                j_comp = ft_img[i,j].imag
                circle = i_comp**2 + j_comp**2
                amp_array[i,j] = circle
            
        #find the threshold values
        lower = np.percentile(amp_array,self.alpha)
        upper = np.percentile(amp_array,100-self.alpha)
        #   look at rows and cols
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
    
    def loop_files(self):
        """
        Loops over all the files in the test images folder
        and turns them black and white and performs the ft and 
        filtering
        Returns the inverse transformed filtered image
        """
        path = '/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/Test_Images'
        files = os.listdir(path)

        for filename in files[0:3]:
            img = os.path.join(path,filename)
            img = mpi.imread(img)
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
            # img = img[:,:,:3].mean(axis=2)
            input = self.calculate_2dft(img)
            amp_array = self.amp_filter(input)
            # result = filter(input)
            img2 = self.inv_fft(amp_array)
            plt.subplot(121)
            plt.imshow(np.log(abs(img2)),cmap='gray')
            plt.subplot(122)
            plt.imshow(img,cmap='gray')
            plt.show()

img = FourierTransform.calculate_2dft('/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/Test_Images/imagetwo')
filt_img = FourierTransform.amp_filter_circle(img)
output = FourierTransform.inv_fft(filt_img)
plt.imshow(np.log(abs(output)),cmap='gray')
plt.imshow(img,cmap='gray')
plt.show()

    
    


