import matplotlib.image as mpi
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def calculate_2dft(input):
    ft = np.fft.fft2(input)
    return ft
    # return np.fft.fftshift(ft)








def filter(input):
    """
    Takes in FT transformed image and filters high and low frequency spectra
    """
    ft_img = input
    # remove high frequency info
    keep_frac = 0.2
    ft_img2 = ft_img.copy()
    ft_img3 = ft_img.copy()
    [row,col] = ft_img2.shape

    # cut out frequencies
    # rows
    ft_img2[int(row*keep_frac):int(row*(1-keep_frac)),:] = 0
    # cols
    ft_img2[:,int(col*keep_frac):int(col*(1-keep_frac))] = 0 

    # Get the new matrix by subtracting the cut out 
    ft_img = ft_img-ft_img2
    return ft_img

def circle_filter(input):
    """
    Takes in a FT image and filters out high and low spectra in a circle
    Returns the filtered image
    """
    ft_img = input
    # remove high frequency info
    keep_frac = 0.2
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
    return ft_img



def inv_fft(result):
    """
    Takes in the result of filter and then performs the 
    inverse fft 
    """
    ft_img = result
    img = np.fft.ifft2(ft_img)
    return img

def amp_filter(input):
    """
    Takes in the transformed image and filters out the highest and lowest 20% 
    of amplitudes. Returns the filtered fourier transform
    """
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
    #find the threshold values
    lower = np.percentile(amp_array,20)
    upper = np.percentile(amp_array,80)
    # look at rows and cols
    for i in range(amp_array.shape[0]):
        for j in range(amp_array.shape[1]):
            if amp_array[i,j] < lower or amp_array[i,j] > upper:
              # if the amplitude isn't in the middle, delete from fourier space
              ft_img[i,j] = 0
    return ft_img

def amp_filter_circle(input):
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
    lower = np.percentile(amp_array,20)
    upper = np.percentile(amp_array,80)
    # look at rows and cols
    for i in range(amp_array.shape[0]):
        for j in range(amp_array.shape[1]):
            if amp_array[i,j] < lower or amp_array[i,j] > upper:
              # if the amplitude isn't in the middle, delete from fourier space
              ft_img[i,j] = 0
    return ft_img

              

def loop_files():
    path = '/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/Test_Images'
    files = os.listdir(path)

    for filename in files[0:3]:
        img = os.path.join(path,filename)
        img = mpi.imread(img)
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        # img = img[:,:,:3].mean(axis=2)
        input = calculate_2dft(img)
        amp_array = amp_filter(input)
        result = filter(input)
        img2 = inv_fft(amp_array)
        plt.subplot(121)
        plt.imshow(np.log(abs(img2)),cmap='gray')
        plt.subplot(122)
        plt.imshow(img,cmap='gray')
        plt.show()







def plot():
    img = mpi.imread('/Users/carsonmcvay/Desktop/GradSchool/Research/turbulence_encryption/test_images/mixed_frequency.png')
    img = img[:,:,:3].mean(axis=2)
    input = calculate_2dft(img)
    amp_array = amp_filter(input)
    result = circle_filter(input)
    # img2 = inv_fft(amp_array)  
    img2 = inv_fft(result) 
    return img2  

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a    
    
def prep_plot():
    img = plot()
    tup = totuple(img)
    abs_tuple = tuple(np.abs(x) for x in tup)
    plt.imshow(np.log(abs_tuple), cmap='gray')
    plt.show()


# img = mpi.imread('/Users/carsonmcvay/Desktop/GradSchool/Research/turbulence_encryption/test_images/imagetwo.jpg')
# img = img[:,:,:3].mean(axis=2)
# input = calculate_2dft(img)
# amp_array = amp_filter_circle(input)
# result = filter(input)
# img2 = inv_fft(amp_array)
# plt.subplot(121)
# plt.imshow(np.log(abs(img2)), cmap='gray')
# plt.subplot(122)
# plt.imshow(img, cmap='gray')
# plt.show()

# # plt.imshow(img)
# plt.show()

# img2 = mpi.imread('/Users/carsonmcvay/Desktop/GradSchool/Research/test_images/imagetwo.jpg')
# img2 = img2[:,:,:3].mean(axis=2)
# input2 = calculate_2dft(img2)
# amp_array2 = amp_filter_circle(input2)
# result2 = filter(input2)
# img22 = inv_fft(amp_array2)
# plt.subplot(121)
# plt.imshow(np.log(abs(img22)))
# plt.subplot(122)
# plt.imshow(img)

# # plt.imshow(img)
# plt.show()








# img = mpi.imread('/Users/carsonmcvay/Desktop/GradSchool/Research/test_images/imageone.jpg')
# img = img[:,:,:3].mean(axis=2)
# plt.set_cmap("gray")

# ft = calculate_2dft(img)

# plt.subplot(121)
# plt.imshow(img)
# plt.axis("off")
# plt.subplot(122)
# plt.imshow(np.log(abs(ft)))
# plt.axis("off")
# plt.show()





