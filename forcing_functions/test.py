import matplotlib.image as mpi
import matplotlib.pyplot as plt
import numpy as np

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    #print(ft[0,1])
    return np.fft.fftshift(ft)








def filter(input):
    """
    Takes in FT transformed image and filters high and low frequency spectra
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

def inv_fft(result):
    """
    Takes in the result of filter and then performs the 
    inverse fft 
    """
    ft_img = result
    img = np.fft.ifft2(ft_img)
    return img



img = mpi.imread('/Users/carsonmcvay/Desktop/GradSchool/Research/test_images/imageone.jpg')
img = img[:,:,:3].mean(axis=2)
input = calculate_2dft(img)
result = filter(input)
img = inv_fft(result)
plt.imshow(np.log(abs(img)))
# plt.imshow(img)
plt.show()










img = mpi.imread('/Users/carsonmcvay/Desktop/GradSchool/Research/test_images/imageone.jpg')
img = img[:,:,:3].mean(axis=2)
plt.set_cmap("gray")

ft = calculate_2dft(img)

plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(np.log(abs(ft)))
plt.axis("off")
plt.show()





