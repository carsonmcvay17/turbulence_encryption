import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter_spectra(image, low_percent=5, high_percent=5):
    """
    Filters out the lowest and highest frequency components from the Fourier transformed image.
    Args:
        image: Input grayscale image.
        low_percent: Percentage of the lowest frequencies to filter out.
        high_percent: Percentage of the highest frequencies to filter out.
    Returns:
        filtered_image: Image after inverse FFT from filtered spectrum.
        I'm pretty sure this is nothing
    """
    # Perform Fourier Transform
    f_transform = np.fft.fftshift(np.fft.fft2(image))  # FFT and shift zero frequency to the center

    # Get the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform)

    # Create a mask to filter out the lowest and highest frequencies
    rows, cols = image.shape
    mask = np.ones((rows, cols))

    # Calculate the cutoff indices
    low_cutoff = int(rows * low_percent / 100)
    high_cutoff = int(rows * high_percent / 100)

    # Apply the mask: set the lowest and highest frequencies to zero
    mask[:low_cutoff, :] = 0
    mask[-low_cutoff:, :] = 0
    mask[:, :low_cutoff] = 0
    mask[:, -low_cutoff:] = 0
    mask[rows - high_cutoff:, cols - high_cutoff:] = 0
    mask[high_cutoff:, :high_cutoff] = 0

    # Apply the mask to the frequency components
    f_transform_filtered = f_transform * mask

    # Perform inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_transform_filtered)
    img_reconstructed = np.abs(np.fft.ifft2(f_ishift))

    return img_reconstructed, magnitude_spectrum

# Load image and convert to grayscale
image = cv2.imread('image_path_here', cv2.IMREAD_GRAYSCALE)

# Check if image loaded successfully
if image is None:
    print("Error: Image not found.")
else:
    # Apply Fourier Transform and filtering
    filtered_image, magnitude_spectrum = filter_spectra(image, low_percent=5, high_percent=5)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Display magnitude spectrum
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.axis('off')

    # Display the reconstructed image
    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image (after inverse FFT)')
    plt.axis('off')

    plt.show()