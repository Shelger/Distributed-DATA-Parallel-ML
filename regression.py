import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('./original_images/page_0.jpg', cv2.IMREAD_GRAYSCALE)

# Flatten the image to get the pixel values as a 1D array
pixels = img.flatten()

# Plot a histogram to see the distribution
plt.hist(pixels, bins=256, range=[0,256], density=True, color='gray', alpha=0.75)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Pixel Intensity Distribution')
plt.show()
