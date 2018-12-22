import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters
import skimage as sk
import skimage.io as skio
import scipy.misc as misc

"""
1.1 WARMUP

"""
# 1.1 Warmup


# SHARP KAREEM ADDUL JABBAR CLASSIC
# Unsharp masking
data_dir = "data/"
image_name = "1_1_kareem.jpg"

# name of the input file
imname = data_dir + image_name

# read in the image
im = misc.imread(imname).copy() / 255

plt.figure(figsize=(14, 10))
plt.imshow(im)
plt.title("Original Image Taken From Youtube Old Nba Footage - The One and Only Kareem Abdul Jabbar")
plt.show()

#from scipy.ndimage.filters import gaussian_filter

blurred = filters.gaussian_filter(im, sigma=9)

plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(blurred)
plt.title("Blurred Kareem")
plt.show()

alpha = 1.5
sharp = np.clip((im + alpha * (im - blurred)), 0, 1)
plt.figure(figsize=(14, 10))
plt.imshow(sharp)
plt.title("Sharp Kareem")
plt.show()
misc.imsave("output/sharp_kareem.jpg", sharp)


# SHARP MAGIC JOHNSON CLASSIC IMAGE
data_dir = "data/"
image_name = "1_1_magic.jpg"

# name of the input file
imname = data_dir + image_name

# read in the image
im = misc.imread(imname).copy() / 255

plt.figure(figsize=(14,10))
plt.imshow(im)
plt.title("Original Magic")
plt.show()


SIGMA = 15

blurred = filters.gaussian_filter(im, sigma=SIGMA)
plt.figure(figsize=(14, 10))
plt.imshow(blurred)
plt.title("Blurred Magic")
plt.show()

alpha = .9
sharp = np.clip((im + alpha * (im - blurred)), 0, 1)
plt.figure(figsize=(14, 10))
plt.imshow(sharp)
plt.title("Sharp Magic")
plt.show()
misc.imsave("output/sharp_magic.jpg", sharp)
