import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
from cv2 import cv2
import scipy.misc
from scipy.signal import convolve2d

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale)
    else:
        im2 = sktr.rescale(im2, 1./dscale)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

def get_gaussian_kernel(l=3, sig=1):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    final = kernel.copy()
    return final

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def overlay(bw1, bw2, L1=15, S1=13, L2=15, S2=13, showfft=False, fname=None):
    # Overlay takes in two black and white images

    gauss1 = np.outer(cv2.getGaussianKernel(L1,S1), cv2.getGaussianKernel(L1,S1))
    gauss1 /=  np.sum(gauss1)
    gauss2 = np.outer(cv2.getGaussianKernel(L2,S2), cv2.getGaussianKernel(L2,S2))
    gauss2 /=  np.sum(gauss2)
    lpf_kernel = gauss1
    impulse = np.zeros(gauss2.shape)
    impulse[impulse.shape[0] // 2, impulse.shape[1] // 2] = 1
    hpf_kernel = cv2.subtract(impulse, gauss2)

    im1_filt = convolve2d(bw1, lpf_kernel, mode="same")
    im2_filt = convolve2d(bw2, hpf_kernel, mode="same")

    if showfft and fname:
        postlpf = np.abs(np.fft.fftshift(np.fft.fft2(im1_filt)))
        posthpf = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_filt))))
        scipy.misc.imsave("output/" + fname + "hpffft.jpg", posthpf)
        scipy.misc.imsave("output/" + fname + "lpffft.jpg", postlpf)

    final = (im1_filt + im2_filt) / 2
    return final

def hybrid_image(im1, im2, L1=15, S1=13, L2=15, S2=13, show=False, fname=None):

    bw1 = rgb2gray(im1)
    bw2 = rgb2gray(im2)
    final = overlay(bw1, bw2, L1, S1, L2, S2, showfft=True, fname=fname)
    if show:
        plt.imshow(final, cmap="gray")
        plt.title("Black and White Filtering Overlaying")
        plt.show()
    if fname:
        scipy.misc.imsave("output/" + fname + "BW.jpg", final)


    final_color = []
    for i in range(3):
        curr = (overlay(im1[:,:,i], im2[:,:,i], L1, S1, L2, S2))
        final_color.append(curr)

    fin_shape = (final_color[0].shape[0], final_color[0].shape[1], 3)
    final = np.zeros(fin_shape)
    final[..., 0] = final_color[0]
    final[..., 1] = final_color[1]
    final[..., 2] = final_color[2]
    final = np.clip(final, 0, 1)

    if show:
        plt.imshow(final)
        plt.title("Final Color Overlay")
        plt.show()
    if fname:
        scipy.misc.imsave("output/" + fname + "RGB.jpg", final)
    return final

def convolve_3d(im, kernel_type="gaussian", L=15, S=13):
    kernel = None
    gauss = np.outer(cv2.getGaussianKernel(L,S), cv2.getGaussianKernel(L,S))
    if kernel_type == "gaussian":
        kernel = gauss / np.sum(gauss)
    else:
        kernel = gauss / np.sum(gauss)
        impulse = np.zeros(gauss.shape)
        impulse[impulse.shape[0] // 2, impulse.shape[1] // 2] = 1
        kernel = cv2.subtract(impulse, kernel)


    final_color = []
    for i in range(3):
        curr = convolve2d(im[:,:,i], kernel, mode="same")
        final_color.append(curr)

    fin_shape = (final_color[0].shape[0], final_color[0].shape[1], 3)
    final = np.zeros(fin_shape)
    final[..., 0] = final_color[0]
    final[..., 1] = final_color[1]
    final[..., 2] = final_color[2]
    final = np.clip(final, 0, 1)
    return final


def pyramids(im, N):
    final = im.copy()
    next_im = final.copy()
    # Calculate Gaussian pyramid
    for i in range(N):
        next_im = convolve_3d(next_im, "gaussian")
        final = np.hstack((final, next_im))

    # Calculate Laplcian pyramid
    l_final = im.copy()
    next_im = l_final.copy() + .1
    for i in range(N):
        next_im = convolve_3d(next_im, "laplacian", L=20, S=20) + .1
        l_final = np.hstack((l_final, next_im))

    # Merge Gaussian and Laplcian Pyramid
    final = np.vstack((final, l_final))
    return final



if __name__ == "__main__":

    """
    PART 1.2

    - Hybridize black and white images
    - save all images and corresponding fourier domain representations
    """

    # 1. load the image
    # 2. align the two images by calling align_images
    # Now you are ready to write your own code for creating hybrid images!

    # Derek + Cat
    im1 = plt.imread("data/DerekPicture.jpg", )
    im2 = plt.imread("data/nutmeg.jpg")
    im1, im2 = align_images(im1, im2)

    ret = hybrid_image(im1, im2, L2 = 50, fname="DerekCat")
    plt.imshow(ret)
    plt.show()


    # Snoopy + Snoop Dogg
    im1 = plt.imread("data/Snoopy.jpg")
    im2 = plt.imread("data/SnoopDogg.jpg")
    im1, im2 = align_images(im1, im2)

    ret = hybrid_image(im1, im2, L1=15, S1=13, L2=5, S2=2, fname="SnoopySnoopDogg")
    plt.imshow(ret)
    plt.show()


    # Prius + Camaro
    im1 = plt.imread("data/prius.jpeg")
    im2 = plt.imread("data/camaro.jpeg")
    im1, im2 = align_images(im1, im2)

    ret = hybrid_image(im1, im2, L1=5, S1=10, L2=12, S2=20, fname="camaropriushybrid")
    plt.imshow(ret)
    plt.show()



    """
    PART 1.2 EC AND 1.3

    - hybridize color images and visualize output
    - generate pyramid of merged images
    """

    # DEREK + CAT PYRAMID
    # high sf
    im1 = plt.imread('data/DerekPicture.jpg')/255.

    # low sf
    im2 = plt.imread('data/nutmeg.jpg')/255

    # Next align images (this code is provided, but may be improved)
    im1_aligned, im2_aligned = align_images(im1, im2)

    ## You will provide the code below. Sigma1 and sigma2 are arbitrary
    ## cutoff values for the high and low frequencies

    sigma1 = 15
    sigma2 = 13
    hybrid = hybrid_image(im1_aligned, im2_aligned, S1 = sigma1, S2 = sigma2)

    plt.imshow(hybrid)
    plt.show()

    ## Compute and display Gaussian and Laplacian Pyramids
    ## You also need to supply this function
    N = 5 # suggested number of pyramid levels (your choice)
    pyramid_img = pyramids(hybrid, N)
    plt.imshow(pyramid_img)
    plt.show()
    scipy.misc.imsave("output/pyramid_DerekCat.jpg", pyramid_img)


    # SNOOPY AND SNOOP DOGG
    # high sf
    im1 = plt.imread('data/Snoopy.jpg')/255.

    # low sf
    im2 = plt.imread('data/SnoopDogg.jpg')/255

    # Next align images (this code is provided, but may be improved)
    im1_aligned, im2_aligned = align_images(im1, im2)

    ## You will provide the code below. Sigma1 and sigma2 are arbitrary
    ## cutoff values for the high and low frequencies
    sigma1 = 15
    sigma2 = 13
    hybrid = hybrid_image(im1_aligned, im2_aligned, S1 = sigma1, S2 = sigma2)

    plt.imshow(hybrid)
    plt.show()


    cheetah = plt.imread("data/cheetah.jpg") / 255
    ## Compute and display Gaussian and Laplacian Pyramids
    ## You also need to supply this function
    N = 5 # suggested number of pyramid levels (your choice)
    pyramid_img = pyramids(cheetah, N)
    plt.imshow(pyramid_img)
    plt.show()
    scipy.misc.imsave("output/pyramid_cheetah.jpg", pyramid_img)

    clownshark = plt.imread("data/clownshark.jpg") / 255
    ## Compute and display Gaussian and Laplacian Pyramids
    ## You also need to supply this function
    N = 5 # suggested number of pyramid levels (your choice)
    pyramid_img = pyramids(clownshark, N)
    plt.imshow(pyramid_img)
    plt.show()
    scipy.misc.imsave("output/pyramid_clownshark.jpg", pyramid_img)

    # # Prius + Camaro
    # im1 = plt.imread("data/prius.jpeg")
    # im2 = plt.imread("data/camaro.jpeg")
    # im2, im1 = align_images(im1, im2)
    #
    # ret = hybrid_image(im1, im2, L1=1, S1=1, L2=100, S2=100, fname="camaropriushybrid")
    # plt.imshow(ret)
    # plt.show()
    # plt.imsave("failedpriuscamaro.jpg", ret)
