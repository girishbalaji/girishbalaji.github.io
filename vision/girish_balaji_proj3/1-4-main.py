import matplotlib.pyplot as plt
#from align_image_code import align_images
#from align_image_code import hybrid_image
from cv2 import cv2
import scipy.misc
from scipy.signal import convolve2d
import numpy as np
from  scipy.misc import imresize
import scipy.misc

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


def pyramids(im, N, kernel_type="laplacian", L=15, S=13):
    #final = None
    final = []
    if kernel_type == "gaussian":
        final = [im.copy()]
        next_im = final[-1].copy()
        for i in range(N):
            next_im = convolve_3d(next_im, "gaussian", L=L, S=S)
            #final = np.hstack((final, next_im))
            final.append(next_im.copy())

    else:
        final = [im.copy()]
        next_im = final[-1].copy()
        for i in range(N):
            next_im = convolve_3d(next_im, "laplacian", L=L, S=S)
            #final = np.hstack((final, next_im))
            final.append(next_im.copy())
    return final

# ORANGE AND APPLE
# high sf
im1 = plt.imread('data/apple.jpeg')/255.

# low sf
im2 = plt.imread('data/orange.jpeg')/255.

im1_w = im1.shape[1]
im2_w = im2.shape[1]

R = np.hstack((np.ones(im1[:,:im1_w //2].shape), np.zeros(im2[:,im2_w//2:].shape)))/1.0
A = im1.copy()
B = im2.copy()

N = 5

LA = pyramids(A, N, "laplacian", L=5, S=15)
LB = pyramids(B, N, "laplacian", L=5, S=15)
GR = pyramids(R, N, "gaussian", L=40, S=80)
LS = [GR[-i-1] * LA[i] + (1 - GR[-i-1]) * LB[i] for i in range(0,N-1)]

LS_fin = LS[0]
for i in range(1,N-1):
    plt.title("Image {0}".format(i))
    LS_fin = np.clip(LS_fin + LS[i], 0, 1)

plt.imshow(LS_fin)
scipy.misc.imsave("output/oraple.jpg", LS_fin)
plt.show()

# SHARK AND CLOWNFISH
# high sf
im1 = plt.imread('data/shark.jpeg')/255.
im1 = imresize(im1, (300,300,3))/255
# low sf
im2 = plt.imread('data/clownfish.jpeg')/255.
im2 = imresize(im2, (300,300,3))/255

im1_w = im1.shape[1]
im2_w = im2.shape[1]

R = np.hstack((np.ones(im1[:,:im1_w //2].shape), np.zeros(im2[:,im2_w//2:].shape)))/1.0
A = im1.copy()
B = im2.copy()

N = 5

LA = pyramids(A, N, "laplacian", L=5, S=15)
LB = pyramids(B, N, "laplacian", L=5, S=15)
GR = pyramids(R, N, "gaussian", L=40, S=80)[::-1]
LS = [GR[i] * LA[i] + (1 - GR[i]) * LB[i] for i in range(0,N-1)]

LS_fin = LS[0]

for i in range(1,N-1):
    plt.title("Image {0}".format(i))
    LS_fin = np.clip(LS_fin + LS[i], 0, 1)

plt.imshow(LS_fin)
scipy.misc.imsave("output/clownshark.jpg", LS_fin)
plt.show()


# CAMARO AND PRIUS
# high sf
im1 = plt.imread('data/camaro.jpeg')/255.
im1 = imresize(im1, (300,300,3))/255
# low sf
im2 = plt.imread('data/prius.jpeg')/255.
im2 = imresize(im2, (300,300,3))/255

im1_w = im1.shape[1]
im2_w = im2.shape[1]

R = np.hstack((np.ones(im1[:,:im1_w //2].shape), np.zeros(im2[:,im2_w//2:].shape)))/1.0
A = im1.copy()
B = im2.copy()

N = 5

LA = pyramids(A, N, "laplacian", L=5, S=15)
LB = pyramids(B, N, "laplacian", L=5, S=15)
GR = pyramids(R, N, "gaussian", L=40, S=40)[::-1]
LS = [GR[i] * LA[i] + (1 - GR[i]) * LB[i] for i in range(0,N-1)]

LS_fin = LS[0]
for i in range(1,N-1):
    plt.title("Image {0}".format(i))
    LS_fin = np.clip(LS_fin + LS[i], 0, 1)

plt.imshow(LS_fin)
scipy.misc.imsave("output/camroprius.jpg", LS_fin)
plt.show()




# SHARK AND personswimming

# CAMARO AND PRIUS
# high sf
im1 = plt.imread('data/personswimming.jpg')/255.
im1 = imresize(im1, (300,300,3))/255
# low sf
im2 = plt.imread('data/shark1.jpg')/255.
im2 = imresize(im2, (300,300,3))/255

im1_w = im1.shape[1]
im2_w = im2.shape[1]

R = np.hstack((np.ones(im1[:,:im1_w //2].shape), np.zeros(im2[:,im2_w//2:].shape)))/1.0
A = im1.copy()
B = im2.copy()

N = 5

LA = pyramids(A, N, "laplacian", L=5, S=15)
LB = pyramids(B, N, "laplacian", L=5, S=15)
GR = pyramids(R, N, "gaussian", L=40, S=40)[::-1]
LS = [GR[i] * LA[i] + (1 - GR[i]) * LB[i] for i in range(0,N-1)]

LS_fin = LS[0]
for i in range(1,N-1):
    plt.title("Image {0}".format(i))
    LS_fin = np.clip(LS_fin + LS[i], 0, 1)

plt.imshow(LS_fin)
scipy.misc.imsave("output/swimshark.jpg", LS_fin)
plt.show()
