import matplotlib.pyplot as plt
#from align_image_code import align_images
#from align_image_code import hybrid_image
from cv2 import cv2
import scipy.misc
from scipy.signal import convolve2d
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as la
import scipy.misc

TOY = True

def get_idx(input, i, j):
    return (i) * input.shape[1] + j

if TOY:
    im = plt.imread("data/toy_problem.jpg") / 255
    og_imshape = im.shape
    plt.imshow(im, cmap="gray")
    plt.show()

    # x length = a.shape[0] * a.shape[1]
    x_shape = im.shape[0] * im.shape[1]

    # we will populate A and b
    A = np.empty((0, x_shape))
    #x = np.zeros((x_shape, 1))
    b = np.empty((0,1))

    row_num = 0

    row = []
    col = []
    data = []
    b = []
    #S_top = -1
    #S_left = -1
    #S_right = im.shape[1]
    #S_bottom = im.shape[0]
    idx0 = get_idx(im, 0, 0)
    row.append(row_num)
    col.append(idx0)
    data.append(1)
    b.append(im[0,0])
    row_num += 1

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            #eq = np.zeros((1, x_shape))
            if i < im.shape[0] - 1:
                idx1 = get_idx(im, i, j)
                idx2 = get_idx(im, i + 1, j)
                # eq[0,idx1] = 1
                # eq[0,idx2] = -1
                # b_2 = [[im[i,j] - im[i+1, j]]]
                row.append(row_num)
                col.append(idx1)
                data.append(1)

                row.append(row_num)
                col.append(idx2)
                data.append(-1)

                b.append(im[i,j] - im[i+1, j])
                row_num += 1


            #eq = np.zeros((1, x_shape))
            if j < im.shape[1] - 1:
                idx1 = get_idx(im, i, j)
                idx2 = get_idx(im, i, j + 1)
                # eq[0,idx1] = 1
                # eq[0,idx2] = -1
                # b_2 = [[im[i,j] - im[i, j + 1]]]
                row.append(row_num)
                col.append(idx1)
                data.append(1)

                row.append(row_num)
                col.append(idx2)
                data.append(-1)

                b.append(im[i,j] - im[i, j+1])
                row_num += 1

            #eq = np.zeros((1, x_shape))
            if i > 0:
                idx1 = get_idx(im, i, j)
                idx2 = get_idx(im, i - 1, j)
                # eq[0,idx1] = 1
                # eq[0,idx2] = -1
                # b_2 = [[im[i,j] - im[i+1, j]]]
                row.append(row_num)
                col.append(idx1)
                data.append(1)

                row.append(row_num)
                col.append(idx2)
                data.append(-1)

                b.append(im[i,j] - im[i-1, j])
                row_num += 1


            #eq = np.zeros((1, x_shape))
            if j > 0:
                idx1 = get_idx(im, i, j)
                idx2 = get_idx(im, i, j - 1)
                # eq[0,idx1] = 1
                # eq[0,idx2] = -1
                # b_2 = [[im[i,j] - im[i, j + 1]]]
                row.append(row_num)
                col.append(idx1)
                data.append(1)

                row.append(row_num)
                col.append(idx2)
                data.append(-1)

                b.append(im[i,j] - im[i, j-1])
                row_num += 1


            print("Finished: ", (i,j))

    row, col, data = np.array(row), np.array(col), np.array(data)

    A = csr_matrix((data, (row, col)))
    b = np.array([b]).T
    print("Constructed A matrix of shape: ", A.shape)
    print("Constructed b matrix of shape: ", b.shape)

    x = la.lsqr(A, b)
    y = x[0]
    print("Pixel values found: ", y.shape)
    y = np.reshape(y, og_imshape)
    plt.imshow(y, cmap="gray")
    plt.show()
    scipy.misc.imsave("output/toyexample.jpg", y)
