import matplotlib.pyplot as plt
#from align_image_code import align_images
#from align_image_code import hybrid_image
from cv2 import cv2
import scipy.misc
from scipy.signal import convolve2d
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as la
from  scipy.misc import imresize
import scipy.misc

PENGUIN = True

def get_idx(i, j, start, end):
    width = end[1] - start[1]
    ret = (i - start[0]) * width + (j - start[1])
    return ret

# snow_source: (1740, 500) --> (2000,1000)
# penguin: (20, 50) --> (450,360)

def proj3_22():
    all_channels = []

    for channel in range(3):
        bg = plt.imread("data/im3.jpg") / 255
        bg = bg[...,channel]

        source = plt.imread("data/penguin-chick.jpeg") / 255
        source = source[...,channel]

        og_bg_shape = bg.shape
        og_source_shape = source.shape

        # plt.imshow(source, cmap="gray")
        # plt.show()
        # plt.imshow(input, cmap="gray")
        # plt.show()

        # How to trim the input image
        sx1 = 0                 # sx1 = 20
        sx2 = source.shape[0]   # sx2 = 450
        sy1 = 0                 # sy1 = 50
        sy2 = source.shape[1]   # sy2 = 360

        # How much to scale the input image down
        K = 1

        # Top left corner on the source image in bg image frame
        full_bgx1 = 1600
        full_bgy1 = 600
        # Corresponding bottom right corner
        full_bgx2 = full_bgx1 + (sx2 - sx1) * K
        full_bgy2 = full_bgy1 + (sy2 - sy1) * K

        bgx1 = full_bgx1 + 25
        bgy1 = full_bgy1 + 25
        bgx2 = full_bgx2 - 25
        bgy2 = full_bgy2 - 25

        #penguin = source[sx1:sx2,sy1:sy2]
        penguin = source
        penguin = imresize(penguin, (full_bgx2 - full_bgx1, full_bgy2 - full_bgy1)) / 255

        bg_with_full_source = bg.copy()
        bg_with_full_source[full_bgx1:full_bgx2, full_bgy1:full_bgy2] = penguin
        #plt.imshow(bg_with_full_source, cmap="gray")
        #plt.show()
        scipy.misc.imsave("output/final_original_penguins.jpg", bg_with_full_source)

        #plt.imshow(bg, cmap="gray")
        #plt.show()


        #exit(0)

        #plt.imshow(penguin)
        #plt.show()

        #bg[bgx1:bgx2, bgy1:bgy2] = penguin
        #plt.imshow(bg, cmap="gray")
        #plt.show()

        # exit(0)

        row_num = 0
        row = []
        col = []
        data = []
        b = []

        # where we're traversing
        start = (bgx1, bgy1)
        end = (bgx2, bgy2)
        source_shape = (bgx2 - bgx1, bgy2 - bgy1)



        for i in range(bgx1, bgx2):
            for j in range(bgy1, bgy2):
                # CHECK DOWN (i + 1)
                if i < bgx2 - 1:
                    idx1 = get_idx(i, j, start, end)
                    idx2 = get_idx(i + 1, j, start, end)

                    row.append(row_num)
                    col.append(idx1)
                    data.append(1)

                    row.append(row_num)
                    col.append(idx2)
                    data.append(-1)

                    b.append(bg_with_full_source[i,j] - bg_with_full_source[i+1, j])
                    row_num += 1
                # if its on the very last row, get the other i+1 equation
                elif i == bgx2 - 1:
                    idx1 = get_idx(i, j, start, end)

                    row.append(row_num)
                    col.append(idx1)
                    data.append(1)

                    b.append(bg_with_full_source[i,j] - bg_with_full_source[i+1, j] + bg[i+1,j])
                    row_num += 1

                # CHECK RIGHT (j + 1)
                if j < bgy2 - 1:
                    idx1 = get_idx(i, j, start, end)
                    idx2 = get_idx(i, j + 1, start, end)
                    row.append(row_num)
                    col.append(idx1)
                    data.append(1)

                    row.append(row_num)
                    col.append(idx2)
                    data.append(-1)

                    b.append(bg_with_full_source[i,j] - bg_with_full_source[i, j+1])
                    row_num += 1
                elif j == bgy2 - 1:
                    idx1 = get_idx(i, j, start, end)

                    row.append(row_num)
                    col.append(idx1)
                    data.append(1)

                    b.append(bg_with_full_source[i,j] - bg_with_full_source[i, j+1] + bg[i,j+1])
                    row_num += 1

                # CHECK UP (i - 1)
                if i > bgx1:
                    idx1 = get_idx(i, j, start, end)
                    idx2 = get_idx(i - 1, j, start, end)

                    row.append(row_num)
                    col.append(idx1)
                    data.append(1)

                    row.append(row_num)
                    col.append(idx2)
                    data.append(-1)

                    b.append(bg_with_full_source[i,j] - bg_with_full_source[i-1, j])
                    row_num += 1
                # if its on the very last row, get the other i+1 equation
                elif i == bgx1:
                    idx1 = get_idx(i, j, start, end)

                    row.append(row_num)
                    col.append(idx1)
                    data.append(1)

                    b.append(bg_with_full_source[i,j] - bg_with_full_source[i-1, j] + bg[i-1,j])
                    row_num += 1

                # CHECK UP (j - 1)
                if j > bgy1:
                    idx1 = get_idx(i, j, start, end)
                    idx2 = get_idx(i, j-1, start, end)

                    row.append(row_num)
                    col.append(idx1)
                    data.append(1)

                    row.append(row_num)
                    col.append(idx2)
                    data.append(-1)

                    b.append(bg_with_full_source[i,j] - bg_with_full_source[i, j-1])
                    row_num += 1
                # if its on the very last row, get the other i+1 equation
                elif j == bgy1:
                    idx1 = get_idx(i, j, start, end)

                    row.append(row_num)
                    col.append(idx1)
                    data.append(1)

                    b.append(bg_with_full_source[i,j] - bg_with_full_source[i, j-1] + bg[i,j-1])
                    row_num += 1





        row, col, data = np.array(row), np.array(col), np.array(data)

        A = csr_matrix((data, (row, col)))
        b = np.array([b]).T
        print("Constructed A matrix of shape: ", A.shape)
        print("Constructed b matrix of shape: ", b.shape)

        x = la.lsqr(A, b)
        y = x[0]
        print("Pixel values found: ", y.shape)
        final_source = np.reshape(y, source_shape)
        #plt.imshow(final_source, cmap="gray")
        #plt.show()

        bg_channel = bg.copy()
        bg_channel[bgx1:bgx2, bgy1:bgy2] = final_source
        #plt.imshow(bg_channel, cmap="gray")
        #plt.show()

        all_channels.append(bg_channel)

    final_im = np.empty((all_channels[0].shape[0], all_channels[0].shape[1], 3))
    for channel in range(3):
        final_im[:,:,channel] = np.clip(all_channels[channel], 0, 1)

    plt.imshow(final_im)
    plt.show()
    scipy.misc.imsave("output/final_penguins.jpg", final_im)
