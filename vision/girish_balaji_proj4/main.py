import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave
from scipy.misc import imresize
import pickle
from skimage.draw import polygon
import os
import fnmatch
import json

PREPROCESS = False

# data/edited_{im1_name} and data/edited_{im2_name} need to exist
GEN_REF = False
GEN_IM2_PTS = False

# Points Exist Needs to be True to run GET_AVG and GET_FULL_MORPH
POINTS_EXIST = False
GET_AVG = False
GET_FULL_MORPH = False

DANE_DATA = False

# Adds Caricature messup to IM2
GET_CARICATURE = True

# Adds smile to IM2
MORE_SMILING = False

# default: me_glass, satya
IM1_NAME = "avg_dane"
IM2_NAME = "me_no_glasses_for_dane"




from scipy.spatial import Delaunay

if PREPROCESS:
    im1 = plt.imread("data/{0}.jpeg".format(IM1_NAME)) / 255.0
    im2 = plt.imread("data/{0}.jpeg".format(IM2_NAME)) / 255.0

    if IM1_NAME == "me_glasses" and IM2_NAME == "satya":

        im1 = im1[200:-80, 60:]
        # plt.imshow(im1)
        # plt.show()
        imsave("data/edited_" + IM1_NAME + ".jpeg", im1)

        im2 = im2[:-200, 200:-200]
        im2 = imresize(im2, im1.shape)
        # plt.imshow(im2)
        # plt.show()
        imsave("data/edited_" + IM2_NAME + ".jpeg", im2)

    elif IM1_NAME == "avg_dane" and IM2_NAME == "me_no_glasses_for_dane":
        im2 = imresize(im2, im1.shape)
        imsave("data/edited_" + IM1_NAME + ".jpeg", im1)
        imsave("data/edited_" + IM2_NAME + ".jpeg", im2)


def get_n_points(n, im, msg):
    print('Please select {0} points in the: {1}.'.format(n, msg))
    plt.title('Please select {0} points in the: {1}.'.format(n, msg))
    plt.imshow(im)
    all_pts = plt.ginput(n, show_clicks=True, timeout=0)
    plt.close()
    return [i for i in all_pts]

im1 = plt.imread("data/edited_" + IM1_NAME + ".jpeg") / 255.0
im2 = plt.imread("data/edited_" + IM2_NAME + ".jpeg") / 255.0

if GEN_REF:
    if IM1_NAME != "avg_dane":
        im1_pts = []

        # get 52 points on im1
        region = "full face"
        num_pts = 52
        im1_pts += get_n_points(num_pts, im1, region)

        # add corners
        h, w, _ = im1.shape
        im1_pts.append((0,0))
        im1_pts.append((w-1,0))
        im1_pts.append((0,h-1))
        im1_pts.append((w-1,h-1))

        # dump the 52 points to disk
        f= open("points/" + IM1_NAME + ".txt","wb")
        pickle.dump(im1_pts, f)
        f.close()
        # backup
        f= open("points/backup_" + IM1_NAME + ".txt","wb")
        pickle.dump(im1_pts, f)
        f.close()

        # Reading the points
        im1_pt_file = open("points/" + IM1_NAME + ".txt", 'rb')
        im1_pts = pickle.load(im1_pt_file)
        im1_pt_file.close()

        # save the 52 points as reference for generating im2 points
        plt.imshow(im1)
        plt.scatter(*zip(*im1_pts))
        for i in range(len(im1_pts)):
            plt.annotate(str(i), im1_pts[i])
        plt.savefig("output/point_reference_" + IM1_NAME + ".jpeg")
        plt.show()
        exit(0)

if GEN_IM2_PTS:
    # Reading IM1 PTS points
    im1_pt_file = open("points/" + IM1_NAME + ".txt", 'rb')
    im1_pts = pickle.load(im1_pt_file)
    im1_pt_file.close()

    im2_pts = []

    # get 52 points on im1
    region = "full face"
    num_pts = 58
    im2_pts += get_n_points(num_pts, im2, region)

    # add the corners
    h, w, _ = im2.shape
    im2_pts.append((0,0))
    im2_pts.append((w-1,0))
    im2_pts.append((0,h-1))
    im2_pts.append((w-1,h-1))

    # dump the 52 points to disk
    f= open("points/" + IM2_NAME + ".txt","wb")
    pickle.dump(im2_pts, f)
    f.close()
    # backup
    f= open("points/backup_" + IM2_NAME + ".txt","wb")
    pickle.dump(im2_pts, f)
    f.close()

    # Reading the points
    im2_pt_file = open("points/" + IM2_NAME + ".txt", 'rb')
    im2_pts = pickle.load(im2_pt_file)
    im2_pt_file.close()

    # save the 52 points as reference for generating im2 points
    plt.imshow(im2)
    plt.scatter(*zip(*im2_pts))
    for i in range(len(im2_pts)):
        plt.annotate(str(i), im2_pts[i])
    plt.savefig("output/point_reference_" + IM2_NAME + ".jpeg")
    plt.show()
    exit(0)


# affine transform of point in 1 to point in 2
class TriAffine:
    def __init__(self, tr1_pts, tr2_pts):


        self.tr1_pts = tr1_pts
        self.tr2_pts = tr2_pts
        p11, p12, p13 = tr1_pts
        p21, p22, p23 = tr2_pts

        source_pt_mat = np.array([
            [p11[0], p12[0], p13[0]],
            [p11[1], p12[1], p13[1]],
            [1.0     ,      1,      1]
        ])
        goal_pt_mat = np.array([
            [p21[0], p22[0], p23[0]],
            [p21[1], p22[1], p23[1]],
            [1.0     ,      1,      1]
        ])

        self.transform_mat = np.dot(goal_pt_mat, np.linalg.inv(source_pt_mat))

    # pts = x, y (output of np.nonzero)
    def transform(self, x_pts, y_pts):
        pts_mat = np.array([x_pts, y_pts])
        pts_mat = np.vstack((pts_mat, np.ones(pts_mat.shape[1])))
        transformed = np.round(np.dot(self.transform_mat, pts_mat)).astype(int)
        # print("Transform Matrix: ", transformed)
        return (transformed[0,:], transformed[1,:])


def get_morph(im1, im2, im1_pts_array, imt2_pts_array, t):
    # Get the average image
    mean_pts_array = (im1_pts_array) * (1-t) + (im2_pts_array) * (t)

    # Delaunay Mean
    tri_mean = Delaunay(mean_pts_array)

    #mean_pts_array[:,0], mean_pts_array[:,1] = mean_pts_array[:,1], mean_pts_array[:,0].copy()
    mean_im = np.zeros(im1.shape).astype(float)

    for tri_vert_idxs in tri_mean.simplices.copy():
        vert_idx1, vert_idx2, vert_idx3 = tri_vert_idxs

        im1_tri = np.array([im1_pts_array[vert_idx1], im1_pts_array[vert_idx2], im1_pts_array[vert_idx3]])
        im2_tri = np.array([im2_pts_array[vert_idx1], im2_pts_array[vert_idx2], im2_pts_array[vert_idx3]])
        im_mean_tri = np.array([mean_pts_array[vert_idx1], mean_pts_array[vert_idx2], mean_pts_array[vert_idx3]])

        Trans1 = TriAffine(im_mean_tri, im1_tri)
        Trans2 = TriAffine(im_mean_tri, im2_tri)

        poly_mean_x_idxs, poly_mean_y_idxs = polygon(im_mean_tri[:,0], im_mean_tri[:,1])

        poly1_x_idxs, poly1_y_idxs = Trans1.transform(poly_mean_x_idxs, poly_mean_y_idxs)
        poly2_x_idxs, poly2_y_idxs = Trans2.transform(poly_mean_x_idxs, poly_mean_y_idxs)

        mean_im[poly_mean_x_idxs, poly_mean_y_idxs] = im1[poly1_x_idxs, poly1_y_idxs] * (1-t) + im2[poly2_x_idxs, poly2_y_idxs] * (t)
    return mean_im


if POINTS_EXIST:
    # Reading IM1 Points
    im1_pt_file = open("points/" + IM1_NAME + ".txt", 'rb')
    im1_pts = pickle.load(im1_pt_file)
    im1_pt_file.close()

    # Reading IM2 Points
    im2_pt_file = open("points/" + IM2_NAME + ".txt", 'rb')
    im2_pts = pickle.load(im2_pt_file)
    im2_pt_file.close()

    # Delaunay IM1
    im1_pts_array = np.array([[a[0], a[1]] for a in im1_pts])
    tri1 = Delaunay(im1_pts_array)

    plt.imshow(im1)
    plt.triplot(im1_pts_array[:,0], im1_pts_array[:,1], tri1.simplices.copy())
    plt.plot(im1_pts_array[:,0], im1_pts_array[:,1], 'o')
    plt.savefig("output/Delaunay_" + IM1_NAME + ".jpeg")
    plt.show()

    # Delaunay IM2
    im2_pts_array = np.array([[a[0], a[1]] for a in im2_pts])
    tri2 = Delaunay(im2_pts_array)

    plt.imshow(im2)
    plt.triplot(im2_pts_array[:,0], im2_pts_array[:,1], tri2.simplices.copy())
    plt.plot(im2_pts_array[:,0], im2_pts_array[:,1], 'o')
    plt.savefig("output/Delaunay_" + IM2_NAME + ".jpeg")
    plt.show()


    # Variables we will use: (those right below) im1, im2
    # switch to [[row, col], ..]
    im1_pts_array[:,0], im1_pts_array[:,1] = im1_pts_array[:,1], im1_pts_array[:,0].copy()
    im2_pts_array[:,0], im2_pts_array[:,1] = im2_pts_array[:,1], im2_pts_array[:,0].copy()


    if GET_AVG:
        # Get the average image
        mean_pts_array = (im1_pts_array + im2_pts_array) / 2

        # Delaunay IM2
        tri_mean = Delaunay(mean_pts_array)

        # NOTE: THE AXES ARE FLIPPED HERE BECAUSE WE ALREADY REVERSED X,Y FOR IM{1,2}_PTS_ARRAY
        plt.imshow(im1)
        plt.triplot(mean_pts_array[:,1], mean_pts_array[:,0], tri_mean.simplices.copy())
        plt.plot(mean_pts_array[:,1], mean_pts_array[:,0], 'o', label="average points")
        plt.plot(im1_pts_array[:,1], im1_pts_array[:,0], '+', label="im1 points")
        plt.plot(im2_pts_array[:,1], im2_pts_array[:,0], 'b+', label="im2 points")
        plt.legend()
        plt.savefig("output/Delaunay_mean_" + IM1_NAME + "_" + IM2_NAME + ".jpeg")
        plt.show()

        #mean_pts_array[:,0], mean_pts_array[:,1] = mean_pts_array[:,1], mean_pts_array[:,0].copy()
        mean_im = get_morph(im1, im2, im1_pts_array, im2_pts_array, 0.5)

        imsave("output/avg_" + IM1_NAME + "_" + IM2_NAME + ".jpeg", mean_im)
        plt.imshow(mean_im)
        plt.show()

    if GET_FULL_MORPH:
        # Data available: im1_pts_array, im2_pts_array
        for i in range(45):
            t = i / 44.0
            mean_im = get_morph(im1, im2, im1_pts_array, im2_pts_array, t)

            imsave("output/" + IM1_NAME + "_" + IM2_NAME + "/" + str(i) + ".jpg", mean_im)
            #plt.imshow(mean_im)
            #plt.show()



def parse_dane_file(fname, dirname):
    all_pts = []
    with open(dirname + fname) as fp:
        all_lines = fp.readlines()
        for line in all_lines:
            if line[0] != "#" and len(line) > 0:
                spl = line.split()
                if len(spl) > 3:
                    # Row, column
                    curr_pt = [spl[3], spl[2]]
                    all_pts.append(curr_pt)
    return np.array(all_pts).astype('float')



if DANE_DATA:
    DANE_DIR = "dane_data/"

    total_pts_arr = None
    num_files = 0
    all_pts_arrs = []
    ims = []

    for file in os.listdir(DANE_DIR):
        if fnmatch.fnmatch(file, '*.asf'):
            fname = file
            im_fname = fname.split(".")[0] + ".bmp"
            # im_fname = "01-1m.bmp"

            im_pts_array = parse_dane_file(fname, DANE_DIR)
            im_data = plt.imread(DANE_DIR + im_fname) / 255.0
            im_pts_array[:,0] = im_pts_array[:,0] * (im_data.shape[0])
            im_pts_array[:,1] = im_pts_array[:,1] * (im_data.shape[1])

            # Add corners
            nr, nc, _ = im_data.shape
            im_pts_array = np.vstack((im_pts_array,
                np.array([[0, 0],
                          [nr - 1, 0],
                          [0, nc - 1],
                          [nr - 1, nc - 1]
                          ])))

            # Generate a sample image
            # plt.imshow(im_data)
            # plt.plot(im_pts_array[:,1], im_pts_array[:,0], 'b+', label="im points")
            # plt.legend()
            # plt.show()
            # exit(0)

            if num_files == 0:
                total_pts_arr = im_pts_array.copy()
            else:
                total_pts_arr += im_pts_array.copy()

            num_files += 1
            all_pts_arrs.append(im_pts_array.copy())
            ims.append(im_data)

    avg_pts_array = total_pts_arr.copy() / num_files


    # THE FOLLOWING CODE SAVES THE AVEREAGE AS IM1_NAME
    # TEMPORARILY FLIP THE ARRAY
    avg_pts_array[:,0], avg_pts_array[:,1] = avg_pts_array[:,1], avg_pts_array[:,0].copy()
    # SAVE THE POINTS
    # dump the 52 points to disk
    f= open("points/" + IM1_NAME + ".txt","wb")
    pickle.dump(avg_pts_array, f)
    f.close()
    # backup
    f= open("points/backup_" + IM1_NAME + ".txt","wb")
    pickle.dump(avg_pts_array, f)
    f.close()

    # Reading the points
    im1_pt_file = open("points/" + IM1_NAME + ".txt", 'rb')
    avg_pts_array = pickle.load(im1_pt_file)
    im1_pt_file.close()

    # save the 52 points as reference for generating im2 points
    plt.imshow(im1)
    plt.scatter(*zip(*avg_pts_array), marker='+')
    for i in range(len(avg_pts_array)):
        plt.annotate(str(i), (avg_pts_array[i]))
    plt.savefig("output/point_reference_" + IM1_NAME + ".jpeg")
    plt.show()

    avg_pts_array[:,0], avg_pts_array[:,1] = avg_pts_array[:,1], avg_pts_array[:,0].copy()
    # END SAVING AVERAGE


    # Avg points
    # plt.imshow(im_data)
    # plt.plot(avg_pts_array[:,1], avg_pts_array[:,0], 'b+', label="average points")
    # plt.title("Sample image with average points plotted")
    # plt.legend()
    # plt.savefig("output/avg_dane_points.jpeg")
    # plt.show()

    # Delaunay Mean
    tri_mean = Delaunay(avg_pts_array)

    #mean_pts_array[:,0], mean_pts_array[:,1] = mean_pts_array[:,1], mean_pts_array[:,0].copy()
    mean_im = np.zeros(im_data.shape).astype(float)
    ind_cont = 1 / num_files

    for tri_vert_idxs in tri_mean.simplices.copy():
        vert_idx1, vert_idx2, vert_idx3 = tri_vert_idxs

        for i in range(len(all_pts_arrs)):
            im_pts_array = all_pts_arrs[i]
            im = ims[i]
            im_tri = np.array([im_pts_array[vert_idx1], im_pts_array[vert_idx2], im_pts_array[vert_idx3]])
            im_mean_tri = np.array([avg_pts_array[vert_idx1], avg_pts_array[vert_idx2], avg_pts_array[vert_idx3]])

            Trans = TriAffine(im_mean_tri, im_tri)
            poly_mean_x_idxs, poly_mean_y_idxs = polygon(im_mean_tri[:,0], im_mean_tri[:,1])
            poly_im_x_idxs, poly_im_y_idxs = Trans.transform(poly_mean_x_idxs, poly_mean_y_idxs)
            mean_im[poly_mean_x_idxs, poly_mean_y_idxs] += im[poly_im_x_idxs, poly_im_y_idxs] * ind_cont

    plt.imshow(mean_im)
    imsave("output/avg_dane.jpg", mean_im)
    plt.show()

    # Morphing Examples
    for im_num in range(3):
        im1 = ims[im_num]
        im1_pts_array = all_pts_arrs[im_num]

        im2 = mean_im.copy()
        im2_pts_array = avg_pts_array

        N = 25
        for i in range(N):
            t = i / float(N - 1)
            sample = get_morph(im1, im2, im1_pts_array, im2_pts_array, t)
            imsave("output/dane_to_avg_example_" + str(im_num) + "/" + str(i) + ".jpg", sample)


# Assumes the im1 is saved as points
# We will create a caricature of im2
if GET_CARICATURE:
    # Reading IM1 Points
    im1_pt_file = open("points/" + IM1_NAME + ".txt", 'rb')
    im1_pts = pickle.load(im1_pt_file)
    im1_pt_file.close()

    # Reading IM2 Points
    im2_pt_file = open("points/" + IM2_NAME + ".txt", 'rb')
    im2_pts = pickle.load(im2_pt_file)
    im2_pt_file.close()

    im1_pts_array = np.array([[a[1], a[0]] for a in im1_pts])
    im2_pts_array = np.array([[a[1], a[0]] for a in im2_pts])

    alpha = 1
    im2_pts_array = im2_pts_array + (im2_pts_array - im1_pts_array) * alpha

    t = .7
    caricature = get_morph(im1, im2, im1_pts_array, im2_pts_array, t)
    plt.imshow(caricature)
    imsave("output/caricature_" + IM1_NAME + "_" + IM2_NAME + ".jpeg", caricature)
    plt.show()

    # for i in range(45):
    #     t = i / 44.0
    #     mean_im = get_morph(im1, im2, im1_pts_array, im2_pts_array, t)
    #
    #     imsave("output/" + IM1_NAME + "_" + IM2_NAME + "/" + str(i) + ".jpg", mean_im)
    #

if MORE_SMILING:
    DANE_SMILING_DIR = "dane_smiling/"

    total_pts_arr = None
    num_files = 0
    all_pts_arrs = []
    ims = []

    for file in os.listdir(DANE_SMILING_DIR):
        if fnmatch.fnmatch(file, '*.asf'):
            fname = file
            im_fname = fname.split(".")[0] + ".bmp"
            # im_fname = "01-1m.bmp"

            im_pts_array = parse_dane_file(fname, DANE_SMILING_DIR)
            im_data = plt.imread(DANE_SMILING_DIR + im_fname) / 255.0
            im_pts_array[:,0] = im_pts_array[:,0] * (im_data.shape[0])
            im_pts_array[:,1] = im_pts_array[:,1] * (im_data.shape[1])

            # Add corners
            nr, nc, _ = im_data.shape
            im_pts_array = np.vstack((im_pts_array,
                np.array([[0, 0],
                          [nr - 1, 0],
                          [0, nc - 1],
                          [nr - 1, nc - 1]
                          ])))

            # Generate a sample image
            # plt.imshow(im_data)
            # plt.plot(im_pts_array[:,1], im_pts_array[:,0], 'b+', label="im points")
            # plt.legend()
            # plt.show()
            # exit(0)

            if num_files == 0:
                total_pts_arr = im_pts_array.copy()
            else:
                total_pts_arr += im_pts_array.copy()

            num_files += 1
            all_pts_arrs.append(im_pts_array.copy())
            ims.append(im_data)

    avg_smiling_pts_array = total_pts_arr.copy() / num_files

    DANE_SMILING_DIR = "dane_not_smiling/"

    total_pts_arr = None
    num_files = 0
    all_pts_arrs = []
    ims = []

    for file in os.listdir(DANE_SMILING_DIR):
        if fnmatch.fnmatch(file, '*.asf'):
            fname = file
            im_fname = fname.split(".")[0] + ".bmp"
            # im_fname = "01-1m.bmp"

            im_pts_array = parse_dane_file(fname, DANE_SMILING_DIR)
            im_data = plt.imread(DANE_SMILING_DIR + im_fname) / 255.0
            im_pts_array[:,0] = im_pts_array[:,0] * (im_data.shape[0])
            im_pts_array[:,1] = im_pts_array[:,1] * (im_data.shape[1])

            # Add corners
            nr, nc, _ = im_data.shape
            im_pts_array = np.vstack((im_pts_array,
                np.array([[0, 0],
                          [nr - 1, 0],
                          [0, nc - 1],
                          [nr - 1, nc - 1]
                          ])))

            # Generate a sample image
            # plt.imshow(im_data)
            # plt.plot(im_pts_array[:,1], im_pts_array[:,0], 'b+', label="im points")
            # plt.legend()
            # plt.show()
            # exit(0)

            if num_files == 0:
                total_pts_arr = im_pts_array.copy()
            else:
                total_pts_arr += im_pts_array.copy()

            num_files += 1
            all_pts_arrs.append(im_pts_array.copy())
            ims.append(im_data)

    avg_not_smiling_pts_array = total_pts_arr.copy() / num_files

    # Reading IM2 Points
    im2_pt_file = open("points/" + IM2_NAME + ".txt", 'rb')
    im2_pts = pickle.load(im2_pt_file)
    im2_pt_file.close()

    # Delaunay IM2
    im2_pts_array = np.array([[a[1], a[0]] for a in im2_pts])

    plt.imshow(im2)
    plt.show()

    for alpha in [0, 1, 2, 4, -1, -2, -4]:
        # Get a new set of points
        #alpha = 2
        new_im2_pts_array = (im2_pts_array) + (avg_smiling_pts_array - avg_not_smiling_pts_array) * alpha

        # Delaunay Mean
        tri_mean = Delaunay(new_im2_pts_array)

        #mean_pts_array[:,0], mean_pts_array[:,1] = mean_pts_array[:,1], mean_pts_array[:,0].copy()
        mean_im = np.zeros(im2.shape).astype(float)

        for tri_vert_idxs in tri_mean.simplices.copy():
            vert_idx1, vert_idx2, vert_idx3 = tri_vert_idxs

            #im1_tri = np.array([im1_pts_array[vert_idx1], im1_pts_array[vert_idx2], im1_pts_array[vert_idx3]])
            im2_tri = np.array([im2_pts_array[vert_idx1], im2_pts_array[vert_idx2], im2_pts_array[vert_idx3]])
            im_mean_tri = np.array([new_im2_pts_array[vert_idx1], new_im2_pts_array[vert_idx2], new_im2_pts_array[vert_idx3]])

            #Trans1 = TriAffine(im_mean_tri, im1_tri)
            Trans2 = TriAffine(im_mean_tri, im2_tri)

            poly_mean_x_idxs, poly_mean_y_idxs = polygon(im_mean_tri[:,0], im_mean_tri[:,1])

            #poly1_x_idxs, poly1_y_idxs = Trans1.transform(poly_mean_x_idxs, poly_mean_y_idxs)
            poly2_x_idxs, poly2_y_idxs = Trans2.transform(poly_mean_x_idxs, poly_mean_y_idxs)

            mean_im[poly_mean_x_idxs, poly_mean_y_idxs] = im2[poly2_x_idxs, poly2_y_idxs]
        plt.imshow(mean_im)
        imsave("output/more_smiling_alpha_" + str(alpha) + ".jpg", mean_im)
        plt.show()
