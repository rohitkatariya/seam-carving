import cv2
import math
import numpy as np,sys
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.stats import entropy as scipy_entropy
from SeamCarving import SeamCarving
# #  energy function

def gen_e1(img_gray):

    # img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    diff_xdir = np.diff(img_gray.astype(np.int64),axis=1)
    diff_xdir = np.concatenate((diff_xdir,diff_xdir[:,-1].reshape(-1,1)),axis=1)
    diff_ydir = np.diff(img_gray.astype(np.int64),axis=0)
    diff_ydir = np.concatenate((diff_ydir,diff_ydir[-1,:].reshape(1,-1)),axis=0)
    e1 = np.absolute(diff_xdir) + np.absolute(diff_ydir)
    return e1.astype(np.uint8)

# sobel energy function

def soble_e1(img_gray):
    horizontal_sobel = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])
    vertical_sobel = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
    img_gray_f = img_gray.astype(np.float32)
    dx = convolve(img_gray_f, horizontal_sobel)
    dy = convolve(img_gray_f, vertical_sobel)
    # dx = convolve(img_gray, horizontal_sobel)
    # dy = convolve(img_gray, vertical_sobel)
    e1 = np.sqrt(dx**2 + dy**2)/np.sqrt(2)
    return e1.astype(np.uint8)

# entropy energy function
# def entropy_e2(img_gray):

#     # img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#     diff_xdir = np.diff(img_gray.astype(np.int64),axis=1)
#     diff_xdir = np.concatenate((diff_xdir,diff_xdir[:,-1].reshape(-1,1)),axis=1)
#     diff_ydir = np.diff(img_gray.astype(np.int64),axis=0)
#     diff_ydir = np.concatenate((diff_ydir,diff_ydir[-1,:].reshape(1,-1)),axis=0)
#     e2 = np.absolute(diff_xdir) + np.absolute(diff_ydir)
#     return e2.astype(np.uint8)
def entropy_e1(img_gray):
    img_gray_pad = cv2.copyMakeBorder(img_gray,8,8,8,8,cv2.BORDER_REPLICATE)
    entropy = np.empty(img_gray_pad.shape)
    for x in range(0,img_gray.shape[0]):
        for y in range(0,img_gray.shape[1]):
            _, counts = np.unique(img_gray_pad[x:x+9, y:y+9], return_counts=True)
            entropy[x, y] = scipy_entropy(counts, base=2)
            
    e1_energy = gen_e1(img_gray)
    entropy_roi = entropy[8:img_gray.shape[0]+8,8:img_gray.shape[1]+8]
    entropy_energy = entropy_roi.astype(np.uint8) + e1_energy
    return entropy_energy

# HOG energy function
# def HOG_e2(img_gray):

#     # img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#     diff_xdir = np.diff(img_gray.astype(np.int64),axis=1)
#     diff_xdir = np.concatenate((diff_xdir,diff_xdir[:,-1].reshape(-1,1)),axis=1)
#     diff_ydir = np.diff(img_gray.astype(np.int64),axis=0)
#     diff_ydir = np.concatenate((diff_ydir,diff_ydir[-1,:].reshape(1,-1)),axis=0)
#     e2 = np.absolute(diff_xdir) + np.absolute(diff_ydir)
#     return e2.astype(np.uint8)
def HOG_e1(img_gray):
    # hist = get_hog(img_gray_f)
    img_gray_f = img_gray.astype(np.float32)/255
    gx = cv2.Sobel(img_gray_f, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img_gray_f, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 8
    bin = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = []
    mag_cells = []
    cellx = celly = 11
    for i in range(0, int(img_gray_f.shape[0]/celly)):
        for j in range(0, int(img_gray_f.shape[1]/cellx)):
            bin_cells.append(bin[i*celly:i*celly+celly, j*cellx:j*cellx+cellx])
            mag_cells.append(mag[i*celly:i*celly+celly, j*cellx:j*cellx+cellx])
    hists = []
    for b, m in zip(bin_cells, mag_cells):
        hists.append(np.bincount(b.ravel(), m.ravel(), bin_n))
    hist = np.hstack(hists)
    eps = 1e-7
    hist = hist/hist.sum() + eps
    hist = np.sqrt(hist)
    hist = hist/np.linalg.norm(hist) + eps
    print(hist)
    maximum = np.max(hist)

    e1 = gen_e1(img_gray)/maximum
    return e1.astype(np.uint8)
# CAUTION
e1=entropy_e1    

def vertical_seam(img_e1):
    [row,col] = np.shape(img_e1)
    verseampixloc = np.empty((row,col,2),dtype=np.int64)
    verseampixloc[0,:,0] = img_e1[0,:]
    verseampixloc[0,:,1] = 0
    for i in range(1,row):
        for j in range(0,col):
            if j == 0:
               verseampixloc[i,j,1] = np.argmin([verseampixloc[i-1,j,0],verseampixloc[i-1,j+1,0]])  
            elif j == col-1:
               verseampixloc[i,j,1]= np.argmin([verseampixloc[i-1,j-1,0],verseampixloc[i-1,j,0]]) - 1
            else:
               verseampixloc[i,j,1]= np.argmin([verseampixloc[i-1,j-1,0],verseampixloc[i-1,j,0],verseampixloc[i-1,j+1,0]]) - 1
           
            verseampixloc[i,j,0] = img_e1[i,j] + verseampixloc[i-1,j+verseampixloc[i,j,1],0]
    minval_lastrow = np.argmin(verseampixloc[-1,:,0])
    seam_path = [(row-1,minval_lastrow)]
    for i in range(row-2,-1,-1):
        minval_lastrow = minval_lastrow+verseampixloc[i+1,minval_lastrow,1]
        seam_path = seam_path + [(i,minval_lastrow)]
    seam_path.reverse()
    return np.array(seam_path,dtype=np.int64)

def multiple_vertical_seams(img,n):
    v_seams = []
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # img_gray = img
    print(img_gray.shape)
    img_copy = img_gray.copy()
    for i in range(n):
        print(i)
        img_energy = e1(img_copy)
        v_seams.append(vertical_seam(img_energy))
        # img[v_seams[i][:, 0], v_seams[i][:, 1]] = [0, 0, 255]
        # cv2.imshow("Image", img)
        # cv2.waitKey()c
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[v_seams[-1][:,0],v_seams[-1][:,1]] = False
        img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
    return v_seams

# k = vertical_seam(gradient)

# print('tt output')
# print (ll)

def numseamslayer(N, l): # calculate how many seams to be removed from each layer
    # l = 3
    L = l - 1
    n = [0]*(L+1)
    for i in range(L, -1, -1):
        sum = 0
        for j in range(i+1, L+1):
            if j > l:
                sum = 0
                continue
            sum += (n[j] * 2**j)
        n[i] = math.floor((N - sum) / 2**i)
        # n[i] = (N - sum) // 2**i
    print(n)
    return n

def get_upsampled_seams(seams, target_shape):
    upsampled_seams = []
    row, col = target_shape
    for seam in seams:
        upsampled_seam = []
        for r, c in seam:
            if 2*r < row and 2*c < col:
                upsampled_seam.append([2*r, 2*c])
            if 2*r+1 < row and 2*c < col:
                upsampled_seam.append([2*r+1, 2*c])
            if 2*r < row and 2*c+1 < col:
                upsampled_seam.append([2*r, 2*c+1])
            if 2*r+1 < row and 2*c+1 < col:
                upsampled_seam.append([2*r+1, 2*c+1])
        upsampled_seams.append(np.array(upsampled_seam))
    return upsampled_seams

def hor_decrease_by_seams(img, v_seams):
    img_copy = img.copy()
    # width_decrease = img.shape[1] - num_seams
    # v_seams = repeated_vert_seams(img,width_decrease,energy_function)
    for j, v_seam in enumerate(v_seams):
        # print(v_seam)
        # print(v_seam.shape, img_copy.shape)
        # img_copy[v_seam[:, 0], v_seam[:, 1]] = [0, 0, 255]
        # cv2.imshow("Image", img_copy)
        # cv2.waitKey()
        mask = np.ones(img_copy.shape, dtype=bool)
        mask[v_seam[:, 0], v_seam[:, 1]] = False
        img_copy = img_copy[mask].reshape(
            img_copy.shape[0], img_copy.shape[1]-1, 3)
        # cv2.imshow("Hor Decrease " + str(i), img_copy)
        # cv2.waitKey()
    return img_copy


def hor_decrease_by_seams2(img, v_seams, target_shape):
    img_copy = img.copy()
    upsampled_seams = []

    for j, v_seam in enumerate(v_seams):
        print(v_seam.shape, img_copy.shape)
        # print(v_seam)
        v_seam = get_upsampled_seams([v_seam], target_shape)
        v_seam = np.array(v_seam[0])
        upsampled_seams.append(v_seam)
        print(v_seam.shape, img_copy.shape)
        mask = np.ones(img_copy.shape, dtype=bool)
        mask[v_seam[:, 0], v_seam[:, 1]] = False
        i = v_seam.shape[0]//mask.shape[0]
        # cv2.imshow("image", img_copy)
        # cv2.waitKey()
        # cv2.imshow("mask", np.array(mask*255, dtype = np.uint8))
        # cv2.waitKey()
        # show_multiple_img(["image", "mask"], [img_copy, np.array(mask*255, dtype=np.uint8)], 0)
        
        img_copy = img_copy[mask].reshape(img_copy.shape[0], img_copy.shape[1]-i, 3)
    # cv2.destroyAllWindows()

    return img_copy, upsampled_seams
def getEvenImage(img,name):
    r,c = 0,0
    r = 4-img.shape[0]%4
    c = 4-img.shape[1]%4
    if r>0 or c>0:
        sc = SeamCarving(img,-r,-c,img_name=name)
    else:
        return (r,c),None
    return (r,c),sc.seam_inserted_image


name = "3.jpg"
img = cv2.imread("Data/input/"+name, 1)
orig_imshape = (img.shape)
(r_added,c_added),img= getEvenImage(img,name)
print(img.shape)
# input()
# print(orig_imshape)
# print(img.shape)
# cv2.imshow("energy image", e1(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)) )
# cv2.waitKey(1)
# input()
# exit()
# print(img.shape)
# cv2.waitKey()
# exit()
# img = img2.copy()
# row, col = False, False
# if img2.shape[0] % 2 != 0:
#     img = img2[:-1, :]
#     # img = cv2.resize(img, (img.shape[1], img.shape[0]-1))
#     row = True
# if img2.shape[1] % 2 != 0:
#     img = img2[:, :-1]
#     # img = cv2.resize(img, (img.shape[1]-1, img.shape[0]))
#     col = True

# print(img.shape)

def adjust_row_col(image):
    row, col = 0, 0
    if image.shape[1] % 2 != 0 and image.shape[0] % 2 != 0:
        image = image[:-1, :-1]
        row = 1
        col = 1
    elif image.shape[0] % 2 != 0:
        image = image[:-1, :]
        row = 1
    elif image.shape[1] % 2 != 0:
        image = image[:, :-1]
        col = 1

    return row, col, image

# generate Gaussian pyramid for image
G = img.copy()

# rows = []
# cols = []
# row, col, G = adjust_row_col(G)
# rows.append(row)
# cols.append(col)
# gpA = [G]
# for i in range(2):
 
#     down_shape = (G.shape[1]//2, G.shape[0]//2)
#     print(G.shape, down_shape)
#     G = cv2.pyrDown(G, dstsize=down_shape)
#     row, col, G = adjust_row_col(G)
#     rows.append(row)
#     cols.append(col)
#     gpA.append(G) #.astype(np.uint8)
#     # cv2.imshow(str(i),G)

rows = []
cols = []
# row, col, G = adjust_row_col(G)
# rows.append(row)
# cols.append(col)
gpA = []
for i in range(2):

    row, col, G = adjust_row_col(G)
    rows.append(row)
    cols.append(col)
    gpA.append(G)  # .astype(np.uint8)
    down_shape = (G.shape[1]//2, G.shape[0]//2)
    print(G.shape, down_shape)
    G = cv2.pyrDown(G, dstsize=down_shape)
    
    # cv2.imshow(str(i),G)
gpA.append(G)

for g in gpA:
    print(g.shape)
print("rows", rows)
print("cols", cols)

# newrow = 153
# newcol = 153
# [r, l, p] = np.shape(img)
# print(np.shape(img))
# print([r, l, p])
# num_hori_seams,num_vert_seams=[int(j) for j in input("enter number of rows and columns to remove").split()] # l-newcol
num_hori_seams,num_vert_seams = 60,70
num_hori_seams += r_added
num_vert_seams += c_added
num_vert_seams_layer = numseamslayer(num_vert_seams,3)
num_hort_seams_layer = numseamslayer(num_hori_seams,3)

print("num_vert_seams_layer", num_vert_seams_layer)
# print("num_vert_seams_layer", num_vert_seams_layer2)
print("num_hort_seams_layer", num_hort_seams_layer)
# exit()

# original_seams = multiple_vertical_seams(img, 1)
# mask = np.ones(img.shape, dtype=bool)
# mask[original_seams[0][:, 0], original_seams[0][:, 1]] = False
# cv2.imshow("original image", img)
# cv2.waitKey()
# cv2.imshow("mask", np.array(mask*255, dtype = np.uint8))
# cv2.waitKey()

num_layers = 3
images_p = []
vseams_p = []
vseams_p_ = [None]


last_img = gpA[num_layers-1]
print("i2 shape: ", last_img.shape)
images_p.append(last_img)
vseams_p.append(multiple_vertical_seams(last_img, num_vert_seams_layer[num_layers-1]))
images_p[-1] = hor_decrease_by_seams(images_p[-1], vseams_p[-1])
# images_p[-1] = cv2.resize(images_p[-1],(images_p[-1].shape[1]+cols[-1], images_p[-1].shape[0]))
print("i2 shape after decrease: ", images_p[-1].shape)

images_p.append(gpA[1])
print("i1 shape: ", gpA[1].shape)
# vseams_p_.append(get_upsampled_seams(vseams_p[0]))
img_j, up_samp_seams_1 = hor_decrease_by_seams2(gpA[1], vseams_p[-1], gpA[1].shape[:-1])
vseams_p_.append(up_samp_seams_1)
vseams_p.append(multiple_vertical_seams(img_j, num_vert_seams_layer[1]))
img_j = hor_decrease_by_seams(img_j, vseams_p[-1])
img_j = cv2.resize(img_j,(img_j.shape[1]+cols[1], img_j.shape[0]))
images_p[-1] = img_j
print("i1 shape after decrease: ", images_p[-1].shape)

images_p.append(gpA[0])
print("i0 shape: ", gpA[0].shape)
# vseams_p_.append(get_upsampled_seams(vseams_p[0]))
img_j, up_samp_seams_0 = hor_decrease_by_seams2(gpA[0], vseams_p_[-1], gpA[0].shape[:-1])
img_j, up_samp_seams_0 = hor_decrease_by_seams2(img_j, vseams_p[-1], img_j.shape[:-1])
vseams_p.append(multiple_vertical_seams(img_j, num_vert_seams_layer[0]))
img_j = hor_decrease_by_seams(img_j, vseams_p[-1])
img_j = cv2.resize(img_j,(img_j.shape[1]+cols[0], img_j.shape[0]))
images_p[-1] = img_j
print("i0 shape after decrease: ", images_p[-1].shape)

hseams_p = []
hseams_p_ = [None]

print("i2 shape: ", images_p[0].shape)
hseams_p.append(multiple_vertical_seams(np.rollaxis(images_p[0], 1), num_hort_seams_layer[num_layers-1]))
images_p[0] = hor_decrease_by_seams(np.rollaxis(images_p[0], 1), hseams_p[-1])
images_p[0] = np.rollaxis(images_p[0], 1)
# images_p[0] = cv2.resize(images_p[0],(images_p[0].shape[1], images_p[0].shape[0]+rows[-1]))
print("i2 shape after decrease: ", images_p[0].shape)

print("i1 shape: ", images_p[1].shape)
img_j, up_samp_seams_1 = hor_decrease_by_seams2(np.rollaxis(images_p[1], 1), hseams_p[-1], images_p[1].shape[:-1][::-1])
hseams_p_.append(up_samp_seams_1)
hseams_p.append(multiple_vertical_seams(img_j, num_hort_seams_layer[1]))
img_j = hor_decrease_by_seams(img_j, hseams_p[-1])
images_p[1] = img_j
images_p[1] = np.rollaxis(images_p[1], 1)
images_p[1] = cv2.resize(images_p[1],(images_p[1].shape[1], images_p[1].shape[0]+rows[0]))
print("i1 shape after decrease: ", images_p[1].shape)

print("i0 shape: ", images_p[0].shape)
img_j, up_samp_seams_0 = hor_decrease_by_seams2(np.rollaxis(images_p[2], 1), hseams_p_[-1], images_p[2].shape[:-1][::-1])
img_j, up_samp_seams_0 = hor_decrease_by_seams2(img_j, hseams_p[-1], images_p[2].shape[:-1][::-1])
hseams_p.append(multiple_vertical_seams(img_j, num_hort_seams_layer[0]))
img_j = hor_decrease_by_seams(img_j, hseams_p[-1])
images_p[2] = img_j
images_p[2] = np.rollaxis(images_p[2], 1)
images_p[2] = cv2.resize(images_p[2],(images_p[2].shape[1], images_p[2].shape[0]+rows[1]))
print("i0 shape after decrease: ", images_p[2].shape)


# for i, image in enumerate(images_p):
#     cv2.imshow(str(i), image)
#     cv2.waitKey()

# for j in range(num_layers-2, -1, -1):
#     num_seams = num_vert_seams_layer[j]
#     layer = gpA[j]
#     # layer_gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
#     seam_j = []
#     seam_p = vseams_p[-1] # if vseams_p[-1] is not None
#     # if seam_p is not None:
#     seam_p_ = get_upsampled_seams(seam_p)
#     vseams_p_.append(seam_p_)
#     seam_j.extend(seam_p_)
#     scj = multiple_vertical_seams(layer, num_seams)
#     seam_j.extend(scj)
#     vseams_p.append(seam_j)
#     img_j = hor_decrease_by_seams(layer, seam_j)
#     images_p.append(img_j)

# images_p = images_p[::-1]
# vseams_p = vseams_p[::-1]
# vseams_p_ = vseams_p_[::-1]

# for i, image in enumerate(images_p):
#     cv2.imshow(str(i), image)
#     cv2.waitKey()
#     # print(len(i))
#     print(image.shape)

# for i, image2 in enumerate(gpA):
#     cv2.imshow(str(i), image2)
#     cv2.waitKey()
#     # print(len(i))
#     print(image2.shape)


if row:
    # img = img[:-1, :]
    images_p[-1] = cv2.resize(images_p[-1], (img.shape[1], img.shape[0]+1))
if col:
    # img = img[:, :-1]
    images_p[-1] = cv2.resize(images_p[-1], (img.shape[1]+1, img.shape[0]))


cv2.imshow("original image", gpA[0])
# cv2.waitKey()
cv2.imshow("resized image", images_p[-1])
cv2.waitKey()



# img_copy = np.rollaxis(img,1)
# cv2.imshow("image",img_copy)
# img_copy2 = np.rollaxis(img_copy,1)
# cv2.imshow("image1",img_copy2)
# cv2.waitKey()
cv2.destroyAllWindows()



