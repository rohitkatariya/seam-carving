import os
import config
import cv2
import math
import numpy as np,sys
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.stats import entropy as scipy_entropy

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

for filename in (os.listdir(config.input_dir)):
    img = cv2.imread(config.input_dir+filename, 0)
    energy_dict = {
    "hog_energy" : HOG_e1(img),
    "entropy_energy" : entropy_e1(img),
    "sobel_energy" : soble_e1(img),
    "gen_energy" : gen_e1(img)
    }
    for es,ei in energy_dict.items():
        mapped_color= cv2.applyColorMap(ei.astype(np.uint8), cv2.COLORMAP_HSV) 
        filename2 = "Data/output/energy/{}{}.jpg".format(filename,es)
        cv2.imwrite( filename2 , mapped_color)
        print (filename2)
