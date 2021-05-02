# This file is not in use
import cv2
import numpy as np,sys
import matplotlib.pyplot as plt

def imshow_and_wait(title_text,img , wait_p =True):
    cv2.imshow(title_text,img)
    if wait_p ==True:
        cv2.waitKey()
        cv2.destroyAllWindows()

name = "6.jpg"
img = cv2.imread("Data/input/"+name, 1)
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# imshow_and_wait("gray_image",img_gray,False)

#  energy function

def e1(img_gray):
    diff_xdir = np.diff(img_gray.astype(np.int64),axis=1)
    diff_xdir = np.concatenate((diff_xdir,diff_xdir[:,-1].reshape(-1,1)),axis=1)
    diff_ydir = np.diff(img_gray.astype(np.int64),axis=0)
    diff_ydir = np.concatenate((diff_ydir,diff_ydir[-1,:].reshape(1,-1)),axis=0)
    e1 = np.absolute(diff_xdir) + np.absolute(diff_ydir)
    return e1.astype(np.uint8)

# generate Gaussian pyramid for image
G = img_gray.copy()

gpA = [G]
for i in range(2):
    G = cv2.pyrDown(G)
    gpA.append(G) #.astype(np.uint8)
    cv2.imshow(str(i),G)


    # plt.imshow(gpA,cmap='gray')
    # plt.show()
[r, l, p] = np.shape(img)   
[r1, l1] = np.shape(gpA[1])   
print(r,l,p,r1,l1)

gradient = e1(gpA[1])
cv2.imshow("energy",gradient)

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
    # img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_gray = img
    print(img_gray.shape)
    img_copy = img_gray.copy()
    for i in range(n):
        print(i)
        img_energy = e1(img_copy)
        v_seams.append(vertical_seam(img_energy))
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[v_seams[-1][:,0],v_seams[-1][:,1]] = False
        img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
    return v_seams
def compose_vertical_seams(v_seam1,v_seam2):
    compose = np.vectorize(lambda x,y : y+1 if x <= y else y)
    composed_seam = np.empty(v_seam2.shape,dtype=np.int64)
    composed_seam[:,0] = v_seam2[:,0]
    composed_seam[:,1] = compose(v_seam1[:,1],v_seam2[:,1])
    return composed_seam

def transform_vertical_seams(v_seams):
    transformed_seams = [v_seams[0]]
    for v_seam in v_seams[1:]:
        temp_seam = v_seam
        for transformed_seam in transformed_seams[::-1]:
            temp_seam = compose_vertical_seams(transformed_seam,temp_seam)
        transformed_seams.append(temp_seam)
    return transformed_seams

k = vertical_seam(gradient)
kl= multiple_vertical_seams(gradient,20)
ll=transform_vertical_seams(kl)
print (k)
print (kl)
print('tt output')
print (ll)


def horizontal_seam(gradient):
    [row, col] =np.shape(gradient)
    horizontalseam_loc = np.empty((row,col,2),dtype=np.int64)
    horizontalseam_loc[:,0,0] = gradient[:,0]
    horizontalseam_loc[:,0,1] = 0
    for j in range(1,col):
        for i in range(0,row):
            if i == 0:
                horizontalseam_loc[i,j,1] = np.argmin([horizontalseam_loc[i,j-1,0],horizontalseam_loc[i+1,j-1,0]]) 
            elif i == row-1:
                horizontalseam_loc[i,j,1]=np.argmin([horizontalseam_loc[i-1,j-1,0],horizontalseam_loc[i,j-1,0]]) - 1 
            else:
                horizontalseam_loc[i,j,1]=np.argmin([horizontalseam_loc[i-1,j-1,0],horizontalseam_loc[i,j-1,0],horizontalseam_loc[i+1,j-1,0]]) - 1
            horizontalseam_loc[i,j,0] = gradient[i,j] + horizontalseam_loc[i+horizontalseam_loc[i,j,1],j-1,0]
    l = np.argmin(horizontalseam_loc[:,-1,0])
    seam_path = [(l,col-1)]
    for j in range(col-2,-1,-1):
        l = l+horizontalseam_loc[l,j+1,1]
        seam_path = seam_path + [(l,j)]
    seam_path.reverse()
    return np.array(seam_path,dtype=np.int64)

def multiple_horizontal_seams(img,n):
    h_seams = []
    print(img.shape)
    img_copy = img.copy()
    for i in range(n):
        print(i)
        gradient=e1(img)
        h_seams.append(horizontal_seam(gradient))
        img_copy = np.rollaxis(img_copy,1)
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[h_seams[-1][:,1],h_seams[-1][:,0]] = False
        img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
        img_copy = np.rollaxis(img_copy,1)
    return h_seams

def repeated_hor_seams(img,n,energy_function=1):
    h_seams = []
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print(img_gray.shape)
    img_copy = img_gray.copy()
    for i in range(n):
        print(i)
        img_energy = get_energy(img_copy,energy_function)
        h_seams.append(hor_seam(img_energy))
        img_copy = np.rollaxis(img_copy,1)
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[h_seams[-1][:,1],h_seams[-1][:,0]] = False
        img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
        img_copy = np.rollaxis(img_copy,1)
    return h_seams


def compose_horizontal_seams(h_seam1,h_seam2):
    compose = np.vectorize(lambda x,y : y+1 if x <= y else y)
    composed_seam = np.empty(h_seam2.shape,dtype=np.int64)
    composed_seam[:,1] = h_seam2[:,1]
    composed_seam[:,0] = compose(h_seam1[:,0],h_seam2[:,0])
    return composed_seam

def transform_horizontal_seams(h_seams):
    transformed_seams = [h_seams[0]]
    for h_seam in h_seams[1:]:
        temp_seam = h_seam
        for transformed_seam in transformed_seams[::-1]:
            temp_seam = compose_horizontal_seams(transformed_seam,temp_seam)
        transformed_seams.append(temp_seam)
    return transformed_seams

kk = horizontal_seam(gradient)
klk= multiple_horizontal_seams(gradient,20)
llk=transform_horizontal_seams(kl)
print (kk)
print (klk)
print('tt output')
print (llk)
# seam removal 
# sum = []
# N=20
# L=2
# n=[0]*(L)
# n[L-1]=N
# for i in range(L,-1,-1):
#     for j in range(i):
#         sum = n[j]* 2**j
#         n[i] = (N - sum)
#     n[i]=n[i]//2**i

# print (n)

img_copy = np.rollaxis(img,1)
cv2.imshow("image",img_copy)
img_copy2 = np.rollaxis(img_copy,1)
cv2.imshow("image1",img_copy2)
cv2.waitKey()
cv2.destroyAllWindows()



