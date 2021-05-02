# This file is not in use
import cv2
import numpy as np,sys
import matplotlib.pyplot as plt
import pdb
import objectRemove
import traceback
def wait_and_destroy():
    cv2.waitKey()
    cv2.destroyAllWindows()

class SeamCarving:
    
    def e1(self,img_gray):
        diff_xdir = np.diff(img_gray.astype(np.int64),axis=1)
        diff_xdir = np.concatenate((diff_xdir,diff_xdir[:,-1].reshape(-1,1)),axis=1)
        diff_ydir = np.diff(img_gray.astype(np.int64),axis=0)
        diff_ydir = np.concatenate((diff_ydir,diff_ydir[-1,:].reshape(1,-1)),axis=0)
        e1 = np.absolute(diff_xdir) + np.absolute(diff_ydir)
        return e1.astype(np.uint8)

    def vertical_seam(self,img_e1):
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

    def multiple_vertical_seams(self,img,n):
        v_seams = []
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # img_gray = img
        print(img_gray.shape)
        img_copy = img_gray.copy()
        if self.object_removal_flag:
            object_removal_mask = self.or_obj.object_mask.copy()
            n = img_copy.shape[1]
        for i in range(n):
            print(i)
            img_energy = self.e1(img_copy)
            if self.object_removal_flag:
                try:
                    # pdb.set_trace()
                    if np.min(object_removal_mask)>0:
                        break
                    img_energy = img_energy.astype(np.int64)
                    img_energy = img_energy + (255-object_removal_mask).astype(bool).astype(img_energy.dtype)*-99999
                    # img_energy = img_energy
                    # img_energy = cv2.bitwise_and(img_energy, object_removal_mask.astype(img_energy.dtype)) 
                except:
                    traceback.print_exc()
                    pdb.set_trace()
            v_seams.append(self.vertical_seam(img_energy))
            mask = np.ones(img_copy.shape,dtype=bool)
            mask[v_seams[-1][:,0],v_seams[-1][:,1]] = False
            if i%1 ==0:
                
                temp_img = cv2.bitwise_and(img, img, mask = mask.astype(np.uint8))
                cv2.imshow("object_removal_mask",object_removal_mask)
                cv2.imshow("{}".format(i//10),temp_img)
                
                cv2.waitKey(1)
                if i%10 ==10:
                    wait_and_destroy()
            img = img[mask].reshape(img.shape[0],img.shape[1]-1,3)
            object_removal_mask = object_removal_mask[mask].reshape(object_removal_mask.shape[0],object_removal_mask.shape[1]-1)
            img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
            # pdb.set_trace()
            
        return v_seams,img_copy

    def horizontal_seam(self,gradient):
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

    def multiple_horizontal_seams(self,img,n):
        h_seams = []
        print(img.shape)
        img_copy = img.copy()
        for i in range(n):
            print(i)
            gradient=self.e1(img)
            h_seams.append(horizontal_seam(gradient))
            img_copy = np.rollaxis(img_copy,1)
            mask = np.ones(img_copy.shape,dtype=bool)
            mask[h_seams[-1][:,1],h_seams[-1][:,0]] = False
            img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
            img_copy = np.rollaxis(img_copy,1)
        return h_seams

    def __init__(self, img, object_removal_flag=False):
        self.img_orig = img
        self.img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        e1_energy = self.e1(self.img_gray)
        self.object_removal_flag = object_removal_flag
        if object_removal_flag:
            self.or_obj = objectRemove.ObjectRemoval(self.img_orig)
            kl= self.multiple_vertical_seams(self.img_orig,-1)
        # pdb.set_trace()
        else:
            kl= self.multiple_vertical_seams(self.img_orig,40)
        wait_and_destroy()
        pdb.set_trace()
        exit()
        # kk = horizontal_seam(gradient)
        # klk= multiple_horizontal_seams(gradient,20)
        # llk=transform_horizontal_seams(kl)
        # print (kk)
        # print (klk)
        # print('tt output')
        # print (llk)

if __name__ == "__main__":
    name = "1.jpg"
    img = cv2.imread("Data/input/"+name, 1)
    sc = SeamCarving(img,object_removal_flag=True)