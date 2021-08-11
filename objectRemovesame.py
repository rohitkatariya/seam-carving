import cv2
import cv2
import numpy as np,sys
import matplotlib.pyplot as plt
import pdb
import config
import SeamCarving

def wait_and_destroy():
    cv2.waitKey()
    cv2.destroyAllWindows()

class ObjectRemoval:
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
        energ_min = verseampixloc[-1,minval_lastrow,0]
        seam_path = [(row-1,minval_lastrow)]
        for i in range(row-2,-1,-1):
            minval_lastrow = minval_lastrow+verseampixloc[i+1,minval_lastrow,1]
            seam_path = seam_path + [(i,minval_lastrow)]
        seam_path.reverse()
        return np.array(seam_path,dtype=np.int64),energ_min

    def remove_vertical_seam(self):
        
        img_gray = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
        print(img_gray.shape)
        img_copy = img_gray.copy()
        object_removal_mask = self.object_mask
        img_energy = self.e1(img_copy)
        img_energy = img_energy.astype(np.int64)
        img_energy = img_energy + (255-object_removal_mask).astype(bool).astype(img_energy.dtype)*-99999
        v_seam_this=self.vertical_seam(img_energy)
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[v_seam_this[:,0],v_seam_this[:,1]] = False
        
        temp_img = cv2.bitwise_and(self.img, self.img, mask = mask.astype(np.uint8))
        cv2.imshow("object_removal_mask",object_removal_mask)
        cv2.imshow("intermediate image",temp_img)
        cv2.waitKey(1)
            # if i%10 ==10:
            #     wait_and_destroy()
        pdb.set_trace()
        self.img = self.img[mask]#.reshape(self.img.shape[0],self.img.shape[1]-1,3)
        self.object_mask = object_removal_mask[mask]#.reshape(object_removal_mask.shape[0],object_removal_mask.shape[1]-1)
        
        # img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
        # return v_seams,img_copy

    def remove_seam(self):
        
        img_gray = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
        print(img_gray.shape)
        img_copy = img_gray.copy()
        
        img_energy = self.e1(img_copy)
        img_energy = img_energy.astype(np.int64)
        img_energy = img_energy + (255-self.object_mask).astype(bool).astype(img_energy.dtype)*-99999
        img_energy =  img_energy + (255-self.object_mask_save).astype(bool).astype(img_energy.dtype)*2*999
        v_seam_this,energy_vert =self.vertical_seam(img_energy)
        h_seam_this,energy_horz = self.horizontal_seam(img_energy)
        vertical_removal=0
        if energy_vert<energy_horz:
            vertical_removal=1
        if vertical_removal==1:
            mask = np.ones(img_copy.shape,dtype=bool)
            mask[v_seam_this[:,0],v_seam_this[:,1]] = False
            temp_img = cv2.bitwise_and(self.img, self.img, mask = mask.astype(np.uint8))
            self.img = self.img[mask].reshape(self.img.shape[0],self.img.shape[1]-1,3)
            self.object_mask = self.object_mask[mask].reshape(self.object_mask.shape[0],self.object_mask.shape[1]-1)
            self.object_mask_save = self.object_mask_save[mask].reshape(self.object_mask_save.shape[0],self.object_mask_save.shape[1]-1)
        else:
            
            img_copy = np.rollaxis(img_copy,1)
            self.img = np.rollaxis(self.img,1)
            self.object_mask =np.rollaxis(self.object_mask,1)
            self.object_mask_save =np.rollaxis(self.object_mask_save,1)

            mask = np.ones(img_copy.shape,dtype=bool)
            mask[h_seam_this[:,1],h_seam_this[:,0]] = False
            
            temp_img = cv2.bitwise_and(np.rollaxis(self.img,1), np.rollaxis(self.img,1), mask = np.rollaxis(mask,1).astype(np.uint8))

            self.img = self.img[mask].reshape(self.img.shape[0],self.img.shape[1]-1,3)
            self.object_mask = self.object_mask[mask].reshape(self.object_mask.shape[0],self.object_mask.shape[1]-1)
            self.object_mask_save = self.object_mask_save[mask].reshape(self.object_mask_save.shape[0],self.object_mask_save.shape[1]-1)
            
            self.img = np.rollaxis(self.img,1)
            self.object_mask =np.rollaxis(self.object_mask,1)
            self.object_mask_save =np.rollaxis(self.object_mask_save,1)

        
        cv2.imshow("object_removal_mask",self.object_mask)
        cv2.imshow("intermediate image",temp_img)
        cv2.waitKey(1)
            # if i%10 ==10:
            #     wait_and_destroy()
        # pdb.set_trace()
        
            
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
        l_energy = horizontalseam_loc[l,-1,0]
        seam_path = [(l,col-1)]
        for j in range(col-2,-1,-1):
            l = l+horizontalseam_loc[l,j+1,1]
            seam_path = seam_path + [(l,j)]
        seam_path.reverse()
        return np.array(seam_path,dtype=np.int64),l_energy

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

    def draw_circle(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("hello")
            cv2.circle(self.img_copy, (x, y), self.radius_circle, (0, 255, 0), -1)
            print(self.radius_circle)
            cv2.circle(self.object_mask, (x, y), self.radius_circle, 0, -1)
            cv2.imshow("orig image", self.img_copy)
            cv2.imshow("mask image", self.object_mask)
    
    def draw_circle_save(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("hello")
            cv2.circle(self.img_copy, (x, y), self.radius_circle, ( 255,0, 0), -1)
            print(self.radius_circle)
            cv2.circle(self.object_mask_save, (x, y), self.radius_circle, 0, -1)
            cv2.imshow("orig image", self.img_copy)
            cv2.imshow("mask image", self.object_mask_save)

    def seamToRemove(self):
        # return 0
        black_pixels = np.array(np.where(self.object_mask == 0))
        first_ = len(set(black_pixels[0]))
        second_ = len(set(black_pixels[1]))
        if first_>second_:
            print("remove_vertical")
            return 1
        print("remove_horizontal")
        return 0
         
    def __init__(self,img,radius_circle,object_removal_flag=True,save_obj_flag=False,min_seams=0,img_name=''):
        self.img = img.copy()
        self.radius_circle =radius_circle
        # Getting masked object
        self.img_copy = img.copy()
        self.object_mask = np.zeros(self.img.shape[:2],np.uint8)+255
        cv2.imshow("mask image", self.object_mask)
        cv2.namedWindow(winname = "orig image")
        cv2.setMouseCallback("orig image", self.draw_circle)
        
        
        cv2.imshow("orig image", self.img)
        cv2.waitKey()          
        # cv2.destroyAllWindows()
        self.object_mask_save = np.zeros(self.img.shape[:2],np.uint8)+255
        if save_obj_flag:
            cv2.namedWindow(winname = "orig image")
            cv2.setMouseCallback("orig image", self.draw_circle_save)
            cv2.imshow("orig image", self.img)
            cv2.waitKey()          
        cv2.destroyAllWindows()
        # cv2.destroyAllWindows()
        name_transformed = "{}/object_remove/masked_image_{}_{}_{}.jpg".format(config.output_dir,img_name,config.date_str,str(min_seams))
        cv2.imwrite( name_transformed , self.img_copy)

        masked_image_final = self.img.copy()
        for i in range(3):
            masked_image_final[:,:,i]=cv2.bitwise_and(255-self.object_mask_save,masked_image_final[:,:,i])
        name_transformed = "{}/object_remove/object_mask_save_{}_{}_{}.jpg".format(config.output_dir,img_name,config.date_str,str(min_seams))
        cv2.imwrite( name_transformed , masked_image_final)

        masked_image_final = self.img.copy()
        for i in range(3):
            masked_image_final[:,:,i]=cv2.bitwise_and(255-self.object_mask,masked_image_final[:,:,i])
        name_transformed = "{}/object_remove/object_mask_remove{}_{}_{}.jpg".format(config.output_dir,img_name,config.date_str,str(min_seams))
        cv2.imwrite( name_transformed , masked_image_final)
           

        # self.object_mask_save = cv2.bitwise_or(self.object_mask_save.astype(np.bool) , 1-self.object_mask.astype(np.bool))
        # pdb.set_trace()
        # Removing Masked object
        seams_removed =0
        print("min_seams",min_seams)
        while np.min(self.object_mask)==0 or seams_removed<min_seams:
            # stream_orientation = self.seamToRemove()
            self.remove_seam()
            seams_removed+=1
            # if stream_orientation == 1:
            #     self.remove_horizontal_seam()
            # else:
            #     self.remove_vertical_seam()
            # pdb.set_trace()
            # self.img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            # self.object_removal_flag = object_removal_flag
            # kl= self.multiple_vertical_seams(self.img_orig,-1)
        cv2.destroyAllWindows()
        cv2.imshow("transformed image", self.img)
        cv2.waitKey()          
        cv2.destroyAllWindows()
        
        name_transformed = "{}/object_remove/{}_{}_{}.jpg".format(config.output_dir,img_name,config.date_str,str(min_seams))
        cv2.imwrite( name_transformed , self.img)
        # self.name_transformed
        # pdb.set_trace()
        # exit()
        # pdb.set_trace()
if __name__=="__main__":
    # obj_removed = ObjectRemoval()
    name = "12.jpg"
    img = cv2.imread("Data/input/"+name, 1)
    orig_imshape = list(img.shape)
    radius_circle = int(input("enter circle radius : "))
    save_obj_flag = input("do you want to save onject?") == 'y'
    min_seams = int(input("enter min number of seams to remove:"))
    sc = ObjectRemoval(img,radius_circle,object_removal_flag=True,save_obj_flag=save_obj_flag,min_seams=min_seams,img_name = name)
    # pdb.set_trace()
    r,c = (orig_imshape[0] -sc.img.shape[0]),(orig_imshape[1] -sc.img.shape[1])
    SeamCarving.SeamCarving(sc.img.copy(),-r,-c,img_name='output_same_size')
    # sc.img

  

          

  


