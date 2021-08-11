import cv2
import cv2
import numpy as np,sys
import matplotlib.pyplot as plt
import pdb
import config
def wait_and_destroy():
    cv2.waitKey()
    cv2.destroyAllWindows()

class SeamCarving:
    def e1(self,img_gray):
        print("COMPUTING ENERGY")
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

    def remove_seam(self,num_ver,num_hor):
        
        img_copy = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
        
        v_seam_this,energy_vert =self.vertical_seam(self.img_energy)
        h_seam_this,energy_horz = self.horizontal_seam(self.img_energy)
        
        if num_ver ==0:
            vertical_removal = 0
        elif num_hor ==0 :
            vertical_removal=1
        elif energy_vert<energy_horz:
            vertical_removal=1
        else:
            vertical_removal=0
        if vertical_removal==1:
            seam_used = v_seam_this
            mask = np.ones(img_copy.shape,dtype=bool)
            mask[v_seam_this[:,0],v_seam_this[:,1]] = False
            if self.show_temp_image:
                temp_img = cv2.bitwise_and(self.img, self.img, mask = mask.astype(np.uint8))
            self.img = self.img[mask].reshape(self.img.shape[0],self.img.shape[1]-1,3)
            self.img_energy = self.img_energy[mask].reshape(self.img_energy.shape[0],self.img_energy.shape[1]-1)
        else:
            seam_used = h_seam_this
            img_copy = np.rollaxis(img_copy,1)
            self.img = np.rollaxis(self.img,1)
            self.img_energy = np.rollaxis(self.img_energy,1)

            mask = np.ones(img_copy.shape,dtype=bool)
            mask[h_seam_this[:,1],h_seam_this[:,0]] = False
            if self.show_temp_image:
                temp_img = cv2.bitwise_and(np.rollaxis(self.img,1), np.rollaxis(self.img,1), mask = np.rollaxis(mask,1).astype(np.uint8))
            self.img = self.img[mask].reshape(self.img.shape[0],self.img.shape[1]-1,3)
            self.img_energy = self.img_energy[mask].reshape(self.img_energy.shape[0],self.img_energy.shape[1]-1)
            
            self.img = np.rollaxis(self.img,1)
            self.img_energy = np.rollaxis(self.img_energy,1)
            
        if self.show_temp_image:
            cv2.imshow("intermediate image".format(((num_hor)//10)*10 , (num_ver//10)*10),temp_img)
            cv2.waitKey(1)
        return vertical_removal,seam_used
            
    
    # def draw_circle(self,event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         # print("hello")
    #         cv2.circle(self.img_copy, (x, y), self.radius_circle, (0, 255, 0), -1)
    #         print(self.radius_circle)
    #         cv2.circle(self.object_mask, (x, y), self.radius_circle, 0, -1)
    #         cv2.imshow("orig image", self.img_copy)
    #         cv2.imshow("mask image", self.object_mask)
    
    # def draw_circle_save(self,event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         # print("hello")
    #         cv2.circle(self.img_copy, (x, y), self.radius_circle, ( 255,0, 0), -1)
    #         print(self.radius_circle)
    #         cv2.circle(self.object_mask_save, (x, y), self.radius_circle, 0, -1)
    #         cv2.imshow("orig image", self.img_copy)
    #         cv2.imshow("mask image", self.object_mask_save)

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
    def computeEnergy(self,img):
        img_copy = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img_energy = self.e1(img_copy)
        img_energy = img_energy.astype(np.int64)
        # img_energy = img_energy + (255-self.object_mask).astype(bool).astype(img_energy.dtype)*-99999
        # img_energy =  img_energy + (255-self.object_mask_save).astype(bool).astype(img_energy.dtype)*2*999 
        return img_energy

    def apply_seam_transform(self, seam_to_be_added,secondary_seam_obj,vertical_removed):
        secondary_seam_orientation, secondary_seam = secondary_seam_obj
        if secondary_seam_orientation==1:
            # vertical seam
            new_secondary_seam =[]
            if len(seam_to_be_added)!= len(secondary_seam):
                print("ERROR(v) seam sizes don't match: you can't remove seams in both the axis and still get transformed seams")
                pdb.set_trace()
                exit()
            for secondary_idx in range(len(seam_to_be_added)):
                if secondary_seam[secondary_idx][1] >= seam_to_be_added[secondary_idx][1]:
                    this_seam_point = (secondary_seam[secondary_idx][0],secondary_seam[secondary_idx][1]+1)
                    new_secondary_seam.append(this_seam_point)
                else:
                    this_seam_point = (secondary_seam[secondary_idx][0],secondary_seam[secondary_idx][1])
                    new_secondary_seam.append(this_seam_point)
            return (secondary_seam_orientation,new_secondary_seam)
        else:
            # Horizontal seam
            new_secondary_seam =[]
            if len(seam_to_be_added)!= len(secondary_seam):
                print("ERROR(h) seam sizes don't match: you can't remove seams in both the axis and still get transformed seams ")
                pdb.set_trace()
                exit()
            for secondary_idx in range(len(seam_to_be_added)):
                if secondary_seam[secondary_idx][0] >= seam_to_be_added[secondary_idx][0]:
                    this_seam_point = (secondary_seam[secondary_idx][0]+1,secondary_seam[secondary_idx][1])
                    new_secondary_seam.append(this_seam_point)
                else:
                    this_seam_point = (secondary_seam[secondary_idx][0],secondary_seam[secondary_idx][1])
                    new_secondary_seam.append(this_seam_point)
            return (secondary_seam_orientation,new_secondary_seam)
                    
                    

    def seams_transform(self,seams_removed):
        seams_transformed = []
        for seam_idx in range(len(seams_removed)):
            vertical_removed,this_seam = seams_removed[seam_idx]
            seams_transformed.append(this_seam.copy())
            secondary_seam_idx = seam_idx+1
            while secondary_seam_idx<len(seams_removed):
               seams_removed[secondary_seam_idx] = self.apply_seam_transform(this_seam,seams_removed[secondary_seam_idx],vertical_removed)
               secondary_seam_idx+=1
        return seams_transformed

    # def transform_all_seams(self,seams_image):
    #     t_img_helper = np.zeros((self.img_orig.shape[0],self.img_orig.shape[1],2), dtype = np.uint32)
    #     pdb.set_trace()
    #     for this_seam_orientation,this_seam in seams_image:
    #         for seam_point in this_seam:
    #             helper_point = t_img_helper[][] 
    #             # seam_point[0] = seam_point[0]+[]
    #             seam_point[1] 

    def reduce_image_size(self,img_,r,c):
        self.img = img_.copy()
        self.img_energy = self.computeEnergy(self.img.copy())
        seams_removed = []
        while r+c>0:
            # stream_orientation = self.seamToRemove()
            vertical_removal,seam_used = self.remove_seam(c,r)
            print("vertical_removal" if vertical_removal==1 else "horizontal_removal",r,c)
            seams_removed.append( (vertical_removal,seam_used) )
            if vertical_removal == 1:
                c-=1
            else:
                r-=1
            if r<0 or c<0:
                print("Error r,c<0",r,c)
                exit()
        return seams_removed
    def insert_transformed_seams_horizontal(self,img,transformed_seams,show_seams = False):
        # pdb.set_trace()
        new_shape = (img.shape[0]+len(transformed_seams),img.shape[1],3)
        transformed_seams_orig = transformed_seams.copy()
        img_new = np.zeros( new_shape ,np.uint8)
        
        if show_seams:
            seams_image = img.copy()
            for this_transformed_seam in transformed_seams:
                for this_point in this_transformed_seam:
                    seams_image[this_point[0]][this_point[1]] = (255,0,0)
            
        for i in range(img.shape[0]):
            img_new[i,:,:]=img[i]
        while len(transformed_seams)>0:
            this_seam = transformed_seams.pop()
            for seam_point in this_seam:
                img_new[seam_point[0]+1:,seam_point[1],:] = img_new[seam_point[0]:-1,seam_point[1],:]
                for color_idx in range(3):
                    a = int(img_new[seam_point[0],seam_point[1],color_idx])
                    b = int(img_new[max(0,seam_point[0]-1),seam_point[1],color_idx])
                    img_new[seam_point[0],seam_point[1],color_idx] = (a+b)//2
            for seam_idx in range(len(transformed_seams)):
                new_this_transformed_seam = []
                # pdb.set_trace()
                for this_point_idx in range(len(transformed_seams[seam_idx])):
                    this_point = transformed_seams[seam_idx][this_point_idx]
                    if this_point[0]>=this_seam[this_point_idx][0]:
                        this_point = (this_point[0]+1,this_point[1])
                    new_this_transformed_seam.append(this_point)
                transformed_seams[seam_idx] = new_this_transformed_seam
        name_transformed = "{}{}_{}_vs_{}.jpg".format(config.output_dir,self.img_name,config.date_str,len(transformed_seams_orig))
        cv2.imwrite( name_transformed , img_new)
        if show_seams:
            name_transformed = "{}{}_{}_seams_image{}.jpg".format(config.output_dir,self.img_name,config.date_str,len(transformed_seams_orig))
            cv2.imwrite( name_transformed , seams_image)
        return img_new
        
    def insert_transformed_seams_vertical(self,img,transformed_seams,show_seams = False):
        new_shape = (img.shape[0],img.shape[1]+len(transformed_seams),3)
        transformed_seams_orig = transformed_seams.copy()
        img_new = np.zeros( new_shape ,np.uint8)
        
        if show_seams:
            seams_image = img.copy()
            for this_transformed_seam in transformed_seams:
                for this_point in this_transformed_seam:
                    seams_image[this_point[0]][this_point[1]] = (255,0,0)
            
        for i in range(img.shape[0]):
            img_new[i,:img.shape[1],:]=img[i]
        while len(transformed_seams)>0:
            this_seam = transformed_seams.pop()
            for seam_point in this_seam:
                img_new[seam_point[0],seam_point[1]+1:,:] = img_new[seam_point[0],seam_point[1]:-1,:]
                for color_idx in range(3):
                    a = int(img_new[seam_point[0],seam_point[1],color_idx])
                    b = int(img_new[seam_point[0],max(0,seam_point[1]-1),color_idx])
                    img_new[seam_point[0],seam_point[1],color_idx] = (a+b)//2
            for seam_idx in range(len(transformed_seams)):
                new_this_transformed_seam = []
                # pdb.set_trace()
                for this_point_idx in range(len(transformed_seams[seam_idx])):
                    this_point = transformed_seams[seam_idx][this_point_idx]
                    if this_point[1]>=this_seam[this_point_idx][1]:
                        this_point = (this_point[0],this_point[1]+1)
                    new_this_transformed_seam.append(this_point)
                transformed_seams[seam_idx] = new_this_transformed_seam
        name_transformed = "{}{}_{}_vs_{}.jpg".format(config.output_dir,self.img_name,config.date_str,len(transformed_seams_orig))
        cv2.imwrite( name_transformed , img_new)
        if show_seams:
            name_transformed = "{}{}_{}_seams_image{}.jpg".format(config.output_dir,self.img_name,config.date_str,len(transformed_seams_orig))
            cv2.imwrite( name_transformed , seams_image)
        return img_new
    def insert_seams(self,img_,r,c):
        seams_removed = self.reduce_image_size(img_.copy(),r,c)
        # pdb.set_trace()
        transformed_seams = self.seams_transform(seams_removed)
        if r>0:
            new_image = self.insert_transformed_seams_horizontal(img_.copy(),transformed_seams)
        else:
            new_image = self.insert_transformed_seams_vertical(img_.copy(),transformed_seams)
        return new_image
    def __init__(self,img,r,c,img_name=""):
        self.img_orig = img
        print("img shape:",img.shape)
        self.show_temp_image=True
        self.img_energy_orig = self.computeEnergy(self.img_orig)
        self.img_name = ".".join(img_name.split(".")[:-1])
        if r>=0 and c>=0:
            # General case
            seams_removed = self.reduce_image_size(self.img_orig.copy(),r,c)
            if r==0 or c==0:
                transformed_seams = self.seams_transform(seams_removed)
                seams_image = self.img_orig.copy()
                for this_transformed_seam in transformed_seams:
                    print("transforming seam")
                    for this_point in this_transformed_seam:
                        seams_image[this_point[0]][this_point[1]] = (0,255,0)
                cv2.imshow("seams image", seams_image)
                name_transformed = "{}{}_{}_seams_image{}_c{}.jpg".format(config.output_dir,self.img_name,config.date_str,r,c)
                cv2.imwrite( name_transformed , seams_image)
            name_transformed = "{}{}_{}_sizedown_r{}_c{}.jpg".format(config.output_dir,self.img_name,config.date_str,r,c)
            cv2.imwrite( name_transformed , self.img)
            
            # cv2.COLORMAP_HOT
            cv2.imshow("energy", cv2.applyColorMap(self.img_energy_orig.astype(np.uint8), cv2.COLORMAP_HSV) )
            cv2.imshow("orig image", self.img_orig)
            cv2.imshow("transformed image", self.img)
            cv2.waitKey()          
            cv2.destroyAllWindows()
        print(r,c)
        if r<0 or c<0:
            new_image = img.copy()
            if c<0:
                c_pair = [c//2,c-c//2]
                print ("removing {} cols".format(c))
                new_image = self.insert_seams(new_image,0,-c_pair[0])
                new_image = self.insert_seams(new_image,0,-c_pair[1])
                print(img.shape)
                print(new_image.shape)
                
            if r<0:
                r_pair = [r//2,r-r//2]
                new_image = self.insert_seams(new_image,-r_pair[0],0)
                new_image = self.insert_seams(new_image,-r_pair[1],0)
            if r>0:
                seams_removed = self.reduce_image_size(new_image,r,0)
                new_image = self.img.copy()
            if c>0:
                seams_removed = self.reduce_image_size(new_image,0,c)
                new_image = self.img.copy()
            name_transformed = "{}{}_{}_halfinsert_r{}_c{}.jpg".format(config.output_dir,self.img_name,config.date_str,r,c)
            cv2.imwrite( name_transformed , new_image)
            self.seam_inserted_image = new_image.copy()

        
if __name__=="__main__":
    # obj_removed = ObjectRemoval()
    name = "6.jpg"
    img = cv2.imread("Data/input/"+name, 1)
    
    r,c = [int(z) for z in input("enter number of rows and columns to remove").split()]
    sc = SeamCarving(img,r,c,img_name=name)

  

          

  


