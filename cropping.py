import cv2
name = "15.jpg"
    
img = cv2.imread("Data/input/"+name, 1)

half = cv2.resize(img, (0, 0), fx = (1.*img.shape[0]-30)/img.shape[0], fy = (1.*img.shape[0]-30)/img.shape[0])


cv2.imshow("energy", half )
cv2.imshow("orig image", img)
# cv2.imshow("transformed image", self.img)
cv2.waitKey()          
cv2.destroyAllWindows()

cv2.imwrite( "Data/output/cropped.jpg" , half)