# seam-carving COL783: Digital Image Processing Assignment 
 
# Look at pdf for better view




The Assignment is an implementation of the articles “Seam Carving for Content-Aware Image Resizing” and “Fast Seam Carving Using Gaussian Pyramid”.



                              Contents of Report


INTRODUCTION                                                                                                                            
IMPLEMENTATION                                                                                                                       
 
INTRODUCTION                                                                                                                            
IMPLEMENTATION                                                                                                                       
 
 Energy Functions
 Seam Removal using Gaussian pyramids
 Seam Insertion
 Object Removal 
 Object Preservation
 EXPERIMENTS & RESULTS
          Varying the parameters
          Comparison of energy functions






INTRODUCTION:
This Assignment we are using seam carving method to resize the image. Since scaling is not considering the image content, it applies uniformly all over the image. This makes scaling an inefficient method. Cropping on the other end considers the image content but it can only be applied on the periphery of the image which makes it difficult to apply where redundant part is not at the center of the image. We also apply fast seam carving method using gaussian pyramids. We also explore the object removal from the image which also works very well. Different energy functions were tried and entropy was working the best.
Orignal image -> Scaling image -> cropped image -> Seam carving
Object remoal 


Original Image


Cropped Image

Rescaled Image



Seam Carving Resize







IMPLEMENTATION:

The assignment is implemented in python, Open CV, numpy and scipy.

Seam Carving Image scale down by m rows, n columns
Steps followed: 
1.Input m, n
2.Converting to grayscale
3.Apply energy function
4.Find seams with dynamic programming in both horizontal and vertical direction
5.Select optimum minimum energy seams from both directions
6.Remove minimum energy seams
Fast Seam Carving
1.Create different layers of Gaussian pyramids from original imput image
2.Estimate the number of seams to be removed from each layer of the pyramid
3.Consider the lowest resolution layer of Gaussian pyramid
4.Apply energy function and seam carving
5.Transform the seams got in the layer i+1and transfrom them to the layer i  and remove the transformed seams from layer i.
6.Apply seam carving for the remaining seams which were calculated in step 2.
7.Repeat the process to all the layers

Eg: 
Lets say the image size was 100*100 and number of layers to be 3. 50 seams to be removed.
So from step 2 we need to remove [ 0 1 12] seams from every layer.

Energy Functions
We looked at 4 energy functions:
1)Gradient energy e1: 
i)Consider the input image and calculate the gradients of image along x-axis and y-axis separately.
ii)Now, calculate the magnitude of gradient using the gradients along x-axis and y-axis. i.e., the magnitude of gradient is the sum of squares of gradient along x-axis and y-axis.
2)Sobel energy:
i)Consider the input image and calculate the gradients of image along x-axis and y-axis separately by convolving with Sobel horizontal and vertical operators accordingly.
ii)Now, the magnitude of gradient is computed as the sum of squares of gradient obtained by horizontal and vertical directions.
3)Entropy:
i)It is similar as gradient energy function but has advantage of employing entropy values to strength the gradient function.
ii)Here we consider the energy function as the sum of entropy value and magnitude of gradients
4)HOG:
i)It is also similar to gradient energy function and generally considered as textural descriptors.
ii)To estimate HOG, we initially estimate gradients in horizontal and vertical directions and convert them into polar coordinates. 
iii)Now, we divide the image into cells of size 11*11. And compute the histogram of cells with gradient values.
iv)Now finally estimate the magnitude of gradients using the gradients of each histogram. As a result, it is termed as Histogram of Gradient (HOG) energy function.
 
Seam insertion
In this part, we apply seam insertion i.e. increasing the width and height of the image.
1.Suppose we are to add m rows and n columns
2.Here we start with the original image and find a sequence of seams to be removed for removing m rows and n columns.
3.Once we know the sequence of removal, we can insert seams to the original imaga at the same places where we found the lowese energy seams removed.
4.We also tried with using 50% removal twice.

Object removal
Here the aim is to remove object using seam removal.
1.To achieve this, first step is to take input from user the parts of image to be removed.
2.We set the energy of this part as minimum energy
3.Then we apply seam removal algorithm, which will choose the best horizontal or vertical seam to be removed.
Object Preservation
Some parts of image are too important to be modified eg. face of a person. So this part saves this part of the image from being modified in seam removal.
1.To achieve this, first step is to take input from user the parts of image to be preserved.
2.We set the energy of this part as maximum energy
3.Then we apply seam removal algorithm, which will choose the best horizontal or vertical seam to be removed.




Experiments and Observations
Energy Functions with HSV colormap
Original Image	Gradient	Sobel	Entropy	HOG
				
				
				
				













Seam removal:
Original Image
	Seams image
	Result

		
		
		
		







Input	Result
	
	
	
	

Seam Removal using Gaussian Pyramid:

Object	Seam Removal using different functions (85, 89)
		
Gradient Energy
		
Sobel		
Entropy		
HOG		









Seam insertion
Input image
	Insert 30 vertical seams

Insert 10 rows
	Insert 30 rows remove 20 columns

Insert 40 rows 30 columns
	Remove 30 rows 30 columns


2 Stage Seam insertion
Original image	1 Shot 100 Seams insertion	2 Stage 100 seams insertion
		

Object Preservation
Saving the interviewee in the following image
Without Mask	With mask
	
	
	




Object Removal

Original image

Mask without object preservation
	Mask with object preservation

Result
	Result




Original Image
	Object image

Mask image
	Result




Fast Seam Carving Computational Benefit
Table 1: Computational time comparison for seam removal with and without gaussian pyramids for Removing 60 row,70 col seams
Image		
Seam Carving	77.29 sec	82.05sec
Fast Seam Carving	9.25 sec	13.49 sec




Observations:
We observed that entropy energy function performs better in seams carving. 
In object removal, the removal of object effects the nearby objects, so using Object preservation, we can reduce the distortion. 
Seam insetion is also working well and 2 stage seam insertion works better when a large number of seams are to be inserted. 
Apart from this, the removal of seam carving using gaussian pyramids reduces the computational time.
