import logging

import time

from scipy.spatial import distance as dist
from collections import OrderedDict	
import numpy as np
from scipy.stats import itemfreq
import cv2
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(filename='software.log',
                        format='[%(funcName)s] - %(levelname)s [%(asctime)s] %(message)s', level=logging.INFO,filemode='w')


logging.info(f'program start \n')


#Function to get the centroid of the Object.
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1 #centroid X
    cy = y + y1 #centroid Y

    return (cx, cy)


#function to detect vehicle/moving object 
#def detect_vehicles(fg_mask, min_contour_width=35, min_contour_height=30):
def detect_vehicles(fg_mask, min_contour_width=35, min_contour_height=30):

    matches = []
    frame_copy = fg_mask
    #finding external contours to draw rectangle aroung vehicles.
    #im, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    #im, contours, hierarchy = cv2.findContours((fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#***	Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.

    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= min_contour_width) and (
            h >= min_contour_height)

        if not contour_valid:
            continue
        
        # getting center of the bounding box
        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid))
    return matches


#function to normalize the image so that the entire blob has the same rgb value
def normalized(down):
		s=down.shape
		x=s[1]
		y=s[0]
		norm=np.zeros((y,x,3),np.float32)
		norm_rgb=np.zeros((y,x,3),np.uint8)
		
                #deletes all the first and second elements in the array
		b=down[:,:,0]
		g=down[:,:,1]
		r=down[:,:,2]

		sum=b+g+r

		norm[:,:,0]=b/sum*255.0
		norm[:,:,1]=g/sum*255.0
		norm[:,:,2]=r/sum*255.0

		norm_rgb=cv2.convertScaleAbs(norm)
		# cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
		#***	Scales, calculates absolute values, and converts the result to 8-bit. On each element of the input array, the function convertScaleAbs performs three operations sequentially: scaling, taking an absolute value, conversion to an unsigned 8-bit type

		return norm_rgb	
	
# initializing color class
colors = OrderedDict({"red":(255,0,0),
                      "dark-red":(153,0,0),
                      "maroon":(107,0,0),
                      "green":(0,255,0),
                      "blue":(0,0,255),
                      "white":(255,255,255),
                      "black":(100,100,100),
                      "silver":(128,128,128),
                      "grey":(65,65,65),
                      "yellow":(255,255,0),
                      "orange":(255,165,0)})
lab = np.zeros((len(colors), 1, 3), dtype="uint8")
colorNames = []

#f=open("output.txt","w")

incre=1
'''
if(len(x)==0):
	#no image name present in the file
	incre=1
else:
	#reding the image number 
	incre=int(x[-1].split(",")[0].split("_")[-1].split(".")[0])
f.close()
'''
#converting the rgb color to lab colors
for (i, (name, rgb)) in enumerate(colors.items()):
			# update the L*a*b* array and the color names list
			lab[i] = rgb
			colorNames.append(name)
lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)
#***	cvtColor() method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV

#function to label car lab color to a perticular color class
def label(image,lab,colorNames):

		# initialize the minimum distance found thus far
		minDist = (np.inf, None)
 
		# loop over the known L*a*b* color values
		for (i, row) in enumerate(lab):
			# compute the distance between the current L*a*b*
			# color value and the mean of the image
			
			d = dist.euclidean(row[0],image)
 
			# if the distance is smaller than the current distance,
			# then update the bookkeeping variable
			if d < minDist[0]:
				minDist = (d, i)
 
		# return the name of the color with the smallest distance
		return colorNames[minDist[1]]

#initialising background object used for background elemination 
background=cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture("sample/long-video.h264")
#cap = cv2.VideoCapture('traffic_video.mp4')

#initialising frame counter
count_frame=0
counter = 0

while(cap.isOpened()):
	
	iii = 0

	_,frame=cap.read()
	#resizing the frame 
	try:
		# 320x240 (mod16)
		# 640x480 (mod16)
		# 704x528 (mod16)
		# 720x540 (mod4)
		# 768x576 (mod16)
		# 960x720 (mod16)
		# 1280x960 (mod16)
		# 1440x1080 (mod16)
		#pass
		#frame=cv2.resize(frame,(640,480))
		#frame=cv2.resize(frame,(1280,720))
		#frame=cv2.resize(frame,(240,320))
		frame=cv2.resize(frame,(480,640))
		#frame=cv2.resize(frame,(720,1280))
	except:
		break
	#creating a copy of the frame
	frame_copy=frame
	frame_copy_copy=copy =frame[:,:]
	
	#applying background elimination
	bg=background.apply(frame)
	
	#additional image processing
	
	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) #structuring element. 
	#***	The function constructs and returns the structuring element that can be further passed to erode, dilate or morphologyEx. But you can also construct an arbitrary binary mask yourself and use it as the structuring element.

	bg= cv2.erode(bg,kernel,iterations = 0) #8  => best 30 for sample 1 and 2 -------- 15 for sample 3
	#***	The function erodes the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the minimum is taken
	#***-	The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In case of multi-channel images, each channel is processed independently.	


	# Fill any small holes
	closing=cv2.morphologyEx(bg,cv2.MORPH_CLOSE,kernel) # Performs advanced morphological transformations.
	#***	The function "morphologyEx" can perform advanced morphological transformations using an erosion and dilation as basic operations. Any of the operations can be done in-place. In case of multi-channel images, each channel is processed independently.

	################################################### cv2.imshow("closing",closing)
	cv2.imshow("closing",closing)
	# Remove noise
	opening=cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
	################################################### cv2.imshow("removing_noise",opening)
	cv2.imshow("removing_noise",opening)
	# Dilate to merge adjacent blobs
	dilation=cv2.dilate(opening, kernel, iterations=2) # Dilates an image by using a specific structuring element.
	#***	The function dilates the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the maximum is taken
	#***-	The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In case of multi-channel images, each channel is processed independently.

	# threshold to remove furthur noise 
	dilation[dilation < 250] = 0
	bg=dilation
	
	#initialising output color list
	output_color=[]
	
	#detecting contour and calculating the co-ordinates of the contours
	contour_list=detect_vehicles(bg)
	# if len(contour_list) != 0 :
	# 	print(f'detected car number: {len(contour_list)}')


	
	#traversing through each detected contour 
	for ele in contour_list:
		x1=ele[0][0]
		y1=ele[0][1]
		x2=x1+ele[0][2]
		y2=y1+ele[0][3]
		#extracting the regions that contains car features
		
		slice_bg=frame_copy[y1:y2,x1:x2]

		#normalising the image so that there is uniform color throughout
		slice_bg=normalized(slice_bg)
		
		arr=np.float32(slice_bg)
		#reshaping the image to a linear form with 3-channels
		pixels=arr.reshape((-1,3))
		
		#number of clusters
		n_colors = 2
		
		#number of iterations
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1) # rule :: https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html	
		#*	cv.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
		#*	cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
		#*	cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
	
		#initialising centroid
		flags = cv2.KMEANS_RANDOM_CENTERS # Select random initial centers in each attempt.

		#applying k-means to detect prominent color in the image
		_, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
		#***	The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters and groups the input samples around the clusters. As an output, bestLabelsi contains a 0-based cluster index for the sample stored in the i-th row of the samples matrix.

		
		palette = np.uint8(centroids)
		quantized = palette[labels.flatten()]
		
		#detecting the centroid with densest cluster  
		dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
		
		r=int(dominant_color[0])
		g=int(dominant_color[1])
		b=int(dominant_color[2])

		
		rgb=np.zeros((1,1,3),dtype='uint8')
		rgb[0]=(r,g,b)
		
		#getting the label of the car color
		color=label(rgb,lab,colorNames)
		
		output_color.append(color)
		

		#drawing rectangle over the detected car 
		frame_copy = cv2.rectangle(frame_copy,(x1,y1),(x2,y2),(r,g,b),3)
		
		#frame_copy = cv2.rectangle(frame_copy,(x1,y1),(x2,y2),(0, 0, 255),10)
		#frame_copy = cv2.rectangle(frame_copy,(x1,y1),(x2,y2),(0, 0, 255),10)
		
		#print(f'detected car number: {frame_copy}')
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame_copy,color,(x1,y1), font, 2,(r,g,b),2,cv2.LINE_AA)
		################################################################
		################################################################ font = cv2.FONT_HERSHEY_SIMPLEX
		#labeling each rectangle with the detected color of the car
		################################################################ cv2.putText(frame_copy,color,(x1,y1), font, 2,(r,g,b),2,cv2.LINE_AA)
		################################################################

	#openinig file to write the ouput of each frame
	f=open("reports.txt","a")

	now = time.localtime()


	#writing onto the file for every 10 frames
	# as my test in this test data each car passed thw frames in 48-60 frams #	https://en.wikipedia.org/wiki/Frame_rate
	if(count_frame%60==0): 
	#if((len(output_color)!=0):
		
		if(len(output_color)!=0):
		#if(count_frame%60==0):

			logging.info(f'Detected Car Number: {incre}')
			#logging.info(f'Detected Car Color: {c}')
			# cv2.imwrite(img,frame)
			
			if now.tm_min == 00 :
					
				# c=",".join(output_color)+'\n'
				
				# print(f'detected car number: {c}')
				#image_name="img_"+str(incre)+".jpg,"+c+'\n'
				# f.write(c)
				time_string = time.strftime( "%m/%d/%Y %H:%M:%S", now )
				print(time_string)
				repo = str(time_string)+' : '+str(incre)+'\n'
				print(repo)
				print(type(repo))
				
				f.write(repo)


			incre=incre+1
			#print(f'incre: {incre}')

			count_frame=0
	count_frame+=1
	#print(f'count frame: {count_frame}')
	cv2.imshow("Car Detection",frame_copy)
	if(cv2.waitKey(30)==27 & 0xff):
		break

cap.release()
cv2.destroyAllWindows()
f.close()