# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:33:54 2019

@author: Swathi
"""
#importing the library. 
import os
import numpy as np
import time
import cv2
import random

os.chdir("C:\\Users\\SouravKr\\Music\\Project\\Project\\")
print (os.getcwd())
 
mrcnn_model_full_path = "RCNN\\model"  
image_dir_master = "Images"
 
confidence_value = 0.5;
threshold_value = 0.3;

#validating the time required by each model to process the image
mask_rcnn_time_taken =0;
##Processing 32 set of image
from os import listdir
from os.path import isfile, join
files = [f for f in listdir(image_dir_master) if isfile(join(image_dir_master, f))]

# load the COCO class labels  our model was trained on 
class_name = os.path.sep.join([mrcnn_model_full_path, "object_detection_classes_coco.txt"])
class_label = open(class_name).read().strip().split("\n")

# assigning the color for each class 
 
color_value = np.random.uniform(0, 255, size=(90, 3))
weights_dir = os.path.sep.join([mrcnn_model_full_path,"frozen_inference_graph.pb"])
config_dir = os.path.sep.join([mrcnn_model_full_path, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
net = cv2.dnn.readNetFromTensorflow(weights_dir, config_dir)
 
for file in files:
  image_path_dir = os.path.sep.join([image_dir_master, file])
  # load our input image_view  and gets its dimension
  image_view = cv2.imread(image_path_dir)
  (height, width) = image_view.shape[:2]
  # pass the bounding_box through the network and obtain the area_detections and
  # predictions
  bounding_box = cv2.dnn.blobFromImage(image_view, swapRB=True, crop=False)
  net.setInput(bounding_box)
  start = time.time()
  (area_of_interest_box_ssd, masks) = net.forward(["detection_out_final", "detection_masks"])
  end = time.time()
  # show timing information and volume information on Mask R-CNN
  print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
   
  

  # loop over the number of detected objects
  for i in range(0, area_of_interest_box_ssd.shape[2]):
  	# extract the class ID of the detection along with the confidence
  	# (i.e., probability) associated with the prediction
  	classID = int(area_of_interest_box_ssd[0, 0, i, 1])
  	confidence = area_of_interest_box_ssd[0, 0, i, 2]
  	count = 0
  	# filter out weak predictions by ensuring the detected probability
  	# is greater than the minimum probability
  	if confidence > confidence_value:
  		# clone our original image so we can draw on it
  		clone = image_view.copy()
  
  		# scale the bounding box coordinates back relative to the
  		# size of the image and then compute the width and the height
  		# of the bounding box
  		box = area_of_interest_box_ssd[0, 0, i, 3:7] * np.array([width, height, width, height])
  		(startX, startY, endX, endY) = box.astype("int")
  		boxW = endX - startX
  		boxH = endY - startY
  
  		# extract the pixel-wise segmentation for the object, resize
  		# the mask such that it's the same dimensions of the bounding
  		# box, and then finally threshold to create a *binary* mask
  		mask = masks[i, classID]
  		mask = cv2.resize(mask, (boxW, boxH),
  			interpolation=cv2.INTER_NEAREST)
  		mask = (mask > threshold_value)
  
  		# extract the ROI of the image
  		roi = clone[startY:endY, startX:endX]
  
  		# check to see if are going to visualize how to extract the
  		# masked region itself
  		if True:
  			# convert the mask from a boolean to an integer mask with
  			# to values: 0 or 255, then apply the mask
  			visMask = (mask * 255).astype("uint8")
  			instance = cv2.bitwise_and(roi, roi, mask=visMask)
  
  			# show the extracted ROI, the mask, along with the
  			# segmented instance
  			#cv2.imshow("img",roi)
  			#cv2.imshow("img",visMask)
  			#cv2.imshow("img",instance)
  
  		# now, extract *only* the masked region of the ROI by passing
  		# in the boolean mask array as our slice condition
  		roi = roi[mask]
  
  		# randomly select a color that will be used to visualize this
  		# particular instance segmentation then create a transparent
  		# overlay by blending the randomly selected color with the ROI
  		color = random.choice(color_value)
  		blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
  
  		# store the blended ROI in the original image
  		clone[startY:endY, startX:endX][mask] = blended
  
  		# draw the bounding box of the instance on the image
  		color = [int(c) for c in color]
  		cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)
  
  		# draw the predicted label and associated probability of the
  		# instance segmentation on the image
  		text = "{}: {:.4f}".format(class_label[classID], confidence)
      
  		cv2.putText(clone, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  		count = count + 1
  
  #show image
  cv2.imshow("imagess",image_view)
  wait=cv2.waitkey(5000)
  if Wait == 27:
    cv2.destroyAllWindows()
    break
  #show text
  #print("Image Name -"+file+", No of object detected "+ str(count))
  		
  
  		 


