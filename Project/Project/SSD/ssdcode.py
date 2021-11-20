# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:30:27 2019

@author: sruth
"""
 
import os
import numpy as np
import time
import cv2
import time
import random


os.chdir("E:\\Project-20190715T063106Z-001\\Project\\SSD")
print (os.getcwd())
#Setting up the configuration values 
image_dir_master = "E:\\Project-20190715T063106Z-001\\Project\\Images"
#image_dir_master = "C:\\Users\\sruth\\OneDrive\\Desktop\\clarity images for comparision"


ssd_prototxt_dir = "models/MobileNetSSD_deploy.prototxt.txt"
ssd_model_dir = "models/MobileNetSSD_deploy.caffemodel"

confidence_value = 0.5;
threshold_value = 0.3;

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(image_dir_master) if isfile(join(image_dir_master, f))]


#considering coco class label with 90 ids
classNames = { 0: 'tie',
            1: 'bench', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
            7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',12 : 'spoon',
            13: 'stop sign', 14: 'parking meter', 15: 'person', 16: 'bird', 17: 'cat',
            18: 'dog', 19: 'horse', 20: 'sheep', 21: 'background', 22: 'elephant', 23: 'bear',
            24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
            32: 'cow', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
            37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
            41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
            46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
            67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }

color_value = np.random.uniform(0, 255, size=(len(classNames), 3))

net = cv2.dnn.readNetFromCaffe(ssd_prototxt_dir,ssd_model_dir)
total_time = 0
total_obj = 0
average =0
loc_count = 150

for file in files:
  image_path_dir = os.path.sep.join([image_dir_master, file])
  # making  an input bounding_box for the image_view with 300 * 300 pixel-wise
  image_view = cv2.imread(image_path_dir)
  (height, width) = image_view.shape[:2]
  bounding_box = cv2.dnn.blobFromImage(cv2.resize(image_view, (512, 512)), 0.007843, (512, 512), 127.5)

  # pass the bounding_box through the network and obtain the area_detections and
  # predictions
  
  net.setInput(bounding_box)
  start = time.time()
  area_detections = net.forward()
  end = time.time()

  # show timing information on YOLO   
  count = 0

  
 
  # iteration over the area_detections
  for i in np.arange(0, area_detections.shape[2]):
      
   	# getting the model_prediction_confidence  associated with the probability prediction
   	model_prediction_confidence = area_detections[0, 0, i, 2]
   
   	# removing out weak detection by ensuring the `model_prediction_confidence` is
   	# greater than the minimum model_prediction_confidence
   	if model_prediction_confidence > confidence_value:
           class_id_x = int(area_detections[0, 0, i, 1])
           area_of_interest_box = area_detections[0, 0, i, 3:7] * np.array([width, height, width, height])
           (start_Xaxis, start_Yaxis, end_Xaxis, end_Yaxis) = area_of_interest_box.astype("int")
           label_text_value = "{}: {:.2f}%".format(classNames[class_id_x], model_prediction_confidence * 100)           
           cv2.rectangle(image_view, (start_Xaxis, start_Yaxis), (end_Xaxis, end_Yaxis),color_value[class_id_x], 2)
           y_axis = start_Yaxis - 15 if start_Yaxis - 15 > 15 else start_Yaxis + 15
           cv2.putText(image_view, label_text_value, (start_Xaxis, y_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_value[class_id_x], 2)
           count = count + 1
           total_time = total_time + (end-start)
           total_obj = total_obj + 1
           average = (total_time/total_obj)
           #cropImage = image_view[start_Xaxis:start_Yaxis, end_Xaxis:end_Yaxis]
           #imageName = "../Testing/localization/SSD"+str(loc_count)+".jpg"
           cv2.imshow("a", image_view)
           cv2.waitKey(0)
           #loc_count = loc_count + 1
           
           

  #image show         
  #cv2.imshow("Images",image_view)
  #Wait = cv2.waitKey(5000)  
  #if Wait == 27:
  #    cv2.destroyAllWindows()
  #    break
  #Text show 
  print("[INFO] SSD took {:.6f} seconds".format(end-start))
  print("Image Name -"+file+", No of Object Detected " + str (count) +"\n")

print("Total time taken")
print(total_time)
print("Total number of objects detected")
print(total_obj)
print("average time taken")
print(average)

  
 