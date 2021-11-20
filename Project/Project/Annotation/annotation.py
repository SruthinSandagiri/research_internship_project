
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:50:52 2019

@author: mouni
"""
import cv2
import os

os.chdir("C:\\Users\\SouravKr\\Music\\Project\\Project")

myfile = open("Annotation/annotation.txt")
area_of_interest_box = []

def sumOfAP(a,d,n):
    return a+d * int(n-1)
    
def find_annotation(image_id):
    for line in myfile:    
        text = line       
        text = text.split(',')
        count = 0
        i = 1
       
        for x in range(len(text)):                     
            if (text[0] == image_id):
                #print(text[x])
                count = text[1]
                
        for y in range(int(count)):            
            if (text[0] == image_id):    
                #area  
                area_of_interest_box.append([text[sumOfAP(3,6,i)], text[sumOfAP(4,6,i)], text[sumOfAP(5,6,i)], text[sumOfAP(6,6,i)]])
                i = i + 1
            else:
                return 0         
    return area_of_interest_box

def coordinatesDescription(shape):     
    x1, y1 = int(shape[0]), int(shape[1])
    x2, y2 = (int(shape[0]) + int(shape[2])), (int(shape[3])+int(shape[1]))
    return x1,y1,x2,y2
    
def Annotation_boundingBoxes(image_view,imgID,):
    boxes = find_annotation(imgID)
    area_of_interest_boundingbox = []
    for box in boxes:        
        #image_view = cv2.resize(image_view, (600,600))
        #print(box)
        x1, y1, x2, y2 = coordinatesDescription(box)
        cv2.rectangle(image_view, (x1,y1),(x2,y2),(64, 224, 208),3)            
        cv2.imshow("s",image_view)
        cv2.waitKey(0)
        area_of_interest_boundingbox.append([x1,y1,x2,y2])
    
        return image_view

image_path_dir = "Images//IMG_50001.jpg"
image_view =  cv2.imread(image_path_dir)
if os.path.exists(image_path_dir) == False:
    print("Image not found")
else:            
    returnvalue = Annotation_boundingBoxes(image_view,"IMG_50001.jpg")
    print(returnvalue)