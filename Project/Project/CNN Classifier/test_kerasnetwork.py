# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from os import listdir
from os.path import isfile, join
import os

# create the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="cow.model",
	help="path to trained model model")
ap.add_argument("-i", "--image", default="C:\\Users\\SouravKr\\Music\\Project\\Project\\LocalizationAndSementation\\localization",help="path to input image")
#ap.add_argument("-i", "--image", default="C:\\Users\\SouravKr\\Music\\Project\\Project\\CNN Classifier\\trdataset",help="path to input image")
args = vars(ap.parse_args())

files = [f for f in listdir(args["image"]) if isfile(join(args["image"], f))]
for file in files:
    print(file)
    # loading the image
    image_path = os.path.sep.join([args["image"], file])
    image = cv2.imread(image_path)
    original = image.copy()

    # for classification purpose, preprocess the image
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # input the convolutional neural network that is trained
    print("[INFO] loading network...")
    model = load_model(args["model"])

    # classification of input data
    (notCow, cow) = model.predict(image)[0]

    # construct the labelname
    labelname = "COW" if cow > notCow else "Not Cow"
    probability = cow if cow > notCow else notCow
    labelname = "{}: {:.2f}%".format(labelname, probability * 100)

    # create the labelname on the image
    output = imutils.resize(original, width=400)
    cv2.putText(output, labelname, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display the output image
    #cv2.imdisplay("Output", output)
    #cv2.waitKey(0)

    #save into directory
    if cow > notCow:
        imageBox = "images/cow/"+ file + " "
        cv2.imwrite(imageBox, original)
    else:
        imageBox = "images/not_cow/"+ file + " " 
        cv2.imwrite(imageBox, original)
