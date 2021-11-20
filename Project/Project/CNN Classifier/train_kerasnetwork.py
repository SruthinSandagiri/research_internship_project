import matplotlib
matplotlib.use("Agg")

# import the required packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from mobnet import MobNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# create the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="trdataset",
	help="path to input dataset")
ap.add_argument("-m", "--model", default="cow.model",
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# load the number of iterations for training, initiate batch size and learning rate,

iterations = 25
INIT_Learningrate = 1e-3
Batch = 32

# load the data and names
print("[INFO] loading images...")
data = []
names = []

# capture the image paths and rearrange them randomly
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image and pre-process the image, and store the preprocessed image in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	
	#name = imagePath.split(os.path.sep)[-2]
	#name = 1 if name == "cow" else 0
	names.append(1)

# extend the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
names = np.array(names)

# split up the data as 75%for training and 25% for testing
(trainingX, testingX, trainingY, testingY) = train_test_split(data, names, test_size=0.25, random_state=42)

# transform the names from integers to vectors
trainingY = to_categorical(trainingY, num_classes=2)
testingY = to_categorical(testingY, num_classes=2)

# create the image generator for data augmentation
augmentation = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# initialize the model
model = MobNet.build(Wmeasure=28, Hmeasure=28, Dmeasure=3, classes=2)
opt = Adam(lr=INIT_Learningrate, decay=INIT_Learningrate / iterations)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
Hist = model.fit_generator(augmentation.flow(trainingX, trainingY, batch_size=Batch),
	validation_data=(testingX, testingY), steps_per_epoch=len(trainingX) // Batch,
	epochs=iterations, verbose=1)


model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
I = iterations
plt.plot(np.arange(0, I), Hist.history["loss"], label="training_loss")
plt.plot(np.arange(0, I), Hist.history["val_loss"], label="validation_loss")
plt.plot(np.arange(0, I), Hist.history["acc"], label="training_accuracy")
plt.plot(np.arange(0, I), Hist.history["val_acc"], label="validation_accuracy")
plt.title("Training Loss and Accuracy on Cow or Not  a Cow")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
