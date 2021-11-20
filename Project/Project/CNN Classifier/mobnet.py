# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers import Dropout

class MobNet:
        @staticmethod
        def build(Wmeasure, Hmeasure, Dmeasure, classes):
                # initialize the model
                model = Sequential()
                ImageShape = (Hmeasure, Wmeasure, Dmeasure)

                # if we are using "channels first", update the input shape
                if K.image_data_format() == "channels_first":
                        ImageShape = (Dmeasure, Hmeasure, Wmeasure)

                # first set of CONV => RELU => POOL layers
                model.add(Conv2D(32, (3, 3), input_shape=ImageShape))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(64, (3, 3), input_shape=ImageShape))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # first (and only) set of FC => RELU layers
                model.add(Flatten())
                model.add(Dense(512))
                #
                model.add(Activation("relu"))
                model.add(Dropout(0.5))
                model.add(Dense(1))

                # softmax classifier
                model.add(Dense(classes))
                model.add(Activation("softmax"))
                
                # return the constructed network architecture
                return model
