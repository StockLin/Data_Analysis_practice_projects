# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:15:39 2018

@author: Stock
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense


# =============================================================================
# # Initialising the CNN
# =============================================================================
classifier = Sequential()

# adding covolution layer with 32 filter which size is 3 * 3 and using relu activation function
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
# adding the maxpooling layer
classifier.add(MaxPool2D(pool_size=(2, 2)))

# adding the second convolution and maxpooling layer
# adding second covolution layer with 32 filter which size is 3 * 3 and using relu activation function
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
# adding second maxpooling layer
classifier.add(MaxPool2D(pool_size=(2, 2)))

# addinf flatten layer
classifier.add(Flatten())

# adding full connection layer
classifier.add(Dense(units=128, activation='relu')) # hidden layer
classifier.add(Dense(units=1, activation='sigmoid')) # output layer




# =============================================================================
# # compiling the CNN
# =============================================================================
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# =============================================================================
# # Fitting images to the CNN model
# =============================================================================
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=250,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=62.5)


# =============================================================================
# Making new predictions
# =============================================================================
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
# adding new dimension bcz the NN model always fit or predit by 'batch', cannot not fit by one image
# so alway one image, we also need to give the batch number.
# here is one
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices # see the class of cats(0) and dogs(1)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'











