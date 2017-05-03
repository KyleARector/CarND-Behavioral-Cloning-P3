import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read in lines of training data and image locations from CSV
lines = []
with open("Training_Data/driving_log.csv") as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        lines.append(line)

# Add images and measurements to lists
images, measurements = [], []
for line in lines:
    data_path = line[0]
    filename = data_path.split('/')[-1]
    curr_path = "Training_Data/IMG/" + filename
    image = cv2.imread(curr_path)
    images.append(image)
    measurements.append(float(line[4]))


# Double the training set by flipping each image and measurement
aug_images, aug_measurements = [], []
for imate, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    aug_images.append(cv2.flip(image, 1))
    aug_measurements.append(measurement * -1.0)


X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

# # # Begin Model # # #
model = Sequential()
# Normalize and zero center pixels
model.add(Lambda(lambda x: x/255.0 - 0.5,
                 input_shape=(160, 320, 3)))
# Crop the images to focus on area of interest
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss="mse",
              optimizer="adam")
model.fit(X_train, y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=3)

# # # End Model # # #

model.save("model.h5")
