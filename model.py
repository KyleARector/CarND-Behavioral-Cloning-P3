import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

batch_size = 32
num_epochs = 2
outer_cam_offset = 0.25

camera_angle_aug = [0, outer_cam_offset, -outer_cam_offset]

# Read in lines of training data and image locations from CSV
# Multiple training sets?
lines = []
with open("Training_Data/driving_log.csv") as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        lines.append(line)

# Split validation set and training set from total number of data points
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# Define generator to release data in batches
# Default batch size of 32
def generator(lines, batch_size=32):
    num_samples = len(lines)
    while True:
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = "Training_Data/IMG/"+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(angle + camera_angle_aug[i])
                    # Append flipped images as well
                    images.append(cv2.flip(image, 1))
                    angles.append(-(angle + camera_angle_aug[i]))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# Instantiate generators for the training and validation sets
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# # # Begin Model # # #
model = Sequential()
# Normalize and zero center pixels
model.add(Lambda(lambda x: x/255.0 - 0.5,
                 input_shape=(160, 320, 3)))
# Crop the images to focus on area of interest
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# # # End Model # # #

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=num_epochs)

model.save("model.h5")
