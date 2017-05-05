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
    for i in range(3):
        data_path = line[i]
        filename = data_path.split('/')[-1]
        curr_path = "Training_Data/IMG/" + filename
        image = cv2.imread(curr_path)
        images.append(image)
        if i == 1:
            measurements.append(float(line[4]) + 0.2)
        elif i == 2:
            measurements.append(float(line[4]) - 0.2)
        else:
            measurements.append(float(line[4]))


# Double the training set by flipping each image and measurement
aug_images, aug_measurements = [], []
for image, measurement in zip(images, measurements):
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
#model.add(Convolution2D(6, 5, 5, activation="relu"))
#model.add(MaxPooling2D())
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

batch_size = 10

def generator(X_train, y_train, batch_size):
    X_batch = np.zeroes((batch_size, 160, 320, 3))
    y_batch = np.zeroes((batch_size, 1))

    while True:
        for i in range(batch_size):
            index = random.choice(len(features), 1)
            X_batch[i] = X_train[index]
            y_batch[i] = y_train[index]
        yield X_batch, y_batch

model.compile(loss="mse",
              optimizer="adam")
'''model.fit(X_train, y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=5)'''

train_generator = generator(X_train, y_train, batch_size)
model.fit_generator(train_generator,
                    samples_per_epoch=100,
                    nb_epoch=10)

model.save("model.h5")
