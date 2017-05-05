import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

batch_size = 64
num_epochs = 10

# Read in lines of training data and image locations from CSV
lines = []
with open("Training_Data/driving_log.csv") as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        lines.append(line)

training_set, valid_set = train_test_split(shuffle(lines),
                                           test_size=0.2)


def generator(lines, batch_size):
    while True:
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            line_items = lines[offset: offset + batch_size]

            images, measurements = [], []

            for line in line_items:
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

            X_train = np.array(aug_images[:batch_size])
            y_train = np.array(aug_measurements[:batch_size])

            yield X_train, y_train


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
model.compile(loss="mse",
              optimizer="adam")
''' model.fit(X_train, y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=5)'''

training_generator = generator(training_set,
                               batch_size=batch_size)
valid_generator = generator(valid_set,
                            batch_size=batch_size)
model.fit_generator(train_generator,
                    steps_per_epoch=len(training_set)/batch_size,
                    validation_data=valid_generator,
                    validation_steps=len(valid_set)/batch_size,
                    nb_epoch=num_epochs)

model.save("model.h5")
