import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Set the path to your video dataset
data_dir = 'splitData\\videoSplit\\'

# Set the parameters for video frames
img_width, img_height = 64, 64
frames_per_clip = 16

# Set the number of classes
num_classes = 5

# Set the batch size and number of epochs
batch_size = 32
epochs = 10

# Create the 3D CNN model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(frames_per_clip, img_width, img_height, 3)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Create a data generator for training
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical',
                                                    frames_per_clip=frames_per_clip, shuffle=True)

# Get the class labels from the generator
class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}

# Train the model
model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=epochs)

# Save the trained model
model.save('video_classification_model.h5')

# Save the class labels to a file
with open('class_labels.txt', 'w') as f:
    for label in class_labels.values():
        f.write(label + '\n')
