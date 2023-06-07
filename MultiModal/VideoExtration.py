import os
import csv
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, TimeDistributed, Flatten, Dense, LSTM

# # Define the paths and parameters
# video_folder = "splittedData/splittedVideo"
# output_csv = "output_features.csv"

# Load the pre-trained 2D CNN model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add temporal information to the model
input_shape = (None, 224, 224, 3)
video_input = Input(shape=input_shape)
temporal_output = TimeDistributed(base_model)(video_input)
flattened_output = TimeDistributed(Flatten())(temporal_output)
lstm_output = LSTM(256)(flattened_output)
output = Dense(512)(lstm_output)  # num_features should match the desired number of extracted features

model = Model(inputs=video_input, outputs=output)
import cv2

def preprocess_frame(frame):
    # Resize the frame to a specific size
    frame = cv2.resize(frame, (224, 224))

    # Convert the frame to floating-point representation
    frame = frame.astype('float32')

    # Normalize the frame
    frame /= 255.0

    # Perform any additional preprocessing steps as needed

    return frame

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        frames.append(processed_frame)

    cap.release()

    return frames
def feature(path):
   
        # Load and preprocess the video frames
        video_frames = load_video_frames(path)
       

        # Reshape the frames to match the input shape of the model
        preprocessed_frames = np.expand_dims(video_frames, axis=0)

        # Extract features using the pre-trained 2D CNN + LSTM model
        features = model.predict(preprocessed_frames)

        # Flatten the features if needed
        features = features.flatten()
        return features

# with open(output_csv, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # writer.writerow(['video_name', 'feature1', 'feature2', ...])  # Add header row

#     for video_file in os.listdir(video_folder):
#         video_path = os.path.join(video_folder, video_file)
#         video_name = os.path.splitext(video_file)[0]

#         # Load and preprocess the video frames
#         video_frames = load_video_frames(video_path)
       

#         # Reshape the frames to match the input shape of the model
#         preprocessed_frames = np.expand_dims(video_frames, axis=0)

#         # Extract features using the pre-trained 2D CNN + LSTM model
#         features = model.predict(preprocessed_frames)

#         # Flatten the features if needed
#         features = features.flatten()

#         # Write features to the CSV file
#         writer.writerow([video_name] + features.tolist())
