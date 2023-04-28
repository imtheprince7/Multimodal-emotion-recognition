import dlib
import cv2
import numpy as np
from deepface import DeepFace
from deepface.basemodels import OpenFace
import openface.openface.align_dlib as openface
from openface.alignment import AlignDlib


# Load the pre-trained model
model_path = 'E:\Multimodal-emotion-recognition\shape_predictor_68_face_landmarks.dat'
face_aligner = openface.AlignDlib(model_path)

# Load an image
image_path = 'test_image.jpg'
image = cv2.imread(image_path)

# Detect the face
face_rect = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
face_landmarks = face_aligner(image, face_rect)

# Extract the facial features
face_descriptor = face_encoder.compute_face_descriptor(image, face_landmarks)
print(face_descriptor)