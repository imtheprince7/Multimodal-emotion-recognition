import openface
from deepface import DeepFace
from deepface.basemodels import OpenFace


# Load the OpenFace neural network model
model = openface.TorchNeuralNet("E:\Multimodal-emotion-recognition\dlib_face_recognition_resnet_model_v1.dat")

# Load an image of a face
img = openface.load_image("E:\Multimodal-emotion-recognition\rabbit.jpg")

# Use the neural network to extract facial features from the image
features = model.forward(img)

# Print the features (this will be a numpy array)
print(features)

!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
