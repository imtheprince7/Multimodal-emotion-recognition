import cv2
import dlib

# Step 1: Face Detection
def detect_faces(frame):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

# Step 2: Face Alignment
def align_face(frame, face):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks = predictor(frame, face)
    aligned_face = dlib.get_face_chip(frame, landmarks)
    return aligned_face

# Step 3: Face Recognition
def extract_features(frame):
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    face_descriptor = face_recognizer.compute_face_descriptor(frame)
    return face_descriptor

# Open video capture
video = cv2.VideoCapture("dataSet\video\Ses01F_impro01.avi")

while True:
    # Read the next frame
    ret, frame = video.read()

    # Check if frame was read successfully
    if not ret:
        break

    # Step 1: Face Detection
    faces = detect_faces(frame)

    for face in faces:
        # Step 2: Face Alignment
        aligned_face = align_face(frame, face)

        # Step 3: Face Recognition
        features = extract_features(aligned_face)

        # Do something with the extracted features
        # e.g., compare features, store in a database, etc.

    # Display the frame with detected faces
    cv2.imshow("Video", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
