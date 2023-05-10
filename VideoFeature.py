from deepface import DeepFace
import cv2

# Function to extract facial features from an image
def extract_features(image):
    # Detect and extract facial features
    try:
        # Using DeepFace to extract facial features
        features = DeepFace.represent(image, model_name='Facenet')
        return features
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Open the video file
video = cv2.VideoCapture('E:\\Multimodal-emotion-recognition\\dataSet\\video\\Ses01F_impro01.avi')
# Process each frame in the video
while True:
    # Read a frame from the video
    ret, frame = video.read()
    
    # Check if the frame was read successfully
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Proceed only if at least one face is detected
    if len(faces) > 0:
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face = frame[y:y+h, x:x+w]

            # Extract facial features from the face region
            features = extract_features(face)
            
            # Check if facial features were successfully extracted
            if features is not None:
                # Print the extracted facial features (you can modify this part to store or process the features)
                print(features)
            
            # Draw a rectangle around the face region (optional)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the frame (optional)
    cv2.imshow('Video', frame)
    
    # Check if the 'q' key was pressed to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close the windows
video.release()
cv2.destroyAllWindows()
