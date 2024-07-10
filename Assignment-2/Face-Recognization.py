# pip install keras-facenet mtcnn 

import cv2
import numpy as np
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FaceNet and MTCNN
facenet = FaceNet()
detector = MTCNN()

# Function to detect and crop face from an image
def face_detection(image):
    original_img = image
    out = detector.detect_faces(original_img)

    if out:
        x, y, w, h = out[0]['box']
        adjustment_factor_w = 0.2
        adjustment_factor_h = 0.2
        new_x = max(0, x - int(w * adjustment_factor_w))
        new_y = max(0, y - int(h * adjustment_factor_h))
        new_w = min(original_img.shape[1] - new_x, int(w * (1 + 2 * adjustment_factor_w)))
        new_h = min(original_img.shape[0] - new_y, int(h * (1 + 2 * adjustment_factor_h)))

        cropped_face = original_img[new_y:new_y + new_h, new_x:new_x + new_w]
        return cropped_face
    else:
        return None

# Function to preprocess the cropped face image
def preprocess_image(img):
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    return img

# Function to generate embeddings using FaceNet
def get_face_embeddings(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_detect = face_detection(img)
    if face_detect is not None:
        img1 = preprocess_image(face_detect)
        embeddings = facenet.embeddings(img1)
        return embeddings
    else:
        return None

# Example function to compare embeddings with stored embeddings
def compare_embeddings(user_embedding, stored_embeddings):
    threshold = 0.56
    for stored_embedding in stored_embeddings:
        similarity = cosine_similarity(user_embedding, stored_embedding.reshape(1, -1))
        if similarity > threshold:
            return True
    return False

def display_image_with_caption(image, caption):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1.2
    color = (0, 0, 255) if caption == "Authorized" else (255, 0, 0)
    thickness = 1
    img = cv2.putText(image, caption, org, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Open webcam or video file
    cap = cv2.VideoCapture(0)  # Change 0 to the video file path if using a file
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Detect faces in the frame
        face_detect = face_detection(frame)
        
        if face_detect is not None:
            # Get embeddings of the detected face
            embedding_user = get_face_embeddings(face_detect)
            if embedding_user is not None:
                # Compare embeddings with database images
                match_found = False
                image_dir = r"folder_path"
                db_image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                  if os.path.isfile(os.path.join(image_dir, f))
                                  and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for db_image_path in db_image_paths:
                    embedding_stored = get_face_embeddings(cv2.imread(db_image_path))
                    if embedding_stored is not None:
                        # Compare embeddings
                        match = compare_embeddings(embedding_user, [embedding_stored])
                        if match:
                            match_found = True
                            display_image_with_caption(frame, "Authorized")
                            break
                if not match_found:
                    display_image_with_caption(frame, "Unauthorized")
            else:
                print("No face detected in user image.")
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
