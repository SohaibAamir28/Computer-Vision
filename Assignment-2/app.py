from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FaceNet and MTCNN
facenet = FaceNet()
detector = MTCNN()

app = Flask(__name__)

class FaceRecognition:
    def __init__(self):
        self.facenet = FaceNet()
        self.detector = MTCNN()

    def face_detection(self, image):
        original_img = image
        out = self.detector.detect_faces(original_img)

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

    def preprocess_image(self, img):
        img = cv2.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        return img

    def get_face_embeddings(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_detect = self.face_detection(img)
        if face_detect is not None:
            img1 = self.preprocess_image(face_detect)
            embeddings = self.facenet.embeddings(img1)
            return embeddings
        else:
            return None

    def compare_embeddings(self, user_embedding, stored_embeddings):
        threshold = 0.56
        for stored_embedding in stored_embeddings:
            similarity = cosine_similarity(user_embedding, stored_embedding.reshape(1, -1))
            if similarity > threshold:
                return True
        return False

face_recognition = FaceRecognition()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        face_detect = face_recognition.face_detection(frame)
        if face_detect is not None:
            embedding_user = face_recognition.get_face_embeddings(face_detect)
            if embedding_user is not None:
                match_found = False
                image_dir = 'static'
                db_image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                  if os.path.isfile(os.path.join(image_dir, f))
                                  and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for db_image_path in db_image_paths:
                    embedding_stored = face_recognition.get_face_embeddings(cv2.imread(db_image_path))
                    if embedding_stored is not None:
                        match = face_recognition.compare_embeddings(embedding_user, [embedding_stored])
                        if match:
                            match_found = True
                            cv2.putText(frame, "Authorized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            break
                if not match_found:
                    cv2.putText(frame, "Unauthorized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
