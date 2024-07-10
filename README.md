# Assignment-2: Real-Time Face Recognition Flask App

This project demonstrates real-time face detection and recognition using Flask, OpenCV, MTCNN, and FaceNet. The application captures video from a webcam, detects faces in the video stream, and performs face recognition using pre-stored face embeddings.

# Project Output in Browser with Authorized label

<p align="center">
  <img src="https://github.com/SohaibAamir28/Computer-Vision/blob/main/Assignment-2/output/output-1.PNG" alt="Banner image" />
</p>

# Project Terminal View

<p align="center">
  <img src="https://github.com/SohaibAamir28/Computer-Vision/blob/main/Assignment-2/output/terminal.PNG" alt="Banner image" />
</p>

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your machine
- Install the necessary Python libraries using the following commands:
  ```bash
  pip install Flask opencv-python numpy mtcnn keras-facenet scikit-learn

## Project Structure

|-- app.py
|-- face_recognition.py
|-- templates
|   |-- index.html
|   |-- result.html
|-- static
    |-- <stored_face_images>

## Installation
1. Clone the repository
git clone https://github.com/SohaibAamir28/Computer-Vision.git
cd Computer-Vision

2. Ensure the directory structure is as mentioned above. Place the pre-stored face images in the /static directory.

## Usage

1. Run the Flask application:
  python app.py

2. Open your web browser and navigate to http://127.0.0.1:5000/ to see the real-time face detection and recognition in action.

## Example Stored Images Directory
Place images of authorized faces in the /static directory. These images should be in .jpg, .jpeg, or .png format.

## Troubleshooting
Ensure that your webcam is connected and recognized by your computer.
Make sure the required directories and files are in place.
Verify that all dependencies are installed correctly.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with any improvements or suggestions.

## License
This project is open-source and available under the MIT License.