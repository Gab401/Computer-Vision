# Real-Time Eye Tracking & Deep Learning Verification System

## Project Overview
This project is an interactive, real-time eye-tracking application developed as a final assessment for the Computer Vision course. It captures a live webcam feed to detect faces, extract eye regions, isolate the pupils, and detect the user's gaze.

In strict adherence to the course guidelines, the core system is built entirely using Traditional Computer Vision techniques. A Deep Learning model (OpenCV DNN ResNet-SSD) is integrated purely as a supplementary verification step to confirm the accuracy of the traditional face detection pipeline.

---

## How to Run the Application

**Prerequisites:**
* Python 3.x installed on your system.
* A functional webcam.

**Step 1: Install Dependencies**

Open your terminal or command prompt and install the required libraries:
```bash
   pip install opencv-python numpy
````

**Step 2: Execution**

Ensure your webcam is not being used by another application. Run the main script from your terminal:
````
python main.py
````

**Step 3: First Launch Note**

Upon the very first execution, the dl_verifier.py script will automatically download the required Deep Learning configuration (deploy.prototxt) and weights (res10_300x300_ssd_iter_140000.caffemodel) from the official OpenCV GitHub repository. Please allow a few seconds for this one-time download to complete.

**Step 4: Interacting with the Application**

Keep your head relatively still and move your eyes to see the gaze detection onto the window.

To quit the application cleanly, press the 'q' key on your keyboard or click the window's close (X) button.

## Concepts and Methodologies Utilized



**Fundamentals of Computer Vision & OpenCV:**
 Used OpenCV to read webcam streams, display real-time frames, and manipulate images.

**Image Representation & Processing:**

- Worked with grayscale and color images (converting BGR to Grayscale for Haar Cascade processing).

- Applied image filtering techniques (Gaussian Blur to reduce noise in the eye Region of Interest).

- Applied thresholding techniques (Inverse Binary Thresholding to isolate the dark pupil).

- Performed image segmentation (isolating the pupil blob from the sclera and skin).

**Contour Detection & Shape Analysis:**

- Detected and analyzed contours (using cv2.findContours on the thresholded image).

- Calculated spatial moments (cv2.moments) to accurately compute the centroid of the pupil contour.

**Object Detection & Tracking**

- Detected objects (Faces and Eyes) using Haar Cascade Classifiers.

**Deep Learning for Computer Vision:**

- Implemented a pre-trained Convolutional Neural Network (ResNet-SSD via the OpenCV DNN module) strictly as a verification step. The application calculates the geometric distance (error margin in pixels) between the classical Haar Cascade face center and the Deep Learning face center to validate the traditional methodology.

**Approach and Pipeline**

**Detection**: The system utilizes haarcascade_frontalface_default.xml and haarcascade_eye_tree_eyeglasses.xml to locate the user's face and eyes.

**Top Cropping**: A custom geometric crop is applied to the upper 30% of the eye bounding box to eliminate eyebrows, which commonly interfere with dark-pixel thresholding.

**Preprocessing**: The isolated Eye Region of Interest (ROI) undergoes grayscale conversion, Gaussian blurring, inverse binary thresholding, and morphological opening to isolate the pupil.

**Centroid Calculation**: The largest contour is selected, and its center of mass is calculated using image moments.

**Temporal Smoothing:** An Exponential Moving Average (EMA) low-pass filter is applied to the spatial coordinates of the pupil to reduce camera jitter and stabilize tracking.

**Deep Learning Verification**: Concurrently, a Caffe-based ResNet-SSD model processes the frame. A line and text overlay display the error vector between the classical CV detection and the DL detection to confirm accuracy in real-time.

## Performance Evaluation

**Strengths**: The system is lightweight and runs at high frame rates on standard CPUs without requiring external GPU acceleration. The temporal smoothing successfully mitigates the inherent jitter of frame-by-frame classical detection. The integration of OpenCV's DNN module proves the classical approach remains highly accurate under normal conditions.

**Limitations**: As a traditional threshold-based eye tracker, it is sensitive to severe lighting changes and extreme head poses. The threshold variable (THRESHOLD = 100) may require manual adjustment depending on the user's ambient room lighting.

## Originality Statement

The core framework relies on standard algorithms and pre-trained models provided within the OpenCV library (Haar Cascades, cv2.dnn ResNet).

**Original Contributions:**
The specific logic combining these concepts is entirely original. Specifically:

- The "Top Cropping" algorithm to mathematically exclude eyebrow interference.

- The Temporal Smoothing algorithm (Exponential Moving Average) to stabilize coordinate tracking across continuous frames without relying on heavy optical flow algorithms.

- The comparative verification module designed to continuously evaluate the error margin between the classical pipeline and the Deep Learning pipeline.