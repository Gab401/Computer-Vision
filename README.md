# Real-Time Eye Tracking & Deep Learning Verification System

## Project Overview
This project is an interactive, real-time eye-tracking application developed as a final assessment for the Computer Vision course. It captures a live webcam feed to detect faces, extract eye regions, isolate the pupils, and localize the user's pupil position frame by frame.

In strict adherence to the course guidelines, the core system is built entirely using Traditional Computer Vision techniques. A Deep Learning model (OpenCV DNN ResNet-SSD) is integrated purely as a supplementary verification step to confirm the accuracy of the traditional face detection pipeline.

---

## How to Run the Application

**Prerequisites:**
* Python 3.x installed on your system.
* A functional webcam.

**Step 1: Install Dependencies**

Open your terminal or command prompt and install the required libraries:
```bash
   pip install -r requirements.txt
````

Alternatively, install the libraries manually:
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

Keep your head relatively still in front of the camera to observe the pupil localization, the shape analysis overlays, the CamShift tracking box, and the live histogram windows.

To quit the application cleanly, press the 'q' key on your keyboard or click the window's close (X) button.

## Concepts and Methodologies Utilized

The project covers 8 of the 10 Learning Objectives listed in the course syllabus.

**1. Fundamentals of Computer Vision & OpenCV:**
Used OpenCV to read webcam streams, display real-time frames, perform matrix manipulation, and apply mirror transformations.

**2. Image Representation & Processing:**

- Worked with grayscale and color images (BGR→Grayscale conversion for Haar Cascade processing; color overlays for visualization).

- Computed and displayed real-time intensity histograms (cv2.calcHist) of each eye Region of Interest in independent windows to analyze pixel distribution.

- Applied image filtering (Gaussian Blur with a 7x7 kernel to reduce sensor noise in the eye ROI).

- Applied adaptive thresholding (Otsu's method via cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) to isolate the dark pupil without hardcoded threshold values.

- Applied morphological transformations (Opening) to clean noise artifacts after thresholding.

- Performed image segmentation (isolating the pupil blob from the sclera and skin).

**3. Contour Detection & Shape Analysis:**

- Detected contours using cv2.findContours with the RETR_TREE retrieval mode, which preserves hierarchical contour relationships.

- Computed convex hulls (cv2.convexHull) of the pupil contour to analyze its outer envelope.

- Performed shape approximation (cv2.approxPolyDP) using a 2% arc-length epsilon to simplify the pupil contour to its dominant vertices.

- Calculated spatial moments (cv2.moments) to compute the precise centroid of the pupil contour.

**4. Object Detection & Tracking:**

- Detected faces and eyes using Haar Cascade Classifiers (haarcascade_frontalface_default.xml and haarcascade_eye_tree_eyeglasses.xml).

- Tracked the pupil region using the CamShift algorithm (cv2.CamShift) operating over the thresholded binary mask as a probability density input.

**5. Deep Learning for Computer Vision:**

- Implemented a pre-trained Convolutional Neural Network (ResNet-SSD via the OpenCV DNN module) strictly as a verification step. The application calculates the Euclidean geometric distance (error margin in pixels) between the classical Haar Cascade face center and the Deep Learning face center to validate the traditional methodology.

**6. Practical Applications:**

- Integrated all of the above into a complete real-world Computer Vision application running in real-time on a standard CPU.

## Approach and Pipeline

**Detection**: The system utilizes haarcascade_frontalface_default.xml and haarcascade_eye_tree_eyeglasses.xml to locate the user's face and eyes.

**Top Cropping**: A custom geometric crop is applied to the upper 30% of the eye bounding box to eliminate eyebrows, which commonly interfere with dark-pixel thresholding.

**Histogram Analysis**: For each detected eye, an intensity histogram of the cropped grayscale eye ROI is computed (cv2.calcHist) and rendered in a dedicated window. This visualization motivates the dark-pupil isolation step.

**Preprocessing**: The isolated Eye Region of Interest (ROI) undergoes Gaussian blurring and Otsu's adaptive inverse binary thresholding, followed by morphological opening to isolate the pupil without requiring manual threshold tuning.

**Shape Analysis**: The largest contour is selected, and both its convex hull (drawn in magenta) and a polygonal approximation (drawn in cyan) are rendered to characterize the pupil's geometry.

**Centroid Calculation**: The center of mass of the largest contour is calculated using image moments (cv2.moments) and drawn in red.

**CamShift Tracking**: The thresholded binary mask is fed into the CamShift algorithm as a probability density input, and the resulting rotated bounding box (drawn in orange) provides an adaptive, scale- and orientation-aware track of the pupil region.

**Temporal Smoothing:** An Exponential Moving Average (EMA) low-pass filter is applied to the spatial coordinates of the pupil centroid to reduce camera jitter and stabilize the displayed position.

**Deep Learning Verification**: Concurrently, a Caffe-based ResNet-SSD model processes the frame. A line and text overlay display the error vector between the classical CV detection and the DL detection to confirm accuracy in real-time. When the DNN fails to detect a face (e.g., in extreme profile), the overlay reports "Face DL Error: N/A" rather than disappearing silently.

## Performance Evaluation

**Strengths**: The system is lightweight and runs at high frame rates on standard CPUs without requiring external GPU acceleration. The Otsu adaptive thresholding eliminates the need for manual threshold tuning across different lighting conditions. The EMA temporal smoothing successfully mitigates the inherent jitter of frame-by-frame classical detection. The integration of OpenCV's DNN module empirically validates the classical pipeline: the DL/CV face-center disagreement is typically only ~2 pixels in head-on poses.

**Limitations**: As a Haar-Cascade-based pipeline, it remains sensitive to extreme head poses where the frontal face cascade fails. In profile poses, the DL/CV center disagreement grows (observed ~17 pixels), reflecting the fundamental limit of frontal-face cascades rather than a flaw in the thresholding logic.

## Originality Statement

The core framework relies on standard algorithms and pre-trained models provided within the OpenCV library (Haar Cascades, cv2.dnn ResNet-SSD).

**Original Contributions:**
The specific logic combining these concepts is entirely original. Specifically:

- The "Top Cropping" algorithm to mathematically exclude eyebrow interference from the eye bounding box.

- The use of the Otsu-thresholded binary mask as a probability density input for the CamShift tracker — an unconventional combination of segmentation and tracking pipelines.

- The Temporal Smoothing algorithm (Exponential Moving Average) applied to the pupil centroid to stabilize coordinate display across continuous frames.

- The comparative verification module designed to continuously evaluate the Euclidean error margin between the classical pipeline and the Deep Learning pipeline, with explicit "no detection" fallback handling.
