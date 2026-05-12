import cv2
import numpy as np
import os
import urllib.request

class DeepLearningVerifier:
    def __init__(self):
        """
        Initializes the OpenCV Deep Learning Face Detector (ResNet-SSD).
        Downloads the pre-trained models automatically if they are missing.
        """
        self.prototxt = "deploy.prototxt"
        self.model = "res10_300x300_ssd_iter_140000.caffemodel"
        self._download_models_if_needed()
        
        # Load the Deep Learning model using OpenCV DNN
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

    def _download_models_if_needed(self):
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(self.prototxt):
            print("Downloading DL model configuration (prototxt)...")
            urllib.request.urlretrieve(prototxt_url, self.prototxt)
        if not os.path.exists(self.model):
            print("Downloading DL model weights (caffemodel)...")
            urllib.request.urlretrieve(model_url, self.model)

    def get_face_center(self, frame):
        """
        Processes the frame through the DNN and returns the center (x,y) of the main face.
        """
        h, w = frame.shape[:2]
        # Pre-process the image for the DNN
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()

        max_confidence = 0
        best_center = None

        # Loop over detections to find the most confident face
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections (threshold > 50%)
            if confidence > 0.5 and confidence > max_confidence:
                max_confidence = confidence
                
                # Compute bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Calculate the center point
                cx = int((startX + endX) / 2)
                cy = int((startY + endY) / 2)
                best_center = (cx, cy)
                
        return best_center
    

