import cv2
import numpy as np


# Constants
THRESHOLD = 100  # Threshold for binary inverse thresholding (tune based on lighting conditions)




def main():
    window_name = 'Eye Tracking'
    
    # Load Haar cascades for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Video stream started. Press 'q' or click the window 'X' to quit.")

    # Define the kernel for morphological operations (3x3 matrix of ones)
    kernel = np.ones((3, 3), np.uint8)

    cv2.namedWindow(window_name)

    # --- SMOOTHING VARIABLES ---
    # Dictionary to store previous center coordinates for each eye
    # Key: Eye index (rough estimation, 0 for left, 1 for right in the loop)
    # Value: (prev_cx, prev_cy)
    prev_centers = {}
    alpha = 0.6  # Smoothing factor (0.0 = max smoothing/lag, 1.0 = no smoothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for Haar Cascades
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.3, 
            minNeighbors=5
        )

        # Loop over detected faces
        for (x, y, w, h) in faces:
            # Draw blue rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the Region of Interest (ROI)
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes within the face ROI
            eyes = eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=10, 
                minSize=(20, 20)
            )

            # Clear previous centers if number of eyes changes drastically (to avoid mixing up eyes)
            if len(eyes) != len(prev_centers):
                prev_centers.clear()

            # We sort eyes by x-coordinate to consistently identify left (0) vs right (1) eye
            eyes = sorted(eyes, key=lambda e: e[0])

            # Loop over detected eyes
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                # Draw green rectangle around each eye
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Extract the specific Eye ROI
                eye_roi_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_roi_color = roi_color[ey:ey + eh, ex:ex + ew]

                # Gaussian Blur to reduce image noise
                blur = cv2.GaussianBlur(eye_roi_gray, (7, 7), 0)

                # Inverse Threshold (Dark pupil becomes white)
                # The threshold value might need tuning based on lighting!
                _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

                # Morphology (Open) to clean up small white dots (noise)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                # Find contours of the white blobs (potential pupils)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Sort contours by area in descending order and pick the largest
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    largest_contour = contours[0]

                    # Calculate moments to find the centroid of the contour
                    M = cv2.moments(largest_contour)
                    
                    # Prevent division by zero
                    if M["m00"] != 0:
                        raw_cx = int(M["m10"] / M["m00"])
                        raw_cy = int(M["m01"] / M["m00"])

                        # Temporal smoothing of the pupil center
                        if i in prev_centers:
                            # Apply Exponential Moving Average formula
                            prev_cx, prev_cy = prev_centers[i]
                            
                            smoothed_cx = int(alpha * raw_cx + (1 - alpha) * prev_cx)
                            smoothed_cy = int(alpha * raw_cy + (1 - alpha) * prev_cy)
                        else:
                            # First time seeing this eye, no smoothing possible yet
                            smoothed_cx = raw_cx
                            smoothed_cy = raw_cy

                        # Store current smoothed position for the next frame
                        prev_centers[i] = (smoothed_cx, smoothed_cy)

                        # Draw the smoothed pupil center
                        cv2.circle(eye_roi_color, (smoothed_cx, smoothed_cy), 2, (0, 0, 255), -1)

        # Display the frame
        cv2.imshow(window_name, frame)

        # Exit condition 1: 'q' key pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing application ('q' pressed)...")
            break
            
        # Exit condition 2: Window 'X' clicked
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Closing application (Window closed)...")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()