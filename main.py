# Main application for eye tracking using OpenCV and Deep Learning verification



# General imports
import math
import cv2
import numpy as np

# Local imports
from dl_verifier import DeepLearningVerifier


# Constants
window_name = 'Eye Tracking - CV vs Deep Learning'




def main():
    
    # Load Haar cascades for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    # Initialize Deep Learning Verifier
    dl_verifier = DeepLearningVerifier()

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

    # Track open histogram windows to manage them cleanly
    open_hist_windows = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for Haar Cascades
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get Deep Learning baseline predictions for verification
        dl_face_center = dl_verifier.get_face_center(frame)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.3, 
            minNeighbors=5
        )

        current_frame_hist_windows = set()
     
        # Loop over detected faces
        for (x, y, w, h) in faces:
            cv_face_cx = x + w // 2
            cv_face_cy = y + h // 2

            # Draw classical face bounding box and center (BLUE)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (cv_face_cx, cv_face_cy), 4, (255, 0, 0), -1)

            # --- DEEP LEARNING VERIFICATION STEP ---
            if dl_face_center:
                dl_cx, dl_cy = dl_face_center
                
                # Draw DL face center (ORANGE)
                cv2.circle(frame, (dl_cx, dl_cy), 4, (0, 165, 255), -1)
                
                # Draw connecting line and calculate error
                distance = math.sqrt((cv_face_cx - dl_cx)**2 + (cv_face_cy - dl_cy)**2)
                cv2.line(frame, (cv_face_cx, cv_face_cy), (dl_cx, dl_cy), (0, 255, 255), 1)
                cv2.putText(frame, f"Face DL Error: {int(distance)}px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Face DL Error: N/A", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # --- EYE TRACKING PIPELINE ---
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
                # Cut the top 30% of the bounding box to remove the eyebrow
                crop_top = int(eh * 0.30)
                
                ey_cropped = ey + crop_top
                eh_cropped = eh - crop_top

                # Draw green rectangle around the cropped eye (without eyebrow)
                cv2.rectangle(roi_color, (ex, ey_cropped), (ex + ew, ey_cropped + eh_cropped), (0, 255, 0), 2)
                
                # Extract the specific Eye ROI using the cropped coordinates
                eye_roi_gray = roi_gray[ey_cropped:ey_cropped + eh_cropped, ex:ex + ew]
                eye_roi_color = roi_color[ey_cropped:ey_cropped + eh_cropped, ex:ex + ew]

                # --- HISTOGRAM ANALYSIS (Independent Windows) ---
                hist = cv2.calcHist([eye_roi_gray], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)
                hist_img = np.zeros((100, 256, 3), dtype=np.uint8)
                
                for j in range(256):
                    cv2.line(hist_img, (j, 100), (j, 100 - int(hist[j][0])), (255, 255, 255), 1)
                
                hist_win_name = f"Histogram - Eye {i}"
                cv2.imshow(hist_win_name, hist_img)
                current_frame_hist_windows.add(hist_win_name)
                open_hist_windows.add(hist_win_name)

                # Gaussian Blur
                blur = cv2.GaussianBlur(eye_roi_gray, (7, 7), 0)

                # Otsu's method
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Morphology (Open) to clean up small white dots (noise)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                # Find contours of the white blobs (potential pupils)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Sort contours by area in descending order and pick the largest
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    largest_contour = contours[0]

                    # --- SHAPE APPROXIMATION AND CONVEX HULL ---
                    # Approximate the shape of the pupil
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx_shape = cv2.approxPolyDP(largest_contour, epsilon, True)
                    cv2.drawContours(eye_roi_color, [approx_shape], -1, (255, 255, 0), 1) # Cyan

                    # Compute and draw the convex hull
                    hull = cv2.convexHull(largest_contour)
                    cv2.drawContours(eye_roi_color, [hull], -1, (255, 0, 255), 1) # Magenta

                    # --- CAMSHIFT TRACKING ---
                    x_c, y_c, w_c, h_c = cv2.boundingRect(largest_contour)
                    if w_c > 0 and h_c > 0:
                        track_window = (x_c, y_c, w_c, h_c)
                        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                        # CamShift uses the binary threshold as a probability map
                        ret_camshift, track_window = cv2.CamShift(thresh, track_window, term_crit)
                        pts = cv2.boxPoints(ret_camshift)
                        pts = np.int32(pts)
                        cv2.polylines(eye_roi_color, [pts], True, (0, 165, 255), 1) # Orange bounding box

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
                        cv2.circle(eye_roi_color, (smoothed_cx, smoothed_cy), 4, (0, 0, 255), -1)

        # Cleanup obsolete histogram windows (if an eye is lost)
        windows_to_close = open_hist_windows - current_frame_hist_windows
        for win in windows_to_close:
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass
        open_hist_windows = current_frame_hist_windows

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