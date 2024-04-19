import cv2
import os
import numpy as np
import time

# Create a directory to store images if it doesn't exist yet
output_dir = './stereo_images'
os.makedirs(output_dir, exist_ok=True)

stereoMapL_x = np.load('./camera_calibration/stereoMapL_x.npy')
stereoMapL_y = np.load('./camera_calibration/stereoMapL_y.npy')
stereoMapR_x = np.load('./camera_calibration/stereoMapR_x.npy')
stereoMapR_y = np.load('./camera_calibration/stereoMapR_y.npy')
roi_left = np.load('./camera_calibration/roi_L.npy')
roi_right = np.load('./camera_calibration/roi_R.npy')
# Extracting ROI information
x_left, y_left, width_left, height_left = roi_left
x_right, y_right, width_right, height_right = roi_right

# Create pipelines to read images from the stereo camera on Jetson Nano and manipulate the images
pipeline_left = cv2.VideoCapture("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
pipeline_right = cv2.VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Start the pipelines
if not pipeline_left.isOpened() or not pipeline_right.isOpened():
    print("Failed to start the pipelines")
    exit()

# Get FPS of the cameras
fps_left = int(pipeline_left.get(cv2.CAP_PROP_FPS))
fps_right = int(pipeline_right.get(cv2.CAP_PROP_FPS))

# Loop to capture and save images from the stereo camera
count = 0
start_time = time.time()
while True:
    # Read images from the left camera pipeline
    ret_left, frame_left = pipeline_left.read()

    # Read images from the right camera pipeline
    ret_right, frame_right = pipeline_right.read()

    # Check if reading images is possible
    if not ret_left or not ret_right:
        print("Failed to read images")
        break

    # Remap image
    undistortedL = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR)
    undistortedR = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR)

    # Find intersection of ROIs
    intersection_x = max(x_left, x_right)
    intersection_y = max(y_left, y_right)
    intersection_width = min(x_left + width_left, x_right + width_right) - intersection_x
    intersection_height = min(y_left + height_left, y_right + height_right) - intersection_y

    # Crop images based on the intersection ROI
    undistortedL_cropped = undistortedL[intersection_y:intersection_y+intersection_height, intersection_x:intersection_x+intersection_width]
    undistortedR_cropped = undistortedR[intersection_y:intersection_y+intersection_height, intersection_x:intersection_x+intersection_width]
    
    # Get the dimensions (width and height) of the cropped frames
    height, width, _ = undistortedL_cropped.shape
    print(height, width)
    
    # Draw horizontal line at the center of the image
    center_y = height // 2
    cv2.line(undistortedL_cropped, (0, center_y), (width, center_y), (0, 255, 0), 2)
    cv2.line(undistortedR_cropped, (0, center_y), (width, center_y), (0, 255, 0), 2)

    # Draw vertical line at the center of the image
    center_x = width // 2
    cv2.line(undistortedL_cropped, (center_x, 0), (center_x, height), (0, 255, 0), 2)
    cv2.line(undistortedR_cropped, (center_x, 0), (center_x, height), (0, 255, 0), 2)

    # Draw circle at center (cx, cy) with radius 50
    cx, cy = width // 2, height // 2
    cv2.circle(undistortedL_cropped, (cx, cy), 50, (0, 0, 255), 2)
    cv2.circle(undistortedR_cropped, (cx, cy), 50, (0, 0, 255), 2)
        
    # # Display images from both cameras
    # cv2.imshow('frame_left', undistortedL_cropped)
    # cv2.imshow('frame_right', undistortedR_cropped)
    
    # Concatenate the left and right images horizontally
    stereo_image = cv2.hconcat([undistortedL_cropped, undistortedR_cropped])
        
    # Calculate FPS
    count += 1
    if count % 10 == 0:
        elapsed_time = time.time() - start_time
        fps = count / elapsed_time
        fps_text = f"FPS: {fps:.2f}"
        # Put FPS text on the stereo image
        cv2.putText(stereo_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Reset count and start time for the next FPS calculation
        count = 0
        start_time = time.time()
        
    # Display the stereo image
    cv2.imshow('Stereo Image', stereo_image)
    
    key = cv2.waitKey(1)
    # Press 'q' to exit the image display
    if key == ord('q'):
        break

# Release the pipelines
pipeline_left.release()
pipeline_right.release()
cv2.destroyAllWindows()