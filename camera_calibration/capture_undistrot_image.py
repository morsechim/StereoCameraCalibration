import cv2
import os
import time

# Prompt the user for the total number of images
total_images = input("Enter the total number of images to capture (1-99) or (q)uit: ")

# Check if the input is 'q' to quit
if total_images.lower() == 'q':
    print("Exiting the program...")
    exit()

# Convert the input to an integer
total_images = int(total_images)

# Validate the input
if total_images <= 0 or total_images >= 100:
    print("Invalid input. Please enter a number between 1 and 99.")
    exit()

# Create directories to store images if they don't exist
left_output_dir = './left_images'
right_output_dir = './right_images'
os.makedirs(left_output_dir, exist_ok=True)
os.makedirs(right_output_dir, exist_ok=True)

# Create pipelines to read images from the stereo camera on Jetson Nano and adjust the images
pipeline_left = cv2.VideoCapture("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
pipeline_right = cv2.VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Start the pipelines
if not pipeline_left.isOpened() or not pipeline_right.isOpened():
    print("Unable to start the pipelines")
    exit()

# Counter for saved images
count = 0

# Start time to measure frame rate
start_time = time.time()

# Loop to capture and save images from the stereo camera
while True:
    # Read images from the left camera pipeline
    ret_left, frame_left = pipeline_left.read()

    # Read images from the right camera pipeline
    ret_right, frame_right = pipeline_right.read()

    # Check if image reading is successful
    if not ret_left or not ret_right:
        print("Unable to read images")
        break

    # Display images from both cameras with frame rate
    fps = 1 / (time.time() - start_time)
    cv2.imshow('Stereo Cameras', cv2.putText(cv2.hconcat([frame_left, frame_right]), f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2))

    # Update start time for next iteration
    start_time = time.time()

    # Press 'c' to capture images
    key = cv2.waitKey(1)
    if key == ord('c'):
        # Save images as .jpg files
        left_filename = os.path.join(left_output_dir, f"left_{count}.jpg")
        right_filename = os.path.join(right_output_dir, f"right_{count}.jpg")
        cv2.imwrite(left_filename, frame_left)
        cv2.imwrite(right_filename, frame_right)
        print(f"Saved images: {left_filename}, {right_filename}")
        count += 1
        # Check if the desired number of images has been captured
        if count == total_images:
            print("All images saved")
            break
    # Press 'q' to exit the image display
    elif key == ord('q'):
        break

# Release the pipelines
pipeline_left.release()
pipeline_right.release()
cv2.destroyAllWindows()
