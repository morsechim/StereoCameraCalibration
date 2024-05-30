# Stereo Image Capture and Display

This Python script captures stereo images from a stereo camera setup using the Jetson Nano. It then displays the captured stereo images, performing camera calibration and distortion correction in real-time.

## Prerequisites

- Python 3.x
- OpenCV (cv2) library
- NumPy library

## Installation

1. Clone this repository to your local machine.
2. Make sure you have all the prerequisites installed.
3. Ensure that the stereo camera setup is correctly connected to the Jetson Nano.
4. Place the camera calibration files (`stereoMapL_x.npy`, `stereoMapL_y.npy`, `stereoMapR_x.npy`, `stereoMapR_y.npy`, `roi_L.npy`, `roi_R.npy`) in the `camera_calibration` directory within the project folder.

## Usage

1. Run the Python script `stereo_image_capture.py`.
2. The script will create a directory named `stereo_images` to store captured images if it doesn't already exist.
3. The stereo cameras will start capturing images and display them in real-time.
4. Press the 'q' key to exit the image display.

## Features

- Camera calibration and distortion correction using pre-calibrated camera parameters.
- Real-time display of stereo images with overlaid centerlines and FPS information.

## File Descriptions

- `stereo_image_capture.py`: Python script for capturing and displaying stereo images.
- `camera_calibration`: Directory containing camera calibration files.
  - `stereoMapL_x.npy`, `stereoMapL_y.npy`: Distortion correction maps for the left camera.
  - `stereoMapR_x.npy`, `stereoMapR_y.npy`: Distortion correction maps for the right camera.
  - `roi_L.npy`, `roi_R.npy`: Region of Interest (ROI) information for the left and right cameras.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

