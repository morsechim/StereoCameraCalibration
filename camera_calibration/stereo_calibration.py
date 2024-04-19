import numpy as np
import cv2
import glob

# Find Chessboard Corners - Object Points and Image Points
chessboard_size = (10, 7)
frame_size = (640, 480)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, (0, 0, 0), (1, 0, 0), (2, 0, 0) ... (9, 6, 0) => chessboard_size = (10, 7)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real-world space.
imgpointsL = []  # 2d point in left image plane.
imgpointsR = []  # 2d point in right image plane.

# Read images from the left_images and right_images directories
left_images = sorted(glob.glob('./left_images/*.jpg'))
right_images = sorted(glob.glob('./right_images/*.jpg'))

# Create an OpenCV window to display stereo images
cv2.namedWindow('Draw Stereo Images', cv2.WINDOW_NORMAL)

# Loop through the images
for left_image_path, right_image_path in zip(left_images, right_images):
    # Read the left and right images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    retL, cornersL = cv2.findChessboardCorners(left_gray, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(right_gray, chessboard_size, None)
    
    # If found, add object points, image points (after refining them)
    if retL and retR:  # Checking if both corners are found
        objpoints.append(objp)
        
        # Refine the corner locations to sub-pixel accuracy
        cornersL = cv2.cornerSubPix(left_gray, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)
        
        cornersR = cv2.cornerSubPix(right_gray, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(left_image, chessboard_size, cornersL, retL)
        cv2.drawChessboardCorners(right_image, chessboard_size, cornersR, retR)
        
        # Concatenate the left and right images
        stereo_image = cv2.hconcat([left_image, right_image])

        # Display the stereo image in the 'Draw Stereo Images' window
        cv2.imshow('Draw Stereo Images', stereo_image)
        cv2.waitKey(500)  # Wait for a key press to proceed to the next image

# Close the window after all images have been displayed
cv2.destroyAllWindows()
        
# Calibration for left camera
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frame_size, None, None)
heightL, widthL, channelsL = left_image.shape
newCameraMatrixL, roiL = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

# Calibration for right camera
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frame_size, None, None)
heightR, widthR, channelsR = right_image.shape
newCameraMatrixR, roiR = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

# Stereo Vision Calibration
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camera matrices so that only Rot, Trans, Emat, and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calculate the transformation between the two cameras and Essential and Fundamental Matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, left_gray.shape[::-1], criteria_stereo, flags)

# Stereo Rectification
rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, left_gray.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, left_gray.shape[::-1], cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, right_gray.shape[::-1], cv2.CV_16SC2)

# Calculate RMS error percentage
rms_error_percentage = (retStereo / (left_image.shape[0] * left_image.shape[1])) * 100

# Save parameters to file
print("Saving parameters!")
np.save('stereoMapL_x.npy', stereoMapL[0])
np.save('stereoMapL_y.npy', stereoMapL[1])
np.save('stereoMapR_x.npy', stereoMapR[0])
np.save('stereoMapR_y.npy', stereoMapR[1])
np.save('roi_L.npy', roi_L)
np.save('roi_R.npy', roi_R)

# Write RMS error and significant adjustment values to a text file
with open("stereo_calibration_parameters.txt", "w") as file:
    file.write("RMS Error: {} ({}%)\n".format(retStereo, rms_error_percentage))
    file.write("\n")
    file.write("===== Params =====")
    file.write("\n")
    file.write("New Camera Matrix Left:\n{}\n".format(newCameraMatrixL))
    file.write("\n")
    file.write("Distortion Coefficients Left:\n{}\n".format(distL))
    file.write("\n")
    file.write("New Camera Matrix Right:\n{}\n".format(newCameraMatrixR))
    file.write("\n")
    file.write("Distortion Coefficients Right:\n{}\n".format(distR))
    file.write("\n")
    file.write("Rotation Matrix:\n{}\n".format(rot))
    file.write("\n")
    file.write("Translation Vector:\n{}\n".format(trans))
    file.write("\n")
    file.write("Essential Matrix:\n{}\n".format(essentialMatrix))
    file.write("\n")
    file.write("Fundamental Matrix:\n{}\n".format(fundamentalMatrix))
    file.write("\n")
    file.write("===== StereoParams =====")
    file.write("Stereo Map Left X:\n{}\n".format(stereoMapL[0]))
    file.write("\n")
    file.write("Stereo Map Left Y:\n{}\n".format(stereoMapL[1]))
    file.write("\n")
    file.write("Stereo Map Right X:\n{}\n".format(stereoMapR[0]))
    file.write("\n")
    file.write("Stereo Map Right Y:\n{}\n".format(stereoMapR[1]))
    file.write("\n")
    file.write("Project Matrix Left :\n{}\n".format(projMatrixL))
    file.write("\n")
    file.write("Project Matrix Right :\n{}\n".format(projMatrixR))
    file.write("\n")
    file.write("Rectify Roi Left :\n{}\n".format(roi_L))
    file.write("\n")
    file.write("Rectify Roi Right :\n{}\n".format(roi_R))