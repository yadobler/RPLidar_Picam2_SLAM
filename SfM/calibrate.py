import cv2
import numpy as np
import time
import glob
import os
from picamera2 import Picamera2

# ==== 1. CAPTURE IMAGES ====
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.configure("preview")
picam2.start()

output_dir = "calib_images"
os.makedirs(output_dir, exist_ok=True)

print("Starting image capture...")

print("Press 'q' to quit early.")
print("Preview running, capturing an image every 5 seconds...")

num_images = 10
capture_interval = 5  # seconds
captured = 0
last_capture_time = time.time()

while captured < num_images:
    frame = picam2.capture_array()
    cv2.imshow("Live Preview", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Early exit.")
        break

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        filename = os.path.join(output_dir, f"chessboard_{captured:02}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[{captured+1}/{num_images}] Saved {filename}")
        captured += 1
        last_capture_time = current_time

cv2.destroyAllWindows()
picam2.stop()

# ==== 2. CAMERA CALIBRATION ====
print("Starting calibration...")

# Chessboard dimensions
chessboard_size = (7, 10)
frame_size = (640, 480)

# Prepare object points like (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = sorted(glob.glob(os.path.join(output_dir, "chessboard_*.jpg")))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(500)
    else:
        print(f"!! Chessboard not found in {fname}")

cv2.destroyAllWindows()

# Calibrate the camera
if len(objpoints) >= 5:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None)

    print("\n=== Calibration Results ===")
    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs.ravel())

    np.savez("calibration_data.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    print("\nSaved to calibration_data.npz")
else:
    print("Not enough valid images for calibration!!!!")
