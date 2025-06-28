import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

# --- Camera and Display Configuration ---
WIDTH = 640
HEIGHT = 480
CALIBRATION_FILE = "calibration_data.npz"

# --- Feature Detection and Matching Configuration ---

# feature_detector = cv2.SIFT_create()
# feature_matcher = cv2.BFMatcher()

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50) # or pass an empty dictionary

feature_detector = cv2.ORB_create(nfeatures=100)
feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)

print(f"Loaded feature detector {feature_detector} and matcher {feature_matcher}")

# --- Load Camera Calibration Data ---
if not os.path.exists(CALIBRATION_FILE):
    print(f"Error: Calibration file '{CALIBRATION_FILE}' not found.")
    print("Please run the calibration script first.")
    exit()

with np.load(CALIBRATION_FILE) as data:
    K = data['camera_matrix']  # Intrinsic parameters
    dist_coeffs = data['dist_coeffs']  # Lens distortion

print("Successfully loaded camera calibration data.")
print(f"K = {K}, distortion = {dist_coeffs}")

# --- State Variables for SfM ---
prev_frame = None
prev_keypoints = None
prev_descriptors = None
camera_poses = [np.eye(4)]  # Initial world-to-camera pose
point_cloud = []

# --- Visualisation function ---
def visualise_output(frame, prev_frame, prev_keypoints, keypoints, good_matches,
                     camera_poses, point_cloud, K, dist_coeffs):
    """Draws match visualization, 2D point projection, and trajectory path view."""
    # 1. Match Visualization
    if prev_keypoints is not None and keypoints is not None:
        match_img = cv2.drawMatches(
            prev_frame if prev_frame is not None else frame,
            prev_keypoints, frame, keypoints,
            good_matches if good_matches else [],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        # If no previous frame or keypoints, create a blank image for match_img
        match_img = np.zeros((HEIGHT, WIDTH * 2, 3), dtype=np.uint8)

    # 2. Draw Camera Path
    path_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    origin_x, origin_z = WIDTH // 2, HEIGHT // 2
    # Convert camera poses (T_world_camera) into a top-down view (x,z plane)
    for i in range(len(camera_poses) - 1):
        # camera_poses[i][0, 3] is x-translation, camera_poses[i][2, 3] is z-translation
        # Add origin_x/z for centering the path on the canvas
        p1 = (int(camera_poses[i][0, 3]) + origin_x, int(camera_poses[i][2, 3]) + origin_z)
        p2 = (int(camera_poses[i + 1][0, 3]) + origin_x, int(camera_poses[i + 1][2, 3]) + origin_z)
        # Only draw if points are within canvas bounds
        if 0 <= p1[0] < WIDTH and 0 <= p1[1] < HEIGHT and \
           0 <= p2[0] < WIDTH and 0 <= p2[1] < HEIGHT:
            cv2.line(path_canvas, p1, p2, (0, 255, 0), 2)

    # 3. Project 3D Point Cloud
    if point_cloud:
        # T_camera_world is needed for projectPoints
        T_camera_world = np.linalg.inv(camera_poses[-1])
        rvec, _ = cv2.Rodrigues(T_camera_world[:3, :3])
        tvec = T_camera_world[:3, 3]
        
        # Project all 3D points
        points_2d, _ = cv2.projectPoints(np.array(point_cloud), rvec, tvec, K, dist_coeffs)

        # Draw projected points on the current frame
        if points_2d is not None: # Ensure projection was successful
            for p_float_array in points_2d:
                x_float, y_float = p_float_array.ravel() # Use ravel for cleaner access
                
                if np.isfinite(x_float) and np.isfinite(y_float):
                    # Clamp values to image boundaries before casting to int
                    x_int = int(np.clip(x_float, 0, WIDTH - 1))
                    y_int = int(np.clip(y_float, 0, HEIGHT - 1))
                    cv2.circle(frame, (x_int, y_int), 1, (0, 0, 255), -1)


    # 4. Combine and Show
    top_view = np.hstack((frame, path_canvas))
    combined_view = np.vstack((top_view, match_img))
    cv2.imshow("Real-time SfM with OpenCV", combined_view)

def main():
    global prev_frame, prev_keypoints, prev_descriptors, camera_poses, point_cloud, path_canvas

    # --- Initialize PiCamera2 ---
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    while True:
        # --- 1. Image Capture and Undistortion ---
        frame_distorted = picam2.capture_array()
        frame = cv2.undistort(frame_distorted, K, dist_coeffs)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 2. Feature Extraction ---
        keypoints, descriptors = feature_detector.detectAndCompute(gray_frame, None)

        # --- 3. Feature Matching w/ Lowe's Ratio Test ---
        good_matches = []
        if prev_frame is not None and descriptors is not None and prev_descriptors is not None:
            # FLANN knnMatch expects prev_descriptors and descriptors to be non-empty
            if prev_descriptors.shape[0] >= 2 and descriptors.shape[0] >= 2: # Check if there are at least 2 descriptors
                matches = feature_matcher.knnMatch(prev_descriptors, descriptors, k=2)
                # Apply Lowe's Ratio Test
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
            
        # --- 4. Prepare Matched Points ---
            if len(good_matches) > 10:
                prev_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                current_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # --- 5. Epipolar Geometry (Estimate Essential Matrix) ---
                E, mask = cv2.findEssentialMat(current_pts, prev_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, current_pts, prev_pts, K)

        # --- 6. Motion to Pose (Chaining Global Camera Transform) ---
                    current_transform = np.hstack((R, t))
                    current_transform = np.vstack((current_transform, [0, 0, 0, 1]))
                    # last @ transform = new
                    prev_pose = camera_poses[-1]
                    new_pose = prev_pose @ np.linalg.inv(current_transform)
                    camera_poses.append(new_pose)

        # --- 7. Triangulation (Reconstruct 3D Points) ---
                    P1 = K @ np.linalg.inv(prev_pose)[:3, :]
                    P2 = K @ np.linalg.inv(new_pose)[:3, :]
                    points_4d_hom = cv2.triangulatePoints(P1, P2, prev_pts.T, current_pts.T)
                    
                    # Convert homogeneous to cartesian
                    points_3d = points_4d_hom[:3] / points_4d_hom[3]
                    point_cloud.extend(points_3d.T)
        
        # --- 8. Draw Matches ---
        visualise_output(
            frame, 
            prev_frame, 
            prev_keypoints, 
            keypoints,
            good_matches,
            camera_poses, 
            point_cloud, 
            K, 
            dist_coeffs
        )
        
        # --- Update for Next Frame ---
        prev_frame = frame.copy()
        prev_keypoints = keypoints
        prev_descriptors = descriptors

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()
