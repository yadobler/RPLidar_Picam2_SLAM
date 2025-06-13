import cv2
import numpy as np
import threading
from myrplidar import MyRPLidar
import time

# --- Globals ---
DEFAULT_X_CENTER_PX = 334           # Default X position for aligning the lidar's center
DEFAULT_Y_CAMERA_CENTER_PX = 240    # Camera's approximate Y center 

fov_angle = 20  
image_size = (640, 480)
min_dist = 10   # Minimum distance in mm
max_dist = 2000 # Maximum distance in mm

# New shared data structure for lidar points: dictionary + lock
# Stores { rounded_angle_deg: {'distance': dist_mm, 'timestamp': time.monotonic()} }
THREAD_SAFE_LIDAR_DATA = {
    "data": {},
    "lock": threading.Lock()
}

# Stale data configuration
STALE_THRESHOLD_SEC = 1.0 # Data older than this will be considered stale and reset to max_dist

# --- Parallax Correction Constants ---
# Derived from your measurements:
# Point 1: (D1, Y1) = (235mm, 255px)
# Point 2: (D2, Y2) = (800mm, 355px)
# Slope (m) = (Y2 - Y1) / (D2 - D1) = (355 - 255) / (800 - 235) = 100 / 565 ≈ 0.17699115
# Intercept (c) = Y1 - m * D1 = 255 - 0.17699115 * 235 ≈ 255 - 41.59292025 ≈ 213.40707975
LIDAR_Y_SLOPE = 100 / 565 # Store as fraction for precision or use float directly
LIDAR_Y_INTERCEPT = 213.40707975

# --- Helper for Angle Normalization (important for robust FOV checks) ---
def normalize_angle_degrees(angle):
    """Normalizes an angle to be within [0, 360) degrees."""
    return angle % 360

# --- Helper for checking if an angle is within a FOV ---
def is_angle_in_fov(angle, center_angle_deg, fov):
    """
    Checks if 'angle' is within the FOV defined by 'center_angle_deg' and 'fov'.
    Handles FOV crossing the 0/360 degree boundary.
    """
    half_fov = fov / 2.0
    norm_center_angle = normalize_angle_degrees(center_angle_deg)

    min_fov_boundary = normalize_angle_degrees(norm_center_angle - half_fov)
    max_fov_boundary = normalize_angle_degrees(norm_center_angle + half_fov)

    norm_angle = normalize_angle_degrees(angle)

    if min_fov_boundary <= max_fov_boundary:
        return min_fov_boundary <= norm_angle <= max_fov_boundary
    else:
        return norm_angle >= min_fov_boundary or norm_angle <= max_fov_boundary

# --- Get Filtered Lidar Points for Drawing ---
def get_filtered_lidar_points(lidar_data_dict, center_angle_deg, image_width):
    """
    Transforms the dictionary of lidar data into (x_screen, distance) pairs
    for drawing, considering the current FOV.
    """
    result = []
    fov = fov_angle
    half_fov = fov / 2.0
    norm_center_angle = normalize_angle_degrees(center_angle_deg)

    for angle_deg_rounded, data_point in lidar_data_dict.items():
        distance = data_point['distance']

        if is_angle_in_fov(angle_deg_rounded, norm_center_angle, fov):
            # Calculate the angular difference relative to the center of the FOV
            angular_diff = normalize_angle_degrees(angle_deg_rounded - norm_center_angle + 180) - 180

            # Map this relative angle [-half_fov, +half_fov] to screen x [0, image_width]
            x_screen = int(((angular_diff + half_fov) / fov) * image_width)
            result.append((x_screen, distance))
    return result

# --- Drawing Overlay ---
def draw_lidar_overlay(frame, lidar_data_dict_for_drawing, center_angle_deg):
    width, height = frame.shape[1], frame.shape[0]
    points = get_filtered_lidar_points(lidar_data_dict_for_drawing, center_angle_deg, width)

    for x, dist in points:
        if 0 <= x < width:
            # Calculate Y position based on distance (parallax correction)
            y_pixel = int(LIDAR_Y_SLOPE * dist + LIDAR_Y_INTERCEPT)
            # Clip Y to stay within image bounds
            y_pixel = np.clip(y_pixel, 0, height - 1)

            # Normalize distance for color mapping
            norm = np.clip((dist - min_dist) / (max_dist - min_dist), 0, 1)
            red = int((1 - norm) * 255)
            green = int(norm * 255)
            color = (0, green, red)  # BGR
            
            cv2.circle(frame, (x, y_pixel), 2, color, -1) # Use the calculated y_pixel

# --- Camera Loop (Main Thread) ---
def camera_loop(initial_center_angle): # Parameter name changed for clarity
    print("Camera loop starting...")

    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": image_size})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    cv2.namedWindow("LIDAR + Camera View")

    while True:
        try:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            current_lidar_data_for_drawing = {}
            current_time = time.monotonic()

            with THREAD_SAFE_LIDAR_DATA["lock"]:
                angles_to_remove = []
                for angle, data in THREAD_SAFE_LIDAR_DATA["data"].items():
                    if (current_time - data['timestamp']) > STALE_THRESHOLD_SEC:
                        current_lidar_data_for_drawing[angle] = {'distance': max_dist, 'timestamp': current_time}
                    else:
                        current_lidar_data_for_drawing[angle] = data

            draw_lidar_overlay(frame, current_lidar_data_for_drawing, initial_center_angle)
            cv2.imshow("LIDAR + Camera View", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        except Exception as e:
            print(f"Error in camera loop: {e}")
            break

    print("Stopping camera...")
    picam2.stop()
    cv2.destroyAllWindows()

# --- Lidar Loop (Separate Thread) ---
def lidar_loop(): 
    print("Starting LIDAR thread...")
    PORT = '/dev/ttyUSB0'
    lidar = None

    try:
        lidar = MyRPLidar(PORT)
        print("LIDAR connected and motor started.")

        for scan in lidar:
            for quality, angle, distance in scan:
                if distance < min_dist or distance > max_dist or quality < 0:
                    continue

                rounded_angle = round(angle) % 360

                with THREAD_SAFE_LIDAR_DATA["lock"]:
                    THREAD_SAFE_LIDAR_DATA["data"][rounded_angle] = {
                        'distance': distance,
                        'timestamp': time.monotonic()
                    }

    except Exception as e:
        print(f"LIDAR Error: {e}")
    finally:
        print("Shutting down LIDAR...")
        if lidar:
            try:
                del lidar
                print("LIDAR cleanup complete.")
            except Exception as e:
                print(f"LIDAR cleanup failed: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize the shared dictionary with max_dist for all angles initially
    for i in range(360):
        THREAD_SAFE_LIDAR_DATA["data"][i] = {'distance': max_dist, 'timestamp': time.monotonic()}

    x_ratio = (DEFAULT_X_CENTER_PX / image_size[0]) - 0.5
    initial_center_angle = (-x_ratio * fov_angle * 2) % 360

    lidar_thread = threading.Thread(target=lidar_loop, daemon=True) 
    lidar_thread.start()

    time.sleep(2)
    camera_loop(initial_center_angle) 