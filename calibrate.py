import cv2
import numpy as np
import threading
from myrplidar import MyRPLidar 
import time

# --- Globals ---
cursor_pos = (334, 261)
fov_angle = 20  # Starting FOV in degrees
image_size = (640, 480)
min_dist = 10   
max_dist = 2000 

# Shared data structure for lidar points: dictionary + lock
# Stores { rounded_angle_deg: {'distance': dist_mm, 'timestamp': time.monotonic()} }
THREAD_SAFE_LIDAR_DATA = {
    "data": {},
    "lock": threading.Lock()
}

# Stale data configuration
STALE_THRESHOLD_SEC = 1.0 # Data older than this will be considered stale and reset to max_dist

# --- Mouse Callback for UI ---
def mouse_callback(event, x, y, flags, param):
    global cursor_pos, fov_angle
    cursor_pos = (x, y)
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0: 
            # Scroll up/forward (increase FOV)
            fov_angle = min(180, fov_angle + 10)
        else: 
            # Scroll down/backward (decrease FOV)
            fov_angle = max(5, fov_angle - 1)

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

    # Calculate FOV boundaries
    min_fov_boundary = normalize_angle_degrees(norm_center_angle - half_fov)
    max_fov_boundary = normalize_angle_degrees(norm_center_angle + half_fov)

    norm_angle = normalize_angle_degrees(angle)

    if min_fov_boundary <= max_fov_boundary: # FOV does not cross 0/360 boundary
        return min_fov_boundary <= norm_angle <= max_fov_boundary
    else: # FOV crosses 0/360 boundary (e.g., min_angle = 350, max_angle = 10)
        return norm_angle >= min_fov_boundary or norm_angle <= max_fov_boundary

# --- Get Filtered Lidar Points for Drawing ---
def get_filtered_lidar_points(lidar_data_dict, center_angle_deg, fov, image_width):
    """
    Transforms the dictionary of lidar data into (x_screen, distance) pairs
    for drawing, considering the current FOV.
    """
    result = []
    half_fov = fov / 2.0
    norm_center_angle = normalize_angle_degrees(center_angle_deg)

    for angle_deg_rounded, data_point in lidar_data_dict.items():
        distance = data_point['distance']

        # Only process points within the current dynamic FOV of the camera
        if is_angle_in_fov(angle_deg_rounded, norm_center_angle, fov):
            # Calculate the angular difference relative to the center of the FOV
            # This ensures smooth projection even when crossing 0/360 boundary
            angular_diff = normalize_angle_degrees(angle_deg_rounded - norm_center_angle + 180) - 180

            # Map this relative angle [-half_fov, +half_fov] to screen x [0, image_width]
            x_screen = int(((angular_diff + half_fov) / fov) * image_width)
            result.append((x_screen, distance))
    return result

# --- Drawing Overlay ---
def draw_lidar_overlay(frame, lidar_data_dict_for_drawing, fov_angle, center_angle_deg):
    width, height = frame.shape[1], frame.shape[0]
    points = get_filtered_lidar_points(lidar_data_dict_for_drawing, center_angle_deg, fov_angle, width)

    for x, dist in points:
        if 0 <= x < width:
            # Normalize distance for color mapping
            norm = np.clip((dist - min_dist) / (max_dist - min_dist), 0, 1)
            red = int((1 - norm) * 255)
            green = int(norm * 255)
            color = (0, green, red)  # BGR
            y = cursor_pos[1] # Draw all points on the same horizontal line
            cv2.circle(frame, (x, y), 2, color, -1)

    # Display text overlays
    cv2.putText(frame, f"FOV: {fov_angle} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Center angle: {center_angle_deg:.1f} deg", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"(x, y): {cursor_pos} of max {image_size}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# --- Camera Loop (Main Thread) ---
def camera_loop(center_angle_deg):
    print("Camera loop starting...")

    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": image_size})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    cv2.namedWindow("LIDAR + Camera View")
    cv2.setMouseCallback("LIDAR + Camera View", mouse_callback)

    while True:
        try:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            current_lidar_data_for_drawing = {}
            current_time = time.monotonic()

            with THREAD_SAFE_LIDAR_DATA["lock"]:
                # Iterate through accumulated data to check for staleness
                angles_to_remove = [] # Store angles to remove or reset
                for angle, data in THREAD_SAFE_LIDAR_DATA["data"].items():
                    if (current_time - data['timestamp']) > STALE_THRESHOLD_SEC:
                        # Data is stale, mark for reset to max_dist
                        current_lidar_data_for_drawing[angle] = {'distance': max_dist, 'timestamp': current_time} 
                    else:
                        # Data is fresh, use it
                        current_lidar_data_for_drawing[angle] = data


            # Pass the processed (fresh/reset) data for drawing
            draw_lidar_overlay(frame, current_lidar_data_for_drawing, fov_angle, center_angle_deg)

            cv2.imshow("LIDAR + Camera View", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC key to exit
                break
        except Exception as e:
            print(f"Error in camera loop: {e}")
            break

    print("Stopping camera...")
    picam2.stop()
    cv2.destroyAllWindows()

# --- Lidar Loop (Separate Thread) ---
def lidar_loop(center_angle_deg):
    print("Starting LIDAR thread...")
    PORT = '/dev/ttyUSB0'
    lidar = None

    try:
        lidar = MyRPLidar(PORT)
        print("LIDAR connected and motor started.")

        for scan in lidar: 
            for quality, angle, distance in scan: # Each point in a scan
                # Apply initial filtering
                if distance < min_dist or distance > max_dist or quality < 0:
                    continue

                # Round angle to nearest degree to use as a dictionary key
                # This groups points very close to each other into the same 'degree bin'
                rounded_angle = round(angle) % 360

                # Update the shared dictionary with the latest valid point
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
                del lidar # Calls MyRPLidar's __del__ for cleanup
                print("LIDAR cleanup complete.")
            except Exception as e:
                print(f"LIDAR cleanup failed: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize the shared dictionary with max_dist for all angles initially
    # This ensures that even angles not yet scanned show as 'far away'
    for i in range(360):
        THREAD_SAFE_LIDAR_DATA["data"][i] = {'distance': max_dist, 'timestamp': time.monotonic()}


    # Fixed center angle from calibrated cursor (calculated once at start)
    x_ratio = (cursor_pos[0] / image_size[0]) - 0.5
    initial_center_angle = (-x_ratio * fov_angle * 2) % 360 # This calculation is for the initial setup.

    lidar_thread = threading.Thread(target=lidar_loop, args=(initial_center_angle,), daemon=True)
    lidar_thread.start()

    time.sleep(2) # Give lidar thread time to populate some data
    camera_loop(initial_center_angle) 