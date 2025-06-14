import cv2
import numpy as np
import threading
import time
from typing import Dict, Any, Tuple, List
from picamera2 import Picamera2
from myrplidar import MyRPLidar

# --- Configuration Class ---
class Config:
    """Groups all static configuration variables for clarity."""
    # Image and Camera
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    
    # LIDAR
    LIDAR_PORT = '/dev/ttyUSB0'
    FOV_DEGREES = 20.0
    MIN_DISTANCE_MM = 10
    MAX_DISTANCE_MM = 5000
    
    # Data Handling
    STALE_THRESHOLD_SEC = 1.0 # Data older than this is ignored

    # Parallax Correction: Y_pixel = m * distance_mm + c
    LIDAR_Y_SLOPE = 100 / 565
    LIDAR_Y_INTERCEPT = 213.407

    # This determines the forward-facing direction of the lidar relative to the camera.
    # A value of 0 means the lidar's 0° mark is dead center.
    LIDAR_CENTER_ANGLE_DEG = 0.0


# --- Shared Data ---
LIDAR_DATA: Dict[int, Dict[str, Any]] = {}
LIDAR_LOCK = threading.Lock()


# --- Lidar Processing Thread ---
def lidar_loop():
    """
    Connects to the LIDAR and continuously updates the shared data structure.
    This loop is optimized to lock only once per full scan, not per point.
    """
    print("Starting LIDAR thread...")
    lidar = None
    try:
        lidar = MyRPLidar(Config.LIDAR_PORT)
        print("LIDAR connected and motor started.")

        for scan in lidar:
            # Process a full scan into a temporary dictionary first
            scan_data = {}
            for quality, angle, distance in scan:
                if not (Config.MIN_DISTANCE_MM < distance < Config.MAX_DISTANCE_MM and quality > 1):
                    continue
                
                # Round angle to nearest integer for dictionary key
                rounded_angle = int(angle + 0.5) % 360
                scan_data[rounded_angle] = {
                    'distance': distance,
                    'timestamp': time.monotonic()
                }

            # Now, acquire the lock ONCE and update the main dictionary
            if scan_data:
                with LIDAR_LOCK:
                    LIDAR_DATA.update(scan_data)

    except Exception as e:
        print(f"LIDAR Error: {e}")
    finally:
        print("Shutting down LIDAR...")
        if lidar:
            del lidar # Rely on the destructor to stop motor and disconnect

# --- Main Application Thread (Camera and Drawing) ---
def get_fov_points(center_angle: float, fov: float) -> List[int]:
    """Calculates the integer angles within the specified Field of View."""
    half_fov = fov / 2.0
    
    # Calculate start and end angles, handling the 0/360 wrap-around
    start_angle = (center_angle - half_fov)
    end_angle = (center_angle + half_fov)

    angles = []
    # Handle wrap-around case (e.g., center=5, fov=20 -> range 355 to 15)
    if start_angle < 0:
        angles.extend(range(int(360 + start_angle), 360))
        start_angle = 0
        
    if end_angle > 360:
        angles.extend(range(0, int(end_angle % 360) + 1))
        end_angle = 360

    angles.extend(range(int(start_angle), int(end_angle) + 1))
    return sorted(list(set(angles))) # Return unique, sorted list


def main_loop():
    """
    Handles camera capture, data processing, and visualization.
    """
    print("Camera loop starting...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": Config.IMAGE_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0) # Allow camera to warm up

    cv2.namedWindow("LIDAR + Camera View")

    # Pre-calculate the angles we care about
    fov_angles_to_check = get_fov_points(Config.LIDAR_CENTER_ANGLE_DEG, Config.FOV_DEGREES)

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        current_time = time.monotonic()
        points_to_draw = []

        # --- Efficiently get and process data ---
        with LIDAR_LOCK:
            for angle_deg in fov_angles_to_check:
                data_point = LIDAR_DATA.get(angle_deg)

                # Check if point exists and is not stale
                if data_point and (current_time - data_point['timestamp']) < Config.STALE_THRESHOLD_SEC:
                    dist = data_point['distance']
                    
                    # Map angle to screen X coordinate
                    # Normalize angle diff from -half_fov to +half_fov
                    angular_diff = (angle_deg - Config.LIDAR_CENTER_ANGLE_DEG + 180) % 360 - 180
                    x_screen = int(
                        ((angular_diff + Config.FOV_DEGREES / 2) / Config.FOV_DEGREES) * Config.IMAGE_WIDTH
                    )

                    # Calculate Y with parallax correction
                    y_screen = int(Config.LIDAR_Y_SLOPE * dist + Config.LIDAR_Y_INTERCEPT)
                    
                    points_to_draw.append(((x_screen, y_screen), dist))

        # --- Drawing Loop (Lock is released) ---
        for (x, y), dist in points_to_draw:
            if 0 <= x < Config.IMAGE_WIDTH:
                # Clip Y to stay within frame
                y_clipped = np.clip(y, 0, Config.IMAGE_HEIGHT - 1)
                
                # Color mapping based on distance
                norm = np.clip((dist - Config.MIN_DISTANCE_MM) / (Config.MAX_DISTANCE_MM - Config.MIN_DISTANCE_MM), 0, 1)
                color = (0, int(norm * 255), int((1 - norm) * 255)) # BGR: Green (close) to Red (far)

                cv2.circle(frame, (x, y_clipped), 3, color, -1)
        
        cv2.imshow("LIDAR + Camera View", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break

    print("Stopping camera...")
    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Start the lidar thread as a daemon so it exits with the main thread
    lidar_thread = threading.Thread(target=lidar_loop, daemon=True)
    lidar_thread.start()

    # Give the lidar a moment to connect and gather initial data
    print("Waiting for initial LIDAR scan...")
    time.sleep(2)
    
    main_loop()