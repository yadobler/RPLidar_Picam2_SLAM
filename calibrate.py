import cv2
import numpy as np
from picamera2 import Picamera2
import threading
from myrplidar import MyRPLidar
from collections import deque
import time

# Globals
cursor_pos = (320, 240)  # default center
fov_angle = 30           # degrees
image_size = (640, 480)

def mouse_callback(event, x, y, flags, param):
    global cursor_pos, fov_angle
    cursor_pos = (x, y)
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            fov_angle = min(180, fov_angle + 5)
        else:
            fov_angle = max(5, fov_angle - 5)

def get_filtered_lidar_points(lidar_points, center_angle_deg, fov):
    """Return lidar points (x, y) within a given angular field."""
    result = []
    for _, angle, distance in lidar_points:
        if distance == 0:
            continue
        rel_angle = angle - center_angle_deg
        if -fov/2 <= rel_angle <= fov/2:
            rad = np.radians(angle)
            x = int(distance * np.cos(rad) / 10)  # scale down
            y = int(distance * np.sin(rad) / 10)
            result.append((x, y))
    return result

def draw_lidar_overlay(frame, lidar_points, center):
    """Draw filtered lidar points as dots centered at mouse cursor."""
    global fov_angle
    angle_est = 0  # Optionally estimate based on cursor_pos
    filtered = get_filtered_lidar_points(lidar_points, angle_est, fov_angle)

    for dx, dy in filtered:
        px = center[0] + dx
        py = center[1] - dy
        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

    cv2.putText(frame, f"FOV: {fov_angle} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def camera_loop(shared_lidar_data):
    print("Camera loop starting...")

    try:
        picam2 = Picamera2()
        print("Picamera2 object created.")
        picam2.configure(picam2.create_preview_configuration(main={"size": image_size}))
        print("Camera configured.")
        picam2.start()
        print("Camera started.")
    except Exception as e:
        print("Camera start failed:", e)
        return

    cv2.namedWindow("LIDAR + Camera View")
    cv2.setMouseCallback("LIDAR + Camera View", mouse_callback)
    print("Camera window opened.")

    while True:
        try:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if shared_lidar_data:
                draw_lidar_overlay(frame, shared_lidar_data, cursor_pos)

            cv2.imshow("LIDAR + Camera View", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC to quit
                print("ESC pressed. Exiting camera loop.")
                break
        except Exception as e:
            print("Error in camera loop:", e)
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera loop exited.")

def lidar_loop(shared_data):
    print("Starting lidar thread...")
    PORT = '/dev/ttyUSB0'

    for attempt in range(3):
        print(f"Attempting to connect to RPLidar on port {PORT} (Attempt {attempt+1}/3)...")
        try:
            lidar = MyRPLidar(PORT)
            print("Lidar connected successfully! Info:", lidar.get_info())
            print("Lidar Health:", lidar.get_health())
            break
        except Exception as e:
            print("RPLidarException during connection attempt:", e)
            print("Stopping lidar motor and disconnecting...")
            try:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
            except:
                pass
            print("Lidar stopped and disconnected.")
            time.sleep(2)
    else:
        print("Failed to connect to RPLidar after 3 attempts.")
        return

    print("LIDAR loop started")

    try:
        for scan in lidar:
            print(f"Got scan with {len(scan)} points")
            shared_data.clear()
            shared_data.extend(scan)
    except Exception as e:
        print("LIDAR error:", e)
    finally:
        print("Shutting down LIDAR...")
        try:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
        except:
            pass
        print("LIDAR cleanup complete.")

# Shared data container
shared_lidar_data = deque(maxlen=720)

# Start threads
threading.Thread(target=lidar_loop, args=(shared_lidar_data,), daemon=True).start()
camera_loop(shared_lidar_data)
