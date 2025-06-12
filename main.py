import cv2
import numpy as np
import threading
from myrplidar import MyRPLidar
import time
import queue

# Globals
cursor_pos = (334, 261)
fov_angle = 20  # Starting FOV in degrees
image_size = (640, 480)
min_dist = 10
max_dist=2000

def get_filtered_lidar_points(lidar_points, center_angle_deg, fov, image_width):
    result = []
    half_fov = fov / 2
    min_angle = center_angle_deg - half_fov
    max_angle = center_angle_deg + half_fov

    for _, angle, distance in lidar_points:
        relative_angle = (angle - min_angle) % 360

        if 0 <= relative_angle <= fov:
            x_screen = int((relative_angle / fov) * image_width)
            result.append((x_screen, distance))
    return result

def draw_lidar_overlay(frame, lidar_points, fov_angle, center_angle_deg):
    width, height = frame.shape[1], frame.shape[0]
    points = get_filtered_lidar_points(lidar_points, center_angle_deg, fov_angle, width)

    for x, dist in points:
        if 0 <= x < width:
            norm = np.clip((dist - min_dist) / (max_dist - min_dist), 0, 1)
            red = int((1 - norm) * 255)
            green = int(norm * 255)
            color = (0, green, red)  # BGR
            y = cursor_pos[1]
            cv2.circle(frame, (x, y), 2, color, -1)

def camera_loop(shared_lidar_data: queue.Queue, center_angle_deg):
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

            lidar_points_copy = list(shared_lidar_data.queue)
            draw_lidar_overlay(frame, lidar_points_copy, fov_angle, center_angle_deg)

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


def lidar_loop(shared_data: queue.Queue, center_angle_deg, fov_angle):
    print("Starting LIDAR thread...")
    PORT = '/dev/ttyUSB0'
    lidar = None
    half_fov = fov_angle / 2
    min_angle = center_angle_deg - half_fov
    max_angle = center_angle_deg + half_fov

    def is_in_fov(angle):
        norm_angle = angle % 360
        relative_angle = (norm_angle - min_angle) % 360
        return 0 <= relative_angle <= fov_angle

    try:
        lidar = MyRPLidar(PORT)
        print("LIDAR connected and motor started.")

        for scan in lidar:
            for point in scan:
                quality, angle, distance = point
                if distance < min_dist or distance > max_dist or not is_in_fov(angle) or quality < 0:
                    continue
                try:
                    shared_data.put_nowait(point)
                except queue.Full:
                    try:
                        shared_data.get_nowait()
                        shared_data.put_nowait(point)
                    except queue.Empty:
                        pass

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


if __name__ == '__main__':
    shared_lidar_data = queue.Queue(maxsize=17*10)

    # Fixed center angle from calibrated cursor
    x_ratio = (cursor_pos[0] / image_size[0]) - 0.5
    center_angle = (-x_ratio * fov_angle * 2) % 360

    lidar_thread = threading.Thread(target=lidar_loop, args=(shared_lidar_data, center_angle, fov_angle), daemon=True)
    lidar_thread.start()

    time.sleep(2)
    camera_loop(shared_lidar_data, center_angle)
