from myrplidar import MyRPLidar
import cv2 # Import OpenCV
import numpy as np
import sys
from math import floor, radians, cos, sin 


# --- Configuration for Lidar and Plotting ---
PORT_NAME = '/dev/ttyUSB0'      # !!! IMPORTANT: VERIFY THIS IS YOUR ACTUAL RPLIDAR PORT !!!
MAX_DISTANCE_MM = 3000          # Max expected distance for plotting (e.g., 3000mm = 3 meters)
MIN_DISTANCE_MM = 1             # Minimum distance to plot (filter out noise very close to lidar)
MIN_QUALITY = 2                 # Minimum quality to plot 
NUM_SCANS_TO_ACCUMULATE = 10    # Number of full scans to accumulate before a single plot update

# --- OpenCV Plotting Parameters ---
WINDOW_NAME = "RPLidar Scan (OpenCV)"
IMG_SIZE = 600 
CENTER_X = IMG_SIZE // 2
CENTER_Y = IMG_SIZE // 2
# Scale factor to map lidar distances (mm) to pixels
# Example: 1 pixel per 5 mm, or 100 pixels per meter.
# MAX_DISTANCE_MM / (IMG_SIZE / 2 - some_margin)
PIXELS_PER_MM = (IMG_SIZE / 2 - 20) / MAX_DISTANCE_MM # leave a 20 pixel margin


# --- Helper function to convert polar to cartesian and scale for image ---
def polar_to_pixel(angle_deg, distance_mm, center_x, center_y, pixels_per_mm):
    """
    Converts polar coordinates (angle, distance) to Cartesian (x, y) pixels
    relative to a given center, suitable for OpenCV image drawing.
    
    Args:
        angle_deg (float): Angle in degrees (0-360).
        distance_mm (float): Distance in millimeters.
        center_x (int): X-coordinate of the image center.
        center_y (int): Y-coordinate of the image center.
        pixels_per_mm (float): Scaling factor from millimeters to pixels.
        
    Returns:
        tuple: (x_pixel, y_pixel)
    """
    angle_rad = radians(angle_deg - 180)
    x = int(center_x + (distance_mm * pixels_per_mm) * sin(angle_rad))
    y = int(center_y - (distance_mm * pixels_per_mm) * cos(angle_rad)) # Subtract for inverted Y-axis
    return x, y


# --- Main Execution ---
if __name__ == '__main__':
    my_lidar = None
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    try:
        my_lidar = MyRPLidar(PORT_NAME)
        
        if not my_lidar.is_connected():
            print("Failed to initialize MyRPLidar object. Exiting script.")
            sys.exit(1)

        print(f"Starting real-time plotting with OpenCV. Plot updates after every {NUM_SCANS_TO_ACCUMULATE} scans.")
        print("Press 'q' to quit the display window.")
        
        accumulated_measurements = {}
        current_scan_count = 0
        # Main loop for real-time updates
        for scan in my_lidar:
            # Check for a key press (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break # Exit the loop if 'q' is pressed

            try:
                # Aggregate data from the current scan into accumulated_measurements
                for q, a, d in scan:
                    if d >= MIN_DISTANCE_MM and d <= MAX_DISTANCE_MM and q >= MIN_QUALITY:
                        accumulated_measurements[floor(a)] = d # Store distance only
                
                current_scan_count += 1
                
            except StopIteration:
                print("\nMyRPLidar iterator indicated end of data or permanent failure. Stopping display.")
                break # Break out of the main while loop
            except Exception as e:
                print(f"\nAn unexpected ERROR during scan or plotting: {e}")
                time.sleep(1) # Small delay to prevent rapid error looping
            
            # Only update the plot (OpenCV image) after collecting enough scans
            if current_scan_count >= NUM_SCANS_TO_ACCUMULATE:
                # Create a blank black image
                image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) 
                
                # Draw a grid for context (optional)
                cv2.circle(image, (CENTER_X, CENTER_Y), int(MAX_DISTANCE_MM * PIXELS_PER_MM), (50, 50, 50), 1) # Max range circle
                cv2.circle(image, (CENTER_X, CENTER_Y), int((MAX_DISTANCE_MM/2) * PIXELS_PER_MM), (30, 30, 30), 1) # Half range circle
                cv2.line(image, (CENTER_X, 0), (CENTER_X, IMG_SIZE), (70, 70, 70), 1) # North/South line
                cv2.line(image, (0, CENTER_Y), (IMG_SIZE, CENTER_Y), (70, 70, 70), 1) # East/West line
                
                # Draw points from accumulated_measurements
                if accumulated_measurements:
                    for angle_deg, distance_mm in accumulated_measurements.items():
                        x_pixel, y_pixel = polar_to_pixel(angle_deg, distance_mm, CENTER_X, CENTER_Y, PIXELS_PER_MM)
                        # Draw a small circle for each point (color is BGR, so blue for all)
                        cv2.circle(image, (x_pixel, y_pixel), 2, (255, 0, 0), -1) # Blue color, filled circle

                # Display the image
                cv2.imshow(WINDOW_NAME, image)

                # Reset accumulation for the next plot frame
                accumulated_measurements.clear()
                current_scan_count = 0

    except Exception as e:
        print(f"\nAn error occurred in the main script execution: {e}")
    finally:
        if my_lidar:
            del my_lidar 
        cv2.destroyAllWindows() # Close all OpenCV windows
        print("Script finished.")