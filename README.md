# RPLiDar + Picamera2 

Small little research for feasibility of collating data from a simple lidar and rgb camera setup, together with my professor.

# Base Materials 

Item|Description|Link
----|-----------|:-:
Raspberry Pi 4|RP4 8GB with 5V@3A power consumption|https://sg.shp.ee/5MKLmDw
mini HDMI Cable|For viewing and setting up RP|"
Cooling Case|To prevent overheating when collecting Lidar data|"
SD Card|32GB SD card - to store OS and code|"
Power Supply|USB-C wired power supply for RP|"
Camera|Pi Camera v2 for RGB image|https://sg.shp.ee/UGsMYSS
Lidar|Slamtec Lidar A1 - for lidar data|https://sg.shp.ee/SkGuan2
USB-B mini cable|To connect Lidar to RPi|-
SD Card Reader|To write OS onto SD card|https://sg.shp.ee/FzrgCZ4

# Getting Started

This project does not use ROS, but raw python to better understand the processes of the hardware.

## Running software

Do this on the Raspberry Pi:

```bash
git clone https://github.com/yadobler/RPLidar_Picam2_SLAM
cd RPLidar_Picam2_SLAM
./first_time_install.sh
```

**Remember to source the .venv/bin/activate file for each new terminal instance!**

## File contents:

```
.
├── COLMAP                  - Folder containing colmap and SfM scripts
├── LIDAR + Camera.png      - Calibration Test 1 
├── LIDAR + Camera_2.png    - Calibration Test 2
├── RPLidar Scan.png        - Example output of running myrplidar.py 
├── fov_measurement.jpg     - Image to get FOV of camera: 
|                             half FOV = sinh(16/3) = 10.5deg, since ruler = 16cm, finger segment = 3cm
├── README.md               - This file
├── LICENSE                 - MIT License
├── first_time_install.sh   - Helper to install packages needed
├── requirements.txt        - Requirement list for python (used by first_time_install.sh)
├── main.py                 - The current script I am working on
├── myrplidar.py            - Import this to have robust LiDAR usage, run to test LiDAR
├── calibrate.py            - Attempt at calibrating the picamera2 image with the LiDAR data
└── run.sh                  - Script to run the python files on RPi without keyboard (by double clicking script with mouse)

```

# Thanks
- My Professor (I'll insert his name once I get his permission here)
- Roboticia for the rplidar python package
- ChatGPT / Gemini for rubberduck debugging