#! /bin/bash
export QT_QPA_PLATFORM=xcb
.venv/bin/python lidar_preview.py # | tee log 2>&1
read -p "Press Enter to continue"