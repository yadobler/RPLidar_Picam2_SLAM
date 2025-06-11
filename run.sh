#! /bin/bash
export QT_QPA_PLATFORM=xcb
source .venv/bin/activate
python picam.py # | tee log 2>&1
read -p "Press Enter to continue"