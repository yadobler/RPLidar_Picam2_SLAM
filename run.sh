#! /bin/bash
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0
source .venv/bin/activate
python -u vo.py | tee log 
read -p "Press Enter to continue"