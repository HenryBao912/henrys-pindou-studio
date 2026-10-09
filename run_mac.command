#!/bin/bash
# Henry's Pindou Studio launcher for macOS. Double-click to start (first time: right-click > Open).
cd "$(dirname "$0")"
PY=python3
if ! $PY -c "import tkinter" 2>/dev/null; then
  echo "Tkinter is missing. Install Python from https://www.python.org/downloads/ (or: brew install python-tk)"
  read -n 1 -s -r -p "Press any key to close"
  exit 1
fi
if ! $PY -c "import PIL, numpy, reportlab" 2>/dev/null; then
  echo "Installing Pillow, numpy and reportlab (one time only)..."
  $PY -m pip install --user pillow numpy reportlab || $PY -m pip install --user --break-system-packages pillow numpy reportlab
fi
exec $PY henrys_pindou_studio.py "$@"
