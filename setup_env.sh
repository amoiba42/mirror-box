#!/bin/bash

# --- CRITICAL CHANGE: Use System Python (3.11.2) ---
# This matches the binary version of picamera2 installed by apt.
PYTHON_BIN="/usr/bin/python3"
VENV_NAME="venv"

# 1. Create the Virtual Environment
echo "--- Creating Virtual Environment ($VENV_NAME) ---"
# We use --system-site-packages so it can find 'picamera2' natively
$PYTHON_BIN -m venv $VENV_NAME --system-site-packages

# 2. Install Software Packages
echo "--- Installing Packages from requirements.txt ---"
$VENV_NAME/bin/pip install --upgrade pip

# We install our specific libraries.
# Note: Because we used system-site-packages, pip will see the system Numpy.
# Our requirements.txt will upgrade it if needed, but keep it < 2.0.
$VENV_NAME/bin/pip install -r requirements.txt

echo "=========================================="
echo "Setup Complete!"
echo "To run your code, use: ./$VENV_NAME/bin/python3 run_trial.py"
echo "=========================================="