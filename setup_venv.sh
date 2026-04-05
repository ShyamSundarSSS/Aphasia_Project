#!/bin/bash

echo "============================================================"
echo "Creating Virtual Environment for Aphasia Detection System"
echo "============================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ ! -f "venv/bin/activate" ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo
echo "Virtual environment created successfully!"
echo
echo "Activating virtual environment..."
source venv/bin/activate

echo
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo
echo "To deactivate, run:"
echo "  deactivate"
echo
