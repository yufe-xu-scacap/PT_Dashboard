#!/bin/bash

echo "============================================="
echo " Setting up Streamlit Environment..."
echo "============================================="

# Check if the virtual environment folder exists, if not, create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment and install dependencies
echo "Activating virtual environment and installing packages..."
source venv/bin/activate
pip install -r requirements.txt

# Run the Streamlit app
echo "============================================="
echo " Starting the Streamlit App..."
echo "============================================="
streamlit run Main_Page.py