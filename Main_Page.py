import streamlit as st
import pandas as pd
import numpy as np
import pygame
import pygetwindow as gw
import time
from datetime import datetime

import pyautogui
import pytesseract
from PIL import Image
import win32gui

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Main Dashboard",
    page_icon="📈💰📊",
    layout="wide"
)

# --- IMPORTANT: TESSERACT CONFIGURATION ---
# The user MUST ensure this path is correct for their system.
# This is the default installation path on Windows.
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    st.error(f"Tesseract not configured: {e}. The OCR alert will not work.")
    st.info("Please install Tesseract-OCR from the official website and ensure the path above is correct.")

# --- CSS STYLING ---
st.markdown(
    """
    <style>
    /* Style for the sidebar navigation links */
    [data-testid="stSidebarNav"] a {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SCRIPT CONSTANTS ---
# Constants for the first alert (High Touch)
HIGH_TOUCH_SOUND_FILE = 'Sounds/notify.wav'
HIGH_TOUCH_SEARCH_TERMS = ["Quote Request"]

# Constants for the second alert (OCR)
OCR_SOUND_FILE = 'Sounds/error.wav'  # Using a different sound for the error alert
OCR_WINDOW_TITLE = "Scalable_Quoting_Version_1.0"
OCR_SEARCH_TERMS = ['halterr', 'unkn', 'malterr', 'malter', 'halter', 'unknown']

# General constant
CHECK_INTERVAL = 5  # Seconds


# --- HELPER FUNCTIONS ---

# Function to initialize sound system (cached)
@st.cache_resource
def initialize_sound(file_path):
    """Initializes the pygame mixer and loads a sound file."""
    try:
        # Initialize mixer only once
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        sound = pygame.mixer.Sound(file_path)
        return sound
    except Exception as e:
        return str(e)


# Functions for the OCR alert
def get_window_rect(window_title):
    """Finds a window by title and returns its rectangle coordinates."""
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        if window:
            return (window.left, window.top, window.width, window.height)
    except IndexError:
        return None


def capture_screenshot(rect):
    """Captures a screenshot of a specific region."""
    return pyautogui.screenshot(region=rect)


def extract_text_from_image(image):
    """Extracts text from an image using Tesseract OCR."""
    config = '--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=config).lower()


# --- INITIALIZE SOUND OBJECTS ---
high_touch_sound = initialize_sound(HIGH_TOUCH_SOUND_FILE)
ocr_alert_sound = initialize_sound(OCR_SOUND_FILE)

# --- INITIALIZE SESSION STATE ---
if 'high_touch_log' not in st.session_state:
    st.session_state.high_touch_log = ["Welcome! The High Touch alert is ready."]
if 'ocr_log' not in st.session_state:
    st.session_state.ocr_log = ["Welcome! The OCR Error alert is ready."]

# --- MAIN PAGE LAYOUT ---
st.title("PT Dashboard")
st.sidebar.success("Select a page above.")
st.info("Powered by Hedge Fund Technology!")

st.divider()

# --- HIGH TOUCH ALERT UI ---
st.header("🔊 High Touch Alert")

# Check if the sound file loaded correctly
if isinstance(high_touch_sound, str):
    st.error(f"Failed to load High Touch sound file: {high_touch_sound}")
    st.warning(f"Please ensure the path is correct. Current Path: {HIGH_TOUCH_SOUND_FILE}")
else:
    col1, col2, col3 = st.columns([2, 3, 6])
    with col1:
        run_high_touch_monitoring = st.toggle("Start High Touch Monitoring", value=False, key="high_touch_toggle")
    with col2:
        st.success(
            "Status: High Touch Monitoring Enabled" if run_high_touch_monitoring else "Status: High Touch Monitoring Disabled")
    with col3:
        if st.session_state.high_touch_log:
            with st.expander(f"Log: {st.session_state.high_touch_log[-1]}", expanded=False):
                log_container = st.container(height=200, border=False)
                for message in reversed(st.session_state.high_touch_log):
                    log_container.text(message)

st.divider()

# --- OCR ALERT UI (NEW SECTION) ---
st.header("🖼️ OCR Error Alert")

# Check if the sound file loaded correctly
if isinstance(ocr_alert_sound, str):
    st.error(f"Failed to load OCR Error sound file: {ocr_alert_sound}")
    st.warning(f"Please ensure the path is correct. Current Path: {OCR_SOUND_FILE}")
else:
    col4, col5, col6 = st.columns([2, 3, 6])
    with col4:
        run_ocr_monitoring = st.toggle("Start OCR Error Monitoring", value=False, key="ocr_toggle")
    with col5:
        st.success("Status: OCR Monitoring Enabled" if run_ocr_monitoring else "Status: OCR Monitoring Disabled")
    with col6:
        if st.session_state.ocr_log:
            with st.expander(f"Log: {st.session_state.ocr_log[-1]}", expanded=False):
                log_container_ocr = st.container(height=200, border=False)
                for message in reversed(st.session_state.ocr_log):
                    log_container_ocr.text(message)

# --- MONITORING LOGIC ---

# 1. High Touch Monitoring Logic
if run_high_touch_monitoring and not isinstance(high_touch_sound, str):
    windows = gw.getAllTitles()
    found_match = False
    for title in windows:
        if any(term in title for term in HIGH_TOUCH_SEARCH_TERMS):
            log_message = f"[{time.strftime('%H:%M:%S')}] Found: '{title}' -> Sound PLAYED."
            high_touch_sound.play()
            if st.session_state.high_touch_log[-1] != log_message:
                st.session_state.high_touch_log.append(log_message)
            found_match = True
            break
    if not found_match:
        log_message = f"[{time.strftime('%H:%M:%S')}] Check complete. No match found."
        if st.session_state.high_touch_log[-1] != log_message:
            st.session_state.high_touch_log.append(log_message)

# 2. OCR Monitoring Logic
if run_ocr_monitoring and not isinstance(ocr_alert_sound, str):
    rect = get_window_rect(OCR_WINDOW_TITLE)
    if not rect:
        log_message = f"[{time.strftime('%H:%M:%S')}] Window '{OCR_WINDOW_TITLE}' not found."
        if st.session_state.ocr_log[-1] != log_message:
            st.session_state.ocr_log.append(log_message)
    else:
        screenshot = capture_screenshot(rect)
        text = extract_text_from_image(screenshot)

        if any(term in text for term in OCR_SEARCH_TERMS):
            log_message = f"[{time.strftime('%H:%M:%S')}] ERROR DETECTED! -> Sound PLAYED."
            ocr_alert_sound.play()
            if st.session_state.ocr_log[-1] != log_message:
                st.session_state.ocr_log.append(log_message)
                st.session_state.ocr_log.append(f"   -> Text found: '{text[:100]}...'")  # Log a snippet of the text
        else:
            log_message = f"[{time.strftime('%H:%M:%S')}] OCR check complete. No errors found."
            if st.session_state.ocr_log[-1] != log_message:
                st.session_state.ocr_log.append(log_message)

# --- MASTER RERUN CONTROLLER ---
# If any monitoring is active, wait and rerun the app to create a loop.
if run_high_touch_monitoring or run_ocr_monitoring:
    time.sleep(CHECK_INTERVAL)
    st.rerun()