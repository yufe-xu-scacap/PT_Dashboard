import streamlit as st
import pygame
import pygetwindow as gw
import time
from datetime import datetime

# Imports for OCR alerts
import pyautogui
import pytesseract
from PIL import Image
import win32gui
import re  # Import regular expressions for number parsing

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Main Dashboard",
    page_icon="📈💰📊",
    layout="wide"
)

# --- IMPORTANT: TESSERACT CONFIGURATION ---
# The user MUST ensure this path is correct for their system.
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    st.error(f"Tesseract not configured: {e}. OCR-based alerts will not work.")
    st.info("Please install Tesseract-OCR and ensure the path above is correct.")

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
# General
CHECK_INTERVAL = 5  # Seconds

# 1. High Touch Alert
HIGH_TOUCH_SOUND_FILE = 'Sounds/notify.wav'
HIGH_TOUCH_SEARCH_TERMS = ["Quote Request"]

# 2. HALTER Alert (Previously OCR Error Alert)
HALTER_SOUND_FILE = 'Sounds/error.wav'
HALTER_WINDOW_TITLE = "Scalable_Quoting_Version_1.0"
HALTER_SEARCH_TERMS = ['halterr', 'unkn', 'malterr', 'malter', 'halter', 'unknown']

# 3. Gross Exposure Alert (NEW)
GROSS_EXP_WINDOW_TITLE = "Scalable_Hedging_OCR"  # Partial title of the target window
GROSS_EXP_SOUND_FILES = {
    6000000: 'Sounds/Alert_6M.mp3',  # Sound for > 6 Million
    8000000: 'Sounds/Alert_8M.mp3',  # Sound for > 8 Million
    10000000: 'Sounds/Alert_10M.mp3'  # Sound for > 10 Million
}
GROSS_EXP_COOLDOWN = 120  # Seconds between alerts to prevent spam


# --- HELPER FUNCTIONS ---

@st.cache_resource
def initialize_sounds(file_paths):
    """Initializes pygame and loads one or more sound files."""
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    sounds = {}
    if isinstance(file_paths, dict):  # For Gross Exposure thresholds
        for key, path in file_paths.items():
            try:
                sounds[key] = pygame.mixer.Sound(path)
            except Exception as e:
                st.error(f"Failed to load sound for threshold {key} at '{path}': {e}")
                sounds[key] = None
        return sounds
    else:  # For single sound files
        try:
            return pygame.mixer.Sound(file_paths)
        except Exception as e:
            return str(e)


# OCR-related functions
def find_window_by_title(title):
    """Finds an exact window title."""
    try:
        window = gw.getWindowsWithTitle(title)[0]
        return (window.left, window.top, window.width, window.height) if window else None
    except IndexError:
        return None


def capture_screenshot(rect):
    """Captures a screenshot of a specific region."""
    return pyautogui.screenshot(region=rect)


def extract_text_from_image(image):
    """Extracts text from an image using Tesseract OCR."""
    config = '--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=config).lower()


def parse_gross_exposure_value(text):
    """Parses the specific numeric value from the OCR text using regex."""
    # Looks for "EUR", followed by space(s), then digits and spaces, ending with "ADS"
    match = re.search(r"eur\s([\d\s]+)ads", text)
    if match:
        value_string = match.group(1).replace(" ", "")
        return int(value_string)
    return None


# --- INITIALIZE SOUND OBJECTS ---
high_touch_sound = initialize_sounds(HIGH_TOUCH_SOUND_FILE)
halter_sound = initialize_sounds(HALTER_SOUND_FILE)
gross_exposure_sounds = initialize_sounds(GROSS_EXP_SOUND_FILES)

# --- INITIALIZE SESSION STATE ---
if 'high_touch_log' not in st.session_state: st.session_state.high_touch_log = ["Ready."]
if 'halter_log' not in st.session_state: st.session_state.halter_log = ["Ready."]
if 'gross_exp_log' not in st.session_state: st.session_state.gross_exp_log = ["Ready."]
if 'gross_exp_last_alert' not in st.session_state: st.session_state.gross_exp_last_alert = 0

# --- MAIN PAGE LAYOUT ---
st.title("PT Dashboard")
st.sidebar.success("Select a page above.")
st.info("Powered by Hedge Fund Technology!")

# --- 1. HIGH TOUCH ALERT UI ---
# --- 1. HIGH TOUCH ALERT UI (CORRECTED) ---
st.divider()
st.header("🔊 Sound Alert")
if isinstance(high_touch_sound, str):
    st.error(f"Failed to load sound file: {high_touch_sound}")
else:
    col1, col2, col3 = st.columns([2, 3, 6])
    with col1:
        run_high_touch = st.toggle("Start High Touch Monitoring", key="ht_toggle")
    with col2:
        # --- FIX STARTS HERE ---
        if run_high_touch:
            st.success("Status: Enabled")
        else:
            st.error("Status: Disabled")
        # --- FIX ENDS HERE ---
    with col3:
        with st.expander(f"Log: {st.session_state.high_touch_log[-1]}", expanded=False):
            log_container = st.container(height=200)
            for msg in reversed(st.session_state.high_touch_log): log_container.text(msg)

# --- 2. HALTER ALERT UI (CORRECTED) ---
st.divider()
if isinstance(halter_sound, str):
    st.error(f"Failed to load sound file: {halter_sound}")
else:
    col4, col5, col6 = st.columns([2, 3, 6])
    with col4:
        run_halter = st.toggle("Start HALTER Monitoring", key="halter_toggle")
    with col5:
        # --- FIX STARTS HERE ---
        if run_halter:
            st.success("Status: Enabled")
        else:
            st.error("Status: Disabled")
        # --- FIX ENDS HERE ---
    with col6:
        with st.expander(f"Log: {st.session_state.halter_log[-1]}", expanded=False):
            log_container = st.container(height=200)
            for msg in reversed(st.session_state.halter_log): log_container.text(msg)

# --- 3. GROSS EXPOSURE ALERT UI (CORRECTED) ---
st.divider()
if not all(gross_exposure_sounds.values()):
    st.error("One or more Gross Exposure sound files failed to load. Check paths.")
else:
    col7, col8, col9 = st.columns([2, 3, 6])
    with col7:
        run_gross_exp = st.toggle("Start Gross Exposure Monitoring", key="gross_exp_toggle")
    with col8:
        # --- FIX STARTS HERE ---
        if run_gross_exp:
            st.success("Status: Enabled")
        else:
            st.error("Status: Disabled")
        # --- FIX ENDS HERE ---
    with col9:
        with st.expander(f"Log: {st.session_state.gross_exp_log[-1]}", expanded=False):
            log_container = st.container(height=200)
            for msg in reversed(st.session_state.gross_exp_log): log_container.text(msg)

# --- MONITORING LOGIC ---
now = time.time()
timestamp = time.strftime('%H:%M:%S')

# 1. High Touch Logic
if run_high_touch and not isinstance(high_touch_sound, str):
    found = any(term in title for term in HIGH_TOUCH_SEARCH_TERMS for title in gw.getAllTitles())
    if found:
        log_msg = f"[{timestamp}] Found Quote Request -> Sound PLAYED."
        if st.session_state.high_touch_log[-1] != log_msg:
            high_touch_sound.play()
            st.session_state.high_touch_log.append(log_msg)
    else:
        log_msg = f"[{timestamp}] Check complete. No match."
        if st.session_state.high_touch_log[-1] != log_msg:
            st.session_state.high_touch_log.append(log_msg)

# 2. HALTER Logic
if run_halter and not isinstance(halter_sound, str):
    rect = find_window_by_title(HALTER_WINDOW_TITLE)
    if not rect:
        log_msg = f"[{timestamp}] Window '{HALTER_WINDOW_TITLE}' not found."
    else:
        text = extract_text_from_image(capture_screenshot(rect))
        if any(term in text for term in HALTER_SEARCH_TERMS):
            log_msg = f"[{timestamp}] HALTER error detected -> Sound PLAYED."
            if st.session_state.halter_log[-1] != log_msg:
                halter_sound.play()
                st.session_state.halter_log.append(log_msg)
        else:
            log_msg = f"[{timestamp}] HALTER check complete. No errors."
    if st.session_state.halter_log[-1] != log_msg:
        st.session_state.halter_log.append(log_msg)

# 3. Gross Exposure Logic
if run_gross_exp and all(gross_exposure_sounds.values()):
    rect = find_window_by_title(GROSS_EXP_WINDOW_TITLE)
    if not rect:
        log_msg = f"[{timestamp}] Window '{GROSS_EXP_WINDOW_TITLE}' not found."
    else:
        text = extract_text_from_image(capture_screenshot(rect))
        value = parse_gross_exposure_value(text)
        if value is not None:
            log_msg = f"[{timestamp}] Gross Exposure Found: {value: ,}"
            # Check thresholds from highest to lowest
            for threshold in sorted(gross_exposure_sounds.keys(), reverse=True):
                if value > threshold:
                    if (now - st.session_state.gross_exp_last_alert) > GROSS_EXP_COOLDOWN:
                        sound_to_play = gross_exposure_sounds[threshold]
                        sound_to_play.play()
                        log_msg += f" -> ALERT! Exceeds {threshold:,}. Sound PLAYED."
                        st.session_state.gross_exp_last_alert = now
                    else:
                        log_msg += f" -> Exceeds {threshold:,}. (Cooldown active)."
                    break  # Only trigger highest applicable alert
        else:
            log_msg = f"[{timestamp}] Gross Exposure check complete. Value not found."
    if st.session_state.gross_exp_log[-1] != log_msg:
        st.session_state.gross_exp_log.append(log_msg)

# --- MASTER RERUN CONTROLLER ---
if run_high_touch or run_halter or run_gross_exp:
    time.sleep(CHECK_INTERVAL)
    st.rerun()