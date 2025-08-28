import streamlit as st
import pygame
import pygetwindow as gw
import time
from datetime import datetime
import re
import json
# Imports for OCR alerts
import pyautogui
import pytesseract
from PIL import Image

# Imports for Click Automation
import win32gui
import win32api
from pynput import mouse

@st.cache_data
def load_config(filepath="config.json"):
    """Loads the configuration from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found at '{filepath}'. Please create it.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding the JSON from '{filepath}'. Please check its format.")
        return None

# Load the configuration
config = load_config()
if not config:
    st.stop() # Stops the script if the config file is missing or invalid


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Main Dashboard",
    page_icon="📈💰📊",
    layout="wide"
)

# --- IMPORTANT: TESSERACT CONFIGURATION ---
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
CHECK_INTERVAL = config['general']['check_interval']

# 1. High Touch Alert
HIGH_TOUCH_SOUND_FILE = 'Sounds/notify.wav'
HIGH_TOUCH_SEARCH_TERMS = ["Quote Request"]
HIGH_TOUCH_COOLDOWN = config['cooldown']['high_touch_cooldown']

# 2. HALTER Alert
HALTER_SOUND_FILE = 'Sounds/error.wav'
HALTER_WINDOW_TITLE = config['windows']['halter_window_title']
HALTER_SEARCH_TERMS = ['halterr', 'unkn', 'malterr', 'malter', 'halter', 'unknown']
HALTER_COOLDOWN = config['cooldown']['halter_cooldown']

# 3. Gross Exposure Alert
GROSS_EXP_WINDOW_TITLE = config['cooldown']['gross_exp_window_title']
GROSS_EXP_SOUND_FILES = {
    6000000: 'Sounds/Alert_6M.mp3',
    8000000: 'Sounds/Alert_8M.mp3',
    10000000: 'Sounds/Alert_10M.mp3'
}
GROSS_EXP_COOLDOWN = config['windows']['gross_exp_cooldown']

# 4. Click Automation
CLICKER_TARGET_WINDOW_TITLE = config['windows']['click_target_window_title']
DELAY_BETWEEN_CLICKS = config['clicker']['delay_between_clicks']

# --- HELPER FUNCTIONS ---

@st.cache_resource
def initialize_sounds(file_paths):
    """Initializes pygame and loads one or more sound files."""
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    sounds = {}
    if isinstance(file_paths, dict):
        for key, path in file_paths.items():
            try:
                sounds[key] = pygame.mixer.Sound(path)
            except Exception as e:
                st.error(f"Failed to load sound for threshold {key} at '{path}': {e}")
                sounds[key] = None
        return sounds
    else:
        try:
            return pygame.mixer.Sound(file_paths)
        except Exception as e:
            return str(e)

# --- Alert Helper Functions ---
def find_window_by_title_substring(title_part):
    """Finds a window where the title contains the given substring."""
    try:
        window = gw.getWindowsWithTitle(title_part)[0]
        return (window.left, window.top, window.width, window.height) if window else None
    except IndexError:
        return None

def capture_screenshot(rect):
    return pyautogui.screenshot(region=rect)

def extract_text_from_image(image):
    config = '--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=config).lower()

def parse_gross_exposure_value(text):
    match = re.search(r"eur\s([\d\s]+)ads", text)
    if match:
        value_string = match.group(1).replace(" ", "")
        return int(value_string)
    return None

# --- Click Automation Helper Functions ---
mouse_controller = mouse.Controller()

def find_exact_window(title):
    """Finds the window handle for a given exact window title."""
    try:
        hwnd = win32gui.FindWindow(None, title)
        return hwnd if hwnd != 0 else None
    except Exception:
        return None

def perform_realistic_click(hwnd, x, y, click_type='left'):
    """Simulates a realistic click and logs actions to the Streamlit session state."""
    log_entry = ""
    try:
        original_pos = mouse_controller.position
        screen_coords = win32gui.ClientToScreen(hwnd, (x, y))
        mouse_controller.position = screen_coords
        time.sleep(0.05)
        button = mouse.Button.right if click_type.lower() == 'right' else mouse.Button.left
        mouse_controller.click(button, 1)
        mouse_controller.position = original_pos
        log_entry = f"Success: Performed {click_type} click at {screen_coords}."
        st.session_state.click_log.append(log_entry)
    except Exception as e:
        log_entry = f"Error: Failed to click at ({x}, {y}). Details: {e}"
        st.session_state.click_log.append(log_entry)
        if 'original_pos' in locals():
            mouse_controller.position = original_pos


# --- INITIALIZE SOUND OBJECTS ---
high_touch_sound = initialize_sounds(HIGH_TOUCH_SOUND_FILE)
halter_sound = initialize_sounds(HALTER_SOUND_FILE)
gross_exposure_sounds = initialize_sounds(GROSS_EXP_SOUND_FILES)

# --- INITIALIZE SESSION STATE ---
# For Alerts
if 'high_touch_log' not in st.session_state: st.session_state.high_touch_log = ["Ready."]
if 'halter_log' not in st.session_state: st.session_state.halter_log = ["Ready."]
if 'gross_exp_log' not in st.session_state: st.session_state.gross_exp_log = ["Ready."]
if 'high_touch_last_alert' not in st.session_state: st.session_state.high_touch_last_alert = 0
if 'halter_last_alert' not in st.session_state: st.session_state.halter_last_alert = 0
if 'gross_exp_last_alert' not in st.session_state: st.session_state.gross_exp_last_alert = 0
# For Click Automation
if 'click_points' not in st.session_state: st.session_state.click_points = []
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'capture_mode' not in st.session_state: st.session_state.capture_mode = False
if 'click_log' not in st.session_state: st.session_state.click_log = ["Ready."]


# --- MAIN PAGE LAYOUT ---
st.title("PT Dashboard")
st.sidebar.success("Select a page above.")
st.info("Powered by Hedge Fund Technology!")

# --- 1. HIGH TOUCH ALERT UI ---
st.divider()
st.header("🔊 Sound Alert")
if isinstance(high_touch_sound, str):
    st.error(f"Failed to load sound file: {high_touch_sound}")
else:
    col1, col2, col3 = st.columns([2, 3, 6])
    with col1:
        run_high_touch = st.toggle("Start High Touch Monitoring", key="ht_toggle")
    with col2:
        if run_high_touch: st.success("Status: Enabled")
        else: st.error("Status: Disabled")
    with col3:
        # <-- CHANGE: Static title for expander
        with st.expander("High Touch Log", expanded=False):
            st.info(f"Latest: {st.session_state.high_touch_log[-1]}")
            log_container = st.container(height=200)
            for msg in reversed(st.session_state.high_touch_log): log_container.text(msg)

# --- 2. HALTER ALERT UI ---
st.divider()
if isinstance(halter_sound, str):
    st.error(f"Failed to load sound file: {halter_sound}")
else:
    col4, col5, col6 = st.columns([2, 3, 6])
    with col4:
        run_halter = st.toggle("Start HALTER Monitoring", key="halter_toggle")
    with col5:
        if run_halter: st.success("Status: Enabled")
        else: st.error("Status: Disabled")
    with col6:
        # <-- CHANGE: Static title for expander
        with st.expander("HALTER Log", expanded=False):
            st.info(f"Latest: {st.session_state.halter_log[-1]}")
            log_container = st.container(height=200)
            for msg in reversed(st.session_state.halter_log): log_container.text(msg)

# --- 3. GROSS EXPOSURE ALERT UI ---
st.divider()
if not all(gross_exposure_sounds.values()):
    st.error("One or more Gross Exposure sound files failed to load. Check paths.")
else:
    col7, col8, col9 = st.columns([2, 3, 6])
    with col7:
        run_gross_exp = st.toggle("Start Gross Exposure Monitoring", key="gross_exp_toggle")
    with col8:
        if run_gross_exp: st.success("Status: Enabled")
        else: st.error("Status: Disabled")
    with col9:
        # <-- CHANGE: Static title for expander
        with st.expander("Gross Exposure Log", expanded=False):
            st.info(f"Latest: {st.session_state.gross_exp_log[-1]}")
            log_container = st.container(height=200)
            for msg in reversed(st.session_state.gross_exp_log): log_container.text(msg)

# --- 4. BACKGROUND CLICK SIMULATOR UI ---
st.divider()
toggle_col, interval_col, status_col, log_col = st.columns([2, 1, 2, 6])

with toggle_col:
    st.session_state.is_running = st.toggle(
        "Start / Stop Clicker",
        value=st.session_state.is_running,
        key='click_run_toggle'
    )

with status_col:
    if st.session_state.is_running:
        st.success("Status: Enabled")
    else:
        st.error("Status: Disabled")

with interval_col:
    click_interval = st.number_input(
        "Interval (sec)",
        min_value=1,
        value=20,
        step=1,
        help="The delay in seconds between each full sequence of clicks."
    )

with log_col:
    # <-- CHANGE: Static title for expander
    with st.expander("Clicker Log", expanded=False):
        st.info(f"Latest: {st.session_state.click_log[-1]}")
        log_container = st.container(height=200)
        for msg in reversed(st.session_state.click_log):
            log_container.text(msg)

# Placeholders for status text and progress bar, which will appear below the controls
status_placeholder = st.empty()
progress_placeholder = st.empty()


# --- MONITORING & AUTOMATION LOGIC ---
now = time.time()
timestamp = time.strftime('%H:%M:%S')

# 1. High Touch Logic
if run_high_touch and not isinstance(high_touch_sound, str):
    found = any(term in title for term in HIGH_TOUCH_SEARCH_TERMS for title in gw.getAllTitles())
    log_msg = ""
    if found:
        if (now - st.session_state.high_touch_last_alert) > HIGH_TOUCH_COOLDOWN:
            high_touch_sound.play()
            st.session_state.high_touch_last_alert = now
            log_msg = f"[{timestamp}] Found Quote Request -> Sound PLAYED."
        else:
            log_msg = f"[{timestamp}] Found Quote Request. (Cooldown active)."
    else:
        log_msg = f"[{timestamp}] Check complete. No match."
        st.session_state.high_touch_last_alert = 0 # Reset cooldown when not found

    if st.session_state.high_touch_log[-1] != log_msg:
        st.session_state.high_touch_log.append(log_msg)


# 2. HALTER Logic
if run_halter and not isinstance(halter_sound, str):
    rect = find_window_by_title_substring(HALTER_WINDOW_TITLE)
    log_msg = ""
    if not rect:
        log_msg = f"[{timestamp}] Window '{HALTER_WINDOW_TITLE}' not found."
        st.session_state.halter_last_alert = 0 # Reset cooldown
    else:
        text = extract_text_from_image(capture_screenshot(rect))
        if any(term in text for term in HALTER_SEARCH_TERMS):
            if (now - st.session_state.halter_last_alert) > HALTER_COOLDOWN:
                halter_sound.play()
                st.session_state.halter_last_alert = now
                log_msg = f"[{timestamp}] HALTER error detected -> Sound PLAYED."
            else:
                log_msg = f"[{timestamp}] HALTER error detected. (Cooldown active)."
        else:
            log_msg = f"[{timestamp}] HALTER check complete. No errors."
            st.session_state.halter_last_alert = 0 # Reset cooldown

    if st.session_state.halter_log[-1] != log_msg:
        st.session_state.halter_log.append(log_msg)


# 3. Gross Exposure Logic
if run_gross_exp and all(gross_exposure_sounds.values()):
    rect = find_window_by_title_substring(GROSS_EXP_WINDOW_TITLE)
    log_msg = ""
    if not rect:
        log_msg = f"[{timestamp}] Window '{GROSS_EXP_WINDOW_TITLE}' not found."
        st.session_state.gross_exp_last_alert = 0 # Reset cooldown
    else:
        text = extract_text_from_image(capture_screenshot(rect))
        value = parse_gross_exposure_value(text)
        if value is not None:
            alert_triggered = False
            log_msg = f"[{timestamp}] Gross Exposure Found: {value: ,}"
            for threshold in sorted(gross_exposure_sounds.keys(), reverse=True):
                if value > threshold:
                    if (now - st.session_state.gross_exp_last_alert) > GROSS_EXP_COOLDOWN:
                        sound_to_play = gross_exposure_sounds[threshold]
                        sound_to_play.play()
                        log_msg += f" -> ALERT! Exceeds {threshold:,}. Sound PLAYED."
                        st.session_state.gross_exp_last_alert = now
                    else:
                        log_msg += f" -> Exceeds {threshold:,}. (Cooldown active)."
                    alert_triggered = True
                    break # Stop checking after the first threshold is met
            if not alert_triggered:
                st.session_state.gross_exp_last_alert = 0 # Reset if below all thresholds
        else:
            log_msg = f"[{timestamp}] Gross Exposure check complete. Value not found."
            st.session_state.gross_exp_last_alert = 0 # Reset if value not parsed

    if st.session_state.gross_exp_log[-1] != log_msg:
        st.session_state.gross_exp_log.append(log_msg)


# 4. Click Automation Logic
if st.session_state.is_running and not st.session_state.click_points:
    st.session_state.capture_mode = True

if st.session_state.capture_mode:
    hwnd = find_exact_window(CLICKER_TARGET_WINDOW_TITLE)
    if not hwnd:
        st.error(f"Target window '{CLICKER_TARGET_WINDOW_TITLE}' not found for click setup.")
        st.session_state.is_running = False
        st.rerun()
    else:
        status_placeholder.warning("ACTION REQUIRED: Define the click sequence in the target window.")
        st.write("1. **Right-click** at the first location. 2. **Right-click** at the second. 3. **Right-click** at the third.")

        captured_points = []
        click_definitions = [("Right Click", "right"), ("Right Click", "right"), ("Right Click", "right")]
        def on_click(x, y, button, pressed):
            if pressed:
                point_index = len(captured_points)
                _, defined_type = click_definitions[point_index]
                client_coords = win32gui.ScreenToClient(hwnd, (x, y))
                captured_points.append({'coords': client_coords, 'type': defined_type})
                if len(captured_points) == 3: return False
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()

        st.session_state.click_points = captured_points
        st.session_state.capture_mode = False
        st.success("✅ Click points saved successfully!")
        st.session_state.click_log.append("Click points defined and saved.")
        time.sleep(2)
        st.rerun()

if st.session_state.is_running and st.session_state.click_points:
    hwnd = find_exact_window(CLICKER_TARGET_WINDOW_TITLE)
    if not hwnd:
        status_placeholder.error(f"Window '{CLICKER_TARGET_WINDOW_TITLE}' not found. Stopping.")
        st.session_state.is_running = False
        time.sleep(2)
        st.rerun()
    else:
        status_placeholder.info("Running click sequence...")
        original_foreground_hwnd = win32gui.GetForegroundWindow()
        for point in st.session_state.click_points:
            perform_realistic_click(hwnd, point['coords'][0], point['coords'][1], point['type'])
            time.sleep(DELAY_BETWEEN_CLICKS)
        if win32gui.IsWindow(original_foreground_hwnd):
            try: win32gui.SetForegroundWindow(original_foreground_hwnd)
            except Exception: pass
        st.session_state.click_log.append("Sequence complete. Waiting...")

        # This section creates the countdown progress bar
        status_placeholder.text(f"Next sequence in {click_interval} seconds...")
        progress_bar = progress_placeholder.progress(0)
        for i in range(100):
            time.sleep(click_interval / 100) # Wait for a fraction of the total interval
            progress_bar.progress(i + 1)     # Update the progress bar
        progress_placeholder.empty()
        st.rerun()
elif not st.session_state.is_running and 'click_run_toggle' in st.session_state:
     status_placeholder.empty() # Clear status text when stopped


# --- MASTER RERUN CONTROLLER ---
if run_high_touch or run_halter or run_gross_exp:
    time.sleep(CHECK_INTERVAL)
    st.rerun()