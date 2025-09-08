import streamlit as st
import pygame
import pygetwindow as gw
import time
from datetime import datetime
import re
import json
import pyautogui
import pytesseract
from PIL import Image
import platform
# Imports for Click Automation
import win32gui
from pynput import mouse
import tkinter as tk
from tkinter import messagebox
import ctypes
import streamlit as st
import pygame
import multiprocessing

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
    page_title="PT Dashboard",
    page_icon="data/icon.png",
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
GROSS_EXP_WINDOW_TITLE = config['windows']['gross_exp_window_title']
GROSS_EXP_SOUND_FILES = {
    config['exposure_threshold']['stage1']: 'Sounds/Alert_stage1.wav',
    config['exposure_threshold']['stage2']: 'Sounds/Alert_stage2.wav',
    config['exposure_threshold']['stage3']: 'Sounds/Alert_stage3.wav'
}
GROSS_EXP_COOLDOWN = config['cooldown']['gross_exp_cooldown']
PNL_THRESHOLD = config['PnL_threshold']['delta_PnL']
PNL_ALERT_COOLDOWN = config['cooldown']['delta_PnL_cooldown']  # Example: Cooldown in seconds for the PnL delta alert
PNL_ALERT_SOUND_PATH = "Sounds/2000delta.wav"  # Path to the PnL alert sound

OCR_FAILED_SOUND_FILE = 'Sounds/OCR_failed.wav'

STALE_GROSS_EXP_SOUND_FILE = 'Sounds/Stale_Gross_Exposure.wav'
STALE_PNL_SOUND_FILE = 'Sounds/Stale_PnL.wav'

NotFound_ALERT_COOLDOWN = config['cooldown']['Not_Found_cooldown']
STALE_DURATION = config['Stale_Duration']['Stale_Duration']

# 4. Click Automation
CLICKER_TARGET_WINDOW_TITLE = config['windows']['click_target_window_title']
DELAY_BETWEEN_CLICKS = config['clicker']['delay_between_clicks']

# --- OCR FAILURE POP-UP FUNCTION ---
# --- OCR FAILURE POP-UP FUNCTION ---
def show_ocr_failed_popup():
    """
    Creates and shows a temporary, non-blocking pop-up window
    that flashes red to inform the user that OCR has failed.
    The window closes automatically or when the user clicks OK.
    """
    try:
        mouse_x, mouse_y = pyautogui.position()
        root = tk.Tk()
        root.withdraw()
        popup = tk.Toplevel(root)
        popup.title("OCR Failure")
        popup.attributes("-topmost", True)
        width = 450
        height = 180  # Made it a little taller for the button
        popup.geometry(f'{width}x{height}+{mouse_x}+{mouse_y}')

        message = (
            "OCR failed to read values from the target window.\n\n"
            "Please ensure the window is visible and not obstructed."
        )

        label = tk.Label(
            popup,
            text=message,
            padx=20,
            pady=10,  # Adjusted padding
            fg='#FFFFFF'
        )
        label.pack(expand=True, fill='both')

        # --- NEW: OK BUTTON ---
        # Create a button with the text "OK".
        # The 'command=popup.destroy' tells the button to run the popup.destroy()
        # function when it's clicked, which closes the window.
        ok_button = tk.Button(
            popup,
            text="OK",
            command=popup.destroy,
            width=10  # Set a fixed width for the button
        )
        # Add the button to the window with some padding below it.
        ok_button.pack(pady=10)

        # --- FLASHING LOGIC ---
        COLOR_RED = '#FF0000'
        COLOR_DEFAULT = '#F0F0F0'

        def flash_color(is_red=True):
            try:
                current_color = COLOR_RED if is_red else COLOR_DEFAULT
                popup.config(bg=current_color)
                label.config(bg=current_color)
                # The button's color will not flash, making it stand out.
                popup.after(500, flash_color, not is_red)
            except tk.TclError:
                pass

        flash_color()

        popup.after(120000, popup.destroy)
        root.after(120100, root.destroy)
        root.mainloop()

    except Exception as e:
        print(f"Error showing pop-up: {e}")



#this needed for stop OCR failed ouput when screen is locked
def is_screen_locked():
    """
    Checks if the user's screen is locked using a more reliable method
    by checking the active desktop name.
    """
    if platform.system() != "Windows":
        return False  # Only implement for Windows

    try:
        user32 = ctypes.WinDLL('user32.dll')

        # Open the desktop that is currently receiving user input
        h_desktop = user32.OpenInputDesktop(0, False, 0x0001)

        if not h_desktop:
            return True

        desktop_name_buffer = ctypes.create_unicode_buffer(256)
        buffer_length = ctypes.c_ulong(ctypes.sizeof(desktop_name_buffer))

        user32.GetUserObjectInformationW(
            h_desktop,
            2,  # UOI_NAME
            ctypes.byref(desktop_name_buffer),
            buffer_length,
            ctypes.byref(buffer_length)
        )

        # Don't forget to close the handle
        user32.CloseDesktop(h_desktop)

        # Get the string value from the buffer
        desktop_name = desktop_name_buffer.value.lower()

        return desktop_name != "default"

    except Exception as e:
        print(f"Error checking screen lock status: {e}")
        return False

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

def parse_PnL_value(text):
    """
    Extrahiert die Zahl zwischen 'SCALMM' und 'EUR', entfernt Leerzeichen und Hashtags.
    """
    # Regex is robust against different casing and optional spaces
    match = re.search(r'scalmm\s*([\d\s#,.]+?)\s*eur\b', text, re.IGNORECASE)
    if match:
        value_string = match.group(1).replace(" ", "").replace("#", "").replace(",", "").replace(".", "")

        # --- FIX ---
        # Add a check to ensure the string is not empty *after* cleaning it.
        if value_string:
            return int(value_string)
        # If the string is empty, we fall through and return None,
        # which is the correct behavior (value not found).

    return None
# --- Click Automation Helper Functions ---
mouse_controller = mouse.Controller()

# NEW FUNCTION FOR THE BACKGROUND PROCESS
def clicker_process_target(window_title, click_points, delay_between, full_interval):
    """
    This function is the target for the background process.
    It runs an infinite loop to perform clicks.
    """
    while True:
        hwnd = find_exact_window(window_title)
        if hwnd:
            # This loop performs the sequence of clicks
            for point in click_points:
                perform_realistic_click(hwnd, point['coords'][0], point['coords'][1], point['type'])
                time.sleep(delay_between)

        # Wait for the full interval before starting the next sequence
        time.sleep(full_interval)


def find_exact_window(title_part):
    """Finds the window handle for a title containing the given substring."""
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if title_part.lower() in window_title.lower():
                hwnds.append(hwnd)
        return True
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds[0] if hwnds else None

def perform_realistic_click(hwnd, x, y, click_type='left'):
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
pnl_alert_sound = initialize_sounds(PNL_ALERT_SOUND_PATH)  # NEU
ocr_failed_sound = initialize_sounds(OCR_FAILED_SOUND_FILE)
stale_gross_exp_sound = initialize_sounds(STALE_GROSS_EXP_SOUND_FILE)
stale_pnl_sound = initialize_sounds(STALE_PNL_SOUND_FILE)

# --- INITIALIZE SESSION STATE ---
if 'high_touch_log' not in st.session_state: st.session_state.high_touch_log = ["Ready."]
if 'halter_log' not in st.session_state: st.session_state.halter_log = ["Ready."]
if 'gross_exp_log' not in st.session_state: st.session_state.gross_exp_log = ["Ready."]
if 'high_touch_last_alert' not in st.session_state: st.session_state.high_touch_last_alert = 0
if 'halter_last_alert' not in st.session_state: st.session_state.halter_last_alert = 0
if 'gross_exp_last_alert' not in st.session_state: st.session_state.gross_exp_last_alert = 0
if 'click_points' not in st.session_state: st.session_state.click_points = []
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'capture_mode' not in st.session_state: st.session_state.capture_mode = False
if 'click_log' not in st.session_state: st.session_state.click_log = ["Ready."]
if 'last_pnl_value' not in st.session_state: st.session_state.last_pnl_value = None
if 'pnl_last_alert' not in st.session_state: st.session_state.pnl_last_alert = 0
if 'parsing_last_alert' not in st.session_state: st.session_state.parsing_last_alert = 0
if 'last_gross_exp_value' not in st.session_state: st.session_state.last_gross_exp_value = None
if 'gross_exp_stale_since' not in st.session_state: st.session_state.gross_exp_stale_since = 0
if 'stale_gross_exp_alerted' not in st.session_state: st.session_state.stale_gross_exp_alerted = False
if 'pnl_stale_since' not in st.session_state: st.session_state.pnl_stale_since = 0
if 'stale_pnl_alerted' not in st.session_state: st.session_state.stale_pnl_alerted = False
if 'clicker_process' not in st.session_state: st.session_state.clicker_process = None
if 'ocr_failure_alerted' not in st.session_state: st.session_state.ocr_failure_alerted = False


# --- MAIN PAGE LAYOUT ---
st.title("PT Dashboard")
st.html("""
  <style>
    [alt=Logo] {
      height: 4.5rem;
    }
  </style>
        """)
st.logo("data/logo.png", size="large")
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
        with st.expander("HALTER Log", expanded=False):
            st.info(f"Latest: {st.session_state.halter_log[-1]}")
            log_container = st.container(height=200)
            for msg in reversed(st.session_state.halter_log): log_container.text(msg)

# --- 3. GROSS EXPOSURE ALERT UI ---
st.divider()
if not all(gross_exposure_sounds.values()):
    st.error("One or more Gross Exposure sound files failed to load. Check paths.")
else:
    # Adjusted columns to make space for the new select box
    col7, col8, col9, col10 = st.columns([2, 1, 2, 6])

    with col7:
        run_gross_exp = st.toggle("Start Gross Exposure Monitoring", key="gross_exp_toggle")

    with col8:
        # NEW: Select box for choosing the alert type for OCR failures
        st.selectbox(
            "OCR Fail Alert Type",
            ("Sound Alert", "Pop-up Window"),
            key="ocr_alert_type",
            help="Choose how to be notified if the app cannot read the window text."
        )

    with col9:
        if run_gross_exp:
            st.success("Status: Enabled")
        else:
            st.error("Status: Disabled")

    with col10:
        with st.expander("Gross Exposure Log", expanded=False):
            # To prevent an error if the log is empty on first run
            if st.session_state.gross_exp_log:
                st.info(f"Latest: {st.session_state.gross_exp_log[-1]}")
            log_container = st.container(height=200)
            for msg in reversed(st.session_state.gross_exp_log):
                log_container.text(msg)

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
    with st.expander("Clicker Log", expanded=False):
        st.info(f"Latest: {st.session_state.click_log[-1]}")
        log_container = st.container(height=200)
        for msg in reversed(st.session_state.click_log):
            log_container.text(msg)

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
        st.session_state.high_touch_last_alert = 0

    if st.session_state.high_touch_log[-1] != log_msg:
        st.session_state.high_touch_log.append(log_msg)

# 2. HALTER Logic
if run_halter and not isinstance(halter_sound, str):
    rect = find_window_by_title_substring(HALTER_WINDOW_TITLE)
    log_msg = ""
    if not rect:
        log_msg = f"[{timestamp}] Window '{HALTER_WINDOW_TITLE}' not found."
        st.session_state.halter_last_alert = 0
    else:
        try:
            # This is the code that might fail
            text = extract_text_from_image(capture_screenshot(rect))

            # If screenshot is successful, continue with normal logic
            if any(term in text for term in HALTER_SEARCH_TERMS):
                if (now - st.session_state.halter_last_alert) > HALTER_COOLDOWN:
                    halter_sound.play()
                    st.session_state.halter_last_alert = now
                    log_msg = f"[{timestamp}] HALTER error detected -> Sound PLAYED."
                else:
                    log_msg = f"[{timestamp}] HALTER error detected. (Cooldown active)."
            else:
                log_msg = f"[{timestamp}] HALTER check complete. No errors."
                st.session_state.halter_last_alert = 0

        except OSError as e:
            # This runs ONLY if the screenshot fails
            log_msg = f"[{timestamp}] OCR FAILED. Please check."
            if not isinstance(ocr_failed_sound, str):
                ocr_failed_sound.play()  # Play the custom error sound

    if st.session_state.halter_log[-1] != log_msg:
        st.session_state.halter_log.append(log_msg)

# 3. Gross Exposure Logic
if run_gross_exp and all(gross_exposure_sounds.values()):
    rect = find_window_by_title_substring(GROSS_EXP_WINDOW_TITLE)
    log_msg = ""

    if not rect:
        # This part remains the same: Window title was not found at all.
        log_msg = f"[{timestamp}] Window '{GROSS_EXP_WINDOW_TITLE}' not found."
        st.session_state.gross_exp_last_alert = 0
        st.session_state.last_pnl_value = None
        st.session_state.last_gross_exp_value = None
        st.session_state.gross_exp_stale_since = 0
        st.session_state.pnl_stale_since = 0
    else:
        try:
            # Attempt to capture the screen and extract text
            screenshot = capture_screenshot(rect)
            text = extract_text_from_image(screenshot)

            # Parse both values
            value = parse_gross_exposure_value(text)
            pnl_value = parse_PnL_value(text)

            # --- Build Gross Exp Log Part ---
            gross_exp_log_part = ""
            if value is not None:
                # (Normal and Stale logic for Gross Exp remains the same)
                gross_exp_log_part = f"Gross Exp: {value:,}"
                alert_triggered = False
                for threshold in sorted(gross_exposure_sounds.keys(), reverse=True):
                    if value > threshold:
                        if (now - st.session_state.gross_exp_last_alert) > GROSS_EXP_COOLDOWN:
                            gross_exposure_sounds[threshold].play()
                            gross_exp_log_part += f" -> ALERT! Exceeds {threshold:,}."
                            st.session_state.gross_exp_last_alert = now
                        else:
                            gross_exp_log_part += f" -> Exceeds {threshold:,} (Cooldown)."
                        alert_triggered = True
                        break
                if not alert_triggered: st.session_state.gross_exp_last_alert = 0
                if value != st.session_state.last_gross_exp_value:
                    st.session_state.last_gross_exp_value = value
                    st.session_state.gross_exp_stale_since = now
                    st.session_state.stale_gross_exp_alerted = False
                elif not st.session_state.stale_gross_exp_alerted and (
                        now - st.session_state.gross_exp_stale_since) > STALE_DURATION:
                    if not isinstance(stale_gross_exp_sound, str): stale_gross_exp_sound.play()
                    gross_exp_log_part += " (STALE!)"
                    st.session_state.stale_gross_exp_alerted = True
            else:
                gross_exp_log_part = "Gross Exp: Not Found"

            # --- Build PnL Log Part ---
            pnl_log_part = ""
            if pnl_value is not None:
                # (Normal and Stale logic for PnL remains the same)
                if st.session_state.last_pnl_value is not None:
                    delta = pnl_value - st.session_state.last_pnl_value
                    pnl_log_part = f"PnL: {pnl_value:,} (Δ {delta:+,})"
                    if abs(delta) > PNL_THRESHOLD:
                        if (now - st.session_state.pnl_last_alert) > PNL_ALERT_COOLDOWN:
                            if not isinstance(pnl_alert_sound, str): pnl_alert_sound.play()
                            pnl_log_part += f" -> DELTA ALERT!"
                            st.session_state.pnl_last_alert = now
                        else:
                            pnl_log_part += f" (Delta Cooldown)"
                    if delta == 0:
                        if st.session_state.pnl_stale_since == 0:
                            st.session_state.pnl_stale_since = now
                        elif not st.session_state.stale_pnl_alerted and (
                                now - st.session_state.pnl_stale_since) > STALE_DURATION:
                            if not isinstance(stale_pnl_sound, str): stale_pnl_sound.play()
                            pnl_log_part += " (STALE!)"
                            st.session_state.stale_pnl_alerted = True
                    else:
                        st.session_state.pnl_stale_since = 0
                        st.session_state.stale_pnl_alerted = False
                else:
                    pnl_log_part = f"PnL: {pnl_value:,} (Tracking started)"
                st.session_state.last_pnl_value = pnl_value
            else:
                pnl_log_part = "PnL: Not Found"
                st.session_state.last_pnl_value = None

            # --- NEW AND IMPROVED: UNIFIED PARSING FAILURE ALERT LOGIC ---
            log_suffix = ""
            if value is None or pnl_value is None:
                # If either value wasn't found, we're in a failure state.

                # NEW LOGIC: Only alert if we haven't already alerted for this failure.
                if not st.session_state.ocr_failure_alerted and not is_screen_locked():

                    # Your existing alert preference logic is perfect.
                    alert_preference = st.session_state.get("ocr_alert_type", "Sound Alert")
                    if alert_preference == "Pop-up Window":
                        show_ocr_failed_popup()
                    else:
                        if not isinstance(ocr_failed_sound, str):
                            ocr_failed_sound.play()

                    log_suffix = " -> PARSE ALERT!"
                    # Latch the state: We have now alerted the user.
                    st.session_state.ocr_failure_alerted = True

                elif is_screen_locked():
                    log_suffix = " (Parse Alert Suppressed - Screen Locked)"
                else:
                    # We are in a failure state, but we've already alerted, so we stay quiet.
                    log_suffix = " (Ongoing Parse Failure)"

            else:
                # SUCCESS! Both values were found.
                # Reset the latch so we are ready to alert for the *next* failure.
                st.session_state.ocr_failure_alerted = False

            log_msg = f"[{timestamp}] {gross_exp_log_part} | {pnl_log_part}{log_suffix}"

        # --- NEW: DIFFERENTIATED EXCEPTION HANDLING ---
        except OSError as e:
            # CHANGED: This now handles screen capture failures (minimized/locked screen).
            # We log it, but we DON'T play an alert sound.
            log_msg = f"[{timestamp}] Capture failed. Window might be minimized or screen locked."
            # Note: We don't reset state here, so if the window becomes visible again,
            # the stale logic will correctly resume.

    # This logging logic remains the same
    if log_msg and (not st.session_state.gross_exp_log or st.session_state.gross_exp_log[-1] != log_msg):
        st.session_state.gross_exp_log.append(log_msg)

# 4. Click Automation Logic
if st.session_state.get('click_run_toggle'):
    # This block runs when the toggle is ON
    if st.session_state.clicker_process is not None:
        # If the process is running, calculate and show the countdown
        now = time.time()

        # Check if the timer needs to be reset for the next interval
        if now > st.session_state.next_click_time:
            st.session_state.next_click_time = now + click_interval

        time_left = round(st.session_state.next_click_time - now)

        if time_left > 0:
            status_placeholder.info(f"Clicker is active. Next sequence in ~{time_left} seconds...")
        else:
            status_placeholder.info("Clicker is active. Executing click sequence...")

    else:  # This runs when the toggle has just been switched ON
        if not st.session_state.click_points:
            status_placeholder.warning("ACTION REQUIRED: You must define the click points first!")
            # This will trigger the capture_mode logic in Part 2
            st.session_state.capture_mode = True
        else:
            status_placeholder.info("Starting clicker process...")
            process = multiprocessing.Process(
                target=clicker_process_target,
                args=(CLICKER_TARGET_WINDOW_TITLE, st.session_state.click_points, DELAY_BETWEEN_CLICKS, click_interval),
                daemon=True
            )
            process.start()
            st.session_state.clicker_process = process
            # Set the initial timer for the countdown display
            st.session_state.next_click_time = time.time() + click_interval
            st.session_state.click_log.append(f"[{timestamp}] Clicker process started.")
            st.rerun()

else:
    # This block runs when the toggle is switched OFF
    if st.session_state.clicker_process is not None:
        status_placeholder.info("Stopping clicker process...")
        st.session_state.clicker_process.terminate()
        st.session_state.clicker_process = None
        st.session_state.click_log.append(f"[{timestamp}] Clicker process stopped.")
        status_placeholder.empty()
        st.rerun()

if st.session_state.capture_mode:
    hwnd = find_exact_window(CLICKER_TARGET_WINDOW_TITLE)
    if not hwnd:
        st.error(f"Target window '{CLICKER_TARGET_WINDOW_TITLE}' not found for click setup.")
        st.session_state.is_running = False  # Turn off the toggle
        st.session_state.capture_mode = False
        st.rerun()
    else:
        status_placeholder.warning("ACTION REQUIRED: Define the click sequence in the target window.")
        st.write(
            "1. **Right-click** at the first location. 2. **Right-click** at the second. 3. **Right-click** at the third.")

        captured_points = []


        # Simplified the click definitions as per your original code's behavior
        def on_click(x, y, button, pressed):
            # We only care about the press event, not the release
            if pressed and button == mouse.Button.right:
                client_coords = win32gui.ScreenToClient(hwnd, (x, y))
                captured_points.append({'coords': client_coords, 'type': 'right'})

                # Stop listening after 3 points are captured
                if len(captured_points) == 3:
                    return False


        with mouse.Listener(on_click=on_click) as listener:
            listener.join()

        st.session_state.click_points = captured_points
        st.session_state.capture_mode = False
        st.success("✅ Click points saved successfully!")
        st.session_state.click_log.append("Click points defined and saved.")
        time.sleep(2)
        st.rerun()

# --- MASTER RERUN CONTROLLER ---
# following code ensures the app reruns at consistent intervals (the alert will play at same time on different PC)
if run_high_touch or run_halter or run_gross_exp:
    # --- Synchronization Logic ---
    # Get the current time, including microseconds for better precision.
    now = datetime.now()

    # Calculate how many seconds have passed since the last interval mark.
    # For example, if it's 12.3 seconds past the minute and CHECK_INTERVAL is 5,
    # the remainder of (12.3 / 5) is 2.3.
    seconds_past_interval = (now.second + now.microsecond / 1_000_000) % CHECK_INTERVAL

    # Calculate the delay needed to reach the *next* interval mark.
    # Continuing the example: 5 - 2.3 = 2.7. So, we wait 2.7 seconds to hit the 15-second mark.
    wait_time = CHECK_INTERVAL - seconds_past_interval

    # Sleep for the calculated duration.
    time.sleep(wait_time)

    # Rerun the Streamlit app, which will now start closer to the desired time.
    st.rerun()
