# app.py
# A Streamlit application to automate a sequence of mouse clicks in a specific window.
#
# Features:
# - Remembers click positions after the first setup.
# - Automatically targets a predefined window.
# - Provides a web UI with a start/stop toggle, configurable interval, and a progress bar.

import streamlit as st
import win32gui
import win32api
import time
from pynput import mouse

# --- CONFIGURATION ---
TARGET_WINDOW_TITLE = "Scalable_Hedging_Version_3.0 - Trading Manager <shared> - \\Remote"
DELAY_BETWEEN_CLICKS = 0.5  # A short delay between individual clicks within the sequence.

# --- SCRIPT LOGIC (Adapted for Streamlit) ---

# Create a mouse controller object to perform clicks
mouse_controller = mouse.Controller()


def find_target_window(title):
    """Finds the window handle for a given exact window title."""
    try:
        hwnd = win32gui.FindWindow(None, title)
        if hwnd == 0:
            return None
        return hwnd
    except Exception:
        return None


def perform_realistic_click(hwnd, x, y, click_type='left'):
    """
    Simulates a realistic click by moving the mouse cursor, clicking, and returning.
    Logs actions to the Streamlit session state.
    """
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
        st.session_state.log.append(log_entry)

    except Exception as e:
        log_entry = f"Error: Failed to click at ({x}, {y}). Details: {e}"
        st.session_state.log.append(log_entry)
        if 'original_pos' in locals():
            mouse_controller.position = original_pos


# --- STREAMLIT UI AND APPLICATION FLOW ---

st.set_page_config(page_title="Click Automation", layout="centered")
st.title("🖱️ Background Click Simulator")

# --- INITIALIZE SESSION STATE ---
# This is crucial for Streamlit to remember variables across reruns.
if 'click_points' not in st.session_state:
    st.session_state.click_points = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'capture_mode' not in st.session_state:
    st.session_state.capture_mode = False
if 'log' not in st.session_state:
    st.session_state.log = ["Welcome! Start the automation to begin."]

# --- UI CONTROLS ---
st.info(f"This tool will automate clicks in the window named:\n**{TARGET_WINDOW_TITLE}**")

col1, col2 = st.columns([1, 1])
with col1:
    # This toggle's value is directly tied to the session state
    st.session_state.is_running = st.toggle(
        "Start / Stop Automation",
        value=st.session_state.is_running,
        key='run_toggle'
    )
with col2:
    click_interval = st.number_input(
        "Interval between sequences (seconds)",
        min_value=1,
        value=20,
        step=1
    )

# Placeholders for dynamic content like status and progress bar
status_placeholder = st.empty()
progress_placeholder = st.empty()

# --- LOGIC FOR CAPTURING CLICKS (First time only) ---
if st.session_state.is_running and not st.session_state.click_points:
    st.session_state.capture_mode = True

if st.session_state.capture_mode:
    hwnd = find_target_window(TARGET_WINDOW_TITLE)
    if not hwnd:
        st.error(f"The target window '{TARGET_WINDOW_TITLE}' could not be found. Please ensure it is open.")
        st.session_state.is_running = False  # Turn off toggle
        st.rerun()
    else:
        status_placeholder.warning("ACTION REQUIRED: Please define the click sequence.")
        st.write("1. **Right-click** at the first desired location inside the target window.")
        st.write("2. **Left-click** at the second location.")
        st.write("3. **Left-click** at the third location.")
        st.write("_The script will automatically detect the clicks and continue..._")

        # This listener will capture the 3 clicks and then stop.
        captured_points = []
        click_definitions = [("Right Click", "right"), ("Left Click", "left"), ("Left Click", "left")]


        def on_click(x, y, button, pressed):
            if pressed:
                point_index = len(captured_points)
                click_type = mouse.Button.right if button == mouse.Button.right else mouse.Button.left

                # Use the correct click type from definitions for saving
                defined_label, defined_type = click_definitions[point_index]

                screen_coords = (x, y)
                client_coords = win32gui.ScreenToClient(hwnd, screen_coords)
                captured_points.append({'coords': client_coords, 'type': defined_type})

                # Stop listener after 3 points
                if len(captured_points) == 3:
                    return False


        with mouse.Listener(on_click=on_click) as listener:
            listener.join()  # This will block here until 3 clicks are captured

        st.session_state.click_points = captured_points
        st.session_state.capture_mode = False
        st.success("✅ All 3 click points have been saved successfully!")
        st.session_state.log.append("Click points defined and saved.")
        time.sleep(2)
        st.rerun()

# --- MAIN AUTOMATION LOOP ---
if st.session_state.is_running and st.session_state.click_points:
    hwnd = find_target_window(TARGET_WINDOW_TITLE)
    if not hwnd:
        status_placeholder.error(f"Window '{TARGET_WINDOW_TITLE}' not found. Stopping.")
        st.session_state.is_running = False  # Turn off the toggle
        time.sleep(2)
        st.rerun()
    else:
        # Perform the click sequence
        status_placeholder.info("Running click sequence...")

        # Save the window that is currently in focus right before the click
        original_foreground_hwnd = win32gui.GetForegroundWindow()

        for point in st.session_state.click_points:
            coords = point['coords']
            click_type = point['type']
            perform_realistic_click(hwnd, coords[0], coords[1], click_type)
            time.sleep(DELAY_BETWEEN_CLICKS)

        # Restore focus to the original window immediately after the sequence
        if win32gui.IsWindow(original_foreground_hwnd):
            try:
                win32gui.SetForegroundWindow(original_foreground_hwnd)
            except Exception:
                pass  # May fail if the original window was closed

        st.session_state.log.append("Sequence complete. Waiting for next interval.")

        # Display the progress bar countdown
        status_placeholder.text(f"Next sequence in {click_interval} seconds...")
        progress_bar = progress_placeholder.progress(0)
        for i in range(100):
            time.sleep(click_interval / 100)
            progress_bar.progress(i + 1)

        progress_placeholder.empty()
        st.rerun()

elif not st.session_state.is_running:
    status_placeholder.success("Status: Stopped. Ready to start.")

# --- DISPLAY LOG ---
with st.expander("Show Activity Log", expanded=True):
    log_container = st.container(height=200)
    for msg in reversed(st.session_state.log):
        log_container.text(msg)