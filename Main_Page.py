import streamlit as st
import pandas as pd
import numpy as np

# Set the page configuration
st.set_page_config(
    page_title="Main Dashboard",
    page_icon="📈💰📊",
    layout="wide"
)

# --- CSS to make sidebar page names larger ---
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] a {
        /* Sets the font size for sidebar navigation links */
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a title and some text to the page
st.title("PT Dashboard")
st.sidebar.success("Select a page above.")


st.info("Powered by Hedge Fund Technology!")
