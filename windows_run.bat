@echo off
echo =============================================
echo  Setting up Streamlit Environment...
echo =============================================

:: Check if the virtual environment folder exists, if not, create it
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment and install dependencies
echo Activating virtual environment and installing packages...
call venv\Scripts\activate
pip install -r requirements.txt

:: Run the Streamlit app
echo =============================================
echo  Starting the Streamlit App...
echo =============================================
streamlit run Main_Page.py

pause