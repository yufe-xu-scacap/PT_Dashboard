@echo off
echo =============================================
echo  Setting up Streamlit Environment...
echo =============================================

:: Use the 'py' command to be more reliable on Windows
:: The '-3' flag specifically looks for a Python 3 installation
py -3 -m venv venv

echo Activating virtual environment and installing packages...
call venv\Scripts\activate
pip install -r requirements.txt

echo =============================================
echo  Starting the Streamlit App...
echo =============================================
streamlit run app.py

pause
