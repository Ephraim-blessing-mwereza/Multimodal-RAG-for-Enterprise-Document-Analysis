@echo off
echo ============================================
echo  Multimodal RAG - Environment Setup
echo ============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo [4/5] Installing dependencies...
pip install -r requirements.txt

echo [5/5] Installing Jupyter kernel...
pip install ipykernel
python -m ipykernel install --user --name=multimodal-rag --display-name="Multimodal RAG (Python)"

echo.
echo ============================================
echo  Setup Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Copy .env.example to .env
echo   2. Add your GOOGLE_API_KEY to .env
echo   3. Activate venv:  venv\Scripts\activate
echo   4. Run Jupyter:    jupyter notebook
echo   5. Select kernel:  "Multimodal RAG (Python)"
echo.
echo Or run the quick start:
echo   python main.py ingest --dir ./data/sample_documents
echo   python main.py query "What are the key findings?"
echo.
pause
