# Multimodal RAG - Environment Setup (PowerShell)
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Multimodal RAG - Environment Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Create venv
Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if (-not $?) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate venv
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "[4/5] Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install Jupyter kernel
Write-Host "[5/5] Installing Jupyter kernel..." -ForegroundColor Yellow
pip install ipykernel jupyter notebook
python -m ipykernel install --user --name=multimodal-rag --display-name="Multimodal RAG (Python)"

# Create .env if not exists
if (-not (Test-Path ".env")) {
    Write-Host "`nCreating .env from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - Please add your GOOGLE_API_KEY!" -ForegroundColor Magenta
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env and add your GOOGLE_API_KEY"
Write-Host "     Get key from: https://makersuite.google.com/app/apikey"
Write-Host ""
Write-Host "  2. Activate venv (if not already):"
Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  3. Run Jupyter Notebook:"
Write-Host "     jupyter notebook notebooks/multimodal_rag_demo.ipynb" -ForegroundColor White
Write-Host ""
Write-Host "  4. In Jupyter, select kernel: 'Multimodal RAG (Python)'"
Write-Host ""
Write-Host "Or use the CLI:" -ForegroundColor Cyan
Write-Host "  python main.py ingest --dir ./data/sample_documents"
Write-Host "  python main.py query `"What are the key findings?`""
Write-Host ""
