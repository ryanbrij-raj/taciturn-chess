# setup_windows.ps1
# Run in PowerShell as Administrator
# Usage: .\setup_windows.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Taciturn Chess Engine — Windows Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check Python
Write-Host "`n[1/4] Checking Python..." -ForegroundColor Yellow
$pyVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: $pyVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "`n[2/4] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
.\venv\Scripts\Activate.ps1
Write-Host "  OK: venv created and activated" -ForegroundColor Green

# Install dependencies
Write-Host "`n[3/4] Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "  OK: Dependencies installed" -ForegroundColor Green

# Check GPU
Write-Host "`n[4/4] Checking GPU..." -ForegroundColor Yellow
python scripts\check_gpu.py

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Cyan  
Write-Host "  Start training: python train.py" -ForegroundColor Cyan
Write-Host "  Play vs engine: python play.py" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
