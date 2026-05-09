# BashAgent Launcher
# Usage: .\start.ps1

$VENV_PYTHON   = ".\.venv\Scripts\python.exe"
$CHECKPOINT    = "checkpoints\bash_agent_best.pth"
$API_PORT      = 8000
$UI_DIR        = "NEXTJS-UI"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   BashAgent Auto Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check virtual environment
Write-Host "  Checking virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "  Creating .venv..." -ForegroundColor Yellow
    python -m venv .venv
}
Write-Host "  OK - venv ready" -ForegroundColor Green

# 2. Install dependencies
Write-Host "  Installing dependencies..." -ForegroundColor Yellow
& $VENV_PYTHON -m pip install -q -r requirements.txt
& $VENV_PYTHON -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
Write-Host "  OK - dependencies ready" -ForegroundColor Green

# 3. Train if no checkpoint found
Write-Host "  Checking for trained model..." -ForegroundColor Yellow
if (-not (Test-Path $CHECKPOINT)) {
    Write-Host "  No model found - starting training..." -ForegroundColor Yellow
    & $VENV_PYTHON train.py
    Write-Host "  OK - training complete" -ForegroundColor Green
} else {
    Write-Host "  OK - model checkpoint found" -ForegroundColor Green
}

# 4. Start FastAPI backend in new window
Write-Host "  Starting FastAPI backend on port $API_PORT..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$VENV_PYTHON' -m uvicorn api:app --host 0.0.0.0 --port $API_PORT --reload"
Start-Sleep -Seconds 3
Write-Host "  OK - backend launching at http://localhost:$API_PORT" -ForegroundColor Green

# 5. Start Next.js frontend in new window
Write-Host "  Starting Next.js frontend..." -ForegroundColor Yellow
if (Test-Path $UI_DIR) {
    if (-not (Test-Path "$UI_DIR\node_modules")) {
        Write-Host "  Running npm install..." -ForegroundColor Yellow
        Set-Location $UI_DIR
        npm install
        Set-Location ..
    }
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$UI_DIR'; npm run dev"
    Start-Sleep -Seconds 2
    Write-Host "  OK - frontend launching at http://localhost:3000" -ForegroundColor Green
} else {
    Write-Host "  WARN - $UI_DIR not found, skipping frontend" -ForegroundColor Yellow
}

# 6. Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   All services launched!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "   Backend  -> http://localhost:$API_PORT"      -ForegroundColor White
Write-Host "   API Docs -> http://localhost:$API_PORT/docs" -ForegroundColor White
Write-Host "   Frontend -> http://localhost:3000"           -ForegroundColor White
Write-Host ""
Write-Host "   Close the opened windows to stop." -ForegroundColor DarkGray
Write-Host ""