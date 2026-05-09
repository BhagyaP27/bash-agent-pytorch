# ============================================================
#  BashAgent — startup script (Windows / PowerShell)
#  Usage: .\start.ps1
#         .\start.ps1 -SkipTrain   (skip training even if no model)
#         .\start.ps1 -TrainOnly   (train only, don't start servers)
# ============================================================

param(
    [switch]$SkipTrain,
    [switch]$TrainOnly
)

$ErrorActionPreference = "Stop"

$VENV_PYTHON   = ".\.venv\Scripts\python.exe"
$VENV_ACTIVATE = ".\.venv\Scripts\Activate.ps1"
$CHECKPOINT    = "checkpoints\bash_agent_best.pth"
$API_PORT      = 8000
$UI_DIR        = "NEXTJS-UI"

function Write-Step($msg) {
    Write-Host "`n  $msg" -ForegroundColor Cyan
}
function Write-Ok($msg) {
    Write-Host "  OK  $msg" -ForegroundColor Green
}
function Write-Warn($msg) {
    Write-Host "  !!  $msg" -ForegroundColor Yellow
}
function Write-Err($msg) {
    Write-Host "  ERR $msg" -ForegroundColor Red
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor DarkCyan
Write-Host "   BashAgent — Auto Launcher" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor DarkCyan

# ── 1. Virtual environment ────────────────────────────────────
Write-Step "Checking virtual environment..."
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Warn ".venv not found — creating one..."
    python -m venv .venv
    Write-Ok "Virtual environment created."
}

# ── 2. Dependencies ───────────────────────────────────────────
Write-Step "Installing Python dependencies..."
& $VENV_PYTHON -m pip install -q -r requirements.txt
& $VENV_PYTHON -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
Write-Ok "Python dependencies ready."

# ── 3. Train if no model ──────────────────────────────────────
if (-not $SkipTrain) {
    if (-not (Test-Path $CHECKPOINT)) {
        Write-Warn "No trained model found at $CHECKPOINT"
        Write-Step "Starting training — this may take a few minutes on CPU..."
        & $VENV_PYTHON train.py
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Training failed. Check the output above."
            exit 1
        }
        Write-Ok "Model trained and saved."
    } else {
        Write-Ok "Model checkpoint found — skipping training."
    }
} else {
    Write-Warn "-SkipTrain flag set — skipping training check."
}

if ($TrainOnly) {
    Write-Ok "Done (TrainOnly mode). Exiting."
    exit 0
}

# ── 4. Verify model exists before launching servers ───────────
if (-not (Test-Path $CHECKPOINT)) {
    Write-Err "No model at $CHECKPOINT. Run training first or remove -SkipTrain."
    exit 1
}

# ── 5. Start FastAPI backend ──────────────────────────────────
Write-Step "Starting FastAPI backend on port $API_PORT..."
$backend = Start-Process -PassThru -WindowStyle Normal powershell -ArgumentList `
    "-NoExit", "-Command", `
    "& '$VENV_PYTHON' -m uvicorn api:app --host 0.0.0.0 --port $API_PORT --reload"

Start-Sleep -Seconds 3

# Quick health check
try {
    $health = Invoke-RestMethod -Uri "http://localhost:$API_PORT/health" -TimeoutSec 5
    if ($health.model_loaded) {
        Write-Ok "Backend running — model loaded (vocab: $($health.input_vocab_size) tokens)"
    } else {
        Write-Warn "Backend running but model not loaded yet."
    }
} catch {
    Write-Warn "Backend may still be starting up — check the backend window."
}

# ── 6. Start Next.js frontend ─────────────────────────────────
if (Test-Path $UI_DIR) {
    Write-Step "Checking Node.js dependencies..."
    Push-Location $UI_DIR

    if (-not (Test-Path "node_modules")) {
        Write-Warn "node_modules not found — running npm install..."
        npm install
        Write-Ok "Node dependencies installed."
    } else {
        Write-Ok "node_modules found."
    }

    Write-Step "Starting Next.js frontend on port 3000..."
    $frontend = Start-Process -PassThru -WindowStyle Normal powershell -ArgumentList `
        "-NoExit", "-Command", "npm run dev"
    Pop-Location

    Start-Sleep -Seconds 2
    Write-Ok "Frontend starting at http://localhost:3000"
} else {
    Write-Warn "NEXTJS-UI directory not found — skipping frontend."
}

# ── 7. Summary ────────────────────────────────────────────────
Write-Host ""
Write-Host "==========================================" -ForegroundColor DarkCyan
Write-Host "   All services launched!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "   Backend API  →  http://localhost:$API_PORT"      -ForegroundColor White
Write-Host "   API Docs     →  http://localhost:$API_PORT/docs" -ForegroundColor White
Write-Host "   Frontend     →  http://localhost:3000"           -ForegroundColor White
Write-Host ""
Write-Host "   Close the backend/frontend windows to stop." -ForegroundColor DarkGray
Write-Host ""