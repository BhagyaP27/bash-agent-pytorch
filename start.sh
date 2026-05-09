#!/bin/bash
# ============================================================
#  BashAgent — startup script (Linux / macOS)
#  Usage: ./start.sh
#         ./start.sh --skip-train
#         ./start.sh --train-only
# ============================================================

set -e

VENV_PYTHON=".venv/bin/python"
CHECKPOINT="checkpoints/bash_agent_best.pth"
API_PORT=8000
UI_DIR="NEXTJS-UI"
SKIP_TRAIN=false
TRAIN_ONLY=false

for arg in "$@"; do
  case $arg in
    --skip-train) SKIP_TRAIN=true ;;
    --train-only) TRAIN_ONLY=true ;;
  esac
done

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step()  { echo -e "\n  ${CYAN}▶ $1${NC}"; }
ok()    { echo -e "  ${GREEN}✓ $1${NC}"; }
warn()  { echo -e "  ${YELLOW}⚠ $1${NC}"; }
err()   { echo -e "  ${RED}✗ $1${NC}"; exit 1; }

echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}   BashAgent — Auto Launcher${NC}"
echo -e "${CYAN}==========================================${NC}"

# ── 1. Virtual environment ────────────────────────────────────
step "Checking virtual environment..."
if [ ! -f "$VENV_PYTHON" ]; then
    warn ".venv not found — creating one..."
    python3 -m venv .venv
    ok "Virtual environment created."
fi
source .venv/bin/activate

# ── 2. Dependencies ───────────────────────────────────────────
step "Installing Python dependencies..."
pip install -q -r requirements.txt
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
ok "Python dependencies ready."

# ── 3. Train if no model ──────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        warn "No trained model found at $CHECKPOINT"
        step "Starting training — this may take a few minutes..."
        python train.py || err "Training failed."
        ok "Model trained and saved."
    else
        ok "Model checkpoint found — skipping training."
    fi
else
    warn "--skip-train set — skipping training check."
fi

if [ "$TRAIN_ONLY" = true ]; then
    ok "Done (train-only mode)."
    exit 0
fi

# ── 4. Verify model ───────────────────────────────────────────
[ ! -f "$CHECKPOINT" ] && err "No model at $CHECKPOINT. Train first."

# ── 5. Start FastAPI backend ──────────────────────────────────
step "Starting FastAPI backend on port $API_PORT..."
python -m uvicorn api:app --host 0.0.0.0 --port $API_PORT --reload &
BACKEND_PID=$!
sleep 3

if kill -0 $BACKEND_PID 2>/dev/null; then
    ok "Backend running (PID $BACKEND_PID) → http://localhost:$API_PORT"
else
    err "Backend failed to start."
fi

# ── 6. Start Next.js frontend ─────────────────────────────────
if [ -d "$UI_DIR" ]; then
    step "Checking Node.js dependencies..."
    cd $UI_DIR
    [ ! -d "node_modules" ] && npm install && ok "Node dependencies installed."
    step "Starting Next.js frontend on port 3000..."
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    sleep 2
    ok "Frontend starting (PID $FRONTEND_PID) → http://localhost:3000"
else
    warn "$UI_DIR not found — skipping frontend."
fi

# ── 7. Summary ────────────────────────────────────────────────
echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${GREEN}   All services launched!${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""
echo -e "   Backend API  →  http://localhost:$API_PORT"
echo -e "   API Docs     →  http://localhost:${API_PORT}/docs"
echo -e "   Frontend     →  http://localhost:3000"
echo ""
echo "   Press Ctrl+C to stop all services."
echo ""

# Keep running and clean up on exit
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT
wait