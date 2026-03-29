#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — One-command setup and launch for the Sentiment Analysis Dashboard
# Usage:  bash run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

VENV_DIR=".venv"
PYTHON="python3"
APP="app.py"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   AI-Driven Sentiment Analysis Dashboard             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check Python ───────────────────────────────────────────────────────────
if ! command -v $PYTHON &>/dev/null; then
    echo "❌  python3 not found. Please install Python 3.9+ and try again."
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅  Python $PY_VERSION detected"

# ── 2. Create virtual environment if needed ───────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "📦  Creating virtual environment in $VENV_DIR …"
    $PYTHON -m venv "$VENV_DIR"
fi

# ── 3. Activate venv ─────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "✅  Virtual environment activated"

# ── 4. Install / upgrade dependencies ────────────────────────────────────────
echo "📥  Installing dependencies from requirements.txt …"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✅  Dependencies installed"

# ── 5. Launch the app ────────────────────────────────────────────────────────
echo ""
echo "🚀  Launching dashboard at http://localhost:8501"
echo "    Press Ctrl+C to stop."
echo ""
streamlit run "$APP"
