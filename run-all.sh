#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/venv"
BACKEND_REQ="$REPO_ROOT/captcha-system/backend-service-requirements.txt"
MODEL_REQ="$REPO_ROOT/captcha-system/model-service-requirements.txt"
ATTACKER_REQ="$REPO_ROOT/attackers/requirements.txt"
LOG_DIR="$REPO_ROOT/logs"
MODEL_PORT=8001
BACKEND_PORT=5174

# The script simply performs checks, installs deps and launches both services in parallel.
usage() {
  echo -e "Usage: $0 [--help]\n"
  echo "This helper will create a venv (if missing), install dependencies, then launch both services in parallel."
  echo "Options:"
  echo "  --help, -h     Show this message"
  exit 1
}

if [ "$#" -gt 0 ]; then
  case "$1" in
    --help|-h) usage ;; 
    *) echo "Unknown option: $1"; usage ;; 
  esac
fi

echo -e "\033[1;34m🔧 Repository root:\033[0m $REPO_ROOT"

echo -e "\n\033[1;34m🔧 1) Check Python is available\033[0m"
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found on PATH. Install Python 3.8+ and try again."
  exit 1
fi

echo -e "\n\033[1;34m🔧 2) Create venv if missing\033[0m"
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
  echo "Created venv: $VENV_DIR"
else
  echo "Using existing venv: $VENV_DIR"
fi

# Prefer using the venv python when available; otherwise use a command helper (python3/python)
if [ -f "$VENV_DIR/bin/python" ]; then
  PY_CMD="$VENV_DIR/bin/python"
elif [ -f "$VENV_DIR/Scripts/python.exe" ]; then
  PY_CMD="$VENV_DIR/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PY_CMD="python"
else
  echo "ERROR: Could not find a usable Python interpreter (python3 or python)."
  exit 1
fi
echo -e "\033[1;32m✅ Using Python command:\033[0m $PY_CMD"

echo -e "\n\033[1;34m🔧 3) Basic pip check (using the chosen python)\033[0m"
if ! "$PY_CMD" -m pip --version >/dev/null 2>&1; then
  echo "ERROR: pip appears unavailable for $PY_CMD; please ensure pip is installed or use ensurepip: $PY_CMD -m ensurepip";
  exit 1
fi

echo -e "\n\033[1;34m🔧 4) Install dependencies (backend + model)\033[0m"
mkdir -p "$LOG_DIR"
if [ -f "$BACKEND_REQ" ]; then
  echo "Installing backend requirements..."
  "$PY_CMD" -m pip install -r "$BACKEND_REQ"
else
  echo "WARNING: Backend requirements file not found: $BACKEND_REQ"
fi
if [ -f "$MODEL_REQ" ]; then
  echo "Installing model microservice requirements..."
  "$PY_CMD" -m pip install -r "$MODEL_REQ"
else
  echo "WARNING: Model requirements file not found: $MODEL_REQ"
fi
if [ -f "$ATTACKER_REQ" ]; then
  echo "Installing optional attacker requirements..."
  "$PY_CMD" -m pip install -r "$ATTACKER_REQ" || true
  echo "To install Playwright browsers: $PY_CMD -m playwright install chromium"
fi

check_port_free() {
  local port="$1"
  # Try lsof -> ss -> netstat fallback
  if command -v lsof >/dev/null 2>&1; then
    if lsof -i :"$port" >/dev/null 2>&1; then
      return 1
    else
      return 0
    fi
  elif command -v ss >/dev/null 2>&1; then
    if ss -ltn | grep -q ":$port\b"; then
      return 1
    else
      return 0
    fi
  elif command -v netstat >/dev/null 2>&1; then
    if netstat -an | grep -q "\.$port\b\|:$port\b"; then
      return 1
    else
      return 0
    fi
  else
    echo "NOTE: Couldn't check if port $port is free (no lsof/ss/netstat). Proceeding anyway."
    return 0
  fi
}

echo -e "\n\033[1;34m🔍 5) Basic port checks\033[0m"
if check_port_free "$MODEL_PORT"; then
  echo "Model port $MODEL_PORT appears free"
else
  echo "WARNING: Model port $MODEL_PORT is in use; you may need to stop the other service"
fi
if check_port_free "$BACKEND_PORT"; then
  echo "Backend port $BACKEND_PORT appears free"
else
  echo "WARNING: Backend port $BACKEND_PORT is in use; you may need to stop the other service"
fi

echo -e "\n\033[1;34mℹ️ 6) Starting both services in parallel (Ctrl+C will stop them):\033[0m"
echo "Model microservice will listen on port $MODEL_PORT, backend on port $BACKEND_PORT. Logs are written to $LOG_DIR."

echo -e "\n\033[1;32m▶️  Launching model microservice (background)\033[0m"
pushd "$REPO_ROOT/captcha-system" >/dev/null
"$PY_CMD" model_service.py >"$LOG_DIR/model_service.log" 2>&1 &
MODEL_PID=$!
popd >/dev/null

echo -e "\n\033[1;32m▶️  Launching backend service (background)\033[0m"
pushd "$REPO_ROOT/captcha-system" >/dev/null
"$PY_CMD" main.py >"$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
popd >/dev/null

cleanup() {
  echo -e "\n\033[1;31m⏹️  Stopping services...\033[0m"
  [ -n "${MODEL_PID-}" ] && kill -TERM "$MODEL_PID" 2>/dev/null || true
  [ -n "${BACKEND_PID-}" ] && kill -TERM "$BACKEND_PID" 2>/dev/null || true
  wait "$MODEL_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  echo -e "\033[1;32m✅ Services stopped.\033[0m"
  exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "\n\033[1;33m📌 Running both services. Press Ctrl+C to stop both.\033[0m"
echo "Model PID: $MODEL_PID; Backend PID: $BACKEND_PID"
echo "Logs: $LOG_DIR/model_service.log and $LOG_DIR/backend.log"

# Wait for both services to exit; trap handles SIGINT to shutdown gracefully
wait "$MODEL_PID" "$BACKEND_PID"
EXIT_CODE=$?
echo -e "\n\033[1;33mℹ️  Services exited with code $EXIT_CODE\033[0m"

# End
