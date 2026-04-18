#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/frontend"
BACKEND_DIR="${REPO_ROOT}/backend"
BACKEND_ENTRY="${BACKEND_DIR}/run.py"
RUNTIME_DIR="${REPO_ROOT}/runtime"
LOG_DIR="${RUNTIME_DIR}/logs"

BACKEND_PID_FILE="${RUNTIME_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUNTIME_DIR}/frontend.pid"
NEO4J_STARTED_FILE="${RUNTIME_DIR}/neo4j.started_by_script"

BACKEND_STDOUT_LOG="${LOG_DIR}/backend.stdout.log"
BACKEND_STDERR_LOG="${LOG_DIR}/backend.stderr.log"
FRONTEND_STDOUT_LOG="${LOG_DIR}/frontend.stdout.log"
FRONTEND_STDERR_LOG="${LOG_DIR}/frontend.stderr.log"
NEO4J_STDOUT_LOG="${LOG_DIR}/neo4j.stdout.log"
NEO4J_STDERR_LOG="${LOG_DIR}/neo4j.stderr.log"

NPM_BIN="${NPM_BIN:-npm}"
NEO4J_BIN="${NEO4J_BIN:-/root/autodl-tmp/med_llm/neo4j-community-5.26.24/bin/neo4j}"
NEO4J_HOME_VALUE="${NEO4J_HOME:-/root/autodl-tmp/med_llm/neo4j-community-5.26.24}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-base}"
HOST="${MED_HOST:-0.0.0.0}"
BACKEND_PORT="5000"
FRONTEND_PORT="${MED_FRONTEND_PORT:-5173}"
NEO4J_URI_VALUE="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER_VALUE="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD_VALUE="${NEO4J_PASSWORD:-ZYDzyd917917}"
PYTHON_BIN=""
BACKEND_PID=""
FRONTEND_PID=""

mkdir -p "${LOG_DIR}"

fail() {
  echo "$1" >&2
  exit 1
}

resolve_conda_sh() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local from_conda_exe
    from_conda_exe="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)/etc/profile.d/conda.sh"
    if [[ -f "${from_conda_exe}" ]]; then
      echo "${from_conda_exe}"
      return 0
    fi
  fi

  local candidate
  for candidate in \
    "/root/miniconda3/etc/profile.d/conda.sh" \
    "/root/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

activate_python_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" || fail "Failed to initialize conda shell hook."
  else
    local conda_sh
    conda_sh="$(resolve_conda_sh)" || fail "Cannot initialize conda automatically. Ensure conda is installed and accessible."
    # shellcheck disable=SC1090
    source "${conda_sh}"
  fi

  conda activate "${CONDA_ENV_NAME}" || fail "Failed to activate conda env: ${CONDA_ENV_NAME}"

  command -v python >/dev/null 2>&1 || fail "python not found after activating conda env: ${CONDA_ENV_NAME}"
  PYTHON_BIN="$(command -v python)"
}

resolve_neo4j_bin() {
  if [[ -n "${NEO4J_HOME_VALUE}" && -x "${NEO4J_HOME_VALUE}/bin/neo4j" ]]; then
    NEO4J_BIN="${NEO4J_HOME_VALUE}/bin/neo4j"
    return 0
  fi

  if [[ "${NEO4J_BIN}" == */* ]]; then
    [[ -x "${NEO4J_BIN}" ]] || fail "Neo4j command not executable: ${NEO4J_BIN}"
    return 0
  fi

  command -v "${NEO4J_BIN}" >/dev/null 2>&1 || fail "Neo4j command not found: ${NEO4J_BIN}"
  NEO4J_BIN="$(command -v "${NEO4J_BIN}")"
}

python_url_ready() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from urllib.request import urlopen

url = sys.argv[1]
try:
    with urlopen(url, timeout=2) as response:
        sys.exit(0 if 200 <= response.status < 500 else 1)
except Exception:
    sys.exit(1)
PY
}

port_is_listening() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

neo4j_is_ready() {
  "${PYTHON_BIN}" - "${NEO4J_URI_VALUE}" <<'PY'
import socket
import sys
from urllib.parse import urlparse

uri = sys.argv[1]
parsed = urlparse(uri)
host = parsed.hostname or "localhost"
port = parsed.port or 7687

sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

stop_pid_if_running() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    for _ in {1..10}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

cleanup_failed_start() {
  local reason="$1"
  echo "${reason}" >&2

  stop_pid_if_running "${FRONTEND_PID}"
  stop_pid_if_running "${BACKEND_PID}"
  rm -f "${FRONTEND_PID_FILE}" "${BACKEND_PID_FILE}"

  if [[ -f "${NEO4J_STARTED_FILE}" ]]; then
    "${NEO4J_BIN}" stop >> "${NEO4J_STDOUT_LOG}" 2>> "${NEO4J_STDERR_LOG}" || true
    rm -f "${NEO4J_STARTED_FILE}"
  fi

  exit 1
}

start_neo4j_if_needed() {
  if neo4j_is_ready; then
    rm -f "${NEO4J_STARTED_FILE}"
    echo "Neo4j is already running."
    return 0
  fi

  resolve_neo4j_bin

  echo "Starting Neo4j..."
  "${NEO4J_BIN}" start >> "${NEO4J_STDOUT_LOG}" 2>> "${NEO4J_STDERR_LOG}" || {
    tail -n 40 "${NEO4J_STDERR_LOG}" >&2 || true
    fail "Failed to start Neo4j."
  }

  for _ in {1..30}; do
    if neo4j_is_ready; then
      echo "started" > "${NEO4J_STARTED_FILE}"
      echo "Neo4j started successfully."
      return 0
    fi
    sleep 1
  done

  tail -n 40 "${NEO4J_STDERR_LOG}" >&2 || true
  fail "Neo4j did not become ready in time."
}

for PID_FILE in "${BACKEND_PID_FILE}" "${FRONTEND_PID_FILE}"; do
  if [[ -f "${PID_FILE}" ]]; then
    EXISTING_PID="$(tr -d '[:space:]' < "${PID_FILE}")"
    if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
      fail "A managed service is already running. Stop it first."
    fi
    rm -f "${PID_FILE}"
  fi
done

[[ -n "${NEO4J_PASSWORD_VALUE}" ]] || fail "NEO4J_PASSWORD is not set."
[[ -f "${BACKEND_ENTRY}" ]] || fail "Backend entry not found: ${BACKEND_ENTRY}"

activate_python_env
resolve_neo4j_bin

if port_is_listening "${BACKEND_PORT}"; then
  fail "Port ${BACKEND_PORT} is already in use. Stop the old backend process first."
fi

if port_is_listening "${FRONTEND_PORT}"; then
  fail "Port ${FRONTEND_PORT} is already in use. Stop the old frontend process first."
fi

export MED_HOST="${HOST}"
export MED_FRONTEND_PORT="${FRONTEND_PORT}"
export NEO4J_URI="${NEO4J_URI_VALUE}"
export NEO4J_USER="${NEO4J_USER_VALUE}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD_VALUE}"
export MED_SERVE_FRONTEND_DIST="0"

start_neo4j_if_needed

(
  cd "${BACKEND_DIR}"
  nohup "${PYTHON_BIN}" "${BACKEND_ENTRY}" >> "${BACKEND_STDOUT_LOG}" 2>> "${BACKEND_STDERR_LOG}" < /dev/null &
  echo $! > "${BACKEND_PID_FILE}"
)

(
  cd "${FRONTEND_DIR}"
  nohup "${NPM_BIN}" run dev -- --host "${HOST}" --port "${FRONTEND_PORT}" >> "${FRONTEND_STDOUT_LOG}" 2>> "${FRONTEND_STDERR_LOG}" < /dev/null &
  echo $! > "${FRONTEND_PID_FILE}"
)

BACKEND_PID="$(tr -d '[:space:]' < "${BACKEND_PID_FILE}")"
FRONTEND_PID="$(tr -d '[:space:]' < "${FRONTEND_PID_FILE}")"

[[ -n "${BACKEND_PID}" ]] || cleanup_failed_start "Backend PID file is empty."
[[ -n "${FRONTEND_PID}" ]] || cleanup_failed_start "Frontend PID file is empty."

for _ in {1..20}; do
  kill -0 "${BACKEND_PID}" 2>/dev/null || {
    tail -n 40 "${BACKEND_STDERR_LOG}" >&2 || true
    cleanup_failed_start "Backend exited during startup."
  }

  kill -0 "${FRONTEND_PID}" 2>/dev/null || {
    tail -n 40 "${FRONTEND_STDERR_LOG}" >&2 || true
    cleanup_failed_start "Frontend exited during startup."
  }

  neo4j_is_ready || {
    cleanup_failed_start "Neo4j became unavailable during startup."
  }

  if python_url_ready "http://127.0.0.1:${BACKEND_PORT}/api/health" && python_url_ready "http://127.0.0.1:${FRONTEND_PORT}"; then
    echo "Frontend, backend, and Neo4j started successfully."
    echo "Python: ${PYTHON_BIN}"
    echo "Neo4j command: ${NEO4J_BIN}"
    echo "Frontend PID: ${FRONTEND_PID}"
    echo "Backend PID: ${BACKEND_PID}"
    echo "Frontend: http://127.0.0.1:${FRONTEND_PORT}"
    echo "Backend:  http://127.0.0.1:${BACKEND_PORT}"
    echo "Frontend stdout log: ${FRONTEND_STDOUT_LOG}"
    echo "Frontend stderr log: ${FRONTEND_STDERR_LOG}"
    echo "Backend stdout log: ${BACKEND_STDOUT_LOG}"
    echo "Backend stderr log: ${BACKEND_STDERR_LOG}"
    echo "Backend entry: ${BACKEND_ENTRY}"
    echo "Neo4j stdout log: ${NEO4J_STDOUT_LOG}"
    echo "Neo4j stderr log: ${NEO4J_STDERR_LOG}"
    exit 0
  fi

  sleep 1
done

cleanup_failed_start "Timed out waiting for frontend/backend/Neo4j to become ready."
