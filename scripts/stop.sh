#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_PID_FILE="${REPO_ROOT}/runtime/backend.pid"
FRONTEND_PID_FILE="${REPO_ROOT}/runtime/frontend.pid"
NEO4J_STARTED_FILE="${REPO_ROOT}/runtime/neo4j.started_by_script"
LOG_DIR="${REPO_ROOT}/runtime/logs"
NEO4J_STDOUT_LOG="${LOG_DIR}/neo4j.stdout.log"
NEO4J_STDERR_LOG="${LOG_DIR}/neo4j.stderr.log"
NEO4J_BIN="${NEO4J_BIN:-neo4j}"
NEO4J_HOME_VALUE="${NEO4J_HOME:-}"
STOPPED_ANY=0

resolve_neo4j_bin() {
  if [[ -n "${NEO4J_HOME_VALUE}" && -x "${NEO4J_HOME_VALUE}/bin/neo4j" ]]; then
    NEO4J_BIN="${NEO4J_HOME_VALUE}/bin/neo4j"
    return 0
  fi

  if [[ "${NEO4J_BIN}" == */* ]]; then
    [[ -x "${NEO4J_BIN}" ]]
    return $?
  fi

  if command -v "${NEO4J_BIN}" >/dev/null 2>&1; then
    NEO4J_BIN="$(command -v "${NEO4J_BIN}")"
    return 0
  fi

  return 1
}

stop_one() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi

  local pid
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
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

  rm -f "${pid_file}"
  echo "${name} stopped. PID=${pid}"
  STOPPED_ANY=1
}

stop_one "Frontend" "${FRONTEND_PID_FILE}"
stop_one "Backend" "${BACKEND_PID_FILE}"

if [[ -f "${NEO4J_STARTED_FILE}" ]]; then
  if resolve_neo4j_bin; then
    "${NEO4J_BIN}" stop >> "${NEO4J_STDOUT_LOG}" 2>> "${NEO4J_STDERR_LOG}" || true
    echo "Neo4j stopped."
    STOPPED_ANY=1
  else
    echo "Neo4j marker exists but command not found: ${NEO4J_BIN}" >&2
  fi
  rm -f "${NEO4J_STARTED_FILE}"
fi

if [[ "${STOPPED_ANY}" -eq 0 ]]; then
  echo "No running frontend, backend, or managed Neo4j found."
fi
