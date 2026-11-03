#!/usr/bin/env bash
# ==============================================================================
# Causal Feature Attribution & Actionable Recourse - Unified Launcher Script
# ==============================================================================

set -e

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python virtualenv
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
    UVICORN_BIN=".venv/bin/uvicorn"
    STREAMLIT_BIN=".venv/bin/streamlit"
    PYTEST_BIN=".venv/bin/pytest"
else
    PYTHON_BIN="python3"
    UVICORN_BIN="uvicorn"
    STREAMLIT_BIN="streamlit"
    PYTEST_BIN="pytest"
fi

show_help() {
    echo "================================================================================"
    echo "  Causal Feature Attribution & Actionable Recourse Engine"
    echo "================================================================================"
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Available Commands:"
    echo "  test        Run complete automated test suite (pytest)"
    echo "  api         Start the FastAPI REST service on http://localhost:8000"
    echo "  dashboard   Start the Streamlit interactive analyst UI on http://localhost:8501"
    echo "  all         Start both FastAPI backend and Streamlit dashboard concurrently"
    echo "  docker-up   Build and launch containerized services via docker-compose"
    echo "  docker-down Stop docker containers"
    echo "  help        Display this help message"
    echo "================================================================================"
}

run_test() {
    echo "--> Running automated test suite with $PYTEST_BIN..."
    $PYTEST_BIN tests -v
}

run_api() {
    echo "--> Starting FastAPI REST Microservice on http://0.0.0.0:8000..."
    $UVICORN_BIN module_4_decision_dashboard.api.main:app --host 0.0.0.0 --port 8000 --reload
}

run_dashboard() {
    echo "--> Starting Streamlit Interactive Dashboard on http://localhost:8501..."
    $STREAMLIT_BIN run module_4_decision_dashboard/dashboard/app.py --server.port 8501
}

run_all() {
    echo "--> Launching FastAPI REST API and Streamlit Dashboard concurrently..."
    $UVICORN_BIN module_4_decision_dashboard.api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    echo "    [FastAPI started with PID: $API_PID]"

    trap "echo 'Shutting down services...'; kill $API_PID 2>/dev/null; exit 0" SIGINT SIGTERM

    echo "    Waiting 2 seconds for API warmup..."
    sleep 2

    $STREAMLIT_BIN run module_4_decision_dashboard/dashboard/app.py --server.port 8501
}

run_docker_up() {
    echo "--> Building and starting docker containers..."
    docker-compose up --build
}

run_docker_down() {
    echo "--> Stopping docker containers..."
    docker-compose down
}

case "$1" in
    test)
        run_test
        ;;
    api)
        run_api
        ;;
    dashboard)
        run_dashboard
        ;;
    all)
        run_all
        ;;
    docker-up)
        run_docker_up
        ;;
    docker-down)
        run_docker_down
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
