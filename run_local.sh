#!/bin/bash

# Federated RL Local Test Script
# Runs server and multiple clients locally for testing

set -e

# Default configuration
NUM_CLIENTS=2
NUM_ROUNDS=3
ENV_ID="CartPole-v1"
SERVER_ADDRESS="localhost:8080"
STEPS_PER_ROUND=500
EVAL_EPISODES=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Federated RL Local Test ===${NC}"
echo "Configuration:"
echo "  - Clients: $NUM_CLIENTS"
echo "  - Rounds: $NUM_ROUNDS"
echo "  - Environment: $ENV_ID"
echo "  - Steps per round: $STEPS_PER_ROUND"
echo "  - Eval episodes: $EVAL_EPISODES"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${GREEN}✓ Using uv for dependency management${NC}"

# Check if required packages are installed
echo "Checking dependencies..."
uv run python -c "
import sys
try:
    import torch, flwr, gymnasium, numpy
    print('✓ All required packages are installed')
except ImportError as e:
    print(f'✗ Missing packages: {e}')
    print('Installing dependencies from requirements.txt...')
    import subprocess
    subprocess.check_call(['uv', 'pip', 'install', '-r', 'requirements.txt'])
    print('✓ Dependencies installed')
" || {
    echo -e "${RED}Failed to install dependencies. Please run manually:${NC}"
    echo "  uv pip install -r requirements.txt"
    exit 1
}

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Cleaning up processes...${NC}"
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Change to project directory
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

echo -e "${BLUE}Starting Flower server...${NC}"

# Start server in background
uv run python apps/orchestrator/server.py \
    --server-address $SERVER_ADDRESS \
    --rounds $NUM_ROUNDS \
    --min-clients $NUM_CLIENTS \
    --env-id $ENV_ID \
    --steps-per-round $STEPS_PER_ROUND \
    --eval-episodes $EVAL_EPISODES \
    --log-level INFO &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait a bit for server to start
echo "Waiting for server to start..."
sleep 3

# Start clients
echo -e "${BLUE}Starting $NUM_CLIENTS clients...${NC}"

CLIENT_PIDS=()
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    echo "Starting client $i..."

    uv run python packages/rl_core/client_runtime/client_main.py \
        --server-address $SERVER_ADDRESS \
        --client-id $i \
        --env-id $ENV_ID \
        --steps-per-round $STEPS_PER_ROUND \
        --eval-episodes $EVAL_EPISODES \
        --dummy-weight-update \
        > "client_${i}.log" 2>&1 &

    CLIENT_PID=$!
    CLIENT_PIDS+=($CLIENT_PID)
    echo "Client $i PID: $CLIENT_PID"

    # Stagger client starts
    sleep 1
done

echo -e "${GREEN}All clients started${NC}"
echo ""

# Wait for server to complete
echo -e "${BLUE}Waiting for federated learning to complete...${NC}"
echo "This may take several minutes depending on your configuration."
echo ""

# Monitor the processes
wait $SERVER_PID
SERVER_EXIT_CODE=$?

if [ $SERVER_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}=== Federated learning completed successfully! ===${NC}"
else
    echo -e "${RED}=== Server exited with error code $SERVER_EXIT_CODE ===${NC}"
fi

# Wait a bit for clients to finish
echo "Waiting for clients to finish..."
sleep 2

# Check client status and show logs
echo "Client status:"
for i in "${!CLIENT_PIDS[@]}"; do
    PID=${CLIENT_PIDS[$i]}
    if kill -0 $PID 2>/dev/null; then
        echo "  Client $i (PID $PID): still running"
    else
        wait $PID 2>/dev/null
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "  Client $i (PID $PID): ${GREEN}completed${NC}"
        else
            echo -e "  Client $i (PID $PID): ${RED}failed ($EXIT_CODE)${NC}"
            echo -e "  ${YELLOW}Client $i error log:${NC}"
            if [ -f "client_${i}.log" ]; then
                tail -10 "client_${i}.log" | sed 's/^/    /'
            fi
        fi
    fi
done

echo ""
echo -e "${BLUE}=== Local test finished ===${NC}"

# Exit with server's exit code
exit $SERVER_EXIT_CODE