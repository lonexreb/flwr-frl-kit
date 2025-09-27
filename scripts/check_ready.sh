#!/bin/sh
# check_ready.sh - Verify Federated RL + Flower client is ready to run

set -e

echo "=== Federated RL Kit Readiness Check ==="
echo

# Set PYTHONPATH
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
echo

# 1. Check required files
echo "1. Checking file presence:"
files="
packages/rl_core/client_runtime/a2c_client.py
packages/rl_core/client_runtime/flower_adapter.py
packages/rl_core/algos/a2c.py
packages/rl_core/algos/utils.py
packages/rl_core/nets/shared_backbone.py
packages/rl_core/nets/heads.py
packages/rl_core/envs/make_env.py
packages/rl_core/evaluate/harness.py
client_entry.py
"

all_files_exist=true
for file in $files; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file"
        all_files_exist=false
    fi
done

# Check optional file
if [ -f "packages/rl_core/scripts/train_local.py" ]; then
    echo "  ✅ packages/rl_core/scripts/train_local.py (optional)"
else
    echo "  ⚠️  packages/rl_core/scripts/train_local.py (optional, missing)"
fi
echo

# 2. Test imports
echo "2. Testing imports:"
python3 -c "
try:
    from packages.rl_core.client_runtime.a2c_client import A2CClient, A2CConfig
    from packages.rl_core.client_runtime.flower_adapter import FlowerClientAdapter
    print('  ✅ Core imports successful')
except Exception as e:
    print(f'  ❌ Import error: {e}')
    exit(1)
"
echo

# 3. Test client instantiation and metrics schema
echo "3. Testing client instantiation and metrics schema:"
python3 -c "
import numpy as np
from packages.rl_core.client_runtime.a2c_client import A2CClient, A2CConfig
from packages.rl_core.client_runtime.flower_adapter import FlowerClientAdapter

# Create client
cfg = A2CConfig(env_id='CartPole-v1', seed=42, rollout_len=128)
client = A2CClient(cfg)
print('  ✅ A2CClient created')

# Test train_for metrics
train_metrics = client.train_for(256)
required_train_keys = {'steps', 'entropy', 'kl', 'loss', 'policy_loss', 'value_loss'}
print(f'  Train metrics: {train_metrics}')
if required_train_keys.issubset(train_metrics.keys()):
    print('  ✅ train_for() metrics schema correct')
else:
    missing = required_train_keys - set(train_metrics.keys())
    print(f'  ❌ train_for() missing keys: {missing}')
    exit(1)

# Test evaluate metrics
eval_metrics = client.evaluate(2)
required_eval_keys = {'avg_return', 'std_return', 'episodes'}
print(f'  Eval metrics: {eval_metrics}')
if required_eval_keys.issubset(eval_metrics.keys()):
    print('  ✅ evaluate() metrics schema correct')
else:
    missing = required_eval_keys - set(eval_metrics.keys())
    print(f'  ❌ evaluate() missing keys: {missing}')
    exit(1)

# Test Flower adapter
adapter = FlowerClientAdapter(client, round_train_steps=256, eval_episodes=2)
print('  ✅ FlowerClientAdapter created')
"
echo

# 4. Run local training if available
if [ -f "packages/rl_core/scripts/train_local.py" ]; then
    echo "4. Running local training test:"
    export ENV_ID=CartPole-v1
    export TOTAL_STEPS=2048
    export TICK=2048
    export ROLLOUT_LEN=128
    
    # Run training
    echo "  Running train_local.py..."
    if python3 -m packages.rl_core.scripts.train_local > /dev/null 2>&1; then
        echo "  ✅ Local training completed"
    else
        echo "  ❌ Local training failed"
    fi
    
    # Check if checkpoint was created
    if [ -f "checkpoints/a2c_step_2048.pt" ]; then
        echo "  ✅ Checkpoint created: checkpoints/a2c_step_2048.pt"
        
        # Run evaluation
        echo "  Running evaluation harness..."
        if python3 -m packages.rl_core.evaluate.harness --env_id CartPole-v1 --ckpt checkpoints/a2c_step_2048.pt --episodes 5 > /dev/null 2>&1; then
            echo "  ✅ Evaluation completed"
        else
            echo "  ❌ Evaluation failed"
        fi
    else
        echo "  ⚠️  No checkpoint found (expected: checkpoints/a2c_step_2048.pt)"
    fi
else
    echo "4. Skipping local training test (train_local.py not found)"
fi
echo

# 5. Test client entry startup
echo "5. Testing client_entry.py startup:"
export FLOWER_SERVER=127.0.0.1:8080
export ENV_ID=CartPole-v1
export SEED=1
export ROUND_STEPS=256
export EVAL_EPISODES=2

echo "  Starting client_entry.py (will timeout after 2 seconds)..."
timeout 2 python3 client_entry.py > /dev/null 2>&1 &
CLIENT_PID=$!

sleep 2
if kill -0 $CLIENT_PID 2>/dev/null; then
    kill $CLIENT_PID 2>/dev/null
    echo "  ✅ Client started successfully (killed after timeout)"
    echo "  ℹ️  Note: A real Flower server at 127.0.0.1:8080 is required for actual federated training"
else
    echo "  ❌ Client failed to start"
fi
echo

# Summary
echo "=== SUMMARY ==="
if [ "$all_files_exist" = true ]; then
    echo "✅ All required files present"
else
    echo "❌ Some required files missing"
fi

echo
echo "=== NEXT ACTIONS ==="
echo "1. If any tests failed, fix the issues and run this script again"
echo "2. To start federated training:"
echo "   a) Start the Flower server: cd apps/orchestrator && python server.py"
echo "   b) Start clients: python client_entry.py (with appropriate env vars)"
echo "3. Monitor training via the backend API and frontend dashboard"
echo

# Final status
if [ "$all_files_exist" = true ]; then
    echo "STATUS: PASS ✅"
    exit 0
else
    echo "STATUS: FAIL ❌"
    exit 1
fi
