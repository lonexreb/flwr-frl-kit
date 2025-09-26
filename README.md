## Federated RL Kit

A reference stack for building and experimenting with federated reinforcement learning. It combines a web UI, a lightweight backend for orchestration and metrics, and a Flower-based federated server with RL-specific strategies and client runtime utilities.

### Features
- Federated RL orchestration with Flower
- RL algorithm scaffolds (A2C, SAC) and network heads/backbones
- Backend API surface for runs, clients, metrics, and control
- Frontend components for dashboards and policy exploration
- Deployment scaffolding for local and containerized runs

### Requirements

#### System Requirements
- **Python**: 3.10+ (required for modern type hints and dependencies)
- **Node.js**: 18+ LTS (for frontend development)
- **Git**: For version control
- **Docker**: 24+ (optional, for containerized deployment)

#### Python Dependencies
Core packages needed for the RL components (verified from virtual environment):
```
torch==2.8.0           # Deep learning framework (latest)
gymnasium==1.2.1       # RL environment interface
numpy==1.26.4          # Numerical computing
flwr==1.10.0           # Federated learning framework
```

Configuration and data handling:
```
hydra-core==1.3.0      # Configuration management
omegaconf==2.3.0       # Configuration files
pydantic==2.11.9       # Data validation and serialization
PyYAML==6.0.3          # YAML configuration support
```

Additional dependencies (automatically installed):
```
grpcio==1.75.1         # RPC communication for Flower
protobuf==4.25.8       # Protocol buffers
cloudpickle==3.1.1     # Serialization
cryptography==42.0.8   # Security features
rich==13.9.4           # Beautiful terminal output
typer==0.9.4           # CLI framework
```

**Complete requirements.txt** has been generated and is available in the repository root.

#### Frontend Dependencies
The frontend uses Next.js 15 with modern React:
```json
{
  "next": "15.5.4",
  "react": "19.1.0", 
  "react-dom": "19.1.0",
  "three": "^0.180.0",
  "three-globe": "^2.44.0",
  "typescript": "^5",
  "tailwindcss": "^4"
}
```

#### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/flwr-frl-kit.git
   cd flwr-frl-kit
   ```

2. **Set up Python environment**:
   ```bash
   # Create virtual environment
   python -m venv flwr-frl
   source flwr-frl/bin/activate  # On Windows: flwr-frl\Scripts\activate
   
   # Install all dependencies from requirements.txt
   pip install -r requirements.txt
   
   # Or install core dependencies manually:
   # pip install torch gymnasium numpy flwr hydra-core omegaconf pydantic
   ```

3. **Set up frontend**:
   ```bash
   cd apps/frontend
   npm install
   ```

4. **Optional: Docker setup**:
   ```bash
   cd deploy
   docker compose up --build
   ```

### Repository structure
```
federated-rl/
├─ apps/
│  ├─ frontend/                # Luis
│  │  ├─ src/
│  │  │  ├─ pages/             # Next.js routes or React Router
│  │  │  ├─ components/        # Charts, StatusCards, PolicyPlayground
│  │  │  ├─ lib/               # API client, sockets
│  │  │  └─ styles/
│  │  └─ package.json
│  ├─ backend/                 # FastAPI/Express for REST + WS + auth
│  │  ├─ app/
│  │  │  ├─ api/
│  │  │  │  ├─ runs.py         # list/create runs, metadata
│  │  │  │  ├─ metrics.py      # ingest/query metrics
│  │  │  │  ├─ clients.py      # client status
│  │  │  │  ├─ control.py      # pause/resume/deploy
│  │  │  │  └─ policies.py     # download policy checkpoints
│  │  │  ├─ sockets.py         # live round updates
│  │  │  └─ storage.py         # S3/minio helpers
│  │  └─ pyproject.toml
│  └─ orchestrator/            # Phil (Flower server)
│     ├─ server.py             # start Flower, wires strategy + hooks
│     ├─ strategy_rl.py        # aggregate (actor, critic), staleness logic
│     ├─ secure_agg.py         # stub -> plug real secure aggregation
│     ├─ registry.yaml         # clients/capabilities
│     └─ pyproject.toml
├─ packages/
│  ├─ rl_core/                 # Shubh
│  │  ├─ algos/                # a2c.py, sac.py, utils.py
│  │  ├─ nets/                 # shared_backbone.py, heads.py
│  │  ├─ envs/                 # wrappers for Gymnasium, custom costs
│  │  ├─ client_runtime/       # FlowerClientAdapter, RLClient base
│  │  ├─ evaluate/             # eval loops, logging
│  │  └─ config/               # hydra/omegaconf configs
│  └─ common/                  # schemas, logging, metrics helpers
│     ├─ metrics_schema.py
│     ├─ serialization.py      # (de)serialize weights/opt state
│     └─ types.py
├─ deploy/
│  ├─ docker/                  # Dockerfiles for frontend/backend/client/server
│  ├─ compose.yaml             # N clients + server + backend + minio + redis
│  └─ k8s/                     # optional manifests
├─ scripts/
│  ├─ run_local.sh             # 1 server + 4 clients
│  ├─ seed_eval.sh
│  └─ export_policy.py
├─ tests/
│  └─ ...                      # unit & integration tests
└─ README.md
```

### Contributing
- Propose changes via PRs with a clear description and rationale
- Keep edits focused; include tests where relevant
- Match existing code style and formatting

### Roadmap (high-level)
- Secure aggregation integration
- Expanded strategy variants for RL aggregation
- More evaluation tooling and dashboards
