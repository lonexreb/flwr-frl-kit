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
