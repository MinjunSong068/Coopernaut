# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coopernaut (UT-Austin-RPL, CVPR 2022): an end-to-end driving model that uses cross-vehicle
(V2V) LiDAR-based cooperative perception. Training and evaluation both run against a live
CARLA simulator via **AutoCastSim** (`AutoCastSim/`, a git submodule — a fork of CARLA's
`scenario_runner` with V2V networking, cooperative-perception scenarios, and latency/HUD
instrumentation layered on top). There is no static offline dataset path for training from
scratch: behavior cloning and DAgger both collect rollouts from the live simulator first.

## Running things

Everything that touches CARLA needs a running simulator first, and the CARLA **client and
server versions must match exactly** — this codebase was originally written against CARLA
0.9.11, and its Python API assumptions (map name format, vehicle blueprint IDs, the
`agents` navigation helper package shape) do not all hold on newer CARLA versions. See
`dockerfiles/README.md`'s "Known 0.9.11 → 0.9.15 breakages" section for the specific
failure modes and fixes if you're running against anything newer than 0.9.11.

### Docker (recommended)

`dockerfiles/` has a two-container Compose setup: `carla-server` (official `carlasim/carla`
image, does the simulation) and `coopernaut` (this repo's training/client container, bind-mounts
your local `Coopernaut`/`AutoCastSim` checkouts rather than baking source into the image, so host
edits take effect immediately with no rebuild). Read `dockerfiles/README.md` before touching
this — it documents the version-pinning gotchas, the `CARLA_VERSION`/`CARLA_AGENTS_VERSION`
build args, and per-frame timing instrumentation.

```bash
cd dockerfiles
docker compose build coopernaut   # or: docker build --builder default --build-arg CARLA_VERSION=0.9.15 -t coopernaut:latest .
docker compose up -d carla-server
docker compose run --rm coopernaut
```

Prefer `docker build --builder default ...` (the classic docker-driver builder) over
`docker compose build` / the default `docker buildx` container-driver builder when
`carlasim/carla` images and prior `coopernaut` layers are already present locally —
the container-driver builder has its own isolated image/layer cache and will silently
redownload multi-GB CARLA images and rebuild from scratch instead of reusing them.

### Non-Docker (conda)

`docs/INSTALL.md` / `scripts/install.sh` (CUDA 11.0) / `scripts/install-cu113.sh` (CUDA 11.3)
set up an `autocast` conda env (Python 3.7, PyTorch 1.7.1, MinkowskiEngine, CARLA 0.9.11
client). `scripts/env.sh` shows the required `CARLA_ROOT`/`SCENARIO_RUNNER_ROOT`/`PYTHONPATH`
env vars — `PYTHONPATH` must include `${CARLA_ROOT}/PythonAPI/carla/agents` (CARLA's own
navigation-helper package; it is not part of this repo).

### The pipeline scripts

Root-level `run_*.sh` scripts are the main entry points, one per model variant:

| Method | Script |
| :----------- | :----------------- |
| Coopernaut (point transformer + V2V fusion) | `run_cooperative_point_transformer.sh` |
| No V2V sharing | `run_point_no_fusion.sh` |
| Early fusion | `run_earlyfusion_point_transformer.sh` |
| Voxel GNN | `run_v2v.sh` |

Each is interactive: it prompts for a mode (`data-train`/`data-val`/`data-expert`/`bc`/`dagger`/`eval`)
and a scene number (6=Overtake, 8=Left turn, 10=Red light violation), then launches CARLA workers
itself (`scripts/launch_carla.sh`) and drives `AutoCastSim/parallel_scenario_runner.py` (data
collection/DAgger) or `parallel_evaluation.py` (eval). All paths you'd actually want to change
(`TrainValFolder`, `DATAFOLDER`, `CHECKPOINT`, `RUN`, `CARLA_WORKERS`) are hardcoded at the top of
each script, not passed as CLI args — edit the script directly. `scripts/quick_run.sh` is the
fastest way to sanity-check an eval run end-to-end.

There is no unit test suite; correctness is validated by actually running a scenario against a
live CARLA instance (`test_run_cooperative_point_transformer.sh` is a distributed-training
variant of the main pipeline script used for this, not an automated test).

## Architecture

**Simulation/scenario layer** (`AutoCastSim/`, submodule): forked CARLA `scenario_runner`.
`scenario_runner.py` is the single-process entry point; `parallel_scenario_runner.py` /
`parallel_evaluation.py` fan it out across multiple CARLA worker processes/ports via `ray`.
`srunner/scenariomanager/scenario_manager.py`'s `run_scenario()` is the actual per-frame tick
loop (sensor update → collaborator/V2V tick → scenario tick → HUD) and is also where
per-frame wall-clock latency gets recorded (`record_latency.py`'s `RecordLatency`, dumped to
`latency_scenario.csv`). `AVR/` holds the V2V simulation layer: `Comm.py`/`BeaconList.py`
(message bus over mosquitto/MQTT), `Collaborator.py` (per-vehicle V2V state), `PCProcess.py`
(LiDAR point-cloud → BEV/occupancy-grid processing), `HUD.py`/`Visualizer.py`. Rule-based
scenario agents live in `AVR/autocast_agents/` (`new_agent.py`'s `NewAgent` is the base class
most agents build on, itself a subclass of CARLA's own `agents.navigation.agent.Agent`).

**CARLA's own `agents` navigation helper package** (top-level `agents/` inside `AutoCastSim/`)
is *not* AutoCastSim source — it's CARLA's PythonAPI helper code (route planning, basic/behavior
agents), expected to come from whatever CARLA version you're running against. It's untracked in
git. If imports like `agents.navigation.agent` fail or `GlobalRoutePlanner` raises a
`TypeError` on argument count, this directory is stale/mismatched relative to your CARLA
server version — see `dockerfiles/README.md`.

**Model layer**: `models/` has the network architectures (point transformer, sparse conv via
MinkowskiEngine, voxel/V2V-GNN fusion). `NeuralAgents/` wraps a trained model into AutoCastSim's
agent interface for eval/DAgger (`dagger_agent.py`, `Lidar*Agent.py` per architecture) — this is
the bridge between the model layer and the simulation layer. `training/` has one train script per
method/variant, including `*_distributed.py` counterparts for multi-GPU/multi-worker training.
`utils/` has the corresponding data loaders.

**Data flow**: pipeline script → `parallel_scenario_runner.py` spins up CARLA workers and drives
scenarios with a rule-based expert agent (data collection) or a `NeuralAgents` model (DAgger) →
rollouts land under `data/AutoCast_<scene>[_Small]` → `training/train_*` consumes that data →
checkpoints land under `ckpts/` → `parallel_evaluation.py` runs a trained checkpoint through
`NeuralAgents` against held-out scenarios → results/trajectories land under `result/`, analyzed
via `scripts/result_analysis.py` / `scripts/compare_trajectory.py`.
