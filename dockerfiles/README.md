# Coopernaut Training Docker

Docker setup for training [UT-Austin-RPL/Coopernaut](https://github.com/UT-Austin-RPL/Coopernaut) —
"COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles" (CVPR 2022).

## Architecture: two containers

CARLA runs as its **own container**, separate from the training/client container —
the same pattern you're likely already using to run CARLA via Docker:

- **`carla-server`** — the official `carlasim/carla:0.9.15` image. Does the actual
  Unreal Engine simulation/rendering.
- **`coopernaut`** — this repo's training container. Conda env, PyTorch, MinkowskiEngine,
  and the CARLA **Python client** (extracted from the matching server image at build
  time, not the full simulator). Connects to `carla-server` over the network as an RPC
  client.

This keeps the training image small and lets you swap the CARLA server independently.

## ⚠️ Client/server version must match — this setup targets 0.9.15

Coopernaut and AutoCastSim were originally built against **CARLA 0.9.11**, but this
setup is pinned to **0.9.15** throughout (`Dockerfile`'s `CARLA_VERSION` build arg,
`docker-compose.yml`'s `carla-server` image, `start_carla.sh`'s default), since that's
the version available here. The CARLA RPC protocol and Python API have changed enough
across 0.9.11 → 0.9.15 that this is not guaranteed to work — this is a very commonly
reported CARLA gotcha, independent of anything Coopernaut-specific.

The client/server *pairing* is still guaranteed to match: the `coopernaut` image's
multi-stage build extracts the CARLA Python client `.egg` straight out of whatever
`carlasim/carla:${CARLA_VERSION}` server image you build against (0.9.15 ships a
`py3.7` egg, same as 0.9.11 did), so client and server are always byte-for-byte the
same version — it's the *Coopernaut/AutoCastSim application code's* assumptions about
the 0.9.11 API surface that may not hold on 0.9.15.

**Known 0.9.11 → 0.9.15 breakages, already fixed in this checkout** (found by actually
running a scenario end-to-end against a dockerized 0.9.15 server):

- **`agents` navigation helper package.** CARLA restructured this starting around
  0.9.13 -- `agents/navigation/agent.py` and `global_route_planner_dao.py` are gone,
  and `GlobalRoutePlanner.__init__` no longer takes a DAO object. AutoCastSim's agent
  code (`new_agent.py`) still needs the old shape (it subclasses `Agent` and uses
  `AgentState`, which no longer exist at all in 0.9.15). Fixed by pulling an
  0.9.11-vintage copy of just this helper package into the image (see "What's in the
  image" below) *and* replacing `AutoCastSim/agents/` on the host with the same
  0.9.11-vintage copy -- the image's copy alone isn't enough because Python puts the
  running script's own directory ahead of `PYTHONPATH`, so a stray `agents/` sitting
  next to `scenario_runner.py` always wins.
- **Map name format.** `CarlaDataProvider.get_map().name` returns `"Carla/Maps/Town03"`
  on 0.9.15 instead of the bare `"Town03"` this code compares against. Fixed in
  `scenario_runner.py`'s `_load_and_wait_for_world` and
  `srunner/scenarios/basic_scenario.py`'s `_check_town` to compare by basename.
- **Renamed vehicle blueprints.** `vehicle.lincoln.mkz2017` (the ego vehicle) and
  `vehicle.dodge_charger.police` no longer exist as blueprint IDs; 0.9.15 has
  `vehicle.lincoln.mkz_2017` and `vehicle.dodge.charger_police`. Fixed in
  `srunner/scenarios/route_scenario.py` and `srunner/scenarios/autocast_test.py`. If
  you hit `ValueError: 'a' cannot be empty unless no samples are taken` from
  `CarlaActorPool.create_blueprint`, that's this class of bug -- grep for the
  `'vehicle.*'` blueprint string in the traceback and check it against
  `world.get_blueprint_library().filter('vehicle.*')` on your server version.

Two more bugs surfaced during this same test run that are **not** CARLA-version-related
(pre-existing/unrelated to 0.9.15), also fixed:
- `scenario_runner.py` called `self.manager.run_scenario()` with no arguments, but
  `ScenarioManager.run_scenario()` requires the `recordlatency` object used for the
  per-frame timing CSV (see "Per-frame timing" below) -- this would crash before a
  single frame ran, regardless of CARLA version.
- `_load_and_wait_for_world` unconditionally printed `eval_seed` even when running
  without `--eval` (route mode), where it's never assigned --
  `UnboundLocalError`. Fixed by moving the print inside the `if self._args.eval` guard.

If you hit further breakage in agent/sensor code beyond the above, that's the same
class of problem -- validate incrementally (connect + tick the world) before trusting
results for actual data collection or training. To instead fall back to the version
Coopernaut was originally built against (sidesteps all of the above, but you lose the
0.9.15 requirement):
```bash
docker pull carlasim/carla:0.9.11
docker build --build-arg CARLA_VERSION=0.9.11 -t coopernaut:latest .
# and in docker-compose.yml, change carla-server's image back to carlasim/carla:0.9.11
# and build.args.CARLA_VERSION back to "0.9.11"
```

## What's in the `coopernaut` image

- Ubuntu 18.04 + CUDA 11.0 (`nvidia/cudagl:11.0-devel-ubuntu18.04`)
- Miniconda with an `autocast` env, Python 3.7
- PyTorch 1.7.1 / torchvision 0.8.2 / torchaudio 0.7.2 (cudatoolkit 11.0)
- torch-scatter, torch-sparse, torch-geometric (pinned to the 1.7.1+cu110 wheel index)
- MinkowskiEngine (installed from upstream `master`; Coopernaut's docs pin an
  exact commit, but that commit is no longer reachable in the repo's git
  history, so we follow NVIDIA's own current install instructions instead)
- mosquitto (MQTT broker used to simulate the V2V message bus)
- CARLA **0.9.15** Python client, extracted from the matching `carlasim/carla:0.9.15`
  server image at build time (see the version note above)
- `agents` navigation helper package, extracted from `carlasim/carla:0.9.11` (pinned
  independently via `CARLA_AGENTS_VERSION`, not `CARLA_VERSION` -- see the version note
  above) -- also on `PYTHONPATH`, but see that same note for why a stray
  `AutoCastSim/agents/` on the host still needs fixing separately
- `fontconfig` + `ttf-ubuntu-font-family` (pygame's HUD needs at least one system font)

This mirrors `docs/INSTALL.md` / `scripts/install.sh` from the repo, just containerized.

**The Coopernaut and AutoCastSim source is *not* baked into the image.** It's bind-mounted
from your local checkouts at runtime (see below), so you can edit code on the host — e.g.
instrument `scenario_runner.py` with timestamps to measure latency — and see it take effect
in the container immediately, no rebuild.

Note: `AutoCastSim/agents.carla0915-stray.bak/` is a backup of a mismatched (0.9.15-vintage,
missing `agent.py`/`global_route_planner_dao.py`) copy of the `agents` helper package that
used to live at `AutoCastSim/agents/` and caused the `ModuleNotFoundError` described above.
`AutoCastSim/agents/` now holds an 0.9.11-vintage copy instead (see "Known 0.9.11 → 0.9.15
breakages" above). Both are untracked/gitignored-equivalent, not part of the AutoCastSim git
history -- safe to delete the `.bak` once you've confirmed things work for you.

## Requirements on the host

- NVIDIA GPU with recent driver
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`nvidia-docker`) so `--gpus all` works
- Docker with `--shm-size` support (PyTorch dataloaders + CARLA both want shared memory)
- `carlasim/carla:0.9.15` pulled (or building the whole thing will fail at the compose step
  when it tries to run a server image you don't have)

## Mount your local repos

Point these at your existing checkouts on the host:

```bash
export COOPERNAUT_DIR=/path/to/your/Coopernaut
export AUTOCASTSIM_DIR=/path/to/your/Coopernaut/AutoCastSim   # or an independent AutoCastSim clone
```

```bash
export COOPERNAUT_DIR=/home/janice/Coopernaut
export AUTOCASTSIM_DIR=/home/janice/Coopernaut/AutoCastSim 
```

`AUTOCASTSIM_DIR` is mounted *on top of* `COOPERNAUT_DIR/AutoCastSim`, so it's fine (and
common) for it to physically be the submodule folder inside your Coopernaut checkout — the
mount just makes that mapping explicit and lets you swap in a separately-cloned/branched
AutoCastSim if you want to hack on `scenario_runner.py` independently of the Coopernaut repo
state.

## Run

With `docker-compose.yml` (recommended — starts both containers on a shared network):

```bash
docker compose build coopernaut
docker compose up -d carla-server        # start the CARLA server in the background
# wait ~15-20s for it to finish booting
docker compose run --rm coopernaut       # drop into the training container
```

The `coopernaut` container can reach the simulator at host `carla-server`, port `2000`
(Docker's internal DNS resolves the service name — **not** `localhost`, since that's a
different container now). `CARLA_HOST=carla-server` and `CARLA_PORT=2000` are already set
as environment variables inside the container for convenience.

Without compose, the plain `docker run` equivalent:

```bash
# 1) start the CARLA server (see also start_carla.sh)
docker run -d --name carla-server --gpus all --shm-size=4g \
    -p 2000-2002:2000-2002 \
    carlasim/carla:0.9.15 \
    /bin/bash CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000

# 2) start the training container on the same network, pointed at it
docker run --gpus all -it --rm \
    --shm-size=8g \
    -e CARLA_HOST=carla-server -e CARLA_PORT=2000 \
    -v ${COOPERNAUT_DIR}:/workspace/Coopernaut \
    -v ${AUTOCASTSIM_DIR}:/workspace/Coopernaut/AutoCastSim \
    -v $(pwd)/data:/workspace/Coopernaut/data \
    -v $(pwd)/result:/workspace/Coopernaut/result \
    -v $(pwd)/ckpts:/workspace/Coopernaut/ckpts \
    coopernaut:latest
```

(`--link` is legacy but simplest for a two-container setup like this; a user-defined bridge
network with `--network` is the modern equivalent if you'd rather not rely on it.)

You land in `/workspace/Coopernaut` with the `autocast` conda env already active, running
your host repo's code directly (Python imports it straight from the mounted directory —
no `pip install -e` step needed, `PYTHONPATH` already points there).

## Pointing Coopernaut's scripts at the right CARLA host/port

Coopernaut's pipeline scripts (`run_cooperative_point_transformer.sh`, etc.) and
`scenario_runner.py` typically default to connecting to `localhost`, which assumed
CARLA ran on the same machine/container. Since the server is now a separate container,
check the `HOST`/`PORT` (or equivalent `--host`/`--port` CLI args) in whichever script or
command you're running and point them at `carla-server:2000` instead of `localhost:2000`.
The `CARLA_HOST`/`CARLA_PORT` env vars are set inside the container so you can reference
them from a wrapper script if you'd rather not hardcode the hostname everywhere.

## Per-frame timing

`scenario_runner.py` already records per-frame wall-clock timing — no extra
instrumentation needed. Every tick of the scenario loop (`ScenarioManager.run_scenario`
in `srunner/scenariomanager/scenario_manager.py`) logs a `"Start Frame"` / `"End Frame"`
timestamp pair (via `record_latency.py`'s `RecordLatency`) and, at the end of the run,
writes them to `latency_scenario.csv` in the working directory — one row per frame event,
with columns `Frame`, `Timestamp`, `Event`, `Agent type`. Subtract `Start Frame` from
`End Frame` for the same `Frame` number to get that frame's completion time.

Run a single-instance scenario against the dockerized `carla-server` to produce it:

```bash
cd AutoCastSim
python3 scenario_runner.py \
    --route srunner/data/routes_training_town03_autocast10.xml \
        srunner/data/towns03_traffic_scenarios_autocast10.json \
    --reloadWorld \
    --bgtraffic 0 \
    --agent AVR/autocast_agents/simple_agent.py \
    --host carla-server \
    --port 2000
```

This lands `AutoCastSim/latency_scenario.csv` on the host (bind mount, not a copy), so
you can inspect it directly, e.g.:

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('AutoCastSim/latency_scenario.csv')
start = df[df.Event == 'Start Frame'].set_index('Frame').Timestamp
end = df[df.Event == 'End Frame'].set_index('Frame').Timestamp
print((end - start).describe())
"
```

For a finer per-stage breakdown within each frame (sensor update / collaborator tick /
scenario tick / HUD update), pass `--profile`, which sets `Utils.TIMEPROFILE = True` and
prints each stage's timing to stdout on every frame.

If you want to add timestamps around other hooks (e.g. the V2V message send/receive
path in `AVR/Collaborator.py`), edit the host copy in your normal editor — changes take
effect immediately in the container, no rebuild, since the source is bind-mounted rather
than baked into the image. `git diff` on the host will show exactly your instrumentation
changes.

## Training workflow

CARLA has to be running before you collect data, train (BC or DAgger), or evaluate,
since Coopernaut trains against the live simulator (behavior cloning + DAgger over
AutoCastSim scenarios), not a static offline dataset.

1. **Start the CARLA server** (see "Run" above). Wait ~15-20s for it to finish booting.

2. **Get training data.** Either download the [official dataset](https://utexas.box.com/v/coopernaut-dataset)
   into the `data/` volume, or collect your own by running the pipeline script and
   selecting `data-train` / `data-val` / `data-expert` plus a scene number (6=Overtake,
   8=Left turn, 10=Red light violation) when prompted:
   ```bash
   bash run_cooperative_point_transformer.sh
   ```

3. **Behavior cloning training.** Same script, choose `bc` at the prompt once you have
   both train and val data collected.

4. **DAgger training.** Set `CHECKPOINT` in the script to your BC checkpoint, then choose
   `dagger` at the prompt.

5. **Evaluation.**
   ```bash
   bash run_cooperative_point_transformer.sh   # choose "eval", set RUN= to your DAgger checkpoint
   python scripts/result_analysis.py --root ${PATH_TO_YOUR_OUTPUT_DIR}
   python scripts/compare_trajectory.py --eval ${PATH_TO_YOUR_OUTPUT_DIR} --expert ${PATH_TO_EXPERT}
   ```

Other pipeline scripts for the paper's baselines: `run_point_no_fusion.sh` (no V2V sharing),
`run_earlyfusion_point_transformer.sh` (early fusion), `run_v2v.sh` (Voxel GNN).

All the paths you'll want to edit (`TrainValFolder`, `DATAFOLDER`, `CHECKPOINT`, `RUN`) are
set directly inside these shell scripts, per the upstream README. These live in your mounted
`COOPERNAUT_DIR`, so edit them on the host too.

## Notes / things you may need to adjust

- **GPU memory:** the repo's own quickstart script (`quick_run.sh`) assumes >6GB free per
  CARLA worker; drop `CARLA_WORKERS` to 1 in that script if you're VRAM-constrained.
- **Multi-GPU:** if the CARLA server and the training process need to live on separate
  GPUs, set `NVIDIA_VISIBLE_DEVICES` on the `carla-server` container and
  `CUDA_VISIBLE_DEVICES` on the `coopernaut` container accordingly.
- **mosquitto:** the broker binary is installed but not started as a system service in this
  image (no systemd in the container). Start it manually if a script needs it:
  ```bash
  mosquitto -d
  ```
- **License:** `carlasim/carla` and CARLA's assets are Epic/CARLA-licensed; review before
  redistributing anything built on top of this image.