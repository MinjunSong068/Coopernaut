# Setup instructions: LegoCarla DAgger training (red light, latency recording)

This doc lives in Coopernaut's `docs/` alongside the companion `latency_reproduction_legocarla.md`
and `latency_reproduction_autocastsim.md`, but — like those — everything it describes (the repo,
scripts, and file paths below) lives in a separate repo, `/home/janice/ScenarioRunner`, not this
one. All relative paths in this doc (`docs/...`, `docker/...`, `training/...`) are relative to
that ScenarioRunner clone.

This is a from-scratch setup guide for a new machine to reproduce the DAgger training run
documented in `ScenarioRunner/docs/dagger_latency_recording_run_2026-08-13.md` — single-node
Docker Swarm is proven working end-to-end (structurally; see that doc's memory-ceiling section
for why no run has actually finished on this particular machine). Multi-machine is the next
thing to test; see "Multi-machine specifics" below for what's known-risky before you start.

If you hit anything not covered here, `ScenarioRunner/docs/dagger_latency_recording_run_2026-08-13.md`
has the full bug-by-bug diagnosis history — this doc is deliberately just the setup checklist.

## Prerequisites

- NVIDIA GPU + driver already installed on the host (`nvidia-smi` works).
- Docker with the NVIDIA container runtime configured. Check:
  ```bash
  docker info | grep -i runtime
  ```
  If `nvidia` isn't listed as available, edit `/etc/docker/daemon.json`:
  ```json
  {
      "default-runtime": "nvidia",
      "node-generic-resources": ["NVIDIA-GPU=<your GPU UUID, from nvidia-smi -L>"],
      "runtimes": {
          "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] }
      }
  }
  ```
  then `sudo systemctl restart docker`.
- Conda/miniforge for the local (non-Docker) training driver process.
- **Memory**: budget `(16 GB × num_agents) + ~4 GB (CARLA) + host OS/desktop overhead` free
  before running anything. The real 2-ego `red_light` scenario needs ~36 GB just for the Docker
  Swarm services — see `dagger_latency_recording_run_2026-08-13.md`'s memory section for how
  this was measured and why less than that reliably fails (including one full host freeze).

## 1. Clone the repo

```bash
git clone https://github.com/UCR-CISL/ScenarioRunner.git
cd ScenarioRunner
git checkout mem_save   # branch this investigation was done on
```

## 2. Docker Swarm

On the machine that will be the swarm manager:
```bash
docker swarm init --advertise-addr <this machine's LAN IP>
```
This prints a `docker swarm join --token ...` command — run that on every other machine that
should join (skip entirely for single-node). Verify:
```bash
docker node ls
```
Note down the `ID` column value(s) — every `docker service create` in this pipeline pins a
service to a specific node via `--constraint node.id==<ID>`, which is how you control which
physical machine runs what.

## 3. Build the Docker images

Two images are needed. Both are unpinned to a registry — build locally on every machine that
will run them (Docker Swarm does not auto-distribute locally-built images across nodes; see the
warning under "Multi-machine specifics").

**`cisl/carla:swarm`** — the CARLA server image, only needed on machines that will run
`carla-server`:
```bash
docker build -t cisl/carla:swarm -f docker/docker_swarm/carla.dockerfile .
```

**`cisl/coopernaut:latest`** — the client image used for both `srunner` (coordinator) and every
`agent_wrapper_*` process, needed on every machine that will run either:
```bash
docker build -t cisl/coopernaut:latest -f docker/debug/coopernaut_fixed.dockerfile .
```
This dockerfile pins `torch==2.2.0+cu121` / `open3d==0.18.0` specifically — the original
`dockerfiles/agent.dockerfile`/`scenario.dockerfile` install a CUDA-12.8 nightly torch build
that (a) doesn't satisfy `open3d.ml.torch`'s `torch==2.2.*` version check and (b) failed to see
the GPU at all on the RTX 4090 this was built against. If you're on different hardware and the
nightly build works fine for you, the original dockerfiles may be simpler — but everything in
this investigation was verified against the `coopernaut_fixed` image specifically.

Confirm both exist: `docker images | grep -E "cisl/carla|cisl/coopernaut"`.

## 4. Python environment for the local training driver

**Steps 4 and 5 only apply to whichever single machine you launch the training script from**
(typically the swarm manager) — see the note at the end of section 5 for why the other
machines in a multi-node setup don't need any of this.

The DAgger training script itself (`training/train_dagger_point_transformer_latency.py`) runs
**on the host**, outside Docker — only the live CARLA rollout (`sampling()`, which the script
shells out to) uses the Docker Swarm services above. Set up a conda env:

```bash
conda create -n lego_coopernaut python=3.7
conda activate lego_coopernaut
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  --index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt -r requirements_coopernaut.txt
```
(Adjust the CUDA wheel to your driver — this is what was verified working on the RTX 4090 this
was built against.) `training/train_dagger_point_transformer_latency.py` also expects `logger`
on the path — pass `PYTHONPATH="$PWD/training:$PYTHONPATH"` at invocation time, not into the env
itself.

## 5. Dataset and checkpoint

You need, under `~/ScenarioRunner`:
- `latency_performance/AutoCast_6/{Train,Val}/` — pre-collected AutoCastSim-format rollout data
  (per-episode `<id>_LIDAR/`, `<id>_LIDARFused/`, `<id>_RGB/`, `measurements/`, `config.json`).
- `coopernaut_config_files/model-100.th` + `config.yaml` — the finetune checkpoint (`cpt: false`,
  ego-only `PointTransformer`).

Neither is produced by anything in this repo — copy them over from a machine that already has
them, or regenerate `AutoCast_6` via AutoCastSim data collection (out of scope for this doc).

**Secondary machines in a multi-node setup do not need any of this.** Checked directly in the
code: `agent_wrapper_main.py`, `scenario_runner_distributed.py`, `NeuralAgents/dagger_agent.py`,
and `srunner/autoagents/autonomous_agent_distributed.py` — the code that actually runs inside
the `srunner`/`agent_wrapper_*` Docker services — never reference `AutoCast_6`, `Train`, `Val`,
or `daggerdata` anywhere. Only the local training-driver process (section 4, a plain Python
process, not a Docker service) reads the dataset and checkpoint, and Docker Swarm doesn't
distribute that process — it runs on exactly the one machine you invoke it from. A secondary
node only needs section 2 (joined to the swarm) and section 3 (the images it'll actually run
built locally there).

## 6. The `srunnerData` Docker volume

`docker/docker_swarm/swarmSetupConstraints_py310.sh` (the swarm launcher this pipeline uses —
a Python-3.10-patched copy of `swarmSetupConstraints.sh`, since `cisl/coopernaut:latest` only
has Python 3.10) auto-creates this volume the first time it runs, **but only if it doesn't
already exist**:
```bash
docker volume create srunnerData
docker run --rm -v srunnerData:/data -v "$PWD":/src busybox cp -r /src/. /data/
```

**This volume does not auto-refresh.** Every container mounts it at
`/home/carla/ScenarioRunner` and runs code from *inside the volume*, not from your live
checkout. If you edit any file that runs inside a container (route XMLs, `scenario_runner_distributed.py`,
`agent_wrapper_main.py`, anything under `srunner/`, `models/`, `NeuralAgents/`, etc.) after the
volume already exists, those edits are invisible until you either:
- delete and recreate the whole volume (`docker service rm` anything using it first, then
  `docker volume rm srunnerData` and rerun the two commands above), or
- copy just the changed file(s) in directly:
  ```bash
  docker run --rm -v srunnerData:/data -v "$PWD":/src busybox \
    cp /src/<path/to/file> /data/<path/to/file>
  ```

**⚠️ Multi-machine warning, untested — read before you set up a second node.** The command
above (`docker volume create srunnerData`, no driver flag) creates a plain **local** Docker
volume, which only exists on whichever single host ran that command — Docker's default `local`
volume driver is not shared across swarm nodes. The *official* LegoCarla docs
(`docker/docker_swarm/swarm.md`) describe a different, multi-host-capable form instead:
```bash
docker volume create --driver overlay --scope multi --sharing all srunnerData
```
Nothing in this investigation exercised that multi-host driver — every run here was single-node,
where the plain-local volume works by accident because there's only ever one host. **If you're
setting up multiple machines, do not assume the plain `srunnerData` volume from
`swarmSetupConstraints_py310.sh` will be visible on a second node** — either switch to the
`overlay --scope multi` form (unverified — confirm your Docker version/setup actually supports
it, it typically requires a real distributed storage backend), or manually keep an independent,
in-sync `srunnerData` volume on every node (rerun the `docker volume create` + `busybox cp`
sequence on each machine from that machine's own checkout, and re-sync manually after every
edit — easy to silently get out of sync).

## 7. Smoke test: single node

```bash
cd ~/ScenarioRunner
mkdir -p latency_performance/smoke_test
conda activate lego_coopernaut
PYTHONPATH="$PWD/training:$PYTHONPATH" python -m training.train_dagger_point_transformer_latency \
  --data "$PWD/latency_performance/AutoCast_6/Train/" \
  --daggerdata "$PWD/latency_performance/AutoCast_6/Dagger/" \
  --eval-data "$PWD/latency_performance/AutoCast_6/Val/" \
  --batch-size 4 --num-dataloader-workers 4 \
  --finetune "$PWD/coopernaut_config_files/model-100.th" \
  --transformer_dim 32 --npoints 2048 --num-epochs 1 --max_num_neighbors 3 \
  --benchmark_config benchmark/scene6.json --bgtraffic 30 \
  --server_node <NODE_ID> --scenario_runner_node <NODE_ID> \
  --agent_wrapper_nodes "<NODE_ID>" \
  --route srunner/examples_dagger_agent/DistributedMultiEgoVehicle_red_light_solo.xml \
  --num_agents 1 \
  --latency_folder_path "latency_performance/smoke_test/" \
  --project smoke-test
```
Use `docker node ls` to get `<NODE_ID>`. This uses the trimmed 1-route scenario
(`DistributedMultiEgoVehicle_red_light_solo.xml`) so `--num_agents 1` is valid — the stock
`DistributedMultiEgoVehicle_red_light.xml` has 2 `<route>` entries and requires
`--num_agents 2` (and `--agent_wrapper_nodes "<NODE_ID> <NODE_ID>"`) or the scenario will spin
forever (see "Known gotchas" below).

**Don't trust the script's own "Scenario completed"/"SAMPLING DONE" console output as a
completion signal** — it returns as soon as the underlying shell script exits, which is almost
immediately, well before the actual Docker services finish. Verify independently:
```bash
docker service ls                          # srunner / agent_wrapper_* / carla-server all 1/1?
docker service logs srunner --tail 50
find latency_performance/AutoCast_6/Dagger -mmin -20 -type f   # did new rollout files land?
```

## Known gotchas (see the other doc for full diagnosis)

- **First `client.load_world()` call after a fresh `carla-server` start can hang indefinitely**
  even though the server is healthy — confirmed via `py-spy dump --pid 1` inside the stuck
  container (genuinely blocked, not slow). Workaround: if `srunner`/`agent_wrapper_*` show zero
  CPU-time growth for more than ~1 minute, remove and recreate just those services (not
  `carla-server`) — a fresh client session against the now-provably-warm server does not
  reproduce the hang.
- **`--num_agents` must exactly equal the number of `<route>` entries in the route XML.**
  `RouteScenario.__init__` (`srunner/scenarios/route_scenario_distributed.py`) busy-waits with
  no sleep and no timeout until enough ego vehicles spawn to match the route count — a mismatch
  spins forever with zero log output, indistinguishable from a hang without a `py-spy` trace.
- **`docker service rm` does not immediately kill the underlying container** — in practice it
  can take several seconds. If you're relying on memory freeing up right after removing
  services (e.g., an automated safety check), also `docker ps -aq | xargs -r docker kill`
  afterward rather than assuming `service rm` alone is synchronous.
- **The actual DAgger rollout data write path may not exist in this codebase.**
  `train_dagger_point_transformer_latency.py`'s `sampling()` creates
  `latency_performance/AutoCast_6/Dagger/{regular,nocollider}/` before launching the scenario,
  but grepping `NeuralAgents/dagger_agent.py`,
  `srunner/autoagents/autonomous_agent_distributed.py`, and every container-side file this
  investigation touched turns up **no code that actually saves rollout data into either
  directory** — no `save`/`makedirs`/`DataLogger`/`to_csv` call referencing them anywhere.
  Every run in this investigation was blocked (by memory, or the bugs above) before reaching
  the point where this would even matter, so it's never been confirmed whether the write path
  genuinely doesn't exist or just wasn't found by this search. **Before relying on a completed
  run to have produced new training data, check that `regular/`/`nocollider/` actually gained
  files** — don't assume a clean "SAMPLING DONE" (once that signal is itself verified real, per
  the note above) means data was collected.
- Full bug list (with root causes and exact fixes already applied to
  `training/train_dagger_point_transformer_latency.py`, `models/point_transformer.py`,
  `NeuralAgents/LidarPointTransformerAgent.py`, `scenario_runner_distributed.py`):
  `docs/dagger_latency_recording_run_2026-08-13.md`.

## Multi-machine specifics

Node targeting itself needs no code change — `--server_node`, `--scenario_runner_node`, and
`--agent_wrapper_nodes` are already just `docker service create --constraint node.id==...`
values, so pointing them at different physical machines' node IDs (from `docker node ls`) is
the entire mechanism. What's actually unverified/risky for a real multi-machine attempt:

1. **The `srunnerData` volume** (see section 6 above) — will silently not be shared across
   nodes unless you switch to a real multi-host volume driver or manually keep per-node copies
   in sync.
2. **Both `cisl/carla:swarm` and `cisl/coopernaut:latest` must be built (or otherwise made
   available) on every node that will run them** — Docker Swarm schedules services onto nodes
   that have the image locally; it does not build or pull on your behalf unless the image name
   references a real registry, and these don't.
3. **A prior 2-machine attempt (2026-08-02, documented in
   `docs/coopernaut_dagger_red_light_2machine_swarm_run_2026-08-02.md`) got the infrastructure
   fully working — nodes joined, cross-machine CARLA clustering worked, agents connected and
   loaded checkpoints — but the scenario itself hung indefinitely on frame 1**, both processes
   spinning at 99% CPU with zero progress for ~22 minutes. Suspected (never confirmed) cause: a
   ZeroMQ PUB/SUB "slow joiner" race in `scenario_manager_distributed.py`'s
   `update_agents_control_status()` — an unthrottled spin loop waiting on a status message a
   separate thread never received, plausible because real inter-host latency exposes a
   subscription-registration race that loopback never does. Distinguishing symptom from gotcha
   #1 above: this is high CPU + zero log progress specifically when running across real separate
   machines, not the single-node "first `load_world()` hangs" quirk (which resolves itself once
   you recreate the affected services against an already-warm server). If you hit this, start by
   instrumenting `comm.py`'s `ServerCommRouter`/`ClientCommDealer` and
   `Publisher`/`Subscriber` connection setup rather than re-diagnosing from scratch.
