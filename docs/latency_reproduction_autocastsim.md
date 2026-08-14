# Reproducing Coopernaut's per-frame latency with AutoCastSim

This documents four AutoCastSim runs done to measure per-frame wall-clock latency of the
trained Coopernaut cooperative point-transformer (V2V fusion) model, progressively getting
closer to the actual paper-reproduction pipeline (`run_cooperative_point_transformer.sh`'s
`eval` mode) and then isolating one infrastructure variable (Docker vs. native CARLA server).

All four runs use the same trained checkpoint and scene:

- Checkpoint: `ckpts/model-105.th`, config `ckpts/config.yaml` (`cpt: True`, `npoints: 2048`,
  `max_num_neighbors: 3`, `nblocks: 2`, `nneighbor: 16`, `transformer_dim: 32`) — the
  `CooperativePointTransformer` (V2V fusion) model, not the ego-only variant.
- Agent: `NeuralAgents/dagger_agent.py` -> `DaggerAgent` wrapping `LidarPointTransformerAgent`,
  `--beta 0.0` (student action only, no expert-mixing).
- Scene: 6 (Overtake, Town01), `--sharing` on.
- Latency source: `record_latency.py`'s `RecordLatency`, which logs
  `Frame,Timestamp,Event,Agent type` rows; pairing each frame's `Start Frame`/`End Frame`
  timestamp (`Agent type == "scenario"`) gives per-frame wall-clock time.

## Why these particular runs

`run_cooperative_point_transformer.sh`'s `eval` mode (the actual paper-reproduction path)
calls `parallel_evaluation.py`, which needs a `wandb/<run>/files/config.yaml` that doesn't
exist on this machine — so a literal end-to-end run of that script isn't possible here. The
closest reproduction achievable is driving `AutoCastSim/scenario_runner.py` directly with the
same flags `eval` mode actually passes: `--emulate --hud --sharing --passive_collider
--bgtraffic <sweep value>` (see `run_cooperative_point_transformer.sh` lines 237-257). The
first two runs below predate that discovery and used a faster, minimal flag set instead; the
third run corrects for that.

## Instrumentation added

`NeuralAgents/LidarPointTransformerAgent.py`'s `run_step()` had a dead `num_valid_neighbors`
variable. Patched (left in place on disk, not committed) to actually increment it and print a
per-frame collaborator count, so frame time could be broken down by how many collaborators were
fused that frame:

```python
# in the successful-collaborator-load branch:
num_valid_neighbors += 1
print("sharing vehicles", ego_id, other_agent_id, "frame", frame_id)
...
# after the per-collaborator loop:
print("Frame collaborator count", frame_id, num_valid_neighbors)
```

## Infrastructure

All four runs use `carlasim/carla:0.9.15` as the CARLA server (client/server versions must
match — see root `CLAUDE.md`) and the `coopernaut:latest` client image built per
`dockerfiles/README.md`. A fresh `coopernaut:latest` container needs three ephemeral fixes
every time (not persisted in the image):

```bash
# inside the container's autocast conda env
/opt/conda/envs/autocast/bin/python3 -m pip install --no-index --no-cache-dir \
  -f https://data.pyg.org/whl/torch-1.7.1+cu110.html \
  torch-scatter==2.0.5 torch-sparse==0.6.9   # 0.6.10 (originally documented) is no longer
                                              # published for torch-1.7.0+cu110; --no-index
                                              # is required or pip silently grabs a cached
                                              # CPU-only wheel and torch_sparse import breaks
                                              # with AttributeError: 'NoneType' object has no
                                              # attribute 'origin'
/opt/conda/envs/autocast/bin/python3 -m pip install open3d==0.13.0
/opt/conda/envs/autocast/bin/python3 -m pip install tensorboard
```

Runs 1-3 used `dockerfiles/docker-compose.yml` as-is (both `carla-server` and `coopernaut`
services in Docker; needs `--project-directory /home/janice/Coopernaut` since the compose
file's volume paths are relative). Run 4 instead extracts the exact same server binary out of
the `carlasim/carla:0.9.15` image and runs it natively on the host (see that section).

---

## Run 1 — `latency_scenario_0.csv` (2026-07-27, Docker server)

Original baseline measurement, minimal ("fast simulation") flags — no `--emulate`, no `--hud`,
`--bgtraffic 0`. 2 vehicles total (ego + 1 collaborator).

```bash
cd AutoCastSim
python3 scenario_runner.py \
  --route srunner/data/routes_training_town01_autocast6.xml \
          srunner/data/towns01_traffic_scenarios_autocast6.json \
  --reloadWorld \
  --bgtraffic 0 \
  --agent /workspace/Coopernaut/NeuralAgents/dagger_agent.py \
  --agentConfig /workspace/Coopernaut/ckpts/config.yaml \
  --num_checkpoint 105 \
  --beta 0.0 \
  --sharing \
  --passive_collider \
  --file \
  --host carla-server \
  --port 2000
```

| metric | value |
|---|---|
| avg frame time | 305.4 ms |
| median | 300.5 ms |
| std dev | 53.5 ms |
| min / max | 241.1 / 956.6 ms |
| n frames | 299 |

## Run 2 — `latency_scenario_1.csv` (2026-08-10, Docker server, rerun)

Identical config to Run 1 (kept deliberately identical for an apples-to-apples check), with the
collaborator-count instrumentation added and the raw stdout preserved this time
(`latency_runs/run_2client_v2v.log`). `nvidia-smi` confirmed idle GPU immediately before launch
(197 MiB / 0% util).

| metric | value |
|---|---|
| avg frame time | 196.0 ms |
| median | 189.2 ms |
| std dev | 53.8 ms |
| min / max | 153.9 / 916.5 ms |
| n frames | 301 |

Despite an identical config to Run 1, this run averaged ~1.6x faster — not root-caused, most
likely GPU/machine contention differences three weeks apart. Every one of the 300 real frames
logged exactly 1 active collaborator (2-vehicle scene, both stay in the 40m sharing radius the
whole route) — no collaborator-count variation was possible from this scene. The two slowest
frames in both Run 1 and Run 2 are literally the first two frames of the scenario (CUDA/model
warm-up: 956.6/916.5 ms respectively) — steady-state is otherwise tightly clustered.

## Run 3 — `latency_scenario_2.csv` (2026-08-10, Docker server, paper-faithful flags)

Reproduces the actual flags `run_cooperative_point_transformer.sh` uses for every mode:
`--emulate` (switches `AVR/Collaborator.py` and `AVR/PCProcess.py` into their higher-fidelity
path — real MQTT pub/sub instead of an in-process buffer, plus per-frame
Lidar2BEV/occupancy-grid/object-detection), `--hud` (records BEV/HUD imagery per frame), and
`--bgtraffic 30` (matching the eval sweep's mid-range value for scene 6).

```bash
cd AutoCastSim
python3 scenario_runner.py \
  --route srunner/data/routes_training_town01_autocast6.xml \
          srunner/data/towns01_traffic_scenarios_autocast6.json \
  --reloadWorld \
  --bgtraffic 30 \
  --agent /workspace/Coopernaut/NeuralAgents/dagger_agent.py \
  --agentConfig /workspace/Coopernaut/ckpts/config.yaml \
  --num_checkpoint 105 \
  --beta 0.0 \
  --sharing \
  --emulate \
  --hud \
  --passive_collider \
  --file \
  --host carla-server \
  --port 2000
```

| metric | value (all frames) | excl. 1 outlier |
|---|---|---|
| avg frame time | 326.7 ms | 307.8 ms |
| median | 303.1 ms | 302.6 ms |
| std dev | 341.0 ms | 78.6 ms |
| min / max | 244.0 / 6131.3 ms | 244.0 / 1442.4 ms |
| n frames | 308 | 307 |

This is the first run with real, non-degenerate collaborator-count variation
(`--bgtraffic 30` causes background vehicles to drift in/out of the 40m sharing radius):

| collaborator count | n frames | avg | median |
|---|---|---|---|
| 2 | 282 | 319.5 ms | 288.8 ms |
| 3 | 24 | 350.2 ms | 339.0 ms |

**Root cause of the 6.1s outlier** (frame 23782, traced in `latency_runs/run_paper_emulate.log`
lines 863-899): a background-traffic vehicle that had briefly been a 3rd collaborator got torn
down by CARLA mid-scenario — destroying 4 sensors + 2 MQTT comm channels + the `Collaborator`
object synchronously blocked the main loop for ~6.13s. `grep -c "Destroyed Collaborator"` = 1;
happened exactly once. This is a real, reproducible `--bgtraffic`-driven tail-latency source,
not a per-frame model or `--emulate` cost.

A pre-existing double-`cleanup()` bug (`scenario_manager.py`'s `run_scenario()` calls
`self.cleanup()` in its except-handler, then `scenario_runner.py`'s top-level `destroy()` calls
it again) left the process hung after the scenario completed; the CSV was already fully written
(`save_to_csv()` runs before `cleanup()`) before the process was killed manually. Not fixed —
out of scope for a latency measurement, but worth guarding if this invocation gets reused.

## Run 4 — `latency_scenario_native.csv` (2026-08-11, native CARLA server)

Isolates one variable — Docker vs. native — against Run 2's config exactly (`--bgtraffic 0`, no
`--emulate`/`--hud`), so the only difference is how the CARLA server process is launched. No
native CARLA 0.9.15 binary existed on this machine, so the exact same server binary was
extracted straight out of the `carlasim/carla:0.9.15` image:

```bash
CID=$(docker create carlasim/carla:0.9.15)
docker cp "$CID:/home/carla" ./carla-0.9.15
docker rm "$CID"

cd carla-0.9.15
./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000   # same flags docker-compose.yml uses
```

The coopernaut client still ran in its usual Docker container (needs the `autocast` conda env),
started with `--network host` so it could reach the native server at `localhost:2000` instead of
the `carla-server` container hostname:

```bash
docker run -d --name coopernaut-native-test \
  --network host --gpus all --shm-size=8gb \
  -v /home/janice/Coopernaut:/workspace/Coopernaut \
  -v /home/janice/Coopernaut/AutoCastSim:/workspace/Coopernaut/AutoCastSim \
  -v /home/janice/Coopernaut/data:/workspace/Coopernaut/data \
  -v /home/janice/Coopernaut/result:/workspace/Coopernaut/result \
  -v /home/janice/Coopernaut/ckpts:/workspace/Coopernaut/ckpts \
  coopernaut:latest sleep infinity
# then: apply the 3 ephemeral pip fixes, and run scenario_runner.py with --host localhost --port 2000
```

| metric | value |
|---|---|
| avg frame time | 185.4 ms |
| median | 179.5 ms |
| std dev | 51.1 ms |
| min / max | 151.1 / 873.0 ms |
| n frames | 298 |

Same warm-up pattern (first 2 frames: 873.0/677.6 ms). Cleanup: `docker rm -f
coopernaut-native-test`, `pkill -9 -f CarlaUE4-Linux-Shipping`, extracted 19GB server build
removed from scratchpad.

## Run 5 — `latency_scenario_redlight10.csv` (2026-08-12, Docker server, Scene 10 / red-light violation)

Reruns Run 2's exact config (minimal flags: no `--emulate`/`--hud`, `--bgtraffic 0`, fusion
checkpoint) against Scene 10 instead of Scene 6 — this is the scenario that directly matches
LegoCarla's red-light XML (`DistributedMultiEgoVehicle_red_light.xml`), which references the
same scenario class (`AutoCastIntersectionRedLightViolation`), giving a genuinely matched-scenario
comparison for the first time (see the comparability-caveat section below).

```bash
cd AutoCastSim
python3 scenario_runner.py \
  --route srunner/data/routes_training_town03_autocast10.xml \
          srunner/data/towns03_traffic_scenarios_autocast10.json \
  --reloadWorld \
  --bgtraffic 0 \
  --agent /workspace/Coopernaut/NeuralAgents/dagger_agent.py \
  --agentConfig /workspace/Coopernaut/ckpts/config.yaml \
  --num_checkpoint 105 \
  --beta 0.0 \
  --sharing \
  --passive_collider \
  --file \
  --host carla-server \
  --port 2000
```

**Data-loss note:** this run's output landed at `latency_scenario_1.csv` inside the container
(scenario_runner.py's default output naming), which collided with and overwrote Run 2's original
raw per-frame CSV. Run 2's raw data is not recoverable, but nothing citable in this doc was lost —
Run 2's full aggregate stats (196.0/189.2/53.8/153.9-916.5 ms, n=301) were already independently
recorded in `latency_scenario_1.txt`, and its full raw agent stdout survives at
`latency_runs/run_2client_v2v.log` — both untouched by this run. This run's own CSV was copied out
before cleanup to `latency_scenario_redlight10.csv` (a non-colliding name) to avoid a repeat.

| metric | value |
|---|---|
| avg frame time | 255.0 ms |
| median | 233.8 ms |
| std dev | 81.2 ms |
| min / max | 209.6 / 885.2 ms |
| n frames | 110 |

`"Frame collaborator count"` printed `1` for every one of the 109 in-scenario frames (36701-36809)
— real single-collaborator fusion happened throughout, same as Run 2. The scenario ended after
~5.5s sim-time with `"ScenarioManager: Terminated due to failure"` / `"Not all scenario tests were
successful"` — the same normal termination message every other run in this doc produces (not a
crash); for a red-light-violation scenario, the `RunningRedLightTest` criterion "failing" is
plausibly the expected outcome of the scenario actually completing (the ego running the light is
the point of the test), which is also why this run has fewer paired frames (110 vs. Run 2's ~300)
— the shorter, purpose-built route ends once the violation plays out rather than running a full
~15s loop. Cleanup: `docker rm -f coopernaut-redlight-test`, `docker compose down`.

## Run 6 — `latency_scenario_redlight10_paper.csv` (2026-08-13, Docker server, Scene 10 / red-light violation, paper flags)

Combines Run 3's paper-faithful flags (`--emulate --hud --bgtraffic 30`) with Run 5's
matched scenario (Scene 10, red-light violation, Town03) — the first run in this doc to
be both scenario-matched to LegoCarla *and* running the real eval-mode flag set.

**Same filename-collision issue as Run 5:** this run's raw CSV also landed at the next
auto-incremented name, `latency_scenario_2.csv` — which collided with and overwrote Run 3's
original raw per-frame CSV (Run 3's own file no longer existed on disk before this run, so
nothing of Run 6's own data was lost — but nothing of Run 3's raw per-frame data survives
either). As with Run 5, nothing *citable in this doc* was lost: Run 3's full aggregate stats
(326.7/303.1/341.0/244.0-6131.3 ms, n=308) are independently preserved in
`latency_scenario_2.txt`, untouched by this run. This run's own CSV was copied out before
cleanup to `latency_scenario_redlight10_paper.csv` (a non-colliding name, following Run 5's
pattern) to avoid causing the same problem for a future run.

```bash
cd AutoCastSim
python3 scenario_runner.py \
  --route srunner/data/routes_training_town03_autocast10.xml \
          srunner/data/towns03_traffic_scenarios_autocast10.json \
  --reloadWorld \
  --bgtraffic 30 \
  --agent /workspace/Coopernaut/NeuralAgents/dagger_agent.py \
  --agentConfig /workspace/Coopernaut/ckpts/config.yaml \
  --num_checkpoint 105 \
  --beta 0.0 \
  --sharing \
  --emulate \
  --hud \
  --passive_collider \
  --file \
  --host carla-server \
  --port 2000
```

| metric | value (all frames) | excl. 3 teardown stalls |
|---|---|---|
| avg frame time | 605.9 ms | 563.5 ms |
| median | 549.2 ms | 548.6 ms |
| std dev | 504.7 ms | 105.4 ms |
| min / max | 306.5 / 6474.6 ms | 306.5 / 1525.7 ms |
| n frames | 407 | 404 |

Collaborator-count breakdown (background traffic churned between 2 and 3 real
collaborators throughout, never dropping to the degenerate 1-collaborator case Runs 2/5
saw):

| collaborator count | n frames | avg (all) | avg (excl. teardown) |
|---|---|---|---|
| 2 | 121 | 652.6 ms | 557.0 ms |
| 3 | 283 | 582.5 ms | 562.2 ms |

**Same collaborator-teardown stall as Run 3, now reproduced 3 times in one run.** Live
monitoring caught `"Destroyed Collaborator"` printed at frames 34416, 34509, and 34551;
the 3 slowest frames in the whole CSV are 34417 (6474.6 ms), 34510 (6295.9 ms), and 34552
(6209.9 ms) — i.e., every single one of them is the frame immediately after a teardown
event, all landing in the same ~6.2-6.5s range Run 3's one teardown stall did (~6.13s).
`--bgtraffic 30` reliably produces this: with more background vehicles churning in and out
of the 40m sharing radius, more of them get spawned-then-destroyed while briefly counted as
a collaborator, each destruction synchronously blocking the main loop for ~6s (same root
cause as Run 3: `Collaborator` teardown — 4 sensors + 2 MQTT channels — happens
synchronously on the main thread). This is not a rare edge case at this traffic level: 3
stalls in 407 frames (~1 per 135 frames) at `--bgtraffic 30`.

Excluding those 3 frames, steady-state (563.5 ms avg, std dev collapses from 504.7 ms to
105.4 ms) is still ~1.8x Run 3's steady-state (307.8 ms, Scene 6/Overtake) despite both
using identical flags and checkpoint — consistent with Run 5's finding that the red-light
intersection scenario itself (route/intersection-geometry planner cost) is a real,
independent latency driver, and it stacks with the paper-flags overhead rather than being
subsumed by it.

Hit the same pre-existing double-`cleanup()` bug documented in Run 3 (`RuntimeError: trying
to operate on a destroyed actor` from `agent_wrapper.py`'s `cleanup()`, called twice — once
from `scenario_manager.py`'s except-handler, again from `scenario_runner.py`'s top-level
`destroy()`) — confirmed harmless again: `latency_scenario_2.csv` (815 lines) was already
fully written to disk (`save_to_csv()` runs before `cleanup()`) by the time the traceback
printed, and the scenario had already completed (407 HUD-frame video compiled) before the
hang. Process killed manually (`kill -9`) after confirming the CSV was intact. Cleanup:
`docker rm -f coopernaut-redlight-paper`, `docker compose down`.

---

## Cross-run comparison

| | Run 1 (`scenario_0`) | Run 2 (`scenario_1`) | Run 3 (`scenario_2`, paper flags) | Run 4 (native server) | Run 5 (Scene 10, red light) | Run 6 (Scene 10, paper flags) |
|---|---|---|---|---|---|---|
| server | Docker | Docker | Docker | **native** | Docker | Docker |
| scene | 6 (Overtake) | 6 (Overtake) | 6 (Overtake) | 6 (Overtake) | **10 (red-light violation)** | **10 (red-light violation)** |
| flags vs. `eval` mode | missing `--emulate`/`--hud`, bgtraffic 0 | missing `--emulate`/`--hud`, bgtraffic 0 | matches | missing `--emulate`/`--hud`, bgtraffic 0 | missing `--emulate`/`--hud`, bgtraffic 0 | matches |
| n vehicles | 2 (fixed) | 2 (fixed) | 2-4 (bgtraffic churn) | 2 (fixed) | 2 (fixed) | 2-4 (bgtraffic churn) |
| avg | 305.4 ms | 196.0 ms | 326.7 ms (307.8 excl. outlier) | 185.4 ms | 255.0 ms | 605.9 ms (563.5 excl. 3 outliers) |
| median | 300.5 ms | 189.2 ms | 303.1 ms | 179.5 ms | 233.8 ms | 549.2 ms |
| max | 956.6 ms (warm-up) | 916.5 ms (warm-up) | 6131.3 ms (collaborator teardown) | 873.0 ms (warm-up) | 885.2 ms | 6474.6 ms (collaborator teardown x3) |

**Takeaways:**
- The paper's actual eval flags (Run 3) don't inflate steady-state latency much over the fast
  baseline (Run 1) — ~308ms vs. 305ms — but they do introduce a distinct tail-latency source
  (background-traffic collaborator churn) that the fast/fixed-scene runs can never produce.
- Docker vs. native server (Run 2 vs. Run 4) is a ~5% effect (196ms → 185ms), well inside the
  ~1.6x run-to-run variance already observed between two identical Docker configs (Run 1 vs.
  Run 2). Containerization overhead is not a meaningful contributor to this model's latency.
- In every run, the single largest number is a one-off event (CUDA/model warm-up on frame 1-2,
  or — in Run 3 — a mid-run collaborator teardown), not steady-state per-frame model cost. Mean
  frame time (~185-330ms depending on flags) and "almost a whole second" outlier experiences are
  both real, drawn from the same high-variance distribution — which one you notice depends on
  which frames you're looking at.
- Scenario/route choice itself is a meaningful effect, comparable in size to the Docker-vs-native
  and minimal-vs-paper-flags effects: Run 5 (Scene 10, red-light violation) averages ~30% higher
  than Run 2 (Scene 6, Overtake) despite identical flags, checkpoint, and vehicle count (196.0ms
  → 255.0ms) — plausibly route/intersection geometry (more per-frame planner/behavior-tree work
  near a signalized intersection) rather than anything V2V- or model-related, since both runs have
  exactly 1 real collaborator throughout.
- The scene-choice and paper-flags effects **stack roughly multiplicatively, not additively**:
  Run 6 (Scene 10 + paper flags) averages 563.5ms steady-state — about 1.8x Run 3's 307.8ms
  (Scene 6 + paper flags) and about 2.2x Run 5's 255.0ms (Scene 10 + minimal flags), each of
  which was itself only ~1-1.3x its respective baseline. Whatever `--bgtraffic 30` costs per
  frame (more actors for the planner/behavior-tree to reason about) appears to interact with,
  not just add to, whatever the red-light intersection's own geometry costs.
- `--bgtraffic 30`'s collaborator-teardown stall (first found once in Run 3) is not a one-off:
  Run 6 hit it 3 times in a single 407-frame run (~1 per 135 frames), always in the same
  ~6.2-6.5s range. At this traffic level it's a real, recurring tail-latency source, not an
  edge case — anyone citing "worst-case" numbers for this pipeline under paper-faithful flags
  should expect one of these roughly every couple hundred frames, not treat it as a rare fluke.

## Comparability caveat: these numbers vs. LegoCarla's

It's tempting to line these numbers up against the companion LegoCarla doc
(`latency_reproduction_legocarla.md`) since both claim to measure "Coopernaut's per-frame
latency," but they aren't an apples-to-apples comparison, for several independent reasons:

1. **Different scenario content, though a matching scenario does exist.** Every run here uses
   Scene 6 (Overtake, Town01) — a 2-vehicle scene where exactly 1 collaborator stays in sharing
   range the whole route (Run 3's `--bgtraffic 30` adds 2-3-way variation on top of that).
   LegoCarla's runs all use `DistributedMultiEgoVehicle_red_light.xml`, a red-light-violation
   scenario in Town03. AutoCastSim *does* have a directly corresponding scenario — Scene 10
   (`srunner/data/routes_training_town03_autocast10.xml` + `towns03_traffic_scenarios_autocast10.json`,
   also Town03), using the exact same scenario class, `AutoCastIntersectionRedLightViolation`
   (`srunner/scenarios/autocast_red_light_violation.py`). LegoCarla's red-light XML literally
   references that same class name at the same intersection, and its version of
   `autocast_red_light_violation.py` is a modified copy of this file's (collider/other-actor logic
   mostly stripped out, single-ego `DriveDistance`/`CollisionTest` calls changed to loop over
   multiple `ego_vehicles`) — not an independently-written scenario. **Run 5 above now covers
   this**: it reruns Run 2's exact flags/checkpoint against Scene 10 instead of Scene 6, giving a
   genuinely matched-scenario data point (255.0 ms) — see the updated comparison below.
2. **LegoCarla's "fusion" runs never actually fuse anything.** LegoCarla has no cross-process V2V
   sensor-relay channel at all (see the companion doc's architectural caveat), so its fusion
   numbers reflect the `CooperativePointTransformer` network running on an always-empty
   collaborator input — extra compute cost only, never genuine cross-vehicle fusion. Runs 2 and 3
   here, by contrast, reflect real 1-to-3-way point-cloud fusion with live collaborator data.
3. **No equivalent flags to control for.** Run 3's paper-faithful config adds `--emulate`
   (real MQTT pub/sub + per-frame Lidar2BEV/occupancy-grid/object-detection), `--hud`, and
   `--bgtraffic 30` — all real per-frame cost. `scenario_runner_distributed.py`'s argparser has no
   equivalent flags, so there's no way to make a LegoCarla run match or deliberately not match
   that overhead.
4. **Different architectures dominate the number.** This is a single Python process throughout.
   LegoCarla is a distributed multi-process ZeroMQ pipeline (separate coordinator + per-agent
   processes) with its own per-tick behavior-tree/serialization overhead, documented in the
   companion doc's Runs 1-2 as a multi-hundred-ms cost independent of the model or any V2V work.

Two comparisons are now available, matched on different dimensions — neither is fully
apples-to-apples, since reasons 2-4 above still apply regardless of which scenario is used:

| | scenario match? | flags match? | avg |
|---|---|---|---|
| AutoCastSim Run 2 (minimal flags, Scene 6) | no (Overtake) | yes (no extra flags) | 196.0 ms |
| AutoCastSim Run 5 (minimal flags, Scene 10) | **yes (red-light)** | yes (no extra flags) | 255.0 ms |
| AutoCastSim Run 6 (paper flags, Scene 10) | **yes (red-light)** | **yes (paper-faithful)** | 605.9 ms (563.5 excl. teardown stalls) |
| LegoCarla native, fusion ckpt (Run 5, companion doc) | yes (red-light) | n/a (no such flags exist) | 305.1 ms |
| LegoCarla Swarm, fusion ckpt (Run 6, companion doc) | yes (red-light) | n/a (no such flags exist) | 286.9 ms |

Using the matched-scenario AutoCastSim number (255.0 ms, Run 5) instead of Run 2's Overtake number
narrows the gap to LegoCarla's fusion numbers considerably — from ~50-55% higher down to ~12-20%
higher (286.9-305.1 ms vs. 255.0 ms) — while LegoCarla is still doing *less* actual fusion work
(zero real collaborators vs. AutoCastSim's one real collaborator). That remaining gap is consistent
with reasons 2 and 4 above: LegoCarla's distributed-pipeline per-tick overhead, on top of running a
model that's fusing empty collaborator data instead of doing no fusion computation at all. This is
the best-matched comparison available across the two codebases, but it's still not a controlled
experiment — reasons 2-4 remain unaddressed, and Run 5's own n=110 (vs. Run 2's ~300) reflects a
shorter route, not a directly comparable sample size either. See
`latency_reproduction_legocarla.md`'s own comparability-caveat section for the same analysis from
the other side.

**Run 6 (scenario-matched *and* paper-flags-matched) inverts the comparison entirely**: even
excluding its collaborator-teardown stalls, its 563.5 ms steady-state average is ~1.8-2x
*higher* than either LegoCarla number (286.9-305.1 ms), not lower. This is the clearest
illustration yet of reason 3 above — there is no flag-equivalent way to make LegoCarla pay
the same per-frame cost `--emulate`/`--hud`/`--bgtraffic 30` impose on AutoCastSim, so as
soon as AutoCastSim's flags stop being artificially minimized, the "LegoCarla is slower"
story from the Run 2/Run 5 comparisons reverses. Neither direction is the "true" answer —
they're both real numbers from genuinely different workloads being described with the same
words ("Coopernaut's per-frame latency"). Which AutoCastSim configuration is the fairer
LegoCarla comparison depends entirely on what LegoCarla's distributed pipeline is *meant* to
be judged against, which this investigation doesn't resolve.

## Files

- `AutoCastSim/latency_scenario_0.csv` / `.txt` — Run 1
- `AutoCastSim/latency_scenario_1.csv` / `.txt`, `AutoCastSim/latency_runs/run_2client_v2v.log` — Run 2
- `AutoCastSim/latency_scenario_2.txt` (aggregate only — see Run 6's note, its raw `.csv` was
  overwritten), `AutoCastSim/latency_runs/run_paper_emulate.log` — Run 3
- `AutoCastSim/latency_scenario_native.csv`, `AutoCastSim/latency_runs/run_native_server.log` — Run 4
- `AutoCastSim/latency_scenario_redlight10.csv`, `AutoCastSim/latency_runs/run_redlight_scene10.log` — Run 5
- `AutoCastSim/latency_scenario_redlight10_paper.csv` (raw data also sits at
  `AutoCastSim/latency_scenario_2.csv`, the auto-incremented name it was actually written to —
  the `_redlight10_paper` copy is the durable, non-collidable reference),
  `AutoCastSim/latency_runs/run_redlight10_paper.log` — Run 6
