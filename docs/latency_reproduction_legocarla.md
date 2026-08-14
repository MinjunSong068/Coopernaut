# Reproducing Coopernaut's per-frame latency with LegoCarla

This consolidates several separate investigation docs into one summary of every LegoCarla
latency measurement done for the Coopernaut DAgger red-light scenario, across three deployment
models: bare native processes on one machine, Docker Swarm on one machine, and Docker Swarm
across two physical machines. LegoCarla is a distributed, ZeroMQ-based fork of `scenario_runner`
that lives in a separate repo on this machine, `/home/janice/ScenarioRunner` — this doc lives in
Coopernaut's `docs/` alongside the companion AutoCastSim doc for easier side-by-side comparison,
but all source docs, checkpoints, and run output referenced below are under
`/home/janice/ScenarioRunner`, not this repo.

**Scenario used throughout**: `srunner/examples_dagger_agent/DistributedMultiEgoVehicle_red_light.xml`
— 2 ego vehicles, both driven by `NeuralAgents/dagger_agent.py` -> `DaggerAgent` wrapping
`LidarPointTransformerAgent`, loading the trained Coopernaut point-transformer checkpoint.
`scenario_manager_distributed.py`'s `run_tick()` is hardcoded to a fixed `num_frames = 100`
coordinator ticks — a deliberate short, bounded latency sample, not an early cutoff.

**Important architectural caveat**: unlike AutoCastSim (see the companion doc,
`/home/janice/Coopernaut/docs/latency_reproduction_autocastsim.md`), this distributed fork has
**no cross-process V2V sensor-sharing channel at all** — `comm.py`'s `ServerComm`/`ClientComm`
(PULL/PUSH), `Publisher`/`Subscriber` (PUB/SUB), and `ServerCommRouter`/`ClientCommDealer`
(ROUTER/DEALER) cover control-sync messaging only, and neither
`srunner/autoagents/agent_wrapper_distributed.py` nor `autonomous_agent_distributed.py`
reference LiDAR or other-actor data in any way. So every number below (except where noted) used
`dagger_agent.py`'s **hardcoded** checkpoint at `coopernaut_config_files/model-100.th`, which on
this machine is the **ego-only** `PointTransformer`, not the `CooperativePointTransformer`
V2V-fusion model — `dagger_agent.py` line 19 hardcodes
`path_to_conf_file = "coopernaut_config_files/config.yaml"` and
`LidarPointTransformerAgent.__init__` hardcodes `num_checkpoint = 100` internally, both marked
`#TODO` in the source, ignoring whatever config/checkpoint is actually passed in. Swapping
models requires literally replacing those two files.

**Did any of this use Docker Swarm?** Yes — three of the six runs below did: a real single-node
Swarm deployment and a real two-machine Swarm deployment, both from prior sessions
(2026-07-30 and 2026-08-02, documented in the individual docs this file summarizes), plus a
single-node Swarm rerun with the fusion checkpoint done this session (Run 6). My other rerun
this session (Run 5) was native, single-machine only.

---

## Run 1 — native, single machine, first working attempt (2026-07-30)

Full doc: `/home/janice/ScenarioRunner/docs/coopernaut_dagger_red_light_native_run_2026-07-30.md`.

`lego_coopernaut` conda env (Python 3.7, `torch==1.13.1+cu117`, `carla==0.9.15`,
`open3d==0.17.0`), CARLA server via plain `docker run --privileged --gpus all --net=host
carlasim/carla:0.9.15`, coordinator (`scenario_runner_distributed.py`) + two agent processes
(`agent_wrapper_main.py --route_id 0` / `--route_id 1`), all via `nohup ... python -u ...`
talking to `127.0.0.1`.

| metric | value |
|---|---|
| avg frame time | **2376.4 ms** |
| median | 2368.8 ms |
| min / max | 1305.1 / 3242.4 ms |
| std dev | 143.1 ms |
| n frames | 101 |

**~7.8x slower than AutoCastSim's single-process baseline (305.4 ms)** — traced down to a real
perf bug, not fundamental cross-process overhead (see Run 2).

## Run 2 — native, single machine, after fixing an unconditional debug print (2026-07-30)

Same doc as Run 1. `scenario_manager_distributed.py`'s `tick_pytree()` called
`py_trees.display.print_ascii_tree()` **unconditionally every tick** (the original
single-process `scenario_manager.py` gates the same call behind `if self._debug_mode:`) —
recursively formatting and printing the entire behavior tree, every tick, for both routes, with
unbuffered stdout (`-u`, needed just to see any progress) turning each line into an immediate
flushed write. Fixed by gating it behind `_debug_mode` like the original. Rerun, identical
setup:

| metric | value | vs. Run 1 |
|---|---|---|
| avg frame time | **858.4 ms** | 2.77x faster |
| median | 842.4 ms | |
| min / max | 809.9 / 1496.3 ms | |
| std dev | 83.5 ms | |
| n frames | 101 | |

Sub-event breakdown showed `Scenario Tree Tick` (behavior-tree tick, client-side Python) was
still ~828 ms of the 858 ms total — `World Tick` (26 ms) and cross-process `Update Controls`
(ZeroMQ agent sync, 12 ms) were both fast throughout, in both Run 1 and Run 2. Of that 828 ms:
`Timeout Update` 186.9 ms, `Scenario Trigger Update` (×2 routes) 276.7 ms,
`RouteCompletionTest` (×2 routes) 180.8 ms — mostly per-actor CARLA API round-trips
(`get_location()` etc.) done twice per tick (2 ego routes vs. AutoCastSim's 1 agent). Still
~2.8x slower than AutoCastSim (858 ms vs. 305 ms) after this fix; remaining gap not chased down
further (candidates: doubled per-tick actor-location RPCs, single-machine GPU/CPU contention
from running server + coordinator + both agents together).

A **separate correctness bug** was also found and fixed in this same investigation (not a perf
issue): `LidarPointTransformerAgent.run_step()` could `return None` when the ego's own actor ID
hadn't yet landed in that frame's `JSONState['other_actors']` snapshot (a registration race),
and the caller (`autonomous_agent_distributed.py`'s `__call__`) doesn't null-check before doing
`control.manual_gear_shift = False` — a hard crash. Fixed by building the two fields actually
needed (`bounding_box.extent_z`, `velocity`) directly from the live `vehicle` actor instead of
bailing out. Verified with a clean rerun: **202.1 ms avg / 187.4 ms median / 145.3-1675.1 ms /
n=101** — faster than the 858.4 ms number above, plausibly because the Docker Swarm services
from the same day's later testing (Run 3) had been stopped by the time this verification ran,
leaving the GPU less contended. Not chased down further.

## Run 3 — Docker Swarm, single node (2026-07-30)

Full doc: `/home/janice/ScenarioRunner/docs/coopernaut_dagger_red_light_docker_swarm_run_2026-07-30.md`. This machine
(`192.168.88.88`) is `r15servera.engr.ucr.edu`, the primary node hardcoded in
`launch_nodes.yaml` — so this was a real (if single-node) Swarm deployment, not an
approximation.

Getting a working Swarm deployment required several fixes:
- `cisl/coopernaut:latest` (what `docker/debug/swarmSetup.sh` targets) doesn't exist on this
  host; the debug images that do exist (`cisl/scenario`, `cisl/agent0`, `cisl/agent1`) derive
  from `nvidia/cuda:12.8.1-devel-ubuntu22.04` and hit `CUDA initialization: ... Error 804:
  forward compatibility was attempted on non supported HW` — isolated to something in that
  `nvidia/cuda:*-devel` base image family itself (same already-working conda env worked fine
  inside a plain `ubuntu:22.04` base, failed inside the `-devel` base with identical driver/GPU).
- Fix: stopped building a torch-containing image at all. `cisl/coopernaut-runtime:latest` is
  just `ubuntu:22.04` + the shared libs `open3d`/`pygame`/`opencv` need at runtime (no Python
  packages baked in); the actual `lego_coopernaut` conda env and the repo source are both
  bind-mounted in at `docker service create` time.
- `swarmSetup.sh`'s `overlay`-driver shared volume isn't available on this host (no such Docker
  volume plugin installed) — sidestepped with plain bind mounts (fine for single-node).
- Swarm services need `NVIDIA_VISIBLE_DEVICES=all`/`NVIDIA_DRIVER_CAPABILITIES=all,compute` set
  explicitly as env vars (`docker service create` has no `--gpus` flag) — omitting this silently
  gave `torch.cuda.is_available() == False` inside the agent.
- `swarmSetup.sh` computes its own reachable IP via `hostname -I | awk '{print $2}'`;
  `swarm.md`'s manual walkthrough uses `$1`. Verified directly on this network that `$1` (the
  overlay-network address) is correct and `$2` (the gateway-bridge address) is not reachable
  from other services.

```bash
docker swarm init --advertise-addr 192.168.88.88
docker network create --driver overlay --attachable legocarla

docker service create --detach --name carla-server --network legocarla \
  --restart-condition any --replicas 1 cisl/carla:swarm

docker service create --detach --name srunner --network legocarla \
  --restart-condition none --replicas 1 \
  --mount type=bind,source=/home/janice/miniforge3/envs/lego_coopernaut,target=/opt/lego_coopernaut \
  --mount type=bind,source=/home/janice/ScenarioRunner,target=/home/carla/ScenarioRunner \
  --workdir /home/carla/ScenarioRunner \
  cisl/coopernaut-runtime:latest \
  bash -c 'SRUNNER_IP=$(hostname -I | awk "{print \$1}"); /opt/lego_coopernaut/bin/python -u scenario_runner_distributed.py --host carla-server --ScenarioRunnerIP $SRUNNER_IP --sync --route srunner/examples_dagger_agent/DistributedMultiEgoVehicle_red_light.xml --latency_folder_path latency_performance/coopernaut_swarm'

docker service create --detach --name agent0 --network legocarla \
  --restart-condition none --replicas 1 \
  --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=all,compute \
  --mount type=bind,source=/home/janice/miniforge3/envs/lego_coopernaut,target=/opt/lego_coopernaut \
  --mount type=bind,source=/home/janice/ScenarioRunner,target=/home/carla/ScenarioRunner \
  --workdir /home/carla/ScenarioRunner \
  cisl/coopernaut-runtime:latest \
  /opt/lego_coopernaut/bin/python -u agent_wrapper_main.py --host carla-server --ScenarioRunnerIP srunner --route srunner/examples_dagger_agent/DistributedMultiEgoVehicle_red_light.xml --route_id 0 --latency_folder_path latency_performance/coopernaut_swarm

# agent1: identical, --route_id 1
```

| metric | value |
|---|---|
| avg frame time | **191.2 ms** |
| median | 178.3 ms |
| min / max | 147.7 / 1300.5 ms |
| std dev | 112.8 ms |
| n frames | 101 |

Fastest of every configuration tried that day, including AutoCastSim's single-process baseline
(305.4 ms) — not root-caused; plausible candidates are a warmed-up GPU/shader cache from
repeated same-day runs, and/or genuinely lower per-tick Python overhead from the minimal
bind-mount-only runtime image vs. whatever else was resident in the full native conda env's
import path. `agent1` hit the same actor-registry race described in Run 2 on its very last
frame (75298) of an otherwise fully-successful run — the high outlier (max 1300 ms) is
consistent with the coordinator being forced through a timeout-wait cycle because of it.

## Run 4 — real 2-machine Docker Swarm (2026-08-02)

Full doc: `/home/janice/ScenarioRunner/docs/coopernaut_dagger_red_light_2machine_swarm_run_2026-08-02.md`. Extended Run 3 to
genuine multi-machine deployment using the repo's own orchestration
(`launch_files/launch_nodes.yaml` + `docker_swarm.py`'s `SwarmManager` + `launch_files/launch.py`)
across two real RTX 4090 UCR-CISL lab machines (`r15servera` as manager/primary CARLA,
`r15serverb` as worker/secondary CARLA).

**Infrastructure side fully worked**: `r15serverb` joined the swarm, CARLA's own
primary/secondary multi-machine clustering came up across both hosts, and both distributed
agent processes connected cross-machine, loaded checkpoints, and spawned. **The scenario itself
hung indefinitely on frame 1** — killed after ~22 minutes of both the coordinator and the
cross-machine agent spinning at 99% CPU with zero progress. **No `latency_scenario.csv` or
frame-time number came out of this run.**

Root cause (not fixed): `scenario_manager_distributed.py`'s `update_agents_control_status()`
has an unconditional, unthrottled spin loop with no sleep and no escape until
`self.agents_sync` first becomes `True` (a 10s timeout only applies *after* that point).
`self.agents_sync` is set by a separate thread that blocks on
`self.servercomm_control.recv_message(timeout=1)` and never logged a single received-status
message in this run despite a live TCP connection — the signature of a ZeroMQ PUB/SUB "slow
joiner" race (a subscriber's `connect()` can return before its subscription is actually
registered on the publisher, silently dropping anything published in that window, no retry).
On loopback (both single-machine runs above) this window is apparently narrow enough to never
trigger; over real inter-host LAN latency it's wide enough to hit reliably. Hypothesis, not
confirmed — didn't trace into the actual ZMQ socket setup to verify.

## Run 5 (this session) — native, single machine, fusion checkpoint swapped in (2026-08-11)

Attempted to get a genuine apples-to-apples comparison against AutoCastSim's V2V-fusion model
(all runs above used the ego-only checkpoint LegoCarla hardcodes) by temporarily replacing the
hardcoded checkpoint files:

```bash
cd /home/janice/ScenarioRunner/coopernaut_config_files
cp -n config.yaml config.yaml.egoonly-backup
cp -n model-100.th model-100.th.egoonly-backup
cp /home/janice/Coopernaut/ckpts/config.yaml config.yaml       # cpt: True, fusion model
cp /home/janice/Coopernaut/ckpts/model-105.th model-100.th     # LidarPointTransformerAgent hardcodes num_checkpoint=100,
                                                                # so the file must be named model-100.th regardless of origin
```

Same instrumentation added as the AutoCastSim side (see the companion doc) — a per-frame
collaborator-count print in `LidarPointTransformerAgent.run_step()` — to directly confirm
whether any real fusion was happening.

First attempt crashed immediately: the instrumentation print referenced `route_id`, which isn't
in scope inside `LidarPointTransformerAgent.run_step()` (only in the outer `DaggerAgent.run_step()`)
— `NameError: name 'route_id' is not defined` on frame 1. Fixed (dropped `route_id` from the
print), coordinator/agents relaunched (CARLA server left running from the crashed attempt).

Completed cleanly this time — `"Agent cleaned up and scenario completed"` for both agents,
`"All frames completed"` / `"Saving scenario latency results"` in the coordinator log.

| metric | value |
|---|---|
| avg frame time | **305.1 ms** |
| median | 285.2 ms |
| std dev | 187.2 ms |
| min / max | 218.0 / 2141.4 ms |
| n frames | 101 |

**`"Frame collaborator count"` printed `0` for all 219 logged lines across both agents** —
confirming empirically (not just from the static `comm.py` analysis) that no real V2V fusion
happens in this pipeline. This number is purely the fusion model's extra compute cost
(`backbone_other` running on an all-zeros/padding input, since no real collaborator lidar is
ever received) relative to the ego-only model, run through the exact same distributed
single-machine setup as Run 2's 202.1 ms — i.e. `CooperativePointTransformer` costs roughly
100 ms/frame more than `PointTransformer` here, independent of collaborator count, because
collaborator count is architecturally always zero.

Original ego-only checkpoint/config were restored afterward (`mv config.yaml.egoonly-backup
config.yaml`, `mv model-100.th.egoonly-backup model-100.th`), verified via
`yaml.safe_load` that `cpt: False` was back. No backup files remain.

## Run 6 (this session) — Docker Swarm, single node, fusion checkpoint swapped in (2026-08-11)

Same idea as Run 5, but through Docker Swarm instead of bare processes — the direct fusion
counterpart to Run 3. Swarm mode had gone inactive since Run 3/4 (checked with `docker info
--format '{{.Swarm.LocalNodeState}}'` → `inactive`, no `legocarla` network, no services), so
setup was redone from scratch on this same host (`r15servera.engr.ucr.edu`, still `192.168.88.88`,
default nvidia runtime, idle GPU):

```bash
cd /home/janice/ScenarioRunner/coopernaut_config_files
cp -n config.yaml config.yaml.egoonly-backup
cp -n model-100.th model-100.th.egoonly-backup
cp /home/janice/Coopernaut/ckpts/config.yaml config.yaml
cp /home/janice/Coopernaut/ckpts/model-105.th model-100.th

docker swarm init --advertise-addr 192.168.88.88
docker network create --driver overlay --attachable legocarla
# then the same 4 `docker service create` commands as Run 3, with
# --latency_folder_path latency_performance/coopernaut_swarm_fusion instead of coopernaut_swarm
```

The `route_id` NameError fix from Run 5 was already on disk (persisted from that session), so
no crash this time. Both `srunner` and both agent services completed cleanly —
`"All frames completed"` / `"Saving scenario latency results"` in the coordinator log,
`"Agent cleaned up and scenario completed"` for both agents, no actor-registry-race crash (the
timing-dependent bug from Runs 2/3 simply didn't trigger this run).

| metric | value |
|---|---|
| avg frame time | **286.9 ms** |
| median | 270.9 ms |
| std dev | 169.7 ms |
| min / max | 194.4 / 1946.1 ms |
| n frames | 101 |

`"Frame collaborator count"` again printed `0` for all 218 logged lines across both agents —
same architectural result as Run 5, just under Swarm instead of bare processes. The fusion
overhead over the matching ego-only baseline is consistent across both deployment models:
+103 ms native (305.1 ms vs. Run 2's 202.1 ms) vs. +96 ms on Swarm (286.9 ms vs. Run 3's
191.2 ms) — reinforcing that the fusion-vs-ego-only model difference, not Docker Swarm vs. bare
processes, is what's driving the gap between Run 5/6 and Run 2/3.

Cleanup: checkpoint/config restored the same way as Run 5 (verified `cpt: False` again);
`docker service rm carla-server srunner agent0 agent1`, `docker network rm legocarla`,
`docker swarm leave --force` — Swarm mode is left inactive again, matching the state before
this run started.

---

## Cross-run comparison

| | Run 1 (native, buggy) | Run 2 (native, fixed) | Run 3 (Swarm, single-node) | Run 4 (Swarm, 2-machine) | Run 5 (native, fusion ckpt) | Run 6 (Swarm, fusion ckpt) |
|---|---|---|---|---|---|---|
| model | ego-only | ego-only | ego-only | ego-only | **fusion, 0 real collaborators** | **fusion, 0 real collaborators** |
| deployment | bare processes | bare processes | Docker Swarm | Docker Swarm | bare processes | Docker Swarm |
| machines | 1 | 1 | 1 | 2 | 1 | 1 |
| avg frame time | 2376.4 ms | 858.4 ms → 202.1 ms (after 2nd fix) | 191.2 ms | — (hung, no data) | 305.1 ms | 286.9 ms |
| n frames | 101 | 101 | 101 | 0 | 101 | 101 |

**Takeaways:**
- The huge early numbers (2.4s, then 858ms) were **client-side logging/perf bugs specific to
  the distributed port**, not fundamental cross-process or Docker overhead — fixing an
  unconditional debug-tree print took 2.77x off by itself.
- Once both bugs were fixed, single-machine LegoCarla (202ms native, 191ms Swarm) lands in the
  same ballpark as AutoCastSim's single-process baseline (see below) — the "Docker Swarm vs.
  bare processes" difference on one machine is consistently small (191ms vs. 202ms ego-only;
  286.9ms vs. 305.1ms fusion), likely just run-to-run noise, consistent with how much variance
  AutoCastSim showed between identical configs on its own machine.
- **No configuration of LegoCarla tested here ever performs real V2V LiDAR fusion** — confirmed
  both by static code analysis (`comm.py` has no sensor-relay channel) and empirically
  (collaborator count logged as 0 for every frame in both Run 5 and Run 6). Any LegoCarla number
  in this whole investigation using the fusion checkpoint measures compute overhead only, never
  genuine cooperative perception.
- The only genuinely new failure mode multi-machine deployment (Run 4) surfaced — a ZMQ
  PUB/SUB slow-joiner race in the control-sync layer — is orthogonal to the model or Docker
  question entirely; it's a real-network-latency bug that single-machine loopback testing
  (Runs 1-3, 5, 6) cannot reproduce.

## Comparability caveat: these numbers vs. AutoCastSim's

It's tempting to line these numbers up against the companion AutoCastSim doc
(`/home/janice/Coopernaut/docs/latency_reproduction_autocastsim.md`) since both claim to measure
"Coopernaut's per-frame latency," but they aren't an apples-to-apples comparison, for several
independent reasons:

1. **Different scenario content, though a matching scenario does exist.** Every run here uses
   `DistributedMultiEgoVehicle_red_light.xml` — a red-light-violation scenario with 2 ego
   vehicles in Town03, `type="AutoCastIntersectionRedLightViolation"`. AutoCastSim's runs all use
   a different scenario instead (Scene 6, Overtake, Town01 — a 2-vehicle scene, 2-4 with
   `--bgtraffic`), but AutoCastSim *does* have the matching red-light scenario available and
   simply didn't use it here: Scene 10 (`srunner/data/routes_training_town03_autocast10.xml` +
   `towns03_traffic_scenarios_autocast10.json`, also Town03) uses the exact same scenario class,
   `AutoCastIntersectionRedLightViolation` (`srunner/scenarios/autocast_red_light_violation.py`).
   This file's version of that class is in fact a modified copy of AutoCastSim's (collider/
   other-actor logic mostly stripped out, single-ego `DriveDistance`/`CollisionTest` calls
   changed to loop over multiple `ego_vehicles`) — not an independently-written scenario. **This
   is no longer a hypothetical**: the companion doc's Run 5 reruns AutoCastSim's minimal-flags
   config against Scene 10 instead of Scene 6, giving a genuinely matched-scenario number
   (255.0 ms); Run 6 goes further and combines that same matched scenario with the paper-faithful
   flags below (605.9 ms, 563.5 ms excl. outliers) — see the updated comparison below.
2. **LegoCarla's "fusion" runs never actually fuse anything.** As the architectural caveat at the
   top of this doc explains, LegoCarla has no cross-process V2V sensor-relay channel at all, so
   Runs 5 and 6 reflect the `CooperativePointTransformer` network running on an always-empty
   collaborator input (confirmed empirically: collaborator count logged as 0 for every frame in
   both runs) — extra compute cost only, never genuine cross-vehicle fusion. AutoCastSim's fusion
   numbers, by contrast, reflect real 1-to-3-way point-cloud fusion with live collaborator data.
3. **No equivalent flags to control for.** AutoCastSim's paper-faithful config adds `--emulate`
   (real MQTT pub/sub + per-frame Lidar2BEV/occupancy-grid/object-detection), `--hud`, and
   `--bgtraffic 30` — all real per-frame cost. `scenario_runner_distributed.py`'s argparser here
   has no equivalent flags at all, so there's no way to make a LegoCarla run match or deliberately
   not match that overhead.
4. **Different architectures dominate the number.** AutoCastSim is a single Python process.
   This is a distributed multi-process ZeroMQ pipeline (separate coordinator + per-agent
   processes) with its own per-tick behavior-tree/serialization overhead, documented in Runs 1-2
   above as a multi-hundred-ms cost independent of the model or any V2V work at all.

With those caveats in view, here's how the fusion-checkpoint numbers actually stack up — it
matters which AutoCastSim number gets used, since AutoCastSim now has four depending on flags
and scenario:

| | model | scenario | avg |
|---|---|---|---|
| AutoCastSim, minimal flags, Scene 6 (`latency_scenario_1`) | fusion, real 1 collaborator | Overtake | 196.0 ms |
| AutoCastSim, paper flags `--emulate --hud --bgtraffic 30`, Scene 6 (`latency_scenario_2`) | fusion, real 2-3 collaborators | Overtake | 326.7 ms (307.8 ms excl. 1 outlier) |
| AutoCastSim, minimal flags, **Scene 10** (`latency_scenario_redlight10`) | fusion, real 1 collaborator | **red-light (matched)** | 255.0 ms |
| AutoCastSim, paper flags, **Scene 10** (`latency_scenario_redlight10_paper`) | fusion, real 2-3 collaborators | **red-light (matched)** | 605.9 ms (563.5 ms excl. 3 outliers) |
| LegoCarla native, fusion ckpt (Run 5) | fusion, 0 real collaborators | red-light | 305.1 ms |
| LegoCarla Swarm, fusion ckpt (Run 6) | fusion, 0 real collaborators | red-light | 286.9 ms |

The best *scenario*-matched comparison, holding flags at "minimal" on the AutoCastSim side since
there's nothing to match them to here, is AutoCastSim's Scene 10 minimal-flags number (255.0 ms)
against these two runs. That narrows the gap considerably from the old Scene-6-based comparison:
these runs (287-305ms) run ~12-20% higher than AutoCastSim's matched-scenario number, down from
~50-55% higher against Scene 6's 196ms — while still doing *less* actual fusion work (zero real
collaborators vs. AutoCastSim's one real collaborator). The remaining gap is consistent with the
per-tick Python/behavior-tree overhead documented in Runs 1-2 still being present, just smaller
relative to the baseline than the Scene 6 comparison suggested.

**But bringing AutoCastSim's paper flags into the same matched scenario (605.9 ms / 563.5 ms)
flips this entirely**: LegoCarla's 287-305ms is now ~1.8-2x *lower* than AutoCastSim's
steady-state, not higher. Nothing about LegoCarla changed between these two comparisons — only
which AutoCastSim configuration it's being held up against did. This is reason 3 above made
concrete: `--emulate`/`--hud`/`--bgtraffic 30` impose real, substantial per-frame cost on
AutoCastSim (`--emulate` alone switches to real MQTT pub/sub plus per-frame Lidar2BEV/
occupancy-grid/object-detection), and this codebase's `scenario_runner_distributed.py` has no
argument that turns on an equivalent cost, so there is no way to make these two pipelines pay
comparable per-frame overhead on top of the model itself. "LegoCarla is faster" and "LegoCarla is
slower" are both defensible conclusions depending entirely on which AutoCastSim run gets chosen
as the reference point — which is itself evidence that neither comparison should be read as a
verdict on which pipeline's *architecture* is faster, only on which specific configured workload
happened to run quicker. Reasons 2 and 4 above (no real fusion happening here, and this codebase's
own distributed-pipeline overhead) still apply on top of that regardless of which AutoCastSim
configuration is used — so even the closest comparison available isn't a controlled experiment,
it's the least-confounded one on offer.

## Source docs this summarizes

All paths below are in `/home/janice/ScenarioRunner`, not this repo:

- `docs/coopernaut_dagger_red_light_native_run_2026-07-30.md` — Runs 1 & 2
- `docs/coopernaut_dagger_red_light_docker_swarm_run_2026-07-30.md` — Run 3
- `docs/coopernaut_dagger_red_light_2machine_swarm_run_2026-08-02.md` — Run 4
- `latency_performance/coopernaut_cpt_fusion_v2/` (`agent0.log`, `agent1.log`,
  `latency_scenario.csv`, `latency_agent_0.csv`, `latency_agent_1.csv`) — Run 5
- `latency_performance/coopernaut_swarm_fusion/` (`latency_scenario.csv`, `latency_agent_0.csv`,
  `latency_agent_1.csv`, `fps/`) — Run 6
