import yaml
import subprocess
import time
import os
import socket
import time
from datetime import datetime

PRIMARY_METRICS_RECORDER = "metrics_recorder_primary"

class SwarmManager:
    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        self.metrics_cfg = config.get('metrics', {})
        # Master switch: when false, primary/secondary CARLA volumes are never
        # mounted (and their host directories never provisioned), regardless
        # of what's listed under each node's `volumes:` in the YAML.
        self.save_logs = config.get('save_logs', True)
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Optional RecordLatency instance (see set_recordlatency) — lets every
        # docker CLI call (service create/rm/ps, all blocking subprocess I/O
        # to the swarm manager) show up in the same latency CSV as sim events.
        self.recordlatency = None

        if config.get('nodes', {}).get('server_node') is not None:
            server_cfg = config['nodes']['server_node']
            self.server_node = server_cfg['hostname']
            self.server_image = server_cfg['image']
            self.server_ports = server_cfg.get('ports', {})
            self.server_command = server_cfg.get('command', None)
            self.server_env = server_cfg.get('env', {})
            self.server_privileged = server_cfg.get('privileged', False)
            self.server_gpus = server_cfg.get('gpus', None)
            self.server_network = server_cfg.get('network', None)

        # MULTI GPU / MULTI MACHINE SUPPORT
        # (placeholder shape today — see module docstring: AutoCastSim has no
        # -carla-primary-host/-carla-primary-port usage yet, so a secondary
        # server node here won't actually cluster with the primary.)
        if (
            config.get('nodes', {}).get('primary_server_node') is not None
            and config.get('nodes', {}).get('secondary_server_nodes') is not None
        ):
            primary_cfg = config['nodes']['primary_server_node']

            self.primary_server_node = primary_cfg['hostname']
            self.primary_server_image = primary_cfg['image']
            self.primary_server_ports = primary_cfg.get('ports', {})
            self.primary_server_command = primary_cfg.get('command', None)
            self.primary_server_env = primary_cfg.get('env', {})
            self.primary_server_privileged = primary_cfg.get('privileged', False)
            self.primary_server_gpus = primary_cfg.get('gpus', None)
            self.primary_server_network = primary_cfg.get('network', None)
            self.primary_server_volumes = primary_cfg.get('volumes', [])

            # MULTIPLE SECONDARY SERVERS
            self.secondary_servers = []

            for cfg in config['nodes']['secondary_server_nodes']:

                if "rpc_port" not in cfg:
                    raise ValueError(
                        f"Missing rpc_port in secondary server config:\n{cfg}"
                    )

                self.secondary_servers.append({
                    "host": cfg["hostname"],
                    "hostname": cfg["hostname"],
                    "port": cfg["rpc_port"],
                    "image": cfg["image"],
                    "ports": cfg.get("ports", {}),
                    "command": cfg.get("command", None),
                    "env": cfg.get("env", {}),
                    "privileged": cfg.get("privileged", False),
                    "gpus": cfg.get("gpus", None),
                    "network": cfg.get("network", "host"),
                    "volumes": cfg.get("volumes", [])
                })

            self.secondary_servers_summary = []
            for cfg in config['nodes']['secondary_server_nodes']:
                self.secondary_servers_summary.append({
                    "host": cfg.get("host", cfg.get("hostname")),
                    "port": cfg["rpc_port"]
                })

        self.agents_cfg = config['nodes'].get('agents', [])

        if (config.get('nodes', {}).get('scenario_runner') is not None):
            self.scenario_cfg = config['nodes']['scenario_runner']

        self.services = []

    def set_recordlatency(self, recordlatency):
        """Attach a RecordLatency instance so docker CLI calls get timestamped (called from the gym once one exists)."""
        self.recordlatency = recordlatency

    def run_command(self, cmd):
        print(">>", " ".join(cmd))
        print("running...")
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end = time.time()
        print("finish running...")

        if self.recordlatency is not None:
            # cmd looks like ["docker", "service", "create"/"rm"/"ps", ...] — use the subcommand as the tag
            subcmd = " ".join(cmd[:3]) if len(cmd) >= 3 else " ".join(cmd)
            self.recordlatency.update_df(event=f"Start Docker Cmd [{subcmd}]", timestamp=start, frame=self.experiment_timestamp, agent_type="swarm_manager")
            self.recordlatency.update_df(event=f"End Docker Cmd [{subcmd}]", timestamp=end, frame=self.experiment_timestamp, agent_type="swarm_manager")

        if result.returncode != 0:
            print("✗ Error:", result.stderr)
            raise RuntimeError(result.stderr)
        return result.stdout.strip()

    def create_service(
        self,
        name,
        image,
        node,
        command=None,
        ports=None,
        env=None,
        gpus=False,
        network=None,
        volumes=None,
        restart=True,
        cap_add=None
    ):
        cmd = [
            "docker", "service", "create",
            "--detach=true",
            "--name", name,
            "--constraint", f"node.hostname=={node}"
        ]

        if not restart:
            cmd += ["--restart-condition", "none"]

        if network:
            cmd += ["--network", network]

        if env:
            for k, v in env.items():
                cmd += ["-e", f"{k}={v}"]

        if ports and network != "host":
            for hp, cp in ports.items():
                cmd += ["-p", f"{hp}:{cp}"]

        if volumes:
            for vol in volumes:
                src, dst = vol.split(":")
                if not os.path.isabs(src):
                    raise ValueError(f"Swarm requires absolute bind paths, got: {src}")
                cmd += ["--mount", f"type=bind,src={src},dst={dst}"]

        if cap_add:
            for cap in cap_add:
                cmd += ["--cap-add", cap]

        cmd.append(image)
        if command:
            cmd += command

        self.run_command(cmd)
        self.services.append(name)

    def start_server(self):
        print("Starting CARLA server service...")
        self.create_service(
            name="carla_server",
            image=self.server_image,
            node=self.server_node,
            command=self.server_command,
            ports=self.server_ports,
            env=self.server_env,
            gpus=True,
            network=self.server_network or "host",
            volumes=None
        )
        print("Waiting for CARLA server to be ready...")
        time.sleep(5)

    def ensure_remote_dirs(self, node, host_paths, tag):
        """Create `host_paths` on `node`'s own local disk before Docker binds them.

        Swarm's `--mount type=bind` (unlike the old `-v` flag) requires the
        source path to already exist on whichever node the container actually
        lands on. Since SSH into every node isn't reliably passwordless, we
        can't just `os.makedirs()` locally (that only touches the machine
        running this script) or shell out to `ssh`. Instead we provision the
        directory from inside Docker itself: a short-lived helper service,
        constrained to `node`, bind-mounts that node's root filesystem and
        mkdir/chmod's the target paths, then gets torn down.
        """
        if not host_paths:
            return

        service_name = f"dir_prep_{tag}"

        mkdir_cmd = " && ".join(
            f"mkdir -p /hostfs{path} && chmod -R 777 /hostfs{path}"
            for path in host_paths
        )

        cmd = [
            "docker", "service", "create",
            "--detach=true",
            "--name", service_name,
            "--constraint", f"node.hostname=={node}",
            "--restart-condition", "none",
            "--mount", "type=bind,src=/,dst=/hostfs",
            "busybox",
            "sh", "-c", mkdir_cmd
        ]

        print(f"Provisioning host directories on {node}: {host_paths}")
        self.run_command(cmd)

        try:
            self.wait_for_service_to_finish(service_name)
        except Exception:
            print(
                f"Leaving {service_name} in place for inspection "
                f"(docker service ps {service_name} --no-trunc); "
                f"remove it manually once done."
            )
            raise
        else:
            try:
                self.run_command(["docker", "service", "rm", service_name])
            except Exception as e:
                print(f"Warning: Failed to remove {service_name}: {e}")

    def inject_unique_csvfile(self, env_dict, prefix):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        updated_env = dict(env_dict)  # copy to avoid mutating original

        if "CARLA_TIMESTAMP_LOG_PATH" in updated_env:

            original_path = updated_env["CARLA_TIMESTAMP_LOG_PATH"]

            if original_path.endswith(".csv"):
                base = original_path[:-4]  # remove .csv
                updated_env["CARLA_TIMESTAMP_LOG_PATH"] = (
                    f"{base}_{prefix}_{timestamp}.csv"
                )

        return updated_env

    # multigpu / multi-machine (placeholder — see module docstring)
    def start_primary_server(self):
        print("Starting multi GPU carla primary server service...")
        env = self.inject_unique_csvfile(
                self.primary_server_env,
                "primary"
            )

        volumes = self.primary_server_volumes if self.save_logs else []

        # Create all host-side mount directories on the node that will actually run this
        host_dirs = [volume.split(":")[0] for volume in volumes]
        self.ensure_remote_dirs(self.primary_server_node, host_dirs, tag="primary")

        self.create_service(
            name="carla_primary_server",
            image=self.primary_server_image,
            node=self.primary_server_node,
            command=self.primary_server_command,
            ports=self.primary_server_ports,
            env=env,
            gpus=True,
            network=self.primary_server_network or "host",
            volumes=volumes
        )

        print("Waiting for CARLA primary server to be ready...")
        time.sleep(5)

    def start_secondary_servers(self):
        print("Starting multi GPU CARLA secondary server services...")

        for idx, secondary in enumerate(self.secondary_servers):

            service_name = f"carla_secondary_server_{idx}"

            env = self.inject_unique_csvfile(
                secondary["env"],
                f"secondary_{idx}"
            )

            volumes = secondary["volumes"] if self.save_logs else []

            # Create all host-side mount directories on the node that will actually run this
            host_dirs = [volume.split(":")[0] for volume in volumes]
            self.ensure_remote_dirs(secondary["hostname"], host_dirs, tag=f"secondary_{idx}")

            self.create_service(
                name=service_name,
                image=secondary["image"],
                node=secondary["hostname"],
                command=secondary["command"],
                ports=secondary["ports"],
                env=secondary["env"],
                gpus=True,
                network=secondary["network"],
                volumes=volumes
            )

            print(f"Started {service_name}")

        print("Waiting for secondary servers to initialize...")
        time.sleep(5)

    def start_metrics_recorder(self, node, name):
        """Start a single CPU/GPU usage recorder (log_docker_usage.py) service on `node`.

        PLACEHOLDER TODAY: ported from the LegoCarla fork's
        latency_eval/gpu_cpu_profiling/log_docker_usage.py and the
        cisl/metrics-recorder image built around it. Coopernaut has neither
        the script (scripts/log_docker_usage.py referenced below doesn't
        exist yet) nor a metrics-recorder image — this needs both before
        `metrics:` in launch_nodes.yaml does anything.

        Node-selection (e.g. skipping a node that's already covered by
        another recorder) is the caller's responsibility, not this method's.
        """
        if not self.metrics_cfg:
            print("No metrics config found, skipping metrics recorder...")
            return

        image = self.metrics_cfg['image']
        interval = self.metrics_cfg.get('interval', 1.0)
        container_latency_path = self.metrics_cfg.get('latency_folder_path', '/tmp')
        env = self.metrics_cfg.get('env', {})
        volumes = self.metrics_cfg.get('volumes', [])

        command = [
            "python3", "-u",
            "scripts/log_docker_usage.py",
            "--latency_folder_path", container_latency_path,
            "--interval", str(interval)
        ]

        # docker.sock is a pre-existing socket, not a directory to provision
        host_dirs = [
            vol.split(":")[0] for vol in volumes
            if vol.split(":")[0] != "/var/run/docker.sock"
        ]

        self.ensure_remote_dirs(node, host_dirs, tag=name)

        self.create_service(
            name=name,
            image=image,
            node=node,
            command=command,
            env=env,
            gpus=True,
            network="host",
            volumes=volumes
        )

        print(f"Started {name} on {node}")

    def stop_metrics_recorder(self, name):
        """Stop a single metrics recorder service created by start_metrics_recorder()."""
        try:
            self.run_command(["docker", "service", "rm", name])
            print(f"✓ Service {name} removed")
            if name in self.services:
                self.services.remove(name)
        except Exception as e:
            print(f"Warning: Failed to remove {name}: {e}")

    def start_metrics_recorders(self):
        """Start a CPU/GPU usage recorder on every secondary server node, so
        contention on colocated machines shows up in pc_metrics/ next to that
        node's secondary_logs/. Does not cover the primary node — see
        start_primary_metrics_recorder().
        """
        if not self.metrics_cfg:
            print("No metrics config found, skipping metrics recorders...")
            return

        print("Starting CPU/GPU metrics recorder services on secondary server nodes...")

        for idx, secondary in enumerate(self.secondary_servers):
            self.start_metrics_recorder(secondary["hostname"], f"metrics_recorder_{idx}")

    def stop_metrics_recorders(self):
        """Stop metrics recorder services created by start_metrics_recorders()."""
        for idx in range(len(self.secondary_servers)):
            self.stop_metrics_recorder(f"metrics_recorder_{idx}")

    def primary_covered_by_secondary(self):
        """True if the primary node is also running as one of the secondary
        server nodes, i.e. a secondary's metrics_recorder_<idx> already covers
        the primary machine and a dedicated primary recorder would double up.
        """
        secondary_hosts = {s["hostname"] for s in self.secondary_servers}
        return self.primary_server_node in secondary_hosts

    def start_primary_metrics_recorder(self):
        """CPU/GPU recording on the primary node via docker swarm. Kept out of
        start_metrics_recorders() (secondaries only) since whether the primary
        needs its own recorder depends on the node layout.

        Set `metrics.enable_primary: false` in the swarm yaml to skip this —
        useful for A/B runs isolating whether the recorder service itself
        (competing for CPU/GPU on the same node as CARLA) affects perf.
        """
        if not self.metrics_cfg.get('enable_primary', True):
            print(f"Skipping {PRIMARY_METRICS_RECORDER}: disabled via metrics.enable_primary: false")
            return

        if self.primary_covered_by_secondary():
            print(f"Skipping {PRIMARY_METRICS_RECORDER}: {self.primary_server_node} is already covered by a secondary metrics recorder")
            return

        self.start_metrics_recorder(self.primary_server_node, PRIMARY_METRICS_RECORDER)

    def stop_primary_metrics_recorder(self):
        if self.primary_covered_by_secondary():
            return

        self.stop_metrics_recorder(PRIMARY_METRICS_RECORDER)

    def start_scenario_runner(self):
        if not hasattr(self, 'scenario_cfg'):
            print("No scenario runner config found, skipping...")
            return

        print("Starting ScenarioRunner service...")

        env = self.scenario_cfg.get('env', {})

        self.create_service(
            name="scenario_runner",
            image=self.scenario_cfg['image'],
            node=self.scenario_cfg['hostname'],
            command=self.scenario_cfg.get('command', None),
            env=env,
            gpus=True,
            network=self.scenario_cfg.get('network', 'host'),
            volumes=self.scenario_cfg.get('volumes', []),
            cap_add=self.scenario_cfg.get('cap_add', [])
        )

    def _resolve_agent_latency_host_dir(self, command, volumes):
        """Map an agent's --latency_folder_path (a path *inside* the container)
        back to the equivalent path on its own node's host filesystem, using
        the agent's volume mounts, so it can be provisioned with
        ensure_remote_dirs() before the container ever starts.
        """
        if "--latency_folder_path" not in command:
            return []

        idx = command.index("--latency_folder_path")
        if idx + 1 >= len(command):
            return []

        container_path = command[idx + 1]

        for vol in volumes:
            src, dst = vol.split(":")
            dst = dst.rstrip("/")
            if container_path == dst or container_path.startswith(dst + "/"):
                return [src.rstrip("/") + container_path[len(dst):]]

        return []

    def start_agents(self, extra_args=None):
        """Start all agent services as listed in YAML (Swarm-safe).

        Each entry under `agents:` needs a `command` that's a standalone,
        long-running per-vehicle process — e.g. the LegoCarla/ScenarioRunner
        fork's agent_wrapper_main.py, connecting to an already-running
        scenario_runner service over the network. AutoCastSim (this repo's
        own submodule) has no such process — its agents run in-process
        inside whichever script drives scenario_runner.py/
        parallel_scenario_runner.py — so this only does something real when
        `agents:`'s image/command point at the ScenarioRunner checkout (see
        launch_nodes.yaml's top-of-file note).

        extra_args: optional list of CLI args appended to every agent's
        command (e.g. ["--visualize"]) so runtime flags don't have to be
        hand-edited into the static YAML on every launch.
        """
        for idx, agent_cfg in enumerate(self.agents_cfg, start=0):
            agent_name = f"rl_agent_{idx}"
            raw_cmd = agent_cfg.get('command')

            if not raw_cmd:
                raise ValueError(f"Agent {idx} missing command")

            command = list(raw_cmd) + list(extra_args) if extra_args else raw_cmd

            volumes = agent_cfg.get('volumes', [])

            # Each agent's node has its own local filesystem with its own
            # locally-assigned UIDs. The agent writes latency CSVs into a
            # path under its bind-mounted repo checkout, whose ownership on
            # that node has no relation to the UID the container runs as --
            # so, same as the primary/secondary CARLA log dirs, provision
            # (mkdir -p + chmod 777) that directory on-node before starting.
            host_dirs = self._resolve_agent_latency_host_dir(command, volumes)
            self.ensure_remote_dirs(agent_cfg['hostname'], host_dirs, tag=f"agent_{idx}")

            self.create_service(
                name=agent_name,
                image=agent_cfg['image'],
                node=agent_cfg['hostname'],
                command=command,
                env=agent_cfg.get('env', {}),
                gpus=True,
                network=agent_cfg.get('network', 'host'),
                volumes=volumes,
                restart=False,
                cap_add=agent_cfg.get('cap_add', [])
            )

    def stop_all(self):
        """Stop all services created."""
        self.services = self.list_services()
        if len(self.services) > 0:
            for service in self.services:
                try:
                    self.run_command(["docker", "service", "rm", service])
                    print(f"✓ Service {service} removed")
                except Exception as e:
                    print(f"Warning: Failed to remove {service}: {e}")
            self.services.clear()

    def list_services(self):
        """Return list of all running Swarm services."""
        output = self.run_command(["docker", "service", "ls"])
        lines = output.strip().split("\n")
        if len(lines) <= 1:
            print("No services are currently running.")
            return []
        else:
            services = []
            for line in lines[1:]:
                service_name = line.split()[1]
                services.append(service_name)
            print("Running services:", services)
            return services

    def _poll_service_tasks(self, service_name):
        """Return [(current_state, error_text), ...] for every task of `service_name`."""
        output = self.run_command([
            "docker", "service", "ps",
            service_name,
            "--no-trunc",
            "--format", "{{.CurrentState}}\t{{.Error}}"
        ])

        rows = []
        for line in output.splitlines():
            parts = line.split("\t", 1)
            state = parts[0]
            error = parts[1] if len(parts) > 1 else ""
            rows.append((state, error))
        return rows

    def wait_for_service_running(self, service_name, poll_interval=2):
        print(f"Waiting for {service_name} to be running...")

        while True:
            rows = self._poll_service_tasks(service_name)

            if any(state.startswith("Running") for state, _ in rows):
                print(f"✓ {service_name} is running")
                return

            failed = [(s, e) for s, e in rows if s.startswith("Failed") or s.startswith("Rejected")]
            if failed:
                state, err = failed[0]
                detail = f": {err}" if err else ""
                raise RuntimeError(f"{service_name} failed to start ({state}){detail}")

            time.sleep(poll_interval)

    def wait_for_service_to_finish(self, service_name, poll_interval=2):
        print(f"Waiting for {service_name} to finish...")

        while True:
            rows = self._poll_service_tasks(service_name)

            if any(state.startswith("Complete") for state, _ in rows):
                print(f"✓ {service_name} finished successfully")
                return

            failed = [(s, e) for s, e in rows if s.startswith("Failed") or s.startswith("Rejected")]
            if failed:
                state, err = failed[0]
                detail = f": {err}" if err else ""
                raise RuntimeError(f"{service_name} failed ({state}){detail}")

            time.sleep(poll_interval)

    def wait_for_scenario_runner_to_save(self, timeout=60, poll_interval=2):
        """Block until scenario_runner's own log shows it has finished writing
        out its latency results.

        Greps for "Saving scenario latency results" — scenario_manager_
        distributed.py's actual print, right before its post-run
        RecordLatency.save_to_csv() call. NOTE: the reference docker_swarm.py
        this was ported from greps for "Scenario latency results saved"
        (past tense) instead, which never appears anywhere in that print's
        source file — a real, previously-latent bug (this wait has
        presumably always timed out and warned in the original, never
        actually detected completion). Fixed here to match the real string.

        Agents finishing (wait_for_agents_to_finish) and scenario_runner
        finishing its own post-run save are two independent signals — without
        this, stop_scenario_runner() can docker-service-rm the container
        while it's still mid-save, silently dropping the CSV output.
        """
        print("Waiting for ScenarioRunner to finish saving latency results...")

        marker = "Saving scenario latency results"
        elapsed = 0

        while elapsed < timeout:
            try:
                output = self.run_command(["docker", "service", "logs", "scenario_runner"])
            except Exception as e:
                print(f"Warning: could not read scenario_runner logs: {e}")
                return

            if marker in output:
                print("✓ ScenarioRunner finished saving latency results")
                return

            time.sleep(poll_interval)
            elapsed += poll_interval

        print(
            f"Warning: ScenarioRunner did not confirm saving latency results "
            f"within {timeout}s, proceeding with teardown anyway"
        )

    def wait_for_scenario_runner_ready(self, timeout=60, poll_interval=2):
        """Block until scenario_runner's own log shows it has finished loading
        the world and started preparing the configured route.

        Uses "Preparing routes" (scenario_runner_distributed.py's own print,
        right after client.load_world()/world.tick() succeed) as the marker —
        unchanged from the fork this was ported from, since that's the exact
        repo/file it's grepping the logs of.

        wait_for_service_running("scenario_runner") only confirms the
        container is scheduled, not that the Python process inside has
        finished loading the world. Starting agents on a fixed sleep instead
        of this marker lets them connect and start ticking while
        scenario_runner is still mid-load_world(), racing it to reset/hold
        the same world.

        KNOWN LIMITATION (see docs/dagger_latency_recording_run_2026-08-13.md
        in the ScenarioRunner checkout): the first client.load_world() call
        after a fresh carla-server start can hang indefinitely even though
        the server is healthy. This method's `timeout` will correctly detect
        that (raising RuntimeError instead of hanging launch.py forever), but
        launch.py's cleanup() call is commented out (matching the fork it was
        ported from) — the recovery used tonight (removing and recreating
        just the scenario_runner/agent services against the now-warm server)
        is a manual step, not automated here.
        """
        print("Waiting for ScenarioRunner to finish loading the world...")

        marker = "Preparing routes"
        elapsed = 0

        while elapsed < timeout:
            try:
                output = self.run_command(["docker", "service", "logs", "scenario_runner"])
            except Exception as e:
                print(f"Warning: could not read scenario_runner logs: {e}")
                return

            if marker in output:
                print("✓ ScenarioRunner finished loading the world")
                return

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise RuntimeError(
            f"ScenarioRunner did not finish loading the world within {timeout}s "
            f"(never printed {marker!r}); not starting agents against a "
            f"possibly-still-loading world"
        )

    def wait_for_agents_to_finish(self):
        for idx in range(0, len(self.agents_cfg)):
            self.wait_for_service_to_finish(f"rl_agent_{idx}")

    def stop_agents(self):
        """Stop agents services created."""
        for idx in range(0, len(self.agents_cfg)):
            try:
                service = f"rl_agent_{idx}"
                self.run_command(["docker", "service", "rm", service])
                print(f"✓ Service {service} removed")
                self.services.remove(service)
            except Exception as e:
                print(f"Warning: Failed to remove {service}: {e}")

    def stop_scenario_runner(self):
        """Stop scenario runner service."""
        try:
            service = "scenario_runner"
            self.run_command(["docker", "service", "rm", service])
            print(f"✓ Service {service} removed")
            if service in self.services:
                self.services.remove(service)
        except Exception as e:
            print(f"Warning: Failed to remove {service}: {e}")

    def stop_server(self):
        """Stop server created."""
        self.services = self.list_services()
        try:
            service = "carla_server"
            if service not in self.services:
                print(f"Cannot find {service} in list of services")
            self.run_command(["docker", "service", "rm", service])
            print(f"✓ Service {service} removed")
            if service in self.services:
                self.services.remove(service)
            time.sleep(5)
        except Exception as e:
            print(f"Warning: Failed to remove {service}: {e}")

    def stop_primary_server(self):
        """Stop the primary CARLA server created by start_primary_server().

        Missing from the fork this was ported from -- only stop_server()
        (the single-GPU "carla_server" name) exists there, with no
        counterpart for "carla_primary_server". Harmless in practice since
        launch.py's normal success path never stops the CARLA server at all
        (by design, so it stays warm for reuse across runs) and only calls
        this kind of teardown from cleanup(), which is commented out — but a
        real gap if you want to tear a run down completely by hand.
        """
        self.services = self.list_services()
        try:
            service = "carla_primary_server"
            if service not in self.services:
                print(f"Cannot find {service} in list of services")
            self.run_command(["docker", "service", "rm", service])
            print(f"✓ Service {service} removed")
            if service in self.services:
                self.services.remove(service)
            time.sleep(5)
        except Exception as e:
            print(f"Warning: Failed to remove {service}: {e}")

    def stop_secondary_servers(self):
        for idx in range(len(self.secondary_servers)):
            service = f"carla_secondary_server_{idx}"

            try:
                self.run_command(["docker", "service", "rm", service])
                print(f"✓ Service {service} removed")

            except Exception as e:
                print(f"Warning: Failed to remove {service}: {e}")

    def wait_for_secondary_servers(self):
        """
        Wait until all secondary CARLA servers are accepting TCP connections.
        """

        print("Gym: Waiting for all secondary servers to be ready...")

        for idx, server in enumerate(self.secondary_servers_summary):

            host = server["host"]
            port = server["port"]

            print(f"Waiting for secondary server {idx} on {host}:{port}")

            backoff = 0.1

            while True:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)

                try:
                    sock.connect((host, port))
                    sock.close()

                    print(f"✓ Secondary server {idx} is ready!")

                    # give CARLA a little extra initialization time
                    time.sleep(15)

                    break

                except Exception:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 2.0)

    def fix_all_permissions(self):
        """Re-run the mkdir/chmod helper against every mounted log directory
        after a run finishes. The containers write as root, so log/CSV files
        created during the run (not just the directories provisioned before
        it) end up root-owned and non-world-writable — this recursively
        chmods everything back to 777 so the host user can read/overwrite/
        delete them afterward.
        """
        print("Fixing permissions on remote log directories...")

        if getattr(self, "primary_server_node", None) is not None:
            volumes = self.primary_server_volumes if self.save_logs else []
            host_dirs = [volume.split(":")[0] for volume in volumes]
            self.ensure_remote_dirs(self.primary_server_node, host_dirs, tag="fix_primary")

        for idx, secondary in enumerate(getattr(self, "secondary_servers", [])):
            volumes = secondary["volumes"] if self.save_logs else []
            host_dirs = [volume.split(":")[0] for volume in volumes]
            self.ensure_remote_dirs(secondary["hostname"], host_dirs, tag=f"fix_secondary_{idx}")

        for idx, agent_cfg in enumerate(self.agents_cfg):
            command = agent_cfg.get('command')
            volumes = agent_cfg.get('volumes', [])
            host_dirs = self._resolve_agent_latency_host_dir(command, volumes)
            self.ensure_remote_dirs(agent_cfg['hostname'], host_dirs, tag=f"fix_agent_{idx}")

    def force_refresh_primary(self):
        """Force restart ONLY primary server."""

        service = "carla_primary_server"

        print(f"Forcing refresh: {service}")

        self.run_command([
            "docker", "service", "update",
            "--detach=true",
            "--force",
            "--image", self.primary_server_image,
            service
        ])

    def force_refresh_secondaries(self):
        """Force restart all secondary servers."""

        for idx, secondary in enumerate(self.secondary_servers):
            service = f"carla_secondary_server_{idx}"

            print(f"Forcing refresh: {service}")

            self.run_command([
                "docker", "service", "update",
                "--detach=true",
                "--force",
                "--image", secondary["image"],  # key part
                service
            ])
