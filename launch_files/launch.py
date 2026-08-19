#!/usr/bin/env python3
#
# Usage: python3 launch_files/launch.py [path/to/launch_nodes.yaml]
# (defaults to launch_files/launch_nodes.yaml)

import sys
import time

from docker_swarm.docker_swarm import SwarmManager


def cleanup(manager):
    print("\nCleaning up...")

    manager.stop_scenario_runner()

    try:
        manager.stop_agents()
    except Exception as e:
        print(e)

    try:
        manager.stop_metrics_recorders()
        manager.stop_primary_metrics_recorder()
    except Exception as e:
        print(e)

    try:
        manager.stop_secondary_servers()
    except Exception as e:
        print(e)

    try:
        manager.stop_server()
    except Exception:
        pass

    try:
        manager.stop_primary_server()
    except Exception:
        pass

    try:
        manager.stop_all()
    except Exception:
        pass

    try:
        manager.fix_all_permissions()
    except Exception as e:
        print(e)


def main():

    config_path = sys.argv[1] if len(sys.argv) > 1 else "launch_files/launch_nodes.yaml"
    manager = SwarmManager(config_path)

    try:

        # Primary CARLA
        manager.start_primary_server()

        # Secondary CARLA (placeholder — see launch_nodes.yaml note)
        manager.start_secondary_servers()

        # Wait for secondary servers
        manager.wait_for_secondary_servers()

        # CPU/GPU recording on secondary server nodes (placeholder)
        manager.start_metrics_recorders()

        # CPU/GPU recording on the primary node (placeholder)
        manager.start_primary_metrics_recorder()

        # Wait for servers to initialize
        time.sleep(10)

        # ScenarioRunner
        manager.start_scenario_runner()

        # Give ScenarioRunner time to initialize
        manager.wait_for_service_running("scenario_runner")
        manager.wait_for_scenario_runner_ready()

        # Agents (placeholder — see launch_nodes.yaml note)
        manager.start_agents()

        # Wait for agents to finish
        manager.wait_for_agents_to_finish()

        print("All agents completed.")

        manager.wait_for_scenario_runner_to_save()

        manager.stop_metrics_recorders()
        manager.stop_primary_metrics_recorder()

        manager.stop_scenario_runner()

        print("Ended Scenario Runner")

        manager.fix_all_permissions()

    except KeyboardInterrupt:
        print("Interrupted.")

    # finally:
    #     cleanup(manager)


if __name__ == "__main__":
    main()
