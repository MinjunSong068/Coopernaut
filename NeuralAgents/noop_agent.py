#!/usr/bin/env python
"""
NoOpAgent -- isolates the CARLA/V2V simulation-side cost of running Coopernaut's
native pipeline from the cost of real model inference.

Same sensor suite as DaggerAgent (RGB + LIDAR, real V2V sharing still happens),
but never loads a checkpoint and never runs a forward pass: run_step() returns
an immediate no-op VehicleControl()
"""

from __future__ import print_function
import carla
from AVR import Utils
from AVR.autocast_agents.new_agent import NewAgent
from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool


class NoopAgent(AutonomousAgent):
    def __init__(self, path_to_conf_file, num_checkpoint=200, beta=0.9):
        super().__init__(path_to_conf_file)
        self._route_assigned = False
        self._target_speed = 20
        if Utils.EvalEnv.ego_speed_kmph is not None:
            self._target_speed = Utils.EvalEnv.ego_speed_kmph

        # AVR/HUD.py's hud_update() (via DataLogger.compile_actor_state, used
        # whenever --hud is passed) assumes the full DaggerAgent attribute
        # surface, including a real NewAgent for local-planner/V2V-filtered-
        # object-list bookkeeping -- match DaggerAgent's own pattern (it
        # creates a NewAgent too, only to set the global plan on its local
        # planner; the actual per-frame action always comes from
        # self.student.run_step(), never from this NewAgent's own run_step())
        # rather than dropping --hud, since the real-inference comparison run
        # also used --hud and per-frame HUD/BEV work is part of what's being
        # measured.
        self._agent = None
        self._agent_control = None
        self._expert_control = None
        self._trainee_planned_path = None
        self._route_assigned = False
        self.agent_trajectory_points_timestamp = []
        self.collider_trajectory_points_timestamp = []
        self.next_target_location = None
        self.drawing_object_list = []

    def run_step(self, input_data, timestamp):
        if not self._agent:
            hero_actor = CarlaActorPool.get_hero_actor()
            if hero_actor:
                self._agent = NewAgent(hero_actor, self._target_speed)

        if not self._route_assigned:
            if self._global_plan:
                plan = []
                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True
                print("Global Plan set")

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        self._agent_control = control
        return control

    def sensors(self):
        sensors = [
            {'type': 'sensor.camera.rgb',
             'x': Utils.LidarRoofForwardDistance, 'y': 0.0, 'z': Utils.LidarRange,
             'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
             'width': 720, 'height': 720, 'fov': 90,
             'id': 'RGB'},

            {'type': 'sensor.lidar.ray_cast',
             'x': Utils.LidarRoofForwardDistance, 'y': 0.0, 'z': Utils.LidarRoofTopDistance,
             'yaw': Utils.LidarYawCorrection, 'pitch': 0.0, 'roll': 0.0,
             'range': Utils.LidarRange,
             'rotation_frequency': 20,
             'channels': 64,
             'upper_fov': 4,
             'lower_fov': -20,
             'points_per_second': 2304000,
             'id': 'LIDAR'},
        ]

        return sensors
