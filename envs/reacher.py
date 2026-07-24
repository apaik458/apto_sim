from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat
from apto import Apto, AptoSimple

REACHER_DOC_STRING = """**Task Description:**
A simple task where the objective is to move the robot's end effector close to a target that is spawned at a random position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the target goal position (marked by a green sphere) has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the Reacher never terminates
"""


@register_env("ReacherApto-v1", max_episode_steps=50)
class ReacherAptoEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["apto, apto_simple"]
    agent: Union[Apto, AptoSimple]
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="apto_simple", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.goal_thresh = 0.0125 * 1.25
        self.max_goal_height = 0.3
        self.sensor_cam_eye_pos = [-0.27, 0, 0.4]
        self.sensor_cam_target_pos = [-0.56, 0, -0.25]
        self.human_cam_eye_pos = [0.5, 0.5, 0.6]
        self.human_cam_target_pos = [0.0, 0.0, 0.0]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.pi/2)))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b)) * 0.2 + 0.05
            goal_xyz[:, 1] = torch.rand((b)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + 0.01

            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose_left=self.agent.tcp_pose_left.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        # if "state" in self.obs_mode:
        #     obs.update(tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp_pose_left.p)
        # print(f"Observation: {obs['tcp_pose_left'].shape}, {obs['goal_pos'].shape}") # torch.Size([8, 7]), torch.Size([8, 3])
        return obs

    def evaluate(self):
        is_goal_reached = (
            torch.linalg.norm(self.agent.tcp_pose_left.p - self.goal_site.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_goal_reached & is_robot_static,
            "is_obj_placed": is_goal_reached,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # reward = reward_distance
        # - reward_distance: tcp distance from goal

        # reward_distance
        tcp_goal_pose = Pose.create_from_pq(
            p=self.goal_site.pose.p + torch.tensor([-self.goal_thresh - 0.005, 0, 0], device=self.device)
        )
        tcp_to_goal_dist = torch.linalg.norm(self.agent.tcp_pose_left.p - tcp_goal_pose.p, axis=1)
        reward_near_weight = 1
        reward_distance = -reward_near_weight * tcp_to_goal_dist

        reward = reward_distance
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
