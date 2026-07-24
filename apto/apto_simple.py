import os
from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class AptoSimple(BaseAgent):
    uid = "apto_simple"
    urdf_path = os.path.join(os.path.dirname(__file__), '../urdf/robot_simple.urdf')

    keyframes = dict(
        home=Keyframe(
            qpos=np.array([
                0.0, # camera
                0.0, # left_1
                0.0, # right_1
                0.0, # left_2
                0.0, # right_2
                0.0, # left_3
                0.0, # right_3
                0.0, # left_4
                0.0, # right_4
                0.0, # left_5
                0.0, # right_5
                0.0, # left_6
                0.0, # right_6
            ]),
            pose=sapien.Pose(),
        )
    )

    camera_joint_names = [
        "camera",
    ]
    left_arm_joint_names = [
        "left_1",
        "left_2",
        "left_3",
        "left_4",
        "left_5",
        "left_6",
    ]
    right_arm_joint_names = [
        "right_1",
        "right_2",
        "right_3",
        "right_4",
        "right_5",
        "right_6",
    ]
    left_ee_link_name = "tcp_left"
    right_ee_link_name = "tcp_right"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100


    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Left Arm
        # -------------------------------------------------------------------------- #
        left_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        left_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(left_arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True


        # -------------------------------------------------------------------------- #
        # Right Arm
        # -------------------------------------------------------------------------- #
        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        right_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(right_arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        
        # -------------------------------------------------------------------------- #
        # Controller Configuration Dictionary
        # -------------------------------------------------------------------------- #
        controller_configs = dict(
            # Dual-arm coordinated control modes
            pd_joint_delta_pos_dual_arm=dict(
                left_arm=left_arm_pd_joint_delta_pos,
                right_arm=right_arm_pd_joint_delta_pos,
            ),
            pd_joint_pos_dual_arm=dict(
                left_arm=left_arm_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
            ),

            # Left arm individual control
            pd_joint_pos_left_arm=dict(
                arm=left_arm_pd_joint_pos,
            ),
            pd_joint_delta_pos_left_arm=dict(
                arm=left_arm_pd_joint_delta_pos,
            ),

            # Right arm individual control
            pd_joint_pos_right_arm=dict(
                arm=right_arm_pd_joint_pos,
            ),
            pd_joint_delta_pos_right_arm=dict(
                arm=right_arm_pd_joint_delta_pos,
            ),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    # @property
    # def _sensor_configs(self):
    #     return [
    #         CameraConfig(
    #             uid="camera",
    #             pose=sapien.Pose(p=[-0.02, 0.05, 0.03], q=[0, 0, 0.7071788, 0.7070348]),
    #             width=128,
    #             height=128,
    #             fov=80.8 * (np.pi / 180),
    #             near=0.01,
    #             far=100,
    #             mount=self.robot.links_map["servo_frame_3"],
    #         )
    #     ]
    
    def _after_init(self):
        self.left_tcp = self.robot.links_map[self.left_ee_link_name]
        self.right_tcp = self.robot.links_map[self.right_ee_link_name]

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()  # Get all joint velocities (12 DOF for dual arm)
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos_left(self):
        return self.left_tcp.pose.p

    @property
    def tcp_pose_left(self):
        return self.left_tcp.pose
    
    @property
    def tcp_pos_right(self):
        return self.right_tcp.pose.p

    @property
    def tcp_pose_right(self):
        return self.right_tcp.pose
    