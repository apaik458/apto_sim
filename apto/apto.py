import os

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class Apto(BaseAgent):
    uid = "apto"
    urdf_path = os.path.join(os.path.dirname(__file__), '../urdf/robot.urdf')

    keyframes = dict(
        home=Keyframe(
            qpos=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            pose=sapien.Pose(),
        )
    )

    camera_joint_names = [
        "camera",
    ]
    arm_joint_names = [
        "left_1",
        "left_2",
        "left_3",
        "left_4",
        "left_5",
        "left_6",
        "right_1",
        "right_2",
        "right_3",
        "right_4",
        "right_5",
        "right_6",
    ]
    left_gripper_joint_names = [
        "gripper_key_left_1",
        "gripper_key_left_2",
    ]
    right_gripper_joint_names = [
        "gripper_key_right_1",
        "gripper_key_right_2",
    ]

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            apto_leftfinger_1=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            apto_leftfinger_2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            apto_rightfinger_1=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            apto_rightfinger_2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        left_gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.left_gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"gripper_key_left_2": {"joint": "gripper_key_left_1"}},
        )
        right_gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.right_gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"gripper_key_right_2": {"joint": "gripper_key_right_1"}},
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=left_gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos, gripper=left_gripper_pd_joint_pos
            ),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="camera",
                pose=sapien.Pose(p=[-0.02, 0.05, 0.03], q=[0, 0, 0.7071788, 0.7070348]),
                width=128,
                height=128,
                fov=80.8 * (np.pi / 180),
                near=0.01,
                far=100,
                mount=self.robot.links_map["servo_frame_3"],
            )
        ]
    
    def _after_init(self):
        self.finger1_link = self.robot.links_map["gripper_pad"]
        self.finger2_link = self.robot.links_map["gripper_pad_2"]
        self.tcp = self.robot.links_map["gripper_pad"]

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
    