# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import string as string_utils
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from agile.rl_env.mdp.commands import UniformVelocityBaseHeightCommand
from agile.rl_env.mdp.utils import get_contact_sensor_cfg, get_robot_cfg


# Note: The command gets updated after the reward is computed resulting in a one-step reward delay.
def track_base_height_exp_smooth(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward the agent for tracking the base height."""
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    base_height_error = torch.square(command_term.base_height - command_term.target_height)
    return torch.exp(-base_height_error / std**2)


def track_lin_vel_xy_yaw_frame_exp_weighted(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel.

    The tracking error is additionally weighted by the command velocity's magnitude. Higher commanded velocities
    receive higher weight, encouraging accurate tracking especially at higher speeds.
    """

    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    command_term = env.command_manager.get_term(command_name)
    vel_cmd = command_term.vel_command_b[:, :2]

    # Compute tracking error
    lin_vel_error = torch.sum(torch.square(vel_cmd - vel_yaw[:, :2]), dim=1)

    # Compute adaptive std based on commanded velocity magnitude
    cmd_vel_magnitude = torch.norm(vel_cmd, dim=1)
    # Define velocity bounds for scaling (assuming min=0.01
    vel_min = 0.01
    vel_max = torch.norm(torch.tensor([command_term.cfg.ranges.lin_vel_x[1], command_term.cfg.ranges.lin_vel_y[1]]))

    # Clamp magnitude to expected range
    cmd_vel_magnitude_clamped = torch.clamp(cmd_vel_magnitude, vel_min, vel_max)

    # Map direct relationship to weight range
    # Using linear interpolation: higher commanded velocity -> higher weight
    weight_min = 1.0
    weight_max = 2.0

    # Normalized direct mapping: high velocity -> high weight, low velocity -> low weight
    normalized = (cmd_vel_magnitude_clamped - vel_min) / (vel_max - vel_min)
    weight = weight_min + (weight_max - weight_min) * normalized

    # Return weighted exponential reward
    return weight * torch.exp(-lin_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp_aligned(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel.
    The reward is scaled by the cosine similarity between the command and the velocity when the command is not null.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])[:, :2]
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]

    cosine_similarity = torch.nn.functional.cosine_similarity(vel_cmd, vel_yaw, dim=1)

    lin_vel_error = torch.sum(
        torch.square(vel_cmd - vel_yaw),
        dim=1,
    )
    is_null_cmd = (vel_cmd == 0).all(dim=1)

    reward = torch.where(
        ~is_null_cmd, torch.exp(-lin_vel_error / std**2) * cosine_similarity, torch.exp(-lin_vel_error / std**2)
    )

    return reward


def vel_xy_in_threshold(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for tracking the linear velocity."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])[:, :2]
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]

    lin_vel_error = torch.linalg.vector_norm(vel_cmd - vel_yaw, dim=1)
    return (lin_vel_error < threshold).float()


def track_base_height(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    command_name: str = "base_velocity_height",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Reward for tracking a target base height.

    Args:
        env: Environment instance.
        std: Standard deviation for the exponential kernel.
        command_name: Name of the command generator.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor.
    """
    # Get the robot asset from the scene
    robot, _ = get_robot_cfg(env, asset_cfg)

    # Get base height from the robot's root state
    base_height = robot.data.root_pos_w[:, 2]

    # Get the command from the command manager
    command = env.command_manager.get_command(command_name)

    is_null_cmd = (command[:, :3] == 0).all(dim=1)

    # Compute height error
    height_error = torch.abs(base_height - command[:, -1])

    # Compute reward (exponential decay with height error)
    reward = torch.exp(-height_error / std**2) * is_null_cmd.float()

    return reward


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward for tracking the target base height with an exponential kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    height_error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-height_error / std**2)


def base_height_in_threshold(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Reward the agent for tracking the base height."""
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    base_height_error = torch.abs(command_term.base_height - command_term.target_height)
    return (base_height_error < threshold).float()


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity_height",
    contact_threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for standing still when velocity commands are near zero.

    Penalizes motion when command velocity is near zero and the robot should be in stance mode.

    Args:
        env: Environment instance.
        command_name: Name of the command generator.
        velocity_threshold: Threshold for considering velocity commands as zero.
        height_threshold: Height threshold for stance mode.
        contact_threshold: Threshold for determining foot contact.
        sensor_cfg: Contact sensor configuration.

    Returns:
        Reward tensor.
    """
    # Create default sensor_cfg if None is provided
    contact_sensor, sensor_cfg = get_contact_sensor_cfg(env, sensor_cfg)

    # Get velocity command from the command manager
    command = env.command_manager.get_command(command_name)

    # Get feet body IDs from sensor config
    feet_body_ids = sensor_cfg.body_ids

    # Get contact forces for feet
    contact_forces = contact_sensor.data.net_forces_w[:, feet_body_ids, 2]

    # Count feet without contact (contact force < threshold)
    feet_without_contact = torch.sum(contact_forces < contact_threshold, dim=-1)

    # Check if robot is in stance mode
    is_null_cmd = (command[:, :3] == 0).all(dim=1)

    # Compute penalty: apply when in stance mode with zero velocity command
    # Penalty is proportional to the number of feet without contact
    # Ensure the reward has shape [num_envs]
    reward = feet_without_contact * is_null_cmd.float()

    return reward


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of base height commands using exponential kernel.
    
    This reward encourages the robot to track the target base height, enabling
    squatting behavior and expanding the operational workspace.
    
    Args:
        env: The environment.
        command_name: Name of the height command.
        std: Standard deviation for the exponential kernel.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        The reward tensor. Shape is (num_envs,).
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get current base height (z position in world frame)
    current_height = asset.data.root_pos_w[:, 2]
    # get target height command
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    target_height = command_term.target_height
    # compute the error
    height_error = torch.square(target_height - current_height)
    return torch.exp(-height_error / std**2)


def track_base_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for height tracking error using L2 norm.
    
    This provides a more direct penalty for height errors, encouraging faster response.
    
    Args:
        env: The environment.
        command_name: Name of the height command.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        The penalty tensor. Shape is (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    target_height = command_term.target_height
    height_error = torch.square(target_height - current_height)
    return height_error  # Negative weight in config makes this a penalty


def track_height_knee_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    knee_joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""Reward for tracking height by encouraging knee flexion/extension.
    
    This reward implements the rknee term from the paper:
    
    rknee = -||(hr,t - ht) × (qknee,t - qknee,min)/(qknee,max - qknee,min - 1/2)||
    
    where:
    - hr,t is the robot's actual height
    - ht is the target height
    - qknee,t is the current positions of robot's knee joints
    - qknee,min and qknee,max are the minimum and maximum positions of knee joints
    
    This reward encourages flexion of the knee joints when hr,t < ht,
    and encourages extension when hr,t > ht.
    
    Args:
        env: The environment.
        command_name: Name of the height command.
        knee_joint_names: List of knee joint names (regex patterns supported).
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        The reward tensor. Shape is (num_envs,).
    """
    # extract the used quantities
    asset = env.scene[asset_cfg.name]
    
    # get current base height (z position in world frame)
    current_height = asset.data.root_pos_w[:, 2]
    # get target height command
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    target_height = command_term.target_height
    
    # find knee joint indices using string matching utilities
    try:
        knee_joint_ids, _ = string_utils.resolve_matching_names(knee_joint_names, asset.joint_names)
    except ValueError:
        # Return zero reward if no knee joints found
        return torch.zeros(env.num_envs, device=env.device)
    
    if len(knee_joint_ids) == 0:
        # Return zero reward if no knee joints found
        return torch.zeros(env.num_envs, device=env.device)
    
    # get knee joint positions
    knee_pos = asset.data.joint_pos[:, knee_joint_ids]  # Shape: (num_envs, num_knee_joints)
    
    # get knee joint limits
    knee_pos_limits = asset.data.default_joint_pos_limits[:, knee_joint_ids]  # Shape: (num_envs, num_knee_joints, 2)
    knee_min = knee_pos_limits[:, :, 0]  # Shape: (num_envs, num_knee_joints)
    knee_max = knee_pos_limits[:, :, 1]  # Shape: (num_envs, num_knee_joints)
    
    # compute normalized knee position: (qknee,t - qknee,min) / (qknee,max - qknee,min) - 1/2
    knee_range = knee_max - knee_min
    # avoid division by zero
    knee_range = torch.clamp(knee_range, min=1e-6)
    normalized_knee = (knee_pos - knee_min) / knee_range - 0.5  # Shape: (num_envs, num_knee_joints)
    
    # compute height error: (hr,t - ht)
    height_error = current_height - target_height  # Shape: (num_envs,)
    
    # compute reward: -||(hr,t - ht) × normalized_knee||
    # Take mean across knee joints for each environment
    reward_term = height_error.unsqueeze(-1) * normalized_knee  # Shape: (num_envs, num_knee_joints)
    reward = -torch.mean(torch.abs(reward_term), dim=1)  # Shape: (num_envs,)
    
    return reward
