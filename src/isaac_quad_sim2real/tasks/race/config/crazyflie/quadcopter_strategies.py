# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

def catmull_rom_point(P0, P1, P2, P3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2*P1 +
        (-P0 + P2) * t +
        (2*P0 - 5*P1 + 4*P2 - P3) * t2 +
        (-P0 + 3*P1 - 3*P2 + P3) * t3
    )

def catmull_rom_tangent(P0, P1, P2, P3, t):
    t2 = t * t
    return 0.5 * (
        (-P0 + P2) +
        2*(2*P0 - 5*P1 + 4*P2 - P3) * t +
        3*(-P0 + 3*P1 - 3*P2 + P3) * t2
    )

class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        # check to change waypoint
        dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        gate_passed = dist_to_gate < 0.1
        ids_gate_passed = torch.where(gate_passed)[0]
        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]

        # set desired positions in the world frame
        self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        # calculate progress via distance to goal
        distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
        distance_to_goal = torch.tanh(distance_to_goal/3.0)
        progress = 1 - distance_to_goal  # distance_to_goal is between 0 and 1 where 0 means the drone reached the goal

        # compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        #### Basic drone states, modify for your needs)
        drone_pose_w = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_quat_w = self.env._robot.data.root_quat_w

        ##### Some example observations you may want to explore using
        # Angular velocities (referred to as body rates)
        # drone_ang_vel_b = self.env._robot.data.root_ang_vel_b  # [roll_rate, pitch_rate, yaw_rate]

        # Current target gate information
        current_gate_idx = self.env._idx_wp
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]  # World position of current gate
        current_gate_yaw = self.env._waypoints[current_gate_idx, -1].unsqueeze(1)    # Yaw orientation of current gate
    
        # Relative position to current gate in gate frame
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Relative position to current gate in body frame
        gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            current_gate_pos_w
        )
        
        # get the position and yaw of next gate
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        next_gate_yaw = self.env._waypoints[next_gate_idx, -1].unsqueeze(1)
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            next_gate_pos_w
        )

        # Previous actions
        # prev_actions = self.env._previous_actions  # Shape: (num_envs, 4)

        # Number of gates passed
        # gates_passed = self.env._n_gates_passed.unsqueeze(1).float()

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                drone_pose_w,       # position in the world frame (3 dims)
                drone_lin_vel_b,    # velocity in the body frame (3 dims)
                drone_quat_w,       # quaternion in the world frame (4 dims)
                drone_pos_gate_frame,
                gate_pos_b,
                current_gate_yaw,
                next_gate_pos_b,
                next_gate_yaw,
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # start from the zeroth waypoint (beginning of the race)
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = -2.0 * torch.ones(n_reset, device=self.device)
        y_local = torch.zeros(n_reset, device=self.device)
        z_local = torch.zeros(n_reset, device=self.device)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate = torch.ones(self.num_envs, device=self.device)

        self.env._crashed[env_ids] = 0
        
        
class PassGateGPT1(DefaultQuadcopterStrategy):
    """A strategy that extends the DefaultQuadcopterStrategy with a simple modification."""

    def get_rewards(self) -> torch.Tensor:
        """Reward function optimized for quadcopter gate tracking and racing."""
        
        # ---------------------------------------------------------
        # 1. Core tracking signals
        # ---------------------------------------------------------
        # (A) Forward progress: distance to goal decreases
        dist_now = torch.linalg.norm(
                    self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
        dist_prev = self.env._last_distance_to_goal                  # (N,)
        progress_delta = dist_prev - dist_now
        progress_reward = torch.tanh(progress_delta * 2.0) # / 2.0

        # (B) Gate position alignment (drone in gate frame)
        pos_gate_frame = self.env._pose_drone_wrt_gate
        lateral_error = torch.linalg.norm(pos_gate_frame[:, :2], dim=1)
        alignment_reward = torch.exp(-3.0 * lateral_error)
        
        # (C) Yaw alignment
        drone_quat = self.env._robot.data.root_quat_w
        drone_yaw = euler_xyz_from_quat(drone_quat)[2]

        gate_quat = self.env._waypoints_quat[self.env._idx_wp, :]
        gate_yaw = euler_xyz_from_quat(gate_quat)[2]
        yaw_error = torch.abs(wrap_to_pi(gate_yaw) - wrap_to_pi(drone_yaw))
        yaw_reward = torch.exp(-1.5 * torch.abs(yaw_error))

        # (D) Velocity alignment (reward forward body velocity)
        vel_body = self.env._robot.data.root_com_lin_vel_b
        forward_vel = vel_body[:, 0]
        velocity_reward = torch.tanh(forward_vel / 1.0).clamp(min=0.0)
        
        # ---------------------------------------------------------
        # 2. Penalties
        # ---------------------------------------------------------
        # Action smoothness penalty
        actions = self.env._actions
        prev_actions = self.env._previous_actions
        action_change = torch.linalg.norm(actions - prev_actions, dim=1)
        smoothness_penalty = -0.05 * action_change

        # Angular rate penalty
        ang_vel = self.env._robot.data.root_ang_vel_b
        ang_rate_penalty = -0.02 * torch.linalg.norm(ang_vel, dim=1)

        # Crash penalty (from contact forces)
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        # crash_penalty = crashed * (-10.0)

        # ---------------------------------------------------------
        # 3. Detect gate crossing
        # ---------------------------------------------------------
        # dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        # gate_passed = (dist_to_gate < 0.15).float()   # larger threshold for racing
        
        dist_to_gate_yz = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, [1, 2]], dim=1)
        x_now = self.env._pose_drone_wrt_gate[:, 0]
        x_prev = self.env._prev_x_drone_wrt_gate
        
        lateral_tolerance, x_tolerance = 0.4, 0.2
        involume = (
            (dist_to_gate_yz < lateral_tolerance) & (x_now < 0) & (x_now > -x_tolerance)
        ).float()
        gate_passed = (
            (dist_to_gate_yz < lateral_tolerance) & ( torch.abs(x_now) < x_tolerance / 4) # (x_prev < 0) & (x_now > 0)
        ).float()
        
        # switch to progress in x_direction
        progress_reward = torch.where(
            involume > 0., 4 * torch.tanh( (x_now - x_prev) * 2.0 ), progress_reward
        )

        # if self.cfg.is_train:
        #     cross_plane = (x_prev < 0) & (x_now > 0)
        #     gate_passed = ((dist_to_gate_yz < 0.4) & cross_plane).float()
        # else:
        #     cross_plane = torch.abs(x_now) < 0.2
        #     gate_passed = ((dist_to_gate_yz < 0.4) & cross_plane).float()
        
        ids_gate_passed = torch.where(gate_passed)[0]
        
        # Advance waypoint for passed envs
        if len(ids_gate_passed) > 0:
            # Advance waypoint
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]

            # Update desired goal
            self.env._desired_pos_w[ids_gate_passed, :3] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
            # self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]
            
            # MUST re-compute pose wrt NEW gate
            new_pos_gate, _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3]
            )       
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pos_gate[:, 0]
        
        # ---------------------------------------------------------
        gate_bonus = gate_passed * torch.exp( - dist_to_gate_yz / 0.4 )

        # ---------------------------------------------------------
        # 4. Combine rewards
        # ---------------------------------------------------------
        reward = (
            + 20.0 * (1 - gate_passed) * progress_reward # if drone passes gate, no more progress reward until next gate
            # + 2.0 * alignment_reward
            + involume * 2.0 * yaw_reward
            # + 0.1 * velocity_reward
            + 40.0 * gate_bonus # gate bonus
            + smoothness_penalty
            + ang_rate_penalty
            + (-10.0) * crashed
        )

        if self.cfg.is_train:
            # Apply terminal penalty
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'],
                                reward)
            
            # ---------- Update last_distance_to_goal for next timestep ----------
            # For envs that passed a gate, reinitialize last_distance_to_goal to the new goal distance
            # (so progress next step is computed relative to the new waypoint)
            if len(ids_gate_passed) > 0:
                # compute distance from current pos to new goal for those envs
                self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(
                    self.env._desired_pos_w[ids_gate_passed, :3] - self.env._robot.data.root_link_pos_w[ids_gate_passed, :3], dim=1
                )
        
            # for all other envs, just update to current distance
            remaining = torch.tensor([i for i in range(self.num_envs) if i not in ids_gate_passed], device=self.device, dtype=torch.long)
            if remaining.numel() > 0:
                self.env._last_distance_to_goal[remaining] = dist_now[remaining].clone()
                self.env._prev_x_drone_wrt_gate[remaining] = x_now[remaining].clone()

            # Logging
            rewards = {
                "progress_goal": progress_reward, 
                "gate_passed": gate_passed,
                "crash": crashed
            }
            for key, value in rewards.items():
                self._episode_sums[key] += value
                
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward
    
    def reset_idx(self, env_ids):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        num_gates = self.env._waypoints.shape[0]

        # 1) pick a random gate for each env
        gate_idx = torch.randint(
            low=0,
            high=num_gates,
            size=(n_reset,),
            device=self.device
        )

        # 2) randomly choose: spawn BEFORE (0) or AFTER (1) the gate
        spawn_after = torch.randint(
            low=0,
            high=2,
            size=(n_reset,),
            device=self.device
        ).float()

        # gate pose
        gate_x = self.env._waypoints[gate_idx][:, 0]
        gate_y = self.env._waypoints[gate_idx][:, 1]
        gate_z = self.env._waypoints[gate_idx][:, 2]
        gate_yaw = self.env._waypoints[gate_idx][:, -1]
        
        # 3) local-frame spawn offsets
        # BEFORE gate: x_local ∈ [-3, -1]
        # AFTER gate:  x_local ∈ [ 1,  3]
        dist_before = -(torch.rand(n_reset, device=self.device) * 2.0 + 1.0)     # [-3, -1]
        dist_after  =   torch.rand(n_reset, device=self.device) * 2.0 + 1.0      # [ 1,  3]
        x_local = dist_before * (1 - spawn_after) + dist_after * spawn_after

        y_local = (torch.rand(n_reset, device=self.device) - 0.5) * 1.0           # ±0.5 m
        z_local = (torch.rand(n_reset, device=self.device) - 0.5) * 0.2           # ±0.1 m

        # 4) rotate from gate frame → world frame
        cos_t = torch.cos(gate_yaw)
        sin_t = torch.sin(gate_yaw)
        x_rot = cos_t * x_local - sin_t * y_local
        y_rot = sin_t * x_local + cos_t * y_local

        spawn_x = gate_x - x_rot
        spawn_y = gate_y - y_rot
        spawn_z = gate_z + z_local

        default_root_state[:, 0] = spawn_x
        default_root_state[:, 1] = spawn_y
        default_root_state[:, 2] = spawn_z

        # point drone towards the zeroth gate
        target_gate = ((gate_idx + spawn_after.long()) % num_gates).int()
        tx = self.env._waypoints[target_gate][:, 0]
        ty = self.env._waypoints[target_gate][:, 1]
        initial_yaw = torch.atan2(ty - spawn_y, tx - spawn_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            target_gate = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = (target_gate)#.int()

        self.env._desired_pos_w[env_ids, :3] = self.env._waypoints[target_gate, :3].clone()
        # self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :3] - self.env._robot.data.root_link_pos_w[env_ids, :3], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].clone()
        self.env._crashed[env_ids] = 0
        
class PassGateGPT2(PassGateGPT1):
    
    def Catmull_Rom_spline(self, gate_idx):
        num_gates = self.env._waypoints.shape[0]
        gate_id_prev = (gate_idx - 1) % num_gates
        gate_id_next = (gate_idx + 1) % num_gates
        gate_id_nextp = (gate_idx + 2) % num_gates
        
        P0 = self.env._waypoints[gate_id_prev, :3]
        P1 = self.env._waypoints[gate_idx, :3]
        P2 = self.env._waypoints[gate_id_next, :3]
        P3 = self.env._waypoints[gate_id_nextp, :3]
        
        noise = torch.randn_like(P1) * self.initial_noise
        t = torch.rand((len(gate_idx), 1), device=self.device)
        spawn_pts = catmull_rom_point(P0, P1, P2, P3, t) + noise

        # point drone towards the zeroth gate
        target_gate = gate_id_next.int()
        tx = self.env._waypoints[target_gate][:, 0]
        ty = self.env._waypoints[target_gate][:, 1]
        initial_yaw = torch.atan2(ty - spawn_pts[:, 1], tx - spawn_pts[:, 0])
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        
        return spawn_pts, quat
    
    def Hermit_spline(self, gate_idx):
        num_gates = self.env._waypoints.shape[0]
        gate_idx_next = (gate_idx + 1) % num_gates
        
        def hermite_segment(p0, p1, m0, m1, t):
            # t = np.linspace(0, 1, n)[:, None]   # shape (n,1)
            t2 = t * t
            t3 = t2 * t

            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2
            return h00*p0 + h10*m0 + h01*p1 + h11*m1
        
        P0 = self.env._waypoints[gate_idx, :3]
        P1 = self.env._waypoints[gate_idx_next, :3]
        
        P0_yaw = self.env._waypoints[gate_idx, -1]
        P1_yaw = self.env._waypoints[gate_idx_next, -1]
        m0 = torch.stack([torch.cos(P0_yaw), torch.sin(P0_yaw), torch.zeros_like(P0_yaw)], dim=1) * 8.0
        m1 = torch.stack([torch.cos(P1_yaw), torch.sin(P1_yaw), torch.zeros_like(P1_yaw)], dim=1) * 8.0
        
        max_ratio = max(0, 0.85 - self.initial_noise)
        t = (0.9 - max_ratio) * torch.rand((len(gate_idx), 1), device=P0.device) + max_ratio
        
        noise = torch.randn_like(P1) * self.initial_noise
        spawn_pts = hermite_segment(P0, P1, m0, m1, t) + noise
        
        # point drone towards the zeroth gate
        target_gate = gate_idx_next.int()
        tx = self.env._waypoints[target_gate][:, 0]
        ty = self.env._waypoints[target_gate][:, 1]
        initial_yaw = torch.atan2(ty - spawn_pts[:, 1], tx - spawn_pts[:, 0])
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        
        return spawn_pts, quat
        
    def reset_idx(self, env_ids):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        gate_passed = self._episode_sums["gate_passed"]
        self.initial_noise = max(0.05, torch.mean(gate_passed).item() * 4)
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.
        num_gates = self.env._waypoints.shape[0]
        gate_idx = torch.randint(0, num_gates, (n_reset,), device=self.device) # .item()
        spawn_pts, quat = self.Hermit_spline(gate_idx)
        target_gate = ((gate_idx + 1) % num_gates).int()
        default_root_state[:, :3] = spawn_pts
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            target_gate = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = (target_gate)#.int()

        self.env._desired_pos_w[env_ids, :3] = self.env._waypoints[target_gate, :3].clone()
        # self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :3] - self.env._robot.data.root_link_pos_w[env_ids, :3], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].clone()
        self.env._crashed[env_ids] = 0
    
    