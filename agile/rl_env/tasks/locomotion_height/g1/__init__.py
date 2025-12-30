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

import gymnasium as gym

from . import agents

###########
# RL envs #
###########

# Variant 1: Current configuration (Moving during walking, With curriculum, Random height during walking)
gym.register(
    id="Velocity-Height-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg:G1LowerVelocityHeightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightPpoRunnerCfg",
    },
)

# Variant 2: Static during walking, No curriculum, Random height during walking
gym.register(
    id="Velocity-Height-G1-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg_static_nocurr:G1LowerVelocityHeightEnvCfgV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightPpoRunnerCfgV2",
    },
)

# Variant 3: Moving during walking, No curriculum, Random height during walking
gym.register(
    id="Velocity-Height-G1-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg_moving_nocurr:G1LowerVelocityHeightEnvCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightPpoRunnerCfgV3",
    },
)

# Variant 4: Moving during walking, With curriculum, Same height during walking
gym.register(
    id="Velocity-Height-G1-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg_moving_curr_sameheight:G1LowerVelocityHeightEnvCfgV4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightPpoRunnerCfgV4",
    },
)

# Variant 5: Static during walking, No curriculum, Same height during walking
gym.register(
    id="Velocity-Height-G1-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg_static_nocurr_sameheight:G1LowerVelocityHeightEnvCfgV5",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightPpoRunnerCfgV5",
    },
)


################################
# Teacher-Student Distillation #
################################

gym.register(
    id="Velocity-Height-G1-Distillation-Recurrent-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg:G1VelocityHeightRecurrentStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightDistillationRecurrentRunnerCfg",
    },
)

gym.register(
    id="Velocity-Height-G1-Distillation-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg:G1VelocityHeightHistoryStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightDistillationHistoryRunnerCfg",
    },
)

# Flat terrain variant: No terrain curriculum, flat terrain only
gym.register(
    id="Velocity-Height-G1-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg_flat:G1LowerVelocityHeightFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightFlatPpoRunnerCfg",
    },
)

# Flat terrain with linear velocity profile variant
gym.register(
    id="Velocity-Height-G1-Flat-Linear-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg_flat_linear:G1LowerVelocityHeightFlatLinearEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightFlatLinearPpoRunnerCfg",
    },
)

# Flat terrain with linear velocity profile, no curriculum variant
gym.register(
    id="Velocity-Height-G1-Flat-Linear-NoCurr-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg_flat_linear_nocurr:G1LowerVelocityHeightFlatLinearNoCurrEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightFlatLinearNoCurrPpoRunnerCfg",
    },
)

# Flat terrain with linear velocity profile, static during walking, no curriculum variant
gym.register(
    id="Velocity-Height-G1-Flat-Linear-Static-NoCurr-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg_flat_linear_static_nocurr:G1LowerVelocityHeightFlatLinearStaticNoCurrEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityHeightFlatLinearStaticNoCurrPpoRunnerCfg",
    },
)
