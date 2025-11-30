# ESE 651: Physical Intelligence - Drone Racing Project

This project repository contains the code for the drone racing environment, training, and evaluation using NVIDIA Isaac Lab. The repo also includes a custom copy of the rsl_rl robot learning library where you will implement Proximal Policy Optimization (PPO).

## Setup Instructions

* Enter your **home** directory, then git clone the project repository. It is critical that the project repo and the Isaac Lab directory are at the same level.

```bash
git clone git@github.com:vineetpasumarti/ese651_project.git
```

## Training

In order to train, we can call the following command from terminal:

```bash
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \
    --max_iterations 5000 \
    --headless

python  scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 2048 \
    --experiment_name gptv1 \
    --max_iterations 2000 \
    --headless \
    --logger wandb

python  scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 16 \
    --experiment_name gptv1 \
    --max_iterations 2000 \
    --logger wandb

python  scripts/rsl_rl/train_race.py \
    hydra.searchpath="[file://home/kostas-lab/Documents/ESE651/ESE651/src/third_parties/rsl_rl_local/config]" \
    --task Isaac-Quadcopter-Race-v0 \
    --config-name dummy_config.yaml \
    --num_envs 8192 \
    --experiment_name gptv1 \
    --max_iterations 20000 \
    --headless \
    --logger wandb

```

## Evaluation

In order to play, we can call the following example command from terminal:

```bash
python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --load_run [YYYY-MM-DD_XX-XX-XX] \  # The run directory is in logs/rsl_rl/quadcopter_direct/
    --checkpoint best_model.pt \
    --headless \
    --video \
    --video_length 800

python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --load_run 2025-11-17_23-23-43 \
    --checkpoint best_model.pt \
    --headless \
    --video \
    --video_length 1200
```

**Note:** Neither command will work until PPO is implemented as per the next section.

## Requirements

- NVIDIA Isaac Lab
- Custom rsl_rl library (included in this repository)
- PPO implementation (to be completed)

## Directory Structure

The project repository should be cloned at the same level as your Isaac Lab installation to ensure proper integration.