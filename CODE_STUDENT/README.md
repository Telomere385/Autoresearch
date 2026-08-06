# CODE_STUDENT File Structure

```text
CODE_STUDENT/
  assets/
    audio/        Game sound effects in .wav and .ogg formats.
    sprites/      Flappy Bird sprites, backgrounds, pipes, score digits, and UI images.

  data/
    prefill/      Prefill experiment average-reward arrays.
    q_tables/     Saved Q-table .npy files used for Q-learning play/training.
    rewards/      Saved reward-history and average-reward .npy files.

  media/          Demo media and packaging assets, including GIF, MP4, screenshot, and icon.

  scripts/        Utility scripts for plotting rewards and inspecting Q-tables.

  src/            Source code for the game, MPC controllers, noisy controller, and Q-learning agent.

  .gitignore      Ignore rules for caches, logs, and local generated outputs.
  LICENSE         Project license.
  Pipfile         Pipenv dependency declaration.
  Pipfile.lock    Locked Pipenv dependency metadata.
  requirements.txt
                  Minimal pip dependency list.
  setup.py        Windows py2exe packaging script.
```

## Unified Experiment Entry

Run a reproducible, non-interactive Q-learning experiment from YAML config:

```powershell
python run_experiment.py --mode train --config configs/default.yaml
```

Run evaluation without Q-value updates:

```powershell
python run_experiment.py --mode eval --config configs/default.yaml --set input_q_table="runs/<run_id>/q_table.npy"
```

The default configuration is in `configs/default.yaml`. Individual values can be overridden without editing source code:

```powershell
python run_experiment.py --mode train --run-id trial_eps_001 --set epsilon_start=0.01 --set training_episodes=500
```

Each run writes structured outputs under `runs/<run_id>/`:

```text
runs/<run_id>/
  config.json
  source_config.yaml
  run.log
  result.json
  q_table.npy
  rewards.npy
  average_rewards.npy
```
