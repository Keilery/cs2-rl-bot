# Setup guide

## 1. Install Python deps

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e '.[dev]'           # core + tests
pip install -e '.[dev,vision]'    # add YOLOv8
pip install -e '.[dev,windows]'   # Windows controller backend
pip install -e '.[dev,linux]'     # Linux controller backend
```

CUDA users: install a `torch` build matching your CUDA toolkit *before* running
`pip install -e .` — see [pytorch.org](https://pytorch.org/get-started/locally/).

## 2. Install the GSI config

Copy `cfg/gamestate_integration_cs2_rl_bot.cfg` into the CS2 cfg directory:

| OS      | Path |
|---------|------|
| Windows | `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg\` |
| Linux   | `~/.steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/cfg/` |
| macOS   | (CS2 not officially supported on macOS) |

**Edit the auth token** in both the cfg file and `cfg/default.yaml` so they
match. Restart CS2 after copying.

## 3. Verify the GSI listener

```bash
cs2-rl-bot gsi --config cfg/default.yaml
```

Then launch CS2 and start an offline match (see step 5). You should see JSON
payloads streaming to the terminal once a round begins.

## 4. Verify screen capture

```bash
cs2-rl-bot capture
```

This prints the captured frame's shape (e.g. `(84, 84, 3)`) and exits. If you
see `No frame captured`, the process is running headless — set
`capture.monitor_index` to the correct display.

## 5. Start an offline match

In CS2, open the developer console (`~`) and run:

```
map de_dust2
mp_warmup_end; mp_freezetime 0
sv_cheats 1
bot_add t; bot_add t; bot_add t; bot_add t
bot_difficulty 0
```

Or use the matchmaking UI: **Play → Practice with Bots**.

## 6. Inference (no learning)

```bash
cs2-rl-bot inference --config cfg/default.yaml --max-steps 500
```

By default this uses the random agent because `agent.algo=ppo` requires a
trained checkpoint. Switch to a deterministic baseline with:

```bash
CS2BOT_AGENT__ALGO=scripted cs2-rl-bot inference
```

To actually move the in-game crosshair you need to disable `dry_run` and set
the controller backend:

```bash
CS2BOT_DRY_RUN=false \
CS2BOT_CONTROLLER__BACKEND=pydirectinput \   # or pynput on Linux
cs2-rl-bot inference
```

## 7. Training

Read [`legal_and_safety.md`](legal_and_safety.md) first. Then:

```bash
CS2BOT_DRY_RUN=false \
CS2BOT_CONTROLLER__BACKEND=pydirectinput \
cs2-rl-bot train --steps 200000 --save-every 10000 --i-understand-the-risks
```

Each invocation creates `runs/<run_id>/`. Tensorboard logs go to
`runs/<run_id>/tensorboard/`, checkpoints go to
`runs/<run_id>/checkpoints/`.

To resume a previous run:

```bash
cs2-rl-bot runs                         # list runs and pick an id
cs2-rl-bot train --resume <run_id> --steps 100000 --i-understand-the-risks
```

## 8. Live monitoring + pause / stop

Open a second terminal and attach the dashboard:

```bash
cs2-rl-bot dashboard <run_id>
```

Hotkeys (active when the dashboard window is focused):

| key | action                                                |
| --- | ----------------------------------------------------- |
|  p  | pause training (the trainer busy-waits between steps) |
|  r  | resume training                                       |
|  s  | save a checkpoint *now*                               |
|  q  | request a graceful stop, then detach                  |

The dashboard never communicates with the trainer directly; it writes commands
to `runs/<run_id>/control.json` and reads metrics from
`runs/<run_id>/status.json`. This means it works fine over SSH, in tmux, or
even on a different machine that shares the run directory.

For an interactive launcher (start / resume / attach / pretrain / yolo) run:

```bash
cs2-rl-bot menu
```

## 9. Optional: vision (YOLOv8)

YOLOv8 is disabled by default. To enable:

1. `pip install -e '.[vision]'`
2. Drop a fine-tuned `*.pt` weights file into `models/` (see
   [`architecture.md`](architecture.md)).
3. Set `vision.enabled: true` and `vision.model_path: models/your_model.pt`
   in `cfg/default.yaml`.

To fine-tune YOLOv8 directly from a Roboflow dataset:

```bash
export ROBOFLOW_API_KEY=...
cs2-rl-bot train-yolo path/to/data.yaml --epochs 60 --batch 16
```

The fine-tuned weights land in `runs/<yolo_run_id>/checkpoints/yolo.pt` and a
`vision_config.yaml` snippet is written next to them — copy that snippet into
`cfg/default.yaml` to use the new model.

## 10. Optional: imitation learning from demos

Install the demos extra and warm-start PPO from `.dem` files:

```bash
pip install -e '.[demos]'
cs2-rl-bot pretrain-bc path/to/demos/ --epochs 5 \
    --steamid 76561198000000000     # optional: only learn from one player
cs2-rl-bot train --resume <bc_run_id> --steps 200000 --i-understand-the-risks
```

The BC step only trains the **scalar** policy head (demos don't contain
frame pixels), so the CNN over rendered frames still learns from scratch
during PPO.
