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
cs2-rl-bot train --steps 200000 --i-understand-the-risks
```

Tensorboard logs go to `tensorboard/`, checkpoints go to `checkpoints/`.

## 8. Optional: vision

YOLOv8 is disabled by default. To enable:

1. `pip install -e '.[vision]'`
2. Drop a fine-tuned `*.pt` weights file into `models/` (see
   [`architecture.md`](architecture.md)).
3. Set `vision.enabled: true` and `vision.model_path: models/your_model.pt`
   in `cfg/default.yaml`.
