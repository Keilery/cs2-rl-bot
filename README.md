# cs2-rl-bot

> ⚠️ **Disclaimer.** This is a research scaffold for reinforcement-learning
> experiments against a *local, offline* Counter-Strike 2 instance. Running it
> on any VAC-protected server (Valve matchmaking, FACEIT, ESEA, most community
> servers) **will permanently ban your Steam account.** See
> [`docs/legal_and_safety.md`](docs/legal_and_safety.md) before running
> anything.

A Gymnasium-compatible environment + PPO agent skeleton for CS2. The agent
combines:

* **GSI** — Counter-Strike 2's [Game State Integration](https://developer.valvesoftware.com/wiki/Counter-Strike_2/Game_State_Integration)
  HTTP feed for HP, money, weapon, kills, round phase, bomb state.
* **Screen capture** — `mss`-based frames at 30 FPS, downsampled to 84x84
  for the policy network.
* **Vision (optional)** — `ultralytics` YOLOv8 enemy detector. Off by default;
  bring your own fine-tuned weights.
* **Controller** — `pydirectinput` (Windows) or `pynput` (Linux). Defaults
  to a `noop` backend so accidental runs don't move the cursor.
* **Reward** — kill / death / damage / round outcome / bomb plant&defuse,
  shaped from GSI deltas.
* **Agent** — `stable-baselines3` PPO with a `MultiInputPolicy` over Dict
  observations.

## What this scaffold does NOT do

* It does not include a trained policy. PPO learns from scratch — expect
  hours-to-days of wall-clock time on a single machine for any meaningful
  competence.
* It does not include a CS2-specific YOLO model. The default `yolov8n.pt`
  weights are COCO-trained and will not detect enemies.
* It does not bypass VAC, hook into the game process, or read game memory. The
  agent observes the game like a human would: through pixels and the
  GSI HTTP stream.
* It does not run training automatically — you must explicitly acknowledge the
  ban risk before `train` will start.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Sanity-check tests (no game required)
pytest

# Print resolved config
cs2-rl-bot info

# Listen for GSI payloads (CS2 must be configured & running)
cs2-rl-bot gsi
```

Full setup, including how to install the GSI config in CS2 and start an
offline bot match, lives in [`docs/setup.md`](docs/setup.md).

## Repository layout

```
cs2-rl-bot/
├── cfg/
│   ├── default.yaml                          # app config (overridable via env)
│   ├── gamestate_integration_cs2_rl_bot.cfg  # CS2 GSI config (KeyValues format)
│   └── autoexec.snippet.cfg                  # optional autoexec lines
├── docs/
│   ├── architecture.md
│   ├── legal_and_safety.md
│   └── setup.md
├── src/cs2_rl_bot/
│   ├── action/         # action space, OS-level controller, kill switch
│   ├── agent/          # PPO/random/scripted agents, replay buffer
│   ├── env/            # Gymnasium env, reward calculator
│   ├── observation/    # GSI listener, screen capture, vision (YOLO)
│   ├── training/       # train + inference entry points
│   ├── utils/          # config + logging
│   └── main.py         # typer CLI (cs2-rl-bot ...)
├── tests/              # pytest suite (no game required)
├── pyproject.toml
└── Makefile
```

## CLI

```
cs2-rl-bot info                                  # print config
cs2-rl-bot dump-cfg cfg/default.yaml             # write defaults to YAML
cs2-rl-bot gsi                                   # GSI listener (debug)
cs2-rl-bot capture                               # capture one frame
cs2-rl-bot inference --max-steps 500             # run agent without learning

# Training (requires --i-understand-the-risks because of VAC)
cs2-rl-bot train --steps 100000 \
    --save-every 10000 --i-understand-the-risks
cs2-rl-bot train --resume <run_id> --steps 100000 \
    --i-understand-the-risks                     # resume + extend a run

# Run management
cs2-rl-bot runs                                  # list training runs
cs2-rl-bot dashboard <run_id>                    # live progress + p/r/s/q hotkeys
cs2-rl-bot menu                                  # interactive launcher

# Imitation learning + YOLO
cs2-rl-bot parse-demo path/to/match.dem          # inspect a demo
cs2-rl-bot pretrain-bc path/to/demos/            # behaviour cloning warm-start
cs2-rl-bot train-yolo path/to/data.yaml          # finetune YOLOv8
```

Override any config value via env vars: `CS2BOT_<SECTION>__<FIELD>=value`.

```bash
CS2BOT_AGENT__ALGO=random \
CS2BOT_DRY_RUN=true \
cs2-rl-bot inference
```

## Training run layout

Every training run gets its own directory under `runs/`:

```
runs/<run_id>/
├── meta.json                # kind / status / parent run
├── config.yaml              # snapshot of the config used
├── status.json              # live metrics (read by dashboard)
├── control.json             # pause / resume / stop / save signals
├── checkpoints/             # ppo_step_*.zip, ppo_final_*.zip, bc_policy.pt, yolo.pt
├── tensorboard/             # SB3 / YOLO tensorboard event files
└── logs/
```

The dashboard polls `status.json` and writes commands to `control.json`. Three
SB3 callbacks are wired in by default:

* **TrainingControlCallback** — honours pause / stop / save commands.
* **RoundCheckpointCallback** — saves at the end of every CS2 round and every
  N steps. Keeps the last K checkpoints by default.
* **MetricsCallback** — records per-round kills / deaths / damage / reward
  to TensorBoard and `status.json`.

## Hybrid training (BC -> PPO)

To warm-start PPO from `.dem` files instead of random weights:

```bash
pip install -e '.[demos]'                        # demoparser2 + pandas
cs2-rl-bot pretrain-bc path/to/demos/            # creates runs/bc-...
cs2-rl-bot train --resume <bc_run_id> \
    --steps 200000 --i-understand-the-risks
```

The behaviour-cloning step trains an MLP over the scalar observation head only
(demo files don't include frame pixels). PPO then continues training that
policy through interaction with the live game.

## License

MIT. See `LICENSE`. (Use of this code in violation of Valve's Terms of Service
is on you.)
