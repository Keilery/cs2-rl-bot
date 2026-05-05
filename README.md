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
cs2-rl-bot train --steps 100000 \
    --i-understand-the-risks                     # PPO training
```

Override any config value via env vars: `CS2BOT_<SECTION>__<FIELD>=value`.

```bash
CS2BOT_AGENT__ALGO=random \
CS2BOT_DRY_RUN=true \
cs2-rl-bot inference
```

## License

MIT. See `LICENSE`. (Use of this code in violation of Valve's Terms of Service
is on you.)
