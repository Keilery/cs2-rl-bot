# Architecture

```
                        ┌────────────────────────┐
                        │       CS2 client       │
                        └────────┬───────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
   ┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ Game State     │  │ Screen capture   │  │ Mouse / keyboard │
   │ Integration    │  │ (mss)            │  │ (pydirectinput)  │
   │ JSON over HTTP │  │ → numpy frame    │  │ ← OS-level events│
   └────────┬───────┘  └─────────┬────────┘  └──────────────────┘
            │                    │                    ▲
            ▼                    ▼                    │
   ┌────────────────┐  ┌──────────────────┐           │
   │ GSIState       │  │ EnemyDetector    │           │
   │ (asyncio)      │  │ (YOLOv8, opt.)   │           │
   └────────┬───────┘  └─────────┬────────┘           │
            └─────────┬──────────┘                    │
                      ▼                               │
              ┌────────────────────┐                  │
              │ Observation        │                  │
              │ (PlayerState +     │                  │
              │  RoundState +      │                  │
              │  Frame)            │                  │
              └─────────┬──────────┘                  │
                        ▼                             │
              ┌────────────────────┐                  │
              │ CS2Env (Gymnasium) │  ◀───── reward ──┤
              └─────────┬──────────┘                  │
                        ▼                             │
              ┌────────────────────┐                  │
              │ PPO agent (SB3)    │ ─── action ──────┘
              └────────────────────┘
```

## Why hybrid (GSI + screen capture)?

| Source        | Latency | Determinism | Information           |
|---------------|---------|-------------|-----------------------|
| GSI           | 50–100 ms | High      | HP, money, score, weapon, bomb state |
| Screen capture| 10–30 ms  | Pixel-level | Enemy positions, environment, crosshair |

GSI alone is insufficient — Valve does not expose enemy positions for the
local player (anti-cheat reasons). A purely vision-based approach works (this
is what most cheats do) but is more complex and harder to reward-shape. The
hybrid approach uses GSI for the reward signal (kills, damage, round outcome)
and screen capture for the observation (so the policy actually has spatial
context).

## Reward design

The reward is computed from GSI deltas between consecutive observations. See
`src/cs2_rl_bot/env/rewards.py:RewardCalculator`.

| Component        | Source                           | Default coef |
|------------------|----------------------------------|--------------|
| Kill             | `player.state.round_kills` Δ     | +1.0 each    |
| Death            | `player.match_stats.deaths` Δ    | -1.0 each    |
| Damage dealt     | `player.state.round_totaldmg` Δ  | +0.01 / hp   |
| Damage taken     | `player.state.health` Δ          | -0.01 / hp   |
| Round win        | `round.win_team` == player team  | +2.0 once    |
| Round loss       | `round.win_team` ≠ player team   | -1.0 once    |
| Bomb plant (T)   | `round.bomb` transition          | +0.5 once    |
| Bomb defuse (CT) | `round.bomb` == "defused"        | +0.5 once    |
| Survival         | `health > 0` and phase=LIVE      | +0.001 / step|
| Kill switch      | F9 pressed                       | -0.1 / step  |

The survival bonus is intentionally small. The dominant signals are kills,
deaths, and round outcome — which corresponds to how human players think
about CS2.

## Action space

A `gymnasium.spaces.Dict` with seven heads:

* `movement`: `Discrete(9)` — stand or one of 8 directional combinations.
* `mouse`: `Box(-1, 1, (2,))` — relative dx, dy scaled by `max_mouse_delta`.
* `attack`, `jump`, `crouch`, `reload`: `Discrete(2)` — binary toggles.
* `weapon`: `Discrete(8)` — slot 1..bomb (no-op = 0).

The dict layout means PPO's `MultiInputPolicy` builds a separate head per
action component, which trains more efficiently than a single MultiDiscrete.

## Vision pipeline (optional)

`ultralytics` is loaded lazily. The default model (`yolov8n.pt`) is the COCO
pre-trained network and **does not detect CS2 enemies**. To make vision useful:

1. Record gameplay (`recordings/`) and use `cv2` or OBS to extract frames.
2. Label enemies with [Roboflow](https://roboflow.com/) or
   [LabelStudio](https://labelstud.io/).
3. Fine-tune `yolov8n.pt` (or `yolov8s.pt` for better accuracy):
   ```bash
   yolo detect train data=cs2.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```
4. Set `vision.model_path` in `cfg/default.yaml` to the trained weights.

For a quick start, [public CS2/CS:GO YOLO datasets](https://universe.roboflow.com/)
exist on Roboflow Universe — download one and skip steps 1-2.

## Round-based learning

Episodes correspond to rounds. The environment's `terminated` flag is True
when GSI reports `round.phase == "over"`. PPO's rollout buffer (`n_steps=2048`)
will span multiple rounds, which is fine — it improves the variance reduction.

For off-policy algorithms (DQN, SAC), use `cs2_rl_bot.agent.replay.RoundBuffer`
to flush transitions per round.

## Real-time performance

The environment runs at wall-clock speed — there is no fast-forward. With
`target_fps=30` and a typical CS2 round (~115s max), one round produces ~3500
transitions. For PPO with `n_steps=2048` you get a gradient update every ~1.2
rounds. Reaching even modest competence (consistently winning vs `bot_easy`)
will require **hundreds of rounds**, i.e. several hours of wall-clock time on
a single machine.

## Run directories

Each training run gets a self-contained directory (`runs/<run_id>/`) holding
its config snapshot, checkpoints, tensorboard logs, and two coordination
files:

* `status.json` — written by the trainer on every step (or every Nth step
  depending on `MetricsCallback.write_every_steps`). Holds current step,
  rounds completed, last K rewards, kills/deaths/damage, capture FPS, error
  message if any.
* `control.json` — written by the dashboard (or `DashboardClient`) to ask
  the trainer to pause / resume / save a checkpoint / stop. The trainer
  polls this file via `TrainingControlCallback`. A monotonic `revision`
  counter dedupes already-handled commands.

Both files use atomic writes (temp file + `fsync` + `rename`) so that a
reader never observes partial JSON.

## Imitation learning (BC) → PPO

For warm-starting PPO, demo files (`.dem`) are parsed via `demoparser2`
into `(observation, action)` pairs. The behaviour cloning step trains a
small MLP over the **scalar observation head only** — demo files don't
contain frame pixels, so the CNN stack still needs to learn from scratch
during PPO. The BC weights are persisted as
`runs/<bc_run_id>/checkpoints/bc_policy.pt` and loaded as initial weights
when PPO is launched with `--resume <bc_run_id>`.

## YOLO fine-tune

`cs2-rl-bot train-yolo` wraps `ultralytics.YOLO.train()` and saves the
resulting weights under `runs/<yolo_run_id>/checkpoints/yolo.pt`. The
Roboflow Universe SDK is used as the default dataset source; supply a
`ROBOFLOW_API_KEY` env var if you want automatic dataset downloads.
