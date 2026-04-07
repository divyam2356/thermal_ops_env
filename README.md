---
title: Thermal Ops Environment
emoji: 🌡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Thermal Ops - Data Center Cooling Simulation

An OpenEnv-compliant RL environment that simulates **data-center thermal management**.
An AI agent controls fan speeds, chiller temperatures, and workload placement across 3 server racks to maintain safe operating temperatures (20–25°C) while minimising energy costs.

## Why This Matters

Data-center cooling accounts for ~40% of total data-center energy consumption. Optimising cooling strategies can save millions in energy costs while preventing expensive hardware damage from overheating. This environment models the core physics that real facility managers deal with daily.

---

## Action Space

The agent acts via **tool calls** — JSON objects with a `tool_name` and `arguments`:

| Tool | Arguments | Description |
|------|-----------|-------------|
| `set_fan_speed` | `rack_id: int`, `rpm: int` | Set fan speed (0–5000 RPM). Higher = more cooling but cubic energy cost. |
| `adjust_chiller` | `chiller_temp: float` | Set chiller setpoint (5–30°C). Lower = more cooling but quadratic energy cost. |
| `migrate_workload` | `source_rack: int`, `target_rack: int` | Move 50% of compute load from one rack to another. |
| `wait` | *(none)* | Advance simulation by 1 tick. **Only action that progresses time.** |

Example action:
```json
{"tool_name": "set_fan_speed", "arguments": {"rack_id": 0, "rpm": 3000}}
```

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ambient_temp` | float | External ambient temperature (°C) |
| `rack_temps` | List[float] | Temperature of each of the 3 racks (°C) |
| `power_loads` | List[float] | Computational load per rack (kW) |
| `fan_rpms` | List[int] | Current fan speed per rack (RPM) |
| `chiller_setpoint` | float | Current chiller temperature target (°C) |
| `broken_fans` | List[int] | IDs of fans that have failed |
| `step_count` | int | Current simulation step |
| `max_steps` | int | Total steps in this episode |
| `status_message` | str | Result of the last action |
| `text_observation` | str | Full LLM-readable text summary |
| `grade` | float \| null | Final episode grade in [0.0, 1.0], populated when the episode ends |

---

## Tasks (3 Difficulty Levels)

### `stable_cooling` (Easy)
- **Ambient**: 20–24°C (mild)
- **Loads**: 5–12 kW (low)
- **Broken fans**: 0
- **Objective**: Keep all racks in [20, 25°C] for 10 steps

### `fan_failure` (Medium)
- **Ambient**: 24–28°C (warm)
- **Loads**: 8–18 kW (moderate)
- **Broken fans**: 1
- **Objective**: Manage cooling with degraded hardware

### `crisis_management` (Hard)
- **Ambient**: 28–32°C (heatwave)
- **Loads**: 15–25 kW (high)
- **Broken fans**: 2
- **Objective**: Prevent critical failures with minimal cooling capacity

---

## Grading

Each episode produces a **grade** (0.0–1.0) based on:

| Component | Description | Easy Weight | Medium Weight | Hard Weight |
|-----------|-------------|:-----------:|:-------------:|:-----------:|
| Temperature Safety | % of steps with all racks in [20, 25°C] | 60% | 50% | 45% |
| Energy Efficiency | Lower total energy = higher score | 25% | 30% | 30% |
| Stability | How close final temps are to ideal 22°C | 15% | 20% | 25% |

### Reward Signal
Per-step rewards provide continuous signal:
- **Energy cost penalty** (proportional to consumption)
- **Drift penalty** (distance from ideal 22°C)
- **Overheat penalty** (escalating: warning at 25°C, critical at 27°C)

---

## Setup & Usage

### Prerequisites
- Docker installed and running
- Python 3.10+
- `pip install openenv-core[cli]`

### Quick Start

```bash
# 1. Build the Docker image
cd thermal_ops_env
docker build -t thermal_ops_env-env:latest .

# 2. Run the server
docker run -d --name thermal_ops -p 8000:8000 thermal_ops_env-env:latest

# 3. Test in browser
open http://localhost:8000/web

# 4. Validate
openenv validate

# 5. Run inference (requires HF token)
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
uv run inference.py
```

### Environment Variables for Inference
| Variable | Required | Default |
|----------|----------|---------|
| `HF_TOKEN` | One of `HF_TOKEN` / `OPENAI_API_KEY` must be set | — |
| `OPENAI_API_KEY` | Optional alias/fallback for `HF_TOKEN` | — |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` |
| `AGENT_TEMPERATURE` | No | `0.0` |

Credential resolution order in `inference.py`:
1. `HF_TOKEN` (preferred for HF Router)
2. `OPENAI_API_KEY` (alias/fallback)
3. `API_KEY` (legacy fallback)

### Reproducible baseline protocol

The baseline runner now uses deterministic reset seeds per task:
- `stable_cooling`: `1101`
- `fan_failure`: `2202`
- `crisis_management`: `3303`

To reproduce baseline output, run:
```bash
uv run inference.py
```

`inference.py` auto-loads `thermal_ops_env/.env` before checking environment variables.

`[START]` logs include the seed used for each task, and `[END]` logs include the final grade.

### Baseline scores

Measured with `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct` via HF Router and the fixed seeds above:

| Task | Seed | Success | Final grade (0.0-1.0) | Notes |
|------|------|---------|------------------------|-------|
| `stable_cooling` | 1101 | true | 0.7046 | Mild scenario is handled reliably by the baseline |
| `fan_failure` | 2202 | true | 0.6765 | Baseline is acceptable under one broken fan |
| `crisis_management` | 3303 | false | 0.1215 | Hard scenario remains intentionally challenging |

Reproduce these values with `uv run inference.py`; the `[START]` / `[END]` logs include the task seed and final grade.

---

## Project Structure

```
thermal_ops_env/
├── Dockerfile              # Container definition
├── openenv.yaml            # OpenEnv spec + task definitions
├── pyproject.toml          # Python project config
├── models.py               # Pydantic Action & Observation types
├── client.py               # EnvClient (WebSocket client)
├── __init__.py             # Package exports
├── inference.py            # Baseline LLM agent script
└── server/
    ├── app.py              # FastAPI server (auto-generated)
    ├── thermal_ops_env_environment.py  # Core simulation
    └── __init__.py
```

---

## Physics Model

Each `wait` tick updates rack temperatures:

```
new_temp = old_temp + heat_generated - fan_cooling - chiller_pull + ambient_bleed

where:
  heat_generated = 0.1 × power_load
  fan_cooling    = (rpm / 1000) × 0.5    (0 if fan broken)
  chiller_pull   = max(0, temp - setpoint) × 0.1
  ambient_bleed  = (ambient - temp) × 0.05
```

Energy consumption:
```
fan_energy     = (rpm / 1000)³ × 0.2     per rack
chiller_energy = 0.5 × max(0, ambient - setpoint)²
```

---

## License

BSD-style license. See LICENSE file.
