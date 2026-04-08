"""
Inference Script — Thermal Ops Environment.

Runs an LLM agent against the Thermal Ops data-center cooling simulation
using an OpenAI-compatible endpoint (HF Router by default).

Credential resolution order:
    1) HF_TOKEN (preferred for HF Router)
    2) OPENAI_API_KEY (alias/fallback)
    3) API_KEY (legacy fallback)

STDOUT FORMAT:
    [START] task=<task_name> env=thermal_ops model=<model_name> seed=<seed>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> grade=<0.0000> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from dotenv import load_dotenv

from thermal_ops_env.client import ThermalOpsEnv
from thermal_ops_env.models import ThermalOpsAction, ThermalOpsObservation

# ── Configuration ──────────────────────────────────────────────────────────

# Load .env but NEVER override system env vars (validator sets API_BASE_URL/API_KEY)
load_dotenv(Path(__file__).with_name(".env"), override=False)

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = (
    (os.getenv("API_KEY") or "").strip()
    or (os.getenv("OPENAI_API_KEY") or "").strip()
    or (os.getenv("HF_TOKEN") or "").strip()
)
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE") or "thermal_ops_env-env:latest"

TASKS = ["stable_cooling", "fan_failure", "crisis_management"]
TASK_SEEDS: Dict[str, int] = {
    "stable_cooling": 1101,
    "fan_failure": 2202,
    "crisis_management": 3303,
}
EPISODES_PER_TASK = 3  # validator requires 3+ episodes per task
MAX_AGENT_STEPS = 25  # max tool calls per episode (wait advances sim)
TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
MAX_TOKENS = 256
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

SAFE_TEMP_MAX = 25.0
IDEAL_TEMP = 22.0
VALID_TOOLS = {"set_fan_speed", "adjust_chiller", "migrate_workload", "wait"}
NUM_RACKS = 3

# ── Helper: strictly clamp a score into (0, 1) ────────────────────────────


def clamp_score(value: float) -> float:
    """Clamp a value to strictly within (0, 1) — never 0.0 or 1.0."""
    if value <= 0.0:
        return 0.01
    elif value >= 1.0:
        return 0.99
    return value


# ── System prompt ──────────────────────────────────────────────────────────


def get_system_prompt(task_name: str) -> str:
    strategy = ""
    if task_name == "crisis_management":
        strategy = "- **CRISIS MODE**: This is a heatwave scenario with broken fans. Immediately drop the chiller temperature to its lowest limit! Migrate workloads completely off broken racks ASAP. Balance extreme cooling against massive energy costs, but safety is the priority."
    elif task_name == "fan_failure":
        strategy = "- Track broken fans and quickly shift workload off those racks. Compensate by increasing fan speeds on working racks and lowering the chiller setpoint."
    else:
        strategy = "- Focus on maintaining ideal temps around 22°C while conserving energy. Avoid massive jumps in RPM or chiller temps to lower energy use."

    return textwrap.dedent(f"""\
You are an expert Data Center Facility Manager.
You manage 3 server racks (IDs 0, 1, 2) that must stay between 20.0°C and 25.0°C.
The ideal temperature is 22.0°C.

## Available Tools
{{"tool_name": "set_fan_speed", "arguments": {{"rack_id": 0, "rpm": 2500}}}}
{{"tool_name": "adjust_chiller", "arguments": {{"chiller_temp": 16.0}}}}
{{"tool_name": "migrate_workload", "arguments": {{"source_rack": 0, "target_rack": 1}}}}
{{"tool_name": "wait", "arguments": {{}}}}

## Rules
- Fan speed range: 0–5000 RPM.  Higher RPM = more cooling but cubic energy cost.
- Broken fans CANNOT be adjusted — work around them.
- The chiller cools all racks toward its setpoint.  Lower setpoint = more energy.
- migrate_workload moves 50% of the source rack's load to the target.
- **wait** advances the simulation by 1 tick — this is the ONLY action that progresses time.
- You MUST call wait regularly to let the physics simulation run.
- Minimise energy cost while keeping all rack temps in [20, 25°C].
- An episode lasts for a fixed number of simulation steps.

## Strategy Tips
- First assess the situation: check which fans are broken, current temps, loads.
{strategy}
- Adjust fans and chiller BEFORE calling wait.
- Avoid repeating wait when any rack is above 24.5C or when broken-fan racks are heavily loaded.
- Use migrate_workload proactively when broken fans are carrying high load.

## Response Format
Respond with **exactly one** JSON object per message expressing your step-by-step thinking and your desired tool call.
Example:
```json
{{
  "thought": "Temperatures are stable at 22°C, and no fans are broken. I should wait one tick and see how temperatures trend.",
  "tool_name": "wait",
  "arguments": {{}}
}}
```
Respond with ONLY a JSON object. No text or markdown tags outside the braces.
""").strip()


# ── Logging helpers ────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err = error if error else "null"
    done_val = str(done).lower()
    clean_action = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={clean_action} reward={reward:.2f} done={done_val} error={err}",
        flush=True,
    )


def log_end(success: bool, score: float, steps: int, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}",
        flush=True,
    )


# ── LLM response parsing ──────────────────────────────────────────────────


def parse_tool_call(text: str) -> Dict[str, Any]:
    """Extract a tool call JSON from the LLM response."""
    text = text.strip()

    # Handle markdown code blocks
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1)

    # Try parse the whole text directly
    try:
        parsed = json.loads(text)
        if "tool_name" in parsed:
            return parsed
        if "name" in parsed:
            return {
                "tool_name": parsed["name"],
                "arguments": parsed.get("arguments", {}),
            }
    except json.JSONDecodeError:
        pass

    # Find the outermost {...} in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if "tool_name" in parsed:
                return parsed
            if "name" in parsed:
                return {
                    "tool_name": parsed["name"],
                    "arguments": parsed.get("arguments", {}),
                }
        except json.JSONDecodeError:
            pass

    # Fallback: just wait
    if DEBUG:
        print(f"[DEBUG] Could not parse tool call from: {text[:200]!r}", flush=True)
    return {"tool_name": "wait", "arguments": {}}


def sanitize_tool_call(
    raw_call: Dict[str, Any], num_racks: int
) -> Optional[Dict[str, Any]]:
    """Validate and sanitize a tool call against both schema and current env state."""
    tool_name = raw_call.get("tool_name")
    arguments = raw_call.get("arguments", {})

    if tool_name not in VALID_TOOLS or not isinstance(arguments, dict):
        if DEBUG:
            print(f"[DEBUG] sanitize: rejected unknown tool {tool_name!r}", flush=True)
        return None

    def valid_rack(x: Any) -> bool:
        return type(x) is int and 0 <= x < num_racks

    if tool_name == "wait":
        return {"tool_name": "wait", "arguments": {}}

    if tool_name == "set_fan_speed":
        rack_id = arguments.get("rack_id")
        rpm = arguments.get("rpm")

        if not valid_rack(rack_id):
            if DEBUG:
                print(f"[DEBUG] sanitize: rack_id {rack_id!r} invalid for {num_racks} racks", flush=True)
            return None
        if not isinstance(rpm, (int, float)) or isinstance(rpm, bool):
            return None

        return {
            "tool_name": "set_fan_speed",
            "arguments": {
                "rack_id": rack_id,
                "rpm": int(max(0, min(5000, rpm))),
            },
        }

    if tool_name == "adjust_chiller":
        temp = arguments.get("chiller_temp")
        if not isinstance(temp, (int, float)) or isinstance(temp, bool):
            return None
        return {
            "tool_name": "adjust_chiller",
            "arguments": {"chiller_temp": float(max(5.0, min(30.0, temp)))},
        }

    if tool_name == "migrate_workload":
        src = arguments.get("source_rack")
        dst = arguments.get("target_rack")

        if not valid_rack(src) or not valid_rack(dst):
            if DEBUG:
                print(f"[DEBUG] sanitize: rack IDs {src!r},{dst!r} invalid for {num_racks} racks", flush=True)
            return None
        if src == dst:
            if DEBUG:
                print(f"[DEBUG] sanitize: src == dst == {src}, rejecting migrate", flush=True)
            return None

        return {
            "tool_name": "migrate_workload",
            "arguments": {"source_rack": src, "target_rack": dst},
        }

    return None


def needs_intervention(observation: Any) -> bool:
    rack_temps = list(observation.rack_temps)
    broken_fans = set(observation.broken_fans)
    power_loads = list(observation.power_loads)

    if any(t > 24.5 for t in rack_temps):
        return True
    if any(t > SAFE_TEMP_MAX for t in rack_temps):
        return True
    if broken_fans:
        for idx in broken_fans:
            if 0 <= idx < len(power_loads) and power_loads[idx] > 10.0:
                return True
    return False


def heuristic_action(task_name: str, observation: Any) -> Dict[str, Any]:
    """Generate a safe fallback action using only valid indices from the observation."""
    rack_temps = list(getattr(observation, "rack_temps", []))
    power_loads = list(getattr(observation, "power_loads", []))
    fan_rpms = list(getattr(observation, "fan_rpms", []))
    broken_fans = set(getattr(observation, "broken_fans", []))
    chiller_setpoint = float(getattr(observation, "chiller_setpoint", 16.0))

    n = min(len(rack_temps), len(power_loads), len(fan_rpms))
    if n <= 0:
        return {"tool_name": "wait", "arguments": {}}

    rack_temps = rack_temps[:n]
    power_loads = power_loads[:n]
    fan_rpms = fan_rpms[:n]
    broken_fans = {i for i in broken_fans if type(i) is int and 0 <= i < n}

    working_racks = [i for i in range(n) if i not in broken_fans]
    hottest_idx = max(range(n), key=lambda i: rack_temps[i])
    coolest_working = (
        min(working_racks, key=lambda i: (rack_temps[i], power_loads[i]))
        if working_racks
        else None
    )

    target_chiller = {
        "stable_cooling": 18.0,
        "fan_failure": 16.0,
        "crisis_management": 12.0,
    }.get(task_name, 16.0)

    if max(rack_temps) > 24.0 and chiller_setpoint > target_chiller:
        return {
            "tool_name": "adjust_chiller",
            "arguments": {"chiller_temp": target_chiller},
        }

    if (
        hottest_idx in broken_fans
        and coolest_working is not None
        and hottest_idx != coolest_working
    ):
        return {
            "tool_name": "migrate_workload",
            "arguments": {
                "source_rack": hottest_idx,
                "target_rack": coolest_working,
            },
        }

    for idx in sorted(working_racks, key=lambda i: rack_temps[i], reverse=True):
        if rack_temps[idx] > IDEAL_TEMP + 1.0 and fan_rpms[idx] < 3200:
            return {
                "tool_name": "set_fan_speed",
                "arguments": {"rack_id": idx, "rpm": 3200},
            }

    loaded_broken = [i for i in broken_fans if power_loads[i] > 2.0]
    if loaded_broken and working_racks:
        src = max(loaded_broken, key=lambda i: power_loads[i])
        dst_candidates = [i for i in working_racks if i != src]
        if dst_candidates:
            dst = min(dst_candidates, key=lambda i: (power_loads[i], rack_temps[i]))
            return {
                "tool_name": "migrate_workload",
                "arguments": {"source_rack": src, "target_rack": dst},
            }

    return {"tool_name": "wait", "arguments": {}}



def compute_wait_energy(observation: Any) -> float:
    chiller_delta = max(
        0.0, float(observation.ambient_temp) - float(observation.chiller_setpoint)
    )
    energy_consumed = 0.5 * (chiller_delta**2)

    fan_rpms = list(getattr(observation, "fan_rpms", []))
    broken_fans = set(getattr(observation, "broken_fans", []))
    for i in range(len(fan_rpms)):
        rpm = fan_rpms[i] if i not in broken_fans else 0
        energy_consumed += ((rpm / 1000.0) ** 3) * 0.2

    return energy_consumed



def compute_grade(
    task_name: str,
    total_energy: float,
    steps_all_safe: int,
    wait_steps: int,
    rack_temps: List[float],
) -> float:
    steps = max(1, wait_steps)
    temp_score = steps_all_safe / steps
    energy_score = max(0.0, 1.0 - (total_energy / 60.0))

    if rack_temps:
        avg_drift = sum(abs(t - IDEAL_TEMP) for t in rack_temps) / len(rack_temps)
        stability_score = max(0.0, 1.0 - avg_drift / 10.0)
    else:
        stability_score = 0.0

    weights = {
        "stable_cooling": (0.60, 0.25, 0.15),
        "fan_failure": (0.50, 0.30, 0.20),
        "crisis_management": (0.45, 0.30, 0.25),
    }
    w_temp, w_energy, w_stab = weights.get(task_name, (0.50, 0.30, 0.20))
    grade = w_temp * temp_score + w_energy * energy_score + w_stab * stability_score
    return clamp_score(grade)


# ── Episode runner ─────────────────────────────────────────────────────────


def run_episode(
    client: OpenAI, env: ThermalOpsEnv, task_name: str, seed: Optional[int] = None
) -> None:
    """Run one episode of the given task."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    grade = 0.01
    total_energy = 0.0
    steps_all_safe = 0
    wait_steps = 0

    if seed is None:
        seed = TASK_SEEDS.get(task_name, 0)

    log_start(task=task_name, env="thermal_ops", model=MODEL_NAME)

    try:
        result = env.reset(task_name=task_name, seed=seed)
        observation = result.observation
        consecutive_waits = 0
        consecutive_failed_actions = 0

        messages = [{"role": "system", "content": get_system_prompt(task_name)}]

        for agent_step in range(1, MAX_AGENT_STEPS + 1):
            if result.done:
                break

            user_msg = observation.text_observation
            messages.append({"role": "user", "content": user_msg})

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": response_text})
            except Exception as exc:
                response_text = '{"tool_name": "wait", "arguments": {}}'
                if DEBUG:
                    print(f"[DEBUG] LLM error: {exc}", flush=True)

            tool_call = parse_tool_call(response_text)
            num_racks = len(getattr(observation, "rack_temps", []))
            sanitized_call = sanitize_tool_call(tool_call, num_racks=num_racks)

            if sanitized_call is None:
                fallback_call = heuristic_action(task_name, observation)
                sanitized_call = sanitize_tool_call(fallback_call, num_racks=num_racks)

            if (
                sanitized_call is not None
                and sanitized_call["tool_name"] == "wait"
                and consecutive_waits >= 2
                and needs_intervention(observation)
            ):
                fallback_call = heuristic_action(task_name, observation)
                sanitized_call = sanitize_tool_call(fallback_call, num_racks=num_racks)

            if consecutive_failed_actions >= 1 and needs_intervention(observation):
                fallback_call = heuristic_action(task_name, observation)
                sanitized_call = sanitize_tool_call(fallback_call, num_racks=num_racks)

            if sanitized_call is None:
                sanitized_call = {"tool_name": "wait", "arguments": {}}

            action = ThermalOpsAction(
                tool_name=sanitized_call.get("tool_name", "wait"),
                arguments=sanitized_call.get("arguments", {}),
            )

            pre_step_observation = observation
            result = env.step(action)
            observation = result.observation
            raw_reward = result.reward or 0.0
            done = result.done

            if action.tool_name == "wait":
                total_energy += compute_wait_energy(pre_step_observation)
                wait_steps += 1
                if all(t <= SAFE_TEMP_MAX for t in observation.rack_temps):
                    steps_all_safe += 1

            reward = clamp_score(raw_reward)
            rewards.append(reward)
            steps_taken = agent_step

            action_str = f"{action.tool_name}({json.dumps(action.arguments)})"
            status_message = getattr(observation, "status_message", "") or ""
            status_lower = status_message.lower()

            error = None
            if (
                "failed" in status_lower
                or "unknown" in status_lower
                or "out of range" in status_lower
                or "range" in status_lower
                or "invalid" in status_lower
            ):
                error = status_message
                consecutive_failed_actions += 1
            else:
                consecutive_failed_actions = 0

            if action.tool_name == "wait":
                consecutive_waits += 1
            else:
                consecutive_waits = 0

            log_step(
                step=agent_step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                grade = observation.grade
                if grade is None and observation.metadata:
                    grade = observation.metadata.get("grade", 0.01)
                if grade is None:
                    grade = compute_grade(
                        task_name=task_name,
                        total_energy=total_energy,
                        steps_all_safe=steps_all_safe,
                        wait_steps=wait_steps,
                        rack_temps=list(observation.rack_temps),
                    )
                grade = clamp_score(float(grade or 0.01))
                success = grade > 0.3
                break
        else:
            success = False

    finally:
        grade = clamp_score(float(grade or 0.01))
        rewards = [clamp_score(r) for r in rewards]
        log_end(success=success, score=grade, steps=steps_taken, rewards=rewards)



# ── Main ───────────────────────────────────────────────────────────────────

ENV_URL = os.getenv("ENV_URL") or "http://localhost:8000"


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Missing API credentials. Set HF_TOKEN (preferred) or OPENAI_API_KEY "
            "(API_KEY is accepted as a legacy fallback)."
        )

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to the already-running Docker container using the sync wrapper
    # (EnvClient is async by default; .sync() gives us synchronous methods)
    env = ThermalOpsEnv(base_url=ENV_URL).sync()

    with env:
        for task_name in TASKS:
            for episode_idx in range(EPISODES_PER_TASK):
                # Use different seed for each episode
                base_seed = TASK_SEEDS.get(task_name, 0)
                episode_seed = base_seed + episode_idx * 1000
                run_episode(llm_client, env, task_name, seed=episode_seed)
                print("", flush=True)  # blank line between episodes


if __name__ == "__main__":
    main()