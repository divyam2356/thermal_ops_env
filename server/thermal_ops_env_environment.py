# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Thermal Ops Environment Implementation.

A data-center cooling simulation with realistic thermodynamics.
The agent controls fan speeds, chiller setpoints, and workload migration
across 3 server racks to keep temperatures safe (20-25°C) while
minimising energy cost.

Three difficulty-tiered tasks:
    - stable_cooling   (Easy)   : mild conditions, no failures
    - fan_failure      (Medium) : one broken fan, moderate heat
    - crisis_management (Hard)  : two broken fans, heatwave, high load
"""

import json
import random
from typing import Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ThermalOpsAction, ThermalOpsObservation
except ImportError:
    from models import ThermalOpsAction, ThermalOpsObservation


# ── Task presets ───────────────────────────────────────────────────────────

TASK_PRESETS: Dict[str, dict] = {
    "stable_cooling": {
        "description": "Easy: keep all racks in [20,25°C] under mild conditions.",
        "ambient_range": (20.0, 24.0),
        "load_range": (5.0, 12.0),
        "broken_fan_count": 0,
        "max_steps": 10,
    },
    "fan_failure": {
        "description": "Medium: manage cooling with one broken fan and moderate heat.",
        "ambient_range": (24.0, 28.0),
        "load_range": (8.0, 18.0),
        "broken_fan_count": 1,
        "max_steps": 10,
    },
    "crisis_management": {
        "description": "Hard: survive a heatwave with 2 broken fans and high load.",
        "ambient_range": (28.0, 32.0),
        "load_range": (15.0, 25.0),
        "broken_fan_count": 2,
        "max_steps": 10,
    },
}


class ThermalOpsEnvironment(Environment):
    """
    Data-center thermal management environment.

    Physics per simulation tick (``wait`` tool):
        - Heat generated per rack:  ``0.1 × power_load``
        - Fan cooling per rack:     ``(rpm / 1000) × 0.5``  (broken → 0)
        - Chiller pull per rack:    ``max(0, rack_temp − chiller_setpoint) × 0.1``
        - Ambient bleed per rack:   ``(ambient_temp − rack_temp) × 0.05``
        - Fan energy per rack:      ``(rpm / 1000)³ × 0.2``
        - Chiller energy:           ``0.5 × max(0, ambient − setpoint)²``

    Reward per step (negative = cost):
        ``−(energy_cost × w1) − overheat_penalty − drift_penalty``
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Tunables ──────────────────────────────────────────────────────────

    NUM_RACKS = 3
    SAFE_TEMP_MAX = 25.0
    CRITICAL_TEMP = 27.0
    IDEAL_TEMP = 22.0
    W1_ENERGY = 0.5
    W2_OVERHEAT = 1000.0

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str = "stable_cooling"
        self._max_steps: int = 10

        # Runtime state (populated in reset)
        self.ambient_temp: float = 22.0
        self.rack_temps: List[float] = []
        self.power_loads: List[float] = []
        self.fan_rpms: List[int] = []
        self.chiller_setpoint: float = 18.0
        self.energy_cost_per_unit: float = 0.15
        self.total_energy: float = 0.0
        self.broken_fans: Set[int] = set()
        self._done: bool = False

        # Per-episode tracking for grading
        self._steps_all_safe: int = 0
        self._step_rewards: List[float] = []

    # ── OpenEnv interface ─────────────────────────────────────────────────

    def reset(
        self, *, task_name: Optional[str] = None, seed: Optional[int] = None
    ) -> ThermalOpsObservation:  # type: ignore[override]
        rng = random.Random(seed) if seed is not None else random

        self._task_name = task_name or rng.choice(list(TASK_PRESETS.keys()))
        preset = TASK_PRESETS[self._task_name]
        self._max_steps = preset["max_steps"]
        self._state = State(episode_id=str(uuid4()), step_count=0)

        lo_a, hi_a = preset["ambient_range"]
        lo_l, hi_l = preset["load_range"]

        self.ambient_temp = rng.uniform(lo_a, hi_a)
        self.rack_temps = [rng.uniform(20.0, 24.0) for _ in range(self.NUM_RACKS)]
        self.power_loads = [rng.uniform(lo_l, hi_l) for _ in range(self.NUM_RACKS)]
        self.fan_rpms = [
            rng.choice([500, 1000, 1500, 2000]) for _ in range(self.NUM_RACKS)
        ]
        self.chiller_setpoint = rng.uniform(12.0, 20.0)
        self.energy_cost_per_unit = rng.uniform(0.10, 0.25)
        self.total_energy = 0.0
        self._done = False

        # Assign broken fans
        n_broken = preset["broken_fan_count"]
        all_ids = list(range(self.NUM_RACKS))
        rng.shuffle(all_ids)
        self.broken_fans = set(all_ids[:n_broken])

        # Tracking
        self._steps_all_safe = 0
        self._step_rewards = []

        return self._make_obs(
            status=f"Thermal Ops [{self._task_name}] ready. Manage {self.NUM_RACKS} racks."
        )

    def step(self, action: ThermalOpsAction) -> ThermalOpsObservation:  # type: ignore[override]
        if self._done:
            return self._make_obs(status="Episode already finished.", step_reward=0.0)

        tool = action.tool_name
        args = action.arguments
        status = "Unknown tool."
        step_reward = 0.0

        if tool == "set_fan_speed":
            status = self._do_set_fan_speed(args)
            if not status.startswith("Failed"):
                step_reward += 1.0
                rack_id = args.get("rack_id", -1)
                if isinstance(rack_id, int) and 0 <= rack_id < self.NUM_RACKS:
                    if self.rack_temps[rack_id] > self.IDEAL_TEMP:
                        step_reward += 2.0  # Proactive cooling bonus

        elif tool == "adjust_chiller":
            status = self._do_adjust_chiller(args)
            if not status.startswith("Failed"):
                step_reward += 1.0
        elif tool == "migrate_workload":
            status = self._do_migrate_workload(args)
            if not status.startswith("Failed"):
                step_reward += 2.0
        elif tool == "wait":
            status, step_reward = self._do_wait()
        else:
            status = f"Unknown tool '{tool}'. Use: set_fan_speed, adjust_chiller, migrate_workload, wait."
            step_reward = -2.0  # Penalty for hallucination

        if tool != "wait":
            self._step_rewards.append(step_reward)

        return self._make_obs(status=status, step_reward=step_reward)

    @property
    def state(self) -> State:
        return self._state

    # ── Tool implementations ──────────────────────────────────────────────

    def _do_set_fan_speed(self, args: dict) -> str:
        rack_id = args.get("rack_id")
        rpm = args.get("rpm")
        if not isinstance(rack_id, int) or not isinstance(rpm, (int, float)):
            return "Failed: rack_id (int) and rpm (int) are required."
        rack_id = int(rack_id)
        rpm = int(rpm)
        if rack_id < 0 or rack_id >= self.NUM_RACKS:
            return f"Failed: rack_id must be 0-{self.NUM_RACKS - 1}."
        if rack_id in self.broken_fans:
            return f"Failed: fan {rack_id} is broken and cannot be adjusted."
        self.fan_rpms[rack_id] = max(0, min(5000, rpm))
        return f"Fan {rack_id} set to {self.fan_rpms[rack_id]} RPM."

    def _do_adjust_chiller(self, args: dict) -> str:
        temp = args.get("chiller_temp")
        if temp is None:
            return "Failed: chiller_temp (float) is required."
        self.chiller_setpoint = max(5.0, min(30.0, float(temp)))
        return f"Chiller setpoint adjusted to {self.chiller_setpoint:.1f}°C."

    def _do_migrate_workload(self, args: dict) -> str:
        src = args.get("source_rack")
        dst = args.get("target_rack")
        if not isinstance(src, int) or not isinstance(dst, int):
            return "Failed: source_rack (int) and target_rack (int) are required."
        if src < 0 or src >= self.NUM_RACKS or dst < 0 or dst >= self.NUM_RACKS:
            return f"Failed: rack IDs must be 0-{self.NUM_RACKS - 1}."
        if src == dst:
            return "Failed: source and target must differ."
        moved = self.power_loads[src] * 0.5
        self.power_loads[src] -= moved
        self.power_loads[dst] += moved
        return f"Migrated {moved:.2f} kW from rack {src} → rack {dst}."

    def _do_wait(self) -> tuple:
        """Advance physics by one tick.  Returns (status, step_reward)."""
        energy_consumed = 0.0
        overheat_penalty = 0.0

        # Chiller energy
        chiller_delta = max(0.0, self.ambient_temp - self.chiller_setpoint)
        energy_consumed += 0.5 * (chiller_delta**2)

        all_safe = True
        for i in range(self.NUM_RACKS):
            heat = 0.1 * self.power_loads[i]
            rpm = self.fan_rpms[i] if i not in self.broken_fans else 0
            cooling = (rpm / 1000.0) * 0.5
            chiller_pull = max(0.0, self.rack_temps[i] - self.chiller_setpoint) * 0.1
            ambient_bleed = (self.ambient_temp - self.rack_temps[i]) * 0.05

            self.rack_temps[i] += heat - cooling - chiller_pull + ambient_bleed

            # Fan energy (cubic)
            energy_consumed += ((rpm / 1000.0) ** 3) * 0.2

            # Drift penalty (gentle)
            drift = abs(self.rack_temps[i] - self.IDEAL_TEMP) * 0.05
            overheat_penalty += drift

            # Overheat penalties
            if self.rack_temps[i] > self.SAFE_TEMP_MAX:
                all_safe = False
                if self.rack_temps[i] > self.CRITICAL_TEMP:
                    overheat_penalty += self.W2_OVERHEAT
                else:
                    overheat_penalty += (
                        self.W2_OVERHEAT
                        * 0.1
                        * (self.rack_temps[i] - self.SAFE_TEMP_MAX)
                    )
            elif self.rack_temps[i] < 18.0:
                # Penalise over-cooling (wastes energy)
                overheat_penalty += abs(self.rack_temps[i] - 18.0) * 0.1

        cost = energy_consumed * self.energy_cost_per_unit
        self.total_energy += energy_consumed

        step_reward = -(self.W1_ENERGY * cost) - overheat_penalty
        self._step_rewards.append(step_reward)
        if all_safe:
            self._steps_all_safe += 1

        self._state.step_count += 1
        if self._state.step_count >= self._max_steps:
            self._done = True

        return "Simulation step advanced.", step_reward

    # ── Observation builder ───────────────────────────────────────────────

    def _make_obs(
        self, status: str = "", step_reward: float = 0.0
    ) -> ThermalOpsObservation:
        rounded_temps = [round(t, 2) for t in self.rack_temps]
        rounded_loads = [round(l, 2) for l in self.power_loads]

        obs_dict = {
            "task": self._task_name,
            "ambient_temp": round(self.ambient_temp, 2),
            "rack_temps": rounded_temps,
            "power_loads": rounded_loads,
            "fan_rpms": list(self.fan_rpms),
            "chiller_setpoint": round(self.chiller_setpoint, 2),
            "broken_fans": sorted(self.broken_fans),
            "step": self._state.step_count,
            "max_steps": self._max_steps,
            "status": status,
        }
        text_obs = f"Observation: {json.dumps(obs_dict)}\nStatus: {status}"

        # Compute final grade when episode ends
        grade = None
        if self._done:
            grade = self._compute_grade()

        return ThermalOpsObservation(
            ambient_temp=round(self.ambient_temp, 2),
            rack_temps=rounded_temps,
            power_loads=rounded_loads,
            fan_rpms=list(self.fan_rpms),
            chiller_setpoint=round(self.chiller_setpoint, 2),
            broken_fans=sorted(self.broken_fans),
            step_count=self._state.step_count,
            max_steps=self._max_steps,
            status_message=status,
            text_observation=text_obs,
            done=self._done,
            reward=step_reward,
            grade=grade,
            metadata={
                "task": self._task_name,
                "total_energy": round(self.total_energy, 4),
                "steps_all_safe": self._steps_all_safe,
                "grade": grade,
            },
        )

    # ── Grader (0.0 – 1.0) ────────────────────────────────────────────────

    def _compute_grade(self) -> float:
        """
        End-of-episode grade in [0.0, 1.0].

        Components:
            - temp_score: fraction of simulation steps where ALL racks were safe
            - energy_score: normalised inverse of total energy (lower = better)
            - stability_score: how close final temps are to the ideal 22°C

        Weights vary by task difficulty.
        """
        steps = max(1, self._state.step_count)

        # 1. Temperature safety score
        temp_score = self._steps_all_safe / steps

        # 2. Energy efficiency score (logistic normalisation)
        # Rough baseline: 30 energy units = mediocre → score ≈ 0.5
        energy_score = max(0.0, 1.0 - (self.total_energy / 60.0))

        # 3. Stability: average |rack_temp − 22| at episode end
        if self.rack_temps:
            avg_drift = sum(abs(t - self.IDEAL_TEMP) for t in self.rack_temps) / len(
                self.rack_temps
            )
            stability_score = max(0.0, 1.0 - avg_drift / 10.0)
        else:
            stability_score = 0.0

        # Weight by task difficulty
        weights = {
            "stable_cooling": (0.60, 0.25, 0.15),
            "fan_failure": (0.50, 0.30, 0.20),
            "crisis_management": (0.45, 0.30, 0.25),
        }
        w_temp, w_energy, w_stab = weights.get(self._task_name, (0.50, 0.30, 0.20))

        grade = w_temp * temp_score + w_energy * energy_score + w_stab * stability_score
        # Clamp to strictly (0, 1) — validator rejects exact 0.0 and 1.0
        return round(max(0.0001, min(0.9999, grade)), 4)
