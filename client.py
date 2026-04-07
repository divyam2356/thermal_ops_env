# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thermal Ops Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ThermalOpsAction, ThermalOpsObservation


class ThermalOpsEnv(
    EnvClient[ThermalOpsAction, ThermalOpsObservation, State]
):
    """
    Client for the Thermal Ops Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with ThermalOpsEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.rack_temps)
        ...
        ...     result = client.step(ThermalOpsAction(
        ...         tool_name="set_fan_speed",
        ...         arguments={"rack_id": 0, "rpm": 3000}
        ...     ))

    Example with Docker:
        >>> client = ThermalOpsEnv.from_docker_image("thermal_ops_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ThermalOpsAction(
        ...         tool_name="wait", arguments={}
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ThermalOpsAction) -> Dict:
        """Convert ThermalOpsAction to JSON payload for the step message."""
        return {
            "tool_name": action.tool_name,
            "arguments": action.arguments,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ThermalOpsObservation]:
        """Parse server response into StepResult[ThermalOpsObservation]."""
        obs_data = payload.get("observation", {})
        done = payload.get("done", False)
        raw_reward = payload.get("reward")

        # Clamp reward/grade strictly into (0, 1) when episode ends
        if done and raw_reward is not None:
            raw_reward = max(0.01, min(0.99, float(raw_reward)))

        # Propagate the grade field from the server observation
        grade = obs_data.get("grade")
        if grade is not None:
            grade = max(0.01, min(0.99, float(grade)))

        observation = ThermalOpsObservation(
            ambient_temp=obs_data.get("ambient_temp", 0.0),
            rack_temps=obs_data.get("rack_temps", []),
            power_loads=obs_data.get("power_loads", []),
            fan_rpms=obs_data.get("fan_rpms", []),
            chiller_setpoint=obs_data.get("chiller_setpoint", 0.0),
            broken_fans=obs_data.get("broken_fans", []),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 10),
            status_message=obs_data.get("status_message", ""),
            text_observation=obs_data.get("text_observation", ""),
            done=done,
            reward=raw_reward,
            grade=grade,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=raw_reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
