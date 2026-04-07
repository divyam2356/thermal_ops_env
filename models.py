# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Thermal Ops Environment.

A data-center cooling simulation where an AI agent controls fans, chillers,
and workload placement to keep 3 server racks within safe temperature
(20-25°C) while minimising energy cost.
"""

import json as _json
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


class ThermalOpsAction(Action):
    """Action for the Thermal Ops environment — represents a tool call.

    Available tools:
        - set_fan_speed(rack_id, rpm)
        - adjust_chiller(chiller_temp)
        - migrate_workload(source_rack, target_rack)
        - wait()
    """

    tool_name: str = Field(
        ...,
        description="Tool function to call: 'set_fan_speed', 'adjust_chiller', 'migrate_workload', or 'wait'.",
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the tool.",
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def _parse_arguments(cls, v: Any) -> Dict[str, Any]:
        """Accept both dict and JSON-string inputs (web UI sends strings)."""
        if isinstance(v, str):
            try:
                parsed = _json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except (_json.JSONDecodeError, TypeError):
                pass
            return {}
        return v


class ThermalOpsObservation(Observation):
    """Observation from the Thermal Ops environment — thermodynamic state.

    The base ``Observation`` class already provides ``done``, ``reward``,
    and ``metadata`` fields.
    """

    ambient_temp: float = Field(0.0, description="Ambient external temperature (°C)")
    rack_temps: List[float] = Field(
        default_factory=list, description="Temperature of each server rack (°C)"
    )
    power_loads: List[float] = Field(
        default_factory=list, description="Computational power load per rack (kW)"
    )
    fan_rpms: List[int] = Field(
        default_factory=list, description="Fan speed per rack (RPM, 0-5000)"
    )
    chiller_setpoint: float = Field(
        0.0, description="Global chiller temperature setpoint (°C)"
    )
    broken_fans: List[int] = Field(
        default_factory=list, description="IDs of broken/failed fans"
    )
    step_count: int = Field(0, description="Elapsed simulation steps")
    max_steps: int = Field(10, description="Maximum steps in this episode")
    status_message: str = Field("", description="Status message from the last action")
    text_observation: str = Field(
        "",
        description="Human/LLM-readable string representation of the full observation",
    )
    grade: Optional[float] = Field(
        default=None,
        description="Final episode grade in (0.01, 0.99) with 2 decimal precision; populated when the episode ends.",
    )
