"""Stateful delay/backlash randomizer consuming an identified model JSON."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from .identification import MODEL_SCHEMA_VERSION


class EquivalentActuatorModelError(ValueError):
    """Raised when an identified model cannot be used for target transformation."""


def load_identified_model(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as stream:
        model = json.load(stream)
    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise EquivalentActuatorModelError(
            f"Unsupported model schema {model.get('schema_version')}; expected {MODEL_SCHEMA_VERSION}."
        )
    if model.get("model_type") != "rnd_encoder_domain_equivalent_actuator":
        raise EquivalentActuatorModelError(f"Unsupported model_type: {model.get('model_type')!r}")
    if not isinstance(model.get("joints"), dict) or not model["joints"]:
        raise EquivalentActuatorModelError("Identified model contains no joints.")
    return model


def _sample_range(value, rng: np.random.Generator, default: float = 0.0) -> float:
    if not isinstance(value, list) or len(value) != 2:
        return default
    lower, upper = (float(item) for item in value)
    if not 0.0 <= lower <= upper or not math.isfinite(lower + upper):
        raise EquivalentActuatorModelError(f"Invalid randomization range: {value}")
    return float(rng.uniform(lower, upper))


class EncoderDomainActuatorRandomizer:
    """Apply sampled command delay and play backlash before position targets.

    This is an integration primitive, not an Isaac action term. A training
    environment should call ``transform`` at its control rate before writing
    position targets and use ``friction_torque_proxy`` only if its actuator
    implementation supports an additive joint torque.
    """

    def __init__(self, model: dict, joint_names: tuple[str, ...], control_hz: float, seed: int | None = None):
        if control_hz <= 0.0:
            raise EquivalentActuatorModelError("control_hz must be positive.")
        missing = sorted(set(joint_names) - set(model["joints"]))
        if missing:
            raise EquivalentActuatorModelError(f"Model is missing joints: {missing}")
        rejected = [
            name
            for name in joint_names
            if not model["joints"][name].get("quality", {}).get("target_randomization_usable", False)
        ]
        if rejected:
            raise EquivalentActuatorModelError(f"Target randomization quality gate failed for joints: {rejected}")
        self.model = model
        self.joint_names = tuple(joint_names)
        self.control_hz = float(control_hz)
        self.rng = np.random.default_rng(seed)
        self.delay_samples = np.zeros(len(joint_names), dtype=np.int32)
        self.half_backlash = np.zeros(len(joint_names), dtype=np.float64)
        self.coulomb_positive_nm = np.zeros(len(joint_names), dtype=np.float64)
        self.coulomb_negative_nm = np.zeros(len(joint_names), dtype=np.float64)
        self.viscous_nm_per_rad_s = np.zeros(len(joint_names), dtype=np.float64)
        self._history: list[deque[float]] = []
        self._play_state = np.zeros(len(joint_names), dtype=np.float64)

    def reset(self, initial_targets_rad: np.ndarray) -> None:
        initial = np.asarray(initial_targets_rad, dtype=np.float64)
        if initial.shape != (len(self.joint_names),) or not np.all(np.isfinite(initial)):
            raise EquivalentActuatorModelError(
                f"initial_targets_rad must be finite with shape {(len(self.joint_names),)}."
            )
        self._history = []
        torque_per_amp = float(self.model["identification_config"]["nominal_torque_per_amp_nm"])
        for index, name in enumerate(self.joint_names):
            joint_model = self.model["joints"][name]
            randomization = joint_model["randomization"]
            delay_s = _sample_range(randomization.get("command_delay_s"), self.rng)
            backlash = _sample_range(randomization.get("effective_backlash_rad"), self.rng)
            friction_usable = joint_model["quality"].get("coulomb_randomization_usable", False)
            positive_a = _sample_range(randomization.get("coulomb_positive_a"), self.rng) if friction_usable else 0.0
            negative_a = _sample_range(randomization.get("coulomb_negative_a"), self.rng) if friction_usable else 0.0
            viscous_usable = joint_model["quality"].get("viscous_randomization_usable", False)
            viscous_a = _sample_range(randomization.get("viscous_a_per_rad_s"), self.rng) if viscous_usable else 0.0
            self.delay_samples[index] = max(0, round(delay_s * self.control_hz))
            self.half_backlash[index] = 0.5 * backlash
            self.coulomb_positive_nm[index] = positive_a * torque_per_amp
            self.coulomb_negative_nm[index] = negative_a * torque_per_amp
            self.viscous_nm_per_rad_s[index] = viscous_a * torque_per_amp
            length = int(self.delay_samples[index]) + 1
            self._history.append(deque([float(initial[index])] * length, maxlen=length))
        self._play_state = initial.copy()

    def transform(self, targets_rad: np.ndarray) -> np.ndarray:
        target = np.asarray(targets_rad, dtype=np.float64)
        if target.shape != (len(self.joint_names),) or not np.all(np.isfinite(target)):
            raise EquivalentActuatorModelError(f"targets_rad must have shape {(len(self.joint_names),)}.")
        if not self._history:
            raise EquivalentActuatorModelError("Call reset before transform.")
        output = np.empty_like(target)
        for index, value in enumerate(target):
            self._history[index].append(float(value))
            delayed = self._history[index][0]
            half_width = self.half_backlash[index]
            if delayed > self._play_state[index] + half_width:
                self._play_state[index] = delayed - half_width
            elif delayed < self._play_state[index] - half_width:
                self._play_state[index] = delayed + half_width
            output[index] = self._play_state[index]
        return output

    def friction_torque_proxy(self, velocity_rad_s: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity_rad_s, dtype=np.float64)
        if velocity.shape != (len(self.joint_names),):
            raise EquivalentActuatorModelError(f"velocity_rad_s must have shape {(len(self.joint_names),)}.")
        coulomb = np.where(velocity >= 0.0, self.coulomb_positive_nm, self.coulomb_negative_nm)
        return -np.sign(velocity) * coulomb - self.viscous_nm_per_rad_s * velocity
