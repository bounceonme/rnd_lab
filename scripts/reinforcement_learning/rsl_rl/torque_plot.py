# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import math

try:
    import omni.kit.app
    import omni.ui as ui
except ImportError:
    ui = None


logger = logging.getLogger(__name__)


class TorquePlotWindow:
    """Live torque monitor window for env[0] using one mini-plot per joint."""

    _WINDOW_TITLE = "Joint Torque Monitor"
    _PRIMARY_DOCK_TARGET_WINDOW = "Viewport"
    _FALLBACK_DOCK_TARGET_WINDOW = "Property"
    _DOCK_POSITION_RATIO = 0.23
    _PLOT_LIMIT = 20.0
    _COMPUTED_COLOR_FACTOR = 0.55
    _ZERO_LINE_COLOR = 0x889E9E9E
    _RMS_LINE_COLOR = 0xFF4040FF
    _LIMIT_LINE_COLOR = 0xFFFE6D53
    _LEFT_JOINT_ORDER = (
        "L_Leg_hip_pitch",
        "L_Leg_hip_roll",
        "L_Leg_hip_yaw",
        "L_Leg_knee",
        "L_Leg_ankle_pitch",
        "L_Leg_ankle_roll",
    )
    _RIGHT_JOINT_ORDER = (
        "R_Leg_hip_pitch",
        "R_Leg_hip_roll",
        "R_Leg_hip_yaw",
        "R_Leg_knee",
        "R_Leg_ankle_pitch",
        "R_Leg_ankle_roll",
    )
    _PLOT_COLORS = [
        0xFFFF5A5A,
        0xFF4FC3F7,
        0xFF81C784,
        0xFFFFD54F,
        0xFFBA68C8,
        0xFFFF8A65,
        0xFF64FFDA,
        0xFF90A4AE,
    ]

    def __init__(
        self,
        robot,
        env_idx: int = 0,
        update_interval: int = 4,
        max_datapoints: int = 240,
    ):
        self._robot = robot
        self._env_idx = max(0, min(env_idx, robot.num_instances - 1))
        self._update_interval = max(1, update_interval)
        self._max_datapoints = max(30, max_datapoints)
        self._step_count = 0

        self._window = None
        self._left_entries: list[dict] = []
        self._right_entries: list[dict] = []
        self._joint_entries: list[dict] = []
        self._history_initialized = False

        if ui is None:
            logger.warning("omni.ui is unavailable. Skipping torque plot window.")
            return

        self._resolve_joints()
        if not self._joint_entries:
            logger.warning("No ordered leg joints found. Skipping torque plot window.")
            return

        self._build_window()
        asyncio.ensure_future(self._dock_window())

    @staticmethod
    def _darken_color(color: int, factor: float) -> int:
        alpha = (color >> 24) & 0xFF
        red = (color >> 16) & 0xFF
        green = (color >> 8) & 0xFF
        blue = color & 0xFF

        red = max(0, min(255, int(red * factor)))
        green = max(0, min(255, int(green * factor)))
        blue = max(0, min(255, int(blue * factor)))

        return (alpha << 24) | (red << 16) | (green << 8) | blue

    @staticmethod
    def _format_scale_label(value: float) -> str:
        if value >= 10.0:
            return f"{value:.0f}"
        return f"{value:.1f}"

    def _resolve_joints(self):
        joint_entry_by_name = {}
        for joint_id, joint_name in enumerate(self._robot.joint_names):
            if joint_name not in self._LEFT_JOINT_ORDER and joint_name not in self._RIGHT_JOINT_ORDER:
                continue

            effort_limit = float(self._robot.data.joint_effort_limits[self._env_idx, joint_id].item())
            effort_limit = abs(effort_limit) if math.isfinite(effort_limit) else 1.0
            effort_limit = max(1.0, effort_limit)
            plot_limit = self._PLOT_LIMIT

            joint_entry_by_name[joint_name] = {
                "joint_id": joint_id,
                "joint_name": joint_name,
                "effort_limit": effort_limit,
                "plot_limit": plot_limit,
                "applied_buffer": [0.0] * self._max_datapoints,
                "computed_buffer": [0.0] * self._max_datapoints,
                "zero_buffer": [0.0] * self._max_datapoints,
                "rms_buffer": [0.0] * self._max_datapoints,
                "rms_value": 0.0,
                "upper_limit_buffer": [effort_limit] * self._max_datapoints,
                "lower_limit_buffer": [-effort_limit] * self._max_datapoints,
                "applied_plot": None,
                "computed_plot": None,
                "zero_plot": None,
                "rms_plot": None,
                "upper_limit_plot": None,
                "lower_limit_plot": None,
                "min_label": None,
                "max_label": None,
                "value_label": None,
                "rms_overlay_label": None,
                "pair_rms_overlay_label": None,
            }

        for joint_name in self._LEFT_JOINT_ORDER:
            entry = joint_entry_by_name.get(joint_name)
            if entry is None:
                logger.warning("Expected torque plot joint '%s' was not found.", joint_name)
                continue
            self._left_entries.append(entry)

        for joint_name in self._RIGHT_JOINT_ORDER:
            entry = joint_entry_by_name.get(joint_name)
            if entry is None:
                logger.warning("Expected torque plot joint '%s' was not found.", joint_name)
                continue
            self._right_entries.append(entry)

        self._joint_entries = [*self._left_entries, *self._right_entries]

    def _build_joint_card(self, entry: dict, color: int):
        computed_color = self._darken_color(color, self._COMPUTED_COLOR_FACTOR)
        with ui.ZStack(height=ui.Fraction(1), width=ui.Fraction(1)):
            ui.Rectangle(
                width=ui.Fraction(1),
                height=ui.Fraction(1),
                style={
                    "background_color": 0xFF303030,
                    "border_color": 0xFF5A5A5A,
                    "border_width": 1.0,
                }
            )
            with ui.VStack(spacing=4, height=ui.Fraction(1), width=ui.Fraction(1)):
                with ui.HStack(height=18, width=ui.Fraction(1)):
                    ui.Label(
                        entry["joint_name"],
                        width=ui.Fraction(1),
                        elided_text=True,
                        style={"font_size": 13, "color": 0xFFFFFFFF},
                    )
                    ui.Spacer(width=4)
                    entry["value_label"] = ui.Label(
                        f"A:+0.00 C:+0.00 R:0.00 / {entry['effort_limit']:.2f} Nm",
                        width=176,
                        elided_text=True,
                        style={"font_size": 11, "color": 0xFFBDBDBD},
                    )

                with ui.ZStack(width=ui.Fraction(1), height=ui.Fraction(1)):
                    entry["computed_plot"] = ui.Plot(
                        ui.Type.LINE,
                        -entry["plot_limit"],
                        entry["plot_limit"],
                        *entry["computed_buffer"],
                        width=ui.Fraction(1),
                        height=ui.Fraction(1),
                        style={"color": computed_color, "background_color": 0x00000000},
                    )
                    entry["applied_plot"] = ui.Plot(
                        ui.Type.LINE,
                        -entry["plot_limit"],
                        entry["plot_limit"],
                        *entry["applied_buffer"],
                        width=ui.Fraction(1),
                        height=ui.Fraction(1),
                        style={"color": color, "background_color": 0x00000000},
                    )
                    entry["upper_limit_plot"] = ui.Plot(
                        ui.Type.LINE,
                        -entry["plot_limit"],
                        entry["plot_limit"],
                        *entry["upper_limit_buffer"],
                        width=ui.Fraction(1),
                        height=ui.Fraction(1),
                        style={"color": self._LIMIT_LINE_COLOR, "background_color": 0x00000000},
                    )
                    entry["lower_limit_plot"] = ui.Plot(
                        ui.Type.LINE,
                        -entry["plot_limit"],
                        entry["plot_limit"],
                        *entry["lower_limit_buffer"],
                        width=ui.Fraction(1),
                        height=ui.Fraction(1),
                        style={"color": self._LIMIT_LINE_COLOR, "background_color": 0x00000000},
                    )
                    entry["zero_plot"] = ui.Plot(
                        ui.Type.LINE,
                        -entry["plot_limit"],
                        entry["plot_limit"],
                        *entry["zero_buffer"],
                        width=ui.Fraction(1),
                        height=ui.Fraction(1),
                        style={"color": self._ZERO_LINE_COLOR, "background_color": 0x00000000},
                    )
                    entry["rms_plot"] = ui.Plot(
                        ui.Type.LINE,
                        -entry["plot_limit"],
                        entry["plot_limit"],
                        *entry["rms_buffer"],
                        width=ui.Fraction(1),
                        height=ui.Fraction(1),
                        style={"color": self._RMS_LINE_COLOR, "background_color": 0x00000000},
                    )
                    with ui.VStack(width=ui.Fraction(1), height=ui.Fraction(1)):
                        with ui.HStack(height=28, width=ui.Fraction(1)):
                            ui.Spacer(width=ui.Fraction(1))
                            with ui.ZStack(width=122, height=26):
                                ui.Rectangle(
                                    width=ui.Fraction(1),
                                    height=ui.Fraction(1),
                                    style={
                                        "background_color": 0xCC202020,
                                        "border_color": self._RMS_LINE_COLOR,
                                        "border_width": 1.0,
                                    },
                                )
                                entry["rms_overlay_label"] = ui.Label(
                                    "RMS 0.00 Nm",
                                    width=ui.Fraction(1),
                                    height=ui.Fraction(1),
                                    alignment=ui.Alignment.CENTER,
                                    style={"font_size": 15, "color": self._RMS_LINE_COLOR},
                                )
                        ui.Spacer(height=ui.Fraction(1))
                    if entry["joint_name"].startswith("R_"):
                        with ui.VStack(width=ui.Fraction(1), height=ui.Fraction(1)):
                            ui.Spacer(height=ui.Fraction(1))
                            with ui.HStack(height=26, width=ui.Fraction(1)):
                                ui.Spacer(width=ui.Fraction(1))
                                with ui.ZStack(width=146, height=24):
                                    ui.Rectangle(
                                        width=ui.Fraction(1),
                                        height=ui.Fraction(1),
                                        style={
                                            "background_color": 0xCC202020,
                                            "border_color": self._RMS_LINE_COLOR,
                                            "border_width": 1.0,
                                        },
                                    )
                                    entry["pair_rms_overlay_label"] = ui.Label(
                                        "L/R AVG 0.00 Nm",
                                        width=ui.Fraction(1),
                                        height=ui.Fraction(1),
                                        alignment=ui.Alignment.CENTER,
                                        style={"font_size": 14, "color": self._RMS_LINE_COLOR},
                                    )

                with ui.HStack(height=12, width=ui.Fraction(1)):
                    entry["min_label"] = ui.Label(
                        f"-{self._format_scale_label(entry['plot_limit'])}",
                        style={"font_size": 10, "color": 0xFF9E9E9E},
                    )
                    ui.Spacer(width=ui.Fraction(1))
                    entry["max_label"] = ui.Label(
                        f"+{self._format_scale_label(entry['plot_limit'])}",
                        style={"font_size": 10, "color": 0xFF9E9E9E},
                    )

    def _build_window(self):
        self._window = ui.Window(
            self._WINDOW_TITLE,
            width=460,
            height=980,
            visible=True,
            dock_preference=ui.DockPreference.LEFT_BOTTOM,
        )

        with self._window.frame:
            with ui.VStack(spacing=6, height=ui.Fraction(1), width=ui.Fraction(1)):
                ui.Label(
                    f"Joint Torque | env[{self._env_idx}]",
                    height=22,
                    style={"font_size": 16, "color": 0xFFFFFFFF},
                )
                ui.Label(
                    "A=Applied, C=Computed, gray=zero, red=applied RMS, blue=applied limit",
                    height=16,
                    style={"font_size": 12, "color": 0xFFB0B0B0},
                )

                with ui.HStack(spacing=8, height=ui.Fraction(1), width=ui.Fraction(1)):
                    with ui.VStack(spacing=6, height=ui.Fraction(1), width=ui.Fraction(1)):
                        for idx, entry in enumerate(self._left_entries):
                            self._build_joint_card(entry, self._PLOT_COLORS[idx % len(self._PLOT_COLORS)])
                    with ui.VStack(spacing=6, height=ui.Fraction(1), width=ui.Fraction(1)):
                        for idx, entry in enumerate(self._right_entries):
                            self._build_joint_card(entry, self._PLOT_COLORS[idx % len(self._PLOT_COLORS)])

    async def _dock_window(self):
        for _ in range(10):
            if ui.Workspace.get_window(self._WINDOW_TITLE):
                break
            await omni.kit.app.get_app().next_update_async()

        custom_window = ui.Workspace.get_window(self._WINDOW_TITLE)
        target_window = ui.Workspace.get_window(self._PRIMARY_DOCK_TARGET_WINDOW)
        if target_window is None:
            target_window = ui.Workspace.get_window(self._FALLBACK_DOCK_TARGET_WINDOW)
        if custom_window and target_window:
            custom_window.dock_in(target_window, ui.DockPosition.LEFT, self._DOCK_POSITION_RATIO)
            custom_window.focus()

    def update(self, force: bool = False):
        if self._window is None or not self._window.visible:
            return

        self._step_count += 1
        if not force and self._step_count % self._update_interval != 0:
            return

        applied_torques = self._robot.data.applied_torque[self._env_idx].detach().cpu().tolist()
        computed_torques = self._robot.data.computed_torque[self._env_idx].detach().cpu().tolist()
        for entry in self._joint_entries:
            applied_value = float(applied_torques[entry["joint_id"]])
            computed_value = float(computed_torques[entry["joint_id"]])
            if not self._history_initialized:
                entry["applied_buffer"] = [applied_value] * self._max_datapoints
                entry["computed_buffer"] = [computed_value] * self._max_datapoints
            else:
                entry["applied_buffer"].pop(0)
                entry["applied_buffer"].append(applied_value)
                entry["computed_buffer"].pop(0)
                entry["computed_buffer"].append(computed_value)

            entry["computed_plot"].set_data(*entry["computed_buffer"])
            entry["applied_plot"].set_data(*entry["applied_buffer"])
            rms_value = math.sqrt(sum(value * value for value in entry["applied_buffer"]) / len(entry["applied_buffer"]))
            entry["rms_value"] = rms_value
            entry["rms_buffer"] = [rms_value] * self._max_datapoints
            entry["zero_plot"].set_data(*entry["zero_buffer"])
            entry["rms_plot"].set_data(*entry["rms_buffer"])
            entry["upper_limit_plot"].set_data(*entry["upper_limit_buffer"])
            entry["lower_limit_plot"].set_data(*entry["lower_limit_buffer"])
            entry["value_label"].text = (
                f"A:{applied_value:+.2f} C:{computed_value:+.2f} R:{rms_value:.2f} / {entry['effort_limit']:.2f} Nm"
            )
            entry["rms_overlay_label"].text = f"RMS {rms_value:.2f} Nm"

        entry_by_name = {entry["joint_name"]: entry for entry in self._joint_entries}
        for right_entry in self._right_entries:
            left_entry = entry_by_name.get(f"L_{right_entry['joint_name'][2:]}")
            pair_label = right_entry.get("pair_rms_overlay_label")
            if left_entry is None or pair_label is None:
                continue
            pair_rms_value = 0.5 * (left_entry["rms_value"] + right_entry["rms_value"])
            pair_label.text = f"L/R AVG {pair_rms_value:.2f} Nm"
        self._history_initialized = True

    def close(self):
        if self._window is not None:
            self._window.visible = False
