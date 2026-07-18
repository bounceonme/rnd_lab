"""Run a deterministic, finite RSL-RL evaluation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a deterministic finite RSL-RL evaluation suite.")
parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint to evaluate.")
parser.add_argument("--task", type=str, required=True, help="Registered Isaac Lab task name.")
parser.add_argument("--suite", type=str, required=True, help="Strict evaluation-suite JSON path.")
parser.add_argument("--cases", nargs="*", default=None, help="Optional case IDs (space- or comma-separated).")
parser.add_argument("--split", choices=("validation", "test"), default="validation", help="Suite split to run.")
parser.add_argument("--num_envs", type=int, default=64, help="Maximum parallel evaluation environments.")
parser.add_argument("--output", type=str, required=True, help="Evaluation artifact directory.")
parser.add_argument(
    "--enable_observation_corruption",
    "--observation-corruption",
    dest="enable_observation_corruption",
    action="store_true",
    default=False,
    help="Enable configured policy observation corruption during evaluation.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent configuration registry key."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip
from evaluation_runtime import (  # isort: skip
    COMMAND_NAME,
    EvaluationArtifactWriter,
    EvaluationRuntimeError,
    FixedDomainSettings,
    FixedSingletonDomainApplicator,
    IsaacFixedDomainBackend,
    MassScaledPhysicalPulse,
    PULSE_PHYSICS_TICKS,
    SafeCommandTargetAdapter,
    checkpoint_sha256,
    command_schedule_from_scenario,
    physical_pulse_spec_from_scenario,
    prepare_deterministic_env_cfg,
    joint_order_permutation,
    reject_legacy_disturbances,
    validate_checkpoint_actor_observation_dimension,
    validate_split_checkpoint,
)
from evaluation_schema import (  # isort: skip
    canonical_json_sha256,
    load_evaluation_suite,
)
from gait_metrics import (  # isort: skip
    aggregate_episode_metrics,
    aggregate_evaluation_results,
    evaluate_episode_metrics,
)
from physics_touchdown_telemetry import attach_physics_touchdown_telemetry  # isort: skip
from play_checkpoint_guard import validate_checkpoint_experiment  # isort: skip


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _requested_case_ids(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    if not result or len(result) != len(set(result)):
        raise EvaluationRuntimeError("--cases must contain unique, non-empty case IDs.")
    return tuple(result)


def _resolve_cases(suite: dict[str, Any], split: str, requested_ids: tuple[str, ...] | None):
    domains = {entry["id"]: entry for entry in suite["domains"]}
    scenarios = {entry["id"]: entry for entry in suite["scenarios"]}
    available = [entry for entry in suite["cases"] if entry["split"] == split]
    available_by_id = {entry["id"]: entry for entry in available}
    if requested_ids is None:
        selected = available
    else:
        missing = [case_id for case_id in requested_ids if case_id not in available_by_id]
        if missing:
            raise EvaluationRuntimeError(f"Requested cases are not present in split {split!r}: {missing}.")
        selected = [available_by_id[case_id] for case_id in requested_ids]
    if not selected:
        raise EvaluationRuntimeError(f"No cases selected from split {split!r}.")
    return [
        {
            "case": case,
            "domain": domains[case["domain_id"]],
            "scenario": scenarios[case["scenario_id"]],
        }
        for case in selected
    ]


def _artifact_checkpoint_sha(suite: dict[str, Any], checkpoint_path: Path) -> str | None:
    matches = []
    for artifact in suite["artifacts"]:
        artifact_path = (REPOSITORY_ROOT / artifact["path"]).resolve()
        if artifact_path == checkpoint_path:
            matches.append(artifact)
    if len(matches) > 1:
        raise EvaluationRuntimeError(f"Suite declares the selected checkpoint more than once: {checkpoint_path}.")
    return None if not matches else str(matches[0]["sha256"])


def _joint_effort_limits(robot: Any) -> torch.Tensor:
    limits = torch.full((robot.num_joints,), torch.nan, dtype=torch.float32, device=robot.device)
    assigned = torch.zeros(robot.num_joints, dtype=torch.bool, device=robot.device)
    for name, actuator in robot.actuators.items():
        indices = actuator.joint_indices
        if isinstance(indices, slice):
            if indices != slice(None):
                raise EvaluationRuntimeError(f"Actuator {name!r} uses an unsupported joint-index slice.")
            joint_ids = torch.arange(robot.num_joints, device=robot.device)
        elif isinstance(indices, torch.Tensor):
            joint_ids = indices.to(device=robot.device, dtype=torch.long)
        else:
            raise EvaluationRuntimeError(f"Actuator {name!r} has ambiguous joint indices.")
        if bool(torch.any(assigned[joint_ids])):
            raise EvaluationRuntimeError(f"Actuator {name!r} overlaps another actuator's joint assignment.")
        values = torch.as_tensor(actuator.effort_limit, device=robot.device, dtype=torch.float32)
        if values.ndim == 2:
            values = values[0]
        if values.shape != (joint_ids.numel(),):
            raise EvaluationRuntimeError(f"Actuator {name!r} effort-limit readback has shape {tuple(values.shape)}.")
        limits[joint_ids] = values
        assigned[joint_ids] = True
    if not bool(torch.all(assigned)) or not bool(torch.isfinite(limits).all()) or bool(torch.any(limits <= 0.0)):
        raise EvaluationRuntimeError("Could not resolve one positive effort limit for every joint.")
    return limits


def _policy_episode_from_raw(
    raw: dict[str, np.ndarray], *, env_column: int, episode_id: int, decimation: int, horizon_steps: int
) -> dict[str, np.ndarray]:
    episode = raw["episode_id"][:, env_column] == episode_id
    policy_sample = raw["substep"][:, env_column] == decimation - 1
    selected = episode & policy_sample
    sample_count = int(np.count_nonzero(selected))
    if sample_count <= 0 or sample_count > horizon_steps:
        raise EvaluationRuntimeError(
            f"Episode env={env_column}, id={episode_id} has {sample_count} policy samples; expected [1, {horizon_steps}]."
        )

    def field(name: str) -> np.ndarray:
        return np.asarray(raw[name][selected, env_column])

    def physics_field(name: str) -> np.ndarray:
        return np.asarray(raw[name][episode, env_column])

    return {
        "termination": field("terminated"),
        "timeout": field("truncated"),
        "command": field("command")[:, :3],
        "root_lin_vel_w": field("root_lin_vel_w"),
        "root_ang_vel_w": field("root_ang_vel_w"),
        "root_quat_w": field("root_quat_w"),
        "foot_contact": field("foot_contact"),
        "foot_pos_w": field("foot_pos_w"),
        "applied_torque": field("applied_torque"),
        "physics_command": physics_field("command")[:, :3],
        "physics_root_quat_w": physics_field("root_quat_w"),
        "physics_foot_contact": physics_field("foot_contact"),
        "physics_foot_pos_w": physics_field("foot_pos_w"),
        "physics_touchdown_event": physics_field("foot_first"),
        "physics_touchdown_air_time_s": physics_field("foot_preceding_air_time_s"),
        "physics_touchdown_preimpact_speed_m_s": physics_field("foot_preimpact_speed"),
    }


def _episode_metrics(
    telemetry: dict[str, np.ndarray],
    *,
    suite: dict[str, Any],
    scenario: dict[str, Any],
    pulse: dict[str, Any] | None,
    pulse_delivery_ticks: int | None,
    effort_limits: np.ndarray,
) -> dict[str, Any]:
    contract = suite["metric_contract"]
    gait = contract["gait"]
    recovery = contract["push_recovery"]
    push_end_step = None
    push_delivery_complete = True
    if pulse is None:
        physical_pulse = {
            "applicable": False,
            "delivery_complete": None,
            "observed_physics_ticks": None,
            "required_physics_ticks": None,
            "delivery_fraction": None,
        }
    else:
        if pulse_delivery_ticks is None:
            raise EvaluationRuntimeError("Pulse scenario is missing per-episode delivery readback.")
        observed_ticks = int(pulse_delivery_ticks)
        if observed_ticks < 0 or observed_ticks > PULSE_PHYSICS_TICKS:
            raise EvaluationRuntimeError(
                f"Pulse delivery readback must lie in [0, {PULSE_PHYSICS_TICKS}]; got {observed_ticks}."
            )
        push_delivery_complete = observed_ticks == PULSE_PHYSICS_TICKS
        physical_pulse = {
            "applicable": True,
            "delivery_complete": push_delivery_complete,
            "observed_physics_ticks": observed_ticks,
            "required_physics_ticks": PULSE_PHYSICS_TICKS,
            "delivery_fraction": observed_ticks / PULSE_PHYSICS_TICKS,
        }
    if pulse is not None:
        push_end_step = min(int(pulse["end_step"]), int(telemetry["command"].shape[0]))
    metrics = evaluate_episode_metrics(
        telemetry,
        horizon_steps=int(scenario["horizon_steps"]),
        step_dt=1.0 / float(suite["rates"]["metric_hz"]),
        touchdown_step_dt=1.0 / float(suite["rates"]["contact_hz"]),
        effort_limits=effort_limits,
        minimum_touchdown_progress_m=float(gait["minimum_touchdown_progress_m"]),
        tap_max_air_time_s=float(gait["tap_max_air_time_s"]),
        command_speed_threshold_m_s=float(gait["command_speed_threshold_m_s"]),
        torque_saturation_threshold_fraction=float(contract["torque_saturation"]["threshold_fraction"]),
        joint_names=suite["task"]["joint_order"],
        timeout=telemetry["timeout"],
        push_end_step=push_end_step,
        push_delivery_complete=push_delivery_complete,
        linear_velocity_error_threshold_m_s=(
            None if pulse is None else float(recovery["linear_velocity_error_threshold_m_s"])
        ),
        yaw_rate_error_threshold_rad_s=(None if pulse is None else float(recovery["yaw_rate_error_threshold_rad_s"])),
        recovery_dwell_s=None if pulse is None else float(recovery["dwell_s"]),
    )
    metrics["physical_pulse"] = physical_pulse
    return metrics


def _mark_manual_horizon(sim_env: Any, telemetry_adapter: Any, active: torch.Tensor) -> None:
    active_ids = torch.nonzero(active, as_tuple=False).flatten()
    if active_ids.numel() == 0:
        return
    false = torch.zeros(sim_env.num_envs, dtype=torch.bool, device=sim_env.device)
    truncated = active.clone()
    monitor = sim_env.physics_touchdown_monitor
    monitor.on_pre_reset(sim_env, active_ids, false, truncated)
    telemetry_adapter.on_pre_reset(sim_env, active_ids, false, truncated)


def _reset_batch(env: RslRlVecEnvWrapper, sim_env: Any, seed: int, policy_module: Any):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    sim_env.seed(seed)
    observations, _ = env.reset()
    env_ids = torch.arange(sim_env.num_envs, device=sim_env.device, dtype=torch.long)
    sim_env.physics_touchdown_monitor.reset(env_ids)
    policy_module.reset(torch.ones(sim_env.num_envs, dtype=torch.long, device=sim_env.device))
    return observations


def _run_case(
    *,
    env: RslRlVecEnvWrapper,
    policy: Any,
    policy_module: Any,
    suite: dict[str, Any],
    resolved: dict[str, Any],
    domain_applicator: FixedSingletonDomainApplicator,
    command_adapter: SafeCommandTargetAdapter,
    checkpoint_path: Path,
    raw_path: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    sim_env = env.unwrapped
    case = resolved["case"]
    scenario = resolved["scenario"]
    schedule = command_schedule_from_scenario(scenario, float(suite["rates"]["policy_hz"]))
    pulse_spec = physical_pulse_spec_from_scenario(scenario, float(suite["rates"]["policy_hz"]))
    domain = FixedDomainSettings.from_mapping(resolved["domain"]["resolved"])
    physics_dt = float(sim_env.physics_dt)
    decimation = int(sim_env.cfg.decimation)
    if not np.isclose(physics_dt, 1.0 / float(suite["rates"]["physics_hz"]), rtol=0.0, atol=1.0e-12):
        raise EvaluationRuntimeError("Suite physics rate does not match the initialized environment.")
    if decimation != int(round(float(suite["rates"]["physics_hz"]) / float(suite["rates"]["policy_hz"]))):
        raise EvaluationRuntimeError("Suite policy decimation does not match the initialized environment.")

    telemetry_adapter = attach_physics_touchdown_telemetry(
        sim_env,
        output_path=raw_path,
        env_ids=None,
        task=args_cli.task,
        checkpoint=checkpoint_path,
        command_name=COMMAND_NAME,
    )
    monitor = sim_env.physics_touchdown_monitor
    targets: list[tuple[int, int]] = []
    target_pulse_delivery_ticks: list[int | None] = []
    domain_readbacks: list[dict[str, Any]] = []
    pulse_readbacks: list[dict[str, Any] | None] = []
    episodes_remaining = int(case["episodes"])
    batch_index = 0
    try:
        while episodes_remaining > 0:
            batch_size = min(env.num_envs, episodes_remaining)
            domain_readback = domain_applicator.apply(domain)
            if domain_readbacks and domain_readback != domain_readbacks[0]:
                raise EvaluationRuntimeError("Fixed-domain readback changed between episode batches.")
            domain_readbacks.append(domain_readback)
            # Apply persistent physics/sensor settings before reset so every history slot is
            # prefilled from the selected fixed domain rather than the previous episode/domain.
            observations = _reset_batch(env, sim_env, int(case["seed"]) + batch_index, policy_module)
            command_adapter.inject(schedule.target_at(0.0))
            observations = env.get_observations()

            episode_ids = monitor.episode_id.detach().cpu().numpy()
            targets.extend((env_index, int(episode_ids[env_index])) for env_index in range(batch_size))
            active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            active[:batch_size] = True
            pulse = None
            if pulse_spec is not None:
                pulse = MassScaledPhysicalPulse(
                    sim_env.scene["robot"],
                    base_body_name=sim_env.cfg.base_link_name,
                    onset_s=float(pulse_spec["onset_s"]),
                    delta_velocity_body_m_s=pulse_spec["delta_velocity_body_m_s"],
                    physics_dt=physics_dt,
                    decimation=decimation,
                    env_ids=torch.arange(batch_size, device=env.device),
                )
                sim_env.add_rnd_physics_observer(pulse)

            try:
                for step_index in range(int(scenario["horizon_steps"])):
                    command_adapter.inject(schedule.target_at(step_index / float(suite["rates"]["policy_hz"])))
                    if pulse is not None:
                        pulse.before_policy_step(step_index)
                    # Isaac Lab updates persistent articulation buffers in-place during step/reset.
                    # no_grad avoids policy autograd work without turning those buffers into
                    # inference tensors that cannot be updated by a later episode reset.
                    with torch.no_grad():
                        actions = policy(observations)
                    actions = actions.clone()
                    actions[~active] = 0.0
                    observations, _, dones, _ = env.step(actions)
                    policy_module.reset(dones)
                    if pulse is not None:
                        pulse.after_policy_step(step_index)
                    active &= ~dones.to(dtype=torch.bool)
                    if not bool(torch.any(active)):
                        break
                _mark_manual_horizon(sim_env, telemetry_adapter, active)
                pulse_readback = None if pulse is None or pulse.readback is None else dict(pulse.readback)
                pulse_readbacks.append(pulse_readback)
                if pulse is None:
                    target_pulse_delivery_ticks.extend([None] * batch_size)
                elif pulse_readback is None:
                    target_pulse_delivery_ticks.extend([0] * batch_size)
                else:
                    readback_env_ids = [int(value) for value in pulse_readback.get("env_ids", [])]
                    readback_ticks = [
                        int(value) for value in pulse_readback.get("observed_physics_ticks_by_env", [])
                    ]
                    if len(readback_env_ids) != len(readback_ticks) or len(readback_env_ids) != len(
                        set(readback_env_ids)
                    ):
                        raise EvaluationRuntimeError("Pulse readback has ambiguous per-environment delivery ticks.")
                    ticks_by_env = dict(zip(readback_env_ids, readback_ticks, strict=True))
                    target_pulse_delivery_ticks.extend(ticks_by_env.get(env_index, 0) for env_index in range(batch_size))
            finally:
                if pulse is not None:
                    sim_env.remove_rnd_physics_observer(pulse)
                    pulse.reset()

            episodes_remaining -= batch_size
            batch_index += 1

        telemetry_adapter.save(raw_path)
    finally:
        sim_env.remove_rnd_physics_observer(telemetry_adapter)
        telemetry_adapter.close()

    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: np.array(archive[key], copy=True) for key in archive.files}
    raw["evaluated_env_index"] = np.asarray([target[0] for target in targets], dtype=np.int64)
    raw["evaluated_episode_id"] = np.asarray([target[1] for target in targets], dtype=np.int64)

    robot = sim_env.scene["robot"]
    runtime_joint_names = tuple(str(name) for name in np.asarray(raw["joint_names"]).tolist())
    metric_joint_names = tuple(str(name) for name in suite["task"]["joint_order"])
    metric_joint_indices = joint_order_permutation(runtime_joint_names, metric_joint_names)
    effort_limits = _joint_effort_limits(robot).detach().cpu().numpy()[metric_joint_indices]
    episode_metrics = []
    if len(target_pulse_delivery_ticks) != len(targets):
        raise EvaluationRuntimeError("Pulse delivery readback count does not match evaluated episodes.")
    for (env_column, episode_id), pulse_delivery_ticks in zip(targets, target_pulse_delivery_ticks, strict=True):
        telemetry = _policy_episode_from_raw(
            raw,
            env_column=env_column,
            episode_id=episode_id,
            decimation=decimation,
            horizon_steps=int(scenario["horizon_steps"]),
        )
        telemetry["applied_torque"] = telemetry["applied_torque"][:, metric_joint_indices]
        episode_metrics.append(
            _episode_metrics(
                telemetry,
                suite=suite,
                scenario=scenario,
                pulse=pulse_spec,
                pulse_delivery_ticks=pulse_delivery_ticks,
                effort_limits=effort_limits,
            )
        )
    runtime_readback = {
        "command_profile": schedule.name,
        "command_ramp_rates": (
            None
            if getattr(sim_env.command_manager.get_term(COMMAND_NAME), "command_ramp_rates", None) is None
            else sim_env.command_manager.get_term(COMMAND_NAME).command_ramp_rates.detach().cpu().tolist()
        ),
        "domain": domain_readbacks[0],
        "pulse": pulse_readbacks,
        "evaluated_episodes": len(targets),
        "parallel_envs": env.num_envs,
        "joint_order": {
            "telemetry": list(runtime_joint_names),
            "metrics": list(metric_joint_names),
            "metric_from_telemetry_indices": metric_joint_indices.tolist(),
        },
    }
    return raw, episode_metrics, runtime_readback


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.num_envs <= 0:
        raise EvaluationRuntimeError("--num_envs must be positive.")
    suite = load_evaluation_suite(
        args_cli.suite,
        verify_artifacts=True,
        repository_root=REPOSITORY_ROOT,
    )
    if suite["task"]["id"] != args_cli.task:
        raise EvaluationRuntimeError(f"Suite task {suite['task']['id']!r} does not match --task {args_cli.task!r}.")
    selected = _resolve_cases(suite, args_cli.split, _requested_case_ids(args_cli.cases))

    resume_path = Path(retrieve_file_path(args_cli.checkpoint)).expanduser().resolve()
    validate_checkpoint_experiment(resume_path, agent_cfg.experiment_name)
    checkpoint_hash = checkpoint_sha256(resume_path)
    validate_split_checkpoint(args_cli.split, _artifact_checkpoint_sha(suite, resume_path), checkpoint_hash)
    validate_checkpoint_actor_observation_dimension(
        resume_path,
        int(suite["task"]["expected_actor_observation_dimension"]),
    )

    env_cfg.seed = int(selected[0]["case"]["seed"])
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    cfg_readback = prepare_deterministic_env_cfg(
        env_cfg,
        num_envs=args_cli.num_envs,
        observation_corruption=args_cli.enable_observation_corruption,
    )
    legacy_readback = reject_legacy_disturbances(env_cfg.events)
    env_cfg.log_dir = str(Path(args_cli.output).expanduser().resolve())

    gym_env = None
    env = None
    try:
        gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(gym_env.unwrapped, DirectMARLEnv):
            gym_env = multi_agent_to_single_agent(gym_env)
        sim_env = gym_env.unwrapped
        reject_legacy_disturbances(sim_env.cfg.events)
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise EvaluationRuntimeError(f"Unsupported RSL-RL runner class: {agent_cfg.class_name!r}.")
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))
        policy = runner.get_inference_policy(device=sim_env.device)
        try:
            policy_module = runner.alg.policy
        except AttributeError:
            policy_module = runner.alg.actor_critic

        command_adapter = SafeCommandTargetAdapter(sim_env.command_manager.get_term(COMMAND_NAME))
        base_body_name = getattr(sim_env.cfg, "base_link_name", None)
        if not isinstance(base_body_name, str) or not base_body_name:
            raise EvaluationRuntimeError("Task configuration does not expose one base_link_name.")
        domain_applicator = FixedSingletonDomainApplicator(
            IsaacFixedDomainBackend(sim_env, base_body_name=base_body_name)
        )
        artifact_writer = EvaluationArtifactWriter(args_cli.output)
        suite_hash = canonical_json_sha256(suite)
        episode_records: list[dict[str, Any]] = []
        case_artifacts: list[dict[str, Any]] = []
        for resolved in selected:
            case = resolved["case"]
            case_id = case["id"]
            raw_path = artifact_writer.output_directory / "cases" / case_id / "raw.npz"
            raw, episode_metrics, runtime_readback = _run_case(
                env=env,
                policy=policy,
                policy_module=policy_module,
                suite=suite,
                resolved=resolved,
                domain_applicator=domain_applicator,
                command_adapter=command_adapter,
                checkpoint_path=resume_path,
                raw_path=raw_path,
            )
            case_metrics = {
                "episode_count": len(episode_metrics),
                "episodes": episode_metrics,
                "aggregate": aggregate_episode_metrics(episode_metrics),
            }
            resolved_config = {
                "suite_id": suite["suite"]["id"],
                "suite_sha256": suite_hash,
                "task": suite["task"],
                "rates": suite["rates"],
                "metric_contract": suite["metric_contract"],
                "checkpoint": {"path": str(resume_path), "sha256": checkpoint_hash},
                "case": case,
                "domain": resolved["domain"],
                "scenario": resolved["scenario"],
                "runtime": {
                    "environment_cfg": cfg_readback,
                    "legacy_disturbance_guard": legacy_readback,
                    "observation_corruption": bool(args_cli.enable_observation_corruption),
                    "readback": runtime_readback,
                },
            }
            case_artifacts.append(
                artifact_writer.write_case(
                    case_id,
                    resolved_config=resolved_config,
                    raw=raw,
                    metrics=case_metrics,
                )
            )
            episode_records.extend(
                {"case_id": case_id, "split": case["split"], "metrics": metrics} for metrics in episode_metrics
            )

        summary = {
            "suite_id": suite["suite"]["id"],
            "suite_sha256": suite_hash,
            "task": args_cli.task,
            "checkpoint": {"path": str(resume_path), "sha256": checkpoint_hash},
            "split": args_cli.split,
            "case_artifacts": case_artifacts,
            "results": aggregate_evaluation_results(episode_records),
        }
        summary_path = artifact_writer.write_summary(summary)
        print(f"[INFO] Evaluation summary: {summary_path}")
    finally:
        if env is not None:
            env.close()
        elif gym_env is not None:
            gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
