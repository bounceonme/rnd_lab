"""RND STEP environment hooks for physics-rate diagnostics and rewards."""

from __future__ import annotations

from typing import Protocol

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class RndPhysicsStepObserver(Protocol):
    """Observer called only after Isaac Lab has refreshed scene and sensor buffers."""

    def on_post_scene_update(
        self,
        env: "RndStepManagerBasedRLEnv",
        action: torch.Tensor,
        policy_step: int,
        substep_index: int,
    ) -> None: ...

    def on_pre_reset(
        self,
        env: "RndStepManagerBasedRLEnv",
        env_ids: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None: ...

    def on_post_reset(self, env: "RndStepManagerBasedRLEnv", env_ids: torch.Tensor) -> None: ...


class RndStepManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with a correctly aligned 200 Hz observer hook."""

    def __init__(self, *args, **kwargs):
        self._rnd_physics_step_observers: list[RndPhysicsStepObserver] = []
        super().__init__(*args, **kwargs)

    def add_rnd_physics_observer(self, observer: RndPhysicsStepObserver) -> None:
        """Register one observer without allowing duplicate callbacks."""
        if observer not in self._rnd_physics_step_observers:
            self._rnd_physics_step_observers.append(observer)

    def remove_rnd_physics_observer(self, observer: RndPhysicsStepObserver) -> None:
        """Remove a previously registered observer."""
        if observer in self._rnd_physics_step_observers:
            self._rnd_physics_step_observers.remove(observer)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one policy step and expose current 200 Hz scene data to observers."""
        action = action.to(self.device)
        self.action_manager.process_action(action)
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        policy_step = int(self.common_step_counter)
        for substep_index in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            # Keep the pinned Isaac Lab recorder behavior unchanged. RND observers are called
            # below, after scene.update(), because cached articulation/sensor data is current there.
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            for observer in tuple(self._rnd_physics_step_observers):
                observer.on_post_scene_update(self, action, policy_step, substep_index)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            for observer in tuple(self._rnd_physics_step_observers):
                observer.on_pre_reset(
                    self,
                    reset_env_ids,
                    self.reset_terminated,
                    self.reset_time_outs,
                )
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)

            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)
            for observer in tuple(self._rnd_physics_step_observers):
                observer.on_post_reset(self, reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def close(self):
        if not self._is_closed:
            for observer in tuple(self._rnd_physics_step_observers):
                close = getattr(observer, "close", None)
                if close is not None:
                    close()
            self._rnd_physics_step_observers.clear()
        super().close()
