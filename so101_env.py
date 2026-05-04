from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import src.mujoco_viewer as mujoco_viewer


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_XML = ROOT / "model" / "so101" / "scene.xml"
DEFAULT_ARM_XML = ROOT / "model" / "so101" / "so101.xml"

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
HOME_QPOS = np.array([0.0, -0.72, 1.2, -0.6, 0.0, 0.35], dtype=np.float64)


class SO101GymEnv(gym.Env):
    """Small Gymnasium environment for the SO101 follower arm.

    The robot assets come from google-deepmind/mujoco_menagerie. Actions are
    absolute position targets in radians for the five arm joints plus the
    gripper hinge.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        scene_xml: str | Path = DEFAULT_SCENE_XML,
        frame_skip: int = 10,
        render_mode: str | None = None,
        obs_width: int = 640,
        obs_height: int = 480,
    ):
        self.scene_xml = Path(scene_xml)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.obs_width = obs_width
        self.obs_height = obs_height

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)

        self.joint_ids = np.array([self.model.joint(name).id for name in JOINT_NAMES])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in JOINT_NAMES])
        self.qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids])
        self.qvel_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids])
        self.ee_site_id = self.model.site("gripperframe").id
        self.target_body_id = self.model.body("target_cube").id

        ctrlrange = self.model.actuator_ctrlrange[self.actuator_ids].astype(np.float32)
        self.action_space = spaces.Box(ctrlrange[:, 0], ctrlrange[:, 1], dtype=np.float32)
        obs_size = len(JOINT_NAMES) * 2 + 3 + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        self.renderer = None
        if render_mode is not None and render_mode != "rgb_array":
            raise ValueError(f"Unsupported render_mode: {render_mode}")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = HOME_QPOS
        self._randomize_target(options)
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_ids]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[self.actuator_ids] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        reward = -info["ee_target_distance"]
        terminated = bool(info["ee_target_distance"] < 0.045)
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.obs_height, width=self.obs_width)
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _randomize_target(self, options: dict | None):
        target_xy = None if options is None else options.get("target_xy")
        if target_xy is None:
            target_xy = self.np_random.uniform([0.25, -0.12], [0.43, 0.12])

        target_qpos_id = self.model.jnt_qposadr[self.model.joint("target_cube_joint").id]
        self.data.qpos[target_qpos_id : target_qpos_id + 7] = np.array(
            [target_xy[0], target_xy[1], 0.035, 1.0, 0.0, 0.0, 0.0]
        )
        target_qvel_id = self.model.jnt_dofadr[self.model.joint("target_cube_joint").id]
        self.data.qvel[target_qvel_id : target_qvel_id + 6] = 0.0

    def _get_obs(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self.data.xpos[self.target_body_id].copy()
        return np.concatenate(
            [
                self.data.qpos[self.qpos_ids],
                self.data.qvel[self.qvel_ids],
                ee_pos,
                target_pos,
                target_pos - ee_pos,
            ]
        ).astype(np.float32)

    def _get_info(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self.data.xpos[self.target_body_id].copy()
        return {
            "ee_position": ee_pos,
            "target_position": target_pos,
            "ee_target_distance": float(np.linalg.norm(target_pos - ee_pos)),
        }


class SO101ViewerEnv(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml: str | Path = DEFAULT_SCENE_XML):
        super().__init__(str(scene_xml), distance=0.9, azimuth=-45, elevation=-25)
        self.home_qpos = HOME_QPOS.copy()

    def runBefore(self):
        self.step_count = 0
        self.data.qpos[: len(JOINT_NAMES)] = self.home_qpos
        self.data.ctrl[: len(JOINT_NAMES)] = self.home_qpos

    def runFunc(self):
        t = self.step_count * self.model.opt.timestep
        target = self.home_qpos.copy()
        target[0] += 0.45 * np.sin(0.8 * t)
        target[1] += 0.25 * np.sin(0.6 * t)
        target[2] += 0.35 * np.sin(0.7 * t + 0.5)
        target[3] += 0.25 * np.sin(1.1 * t)
        target[4] += 0.8 * np.sin(1.2 * t)
        target[5] = 0.4 + 0.25 * np.sin(1.5 * t)
        self.data.ctrl[: len(JOINT_NAMES)] = np.clip(
            target,
            self.model.actuator_ctrlrange[: len(JOINT_NAMES), 0],
            self.model.actuator_ctrlrange[: len(JOINT_NAMES), 1],
        )
        self.step_count += 1


if __name__ == "__main__":
    env = SO101ViewerEnv()
    env.run_loop()
