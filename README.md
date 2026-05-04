# mujoco-sim

## SO101 simulation

This repo includes the Robot Studio SO101 follower-arm model from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
under `model/so101/`. The exposed motors follow the LeRobot-style names:

- `shoulder_pan`
- `shoulder_lift`
- `elbow_flex`
- `wrist_flex`
- `wrist_roll`
- `gripper`

Run the viewer demo with the existing conda environment:

```bash
conda run -n mujoco-sim python so101_env.py
```

Gymnasium usage:

```python
from so101_env import SO101GymEnv

env = SO101GymEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```
