import gym
import numpy as np
from gym.envs.mujoco import ReacherEnv

class SparseReacherControlEnv(ReacherEnv):
    def __init__(self, reward_control_weight=0.05):
        self.reward_control_weight = reward_control_weight
        super(SparseReacherControlEnv, self).__init__()

    def step(self, a):
        thre = 0.01
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = self.reward_control_weight * -np.square(a - np.array([0.25, -0.45])).sum()
        reward_sparse = -float(reward_dist < -thre)
        reward = reward_sparse + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

# Use gym.make to get proper spec injection
def env_creator(**kwargs):
    try:
        weight = float(kwargs.pop("reward_control_weight", 0.05))
        return SparseReacherControlEnv(reward_control_weight=weight)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Warning: mujoco, mujoco-py and MSVC are not installed properly"
        )
    except Exception as e:
        raise Exception(f"Failed to create environment: {e}")

if __name__ == "__main__":
    env = env_creator()
    env.reset()
    for _ in range(100):
        a = env.action_space.sample()
        env.step(a)
        env.render()
