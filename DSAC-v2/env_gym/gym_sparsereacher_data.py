import gym
import numpy as np
from gym.envs.mujoco import ReacherEnv

class SparseReacherEnv(ReacherEnv):

    def step(self, a):
        thre = 0.01
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward_sparse = -float(reward_dist < -thre)
        reward = reward_sparse
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        

# Use gym.make to get proper spec injection
def env_creator(**kwargs):
    try:
        return SparseReacherEnv()
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
