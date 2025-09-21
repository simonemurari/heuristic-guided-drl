import os
import numpy as np
import torch
import csv
from utils.initialization import create_env
from utils.common_utils import set_seed
from utils.tensorboard_setup import tb_tags



class Evaluator:
    def __init__(self, index=0, **kwargs):
        kwargs.update(
            {"reward_scale": None, "repeat_num": None}
        )  # evaluation don't need to scale reward
        self.env = create_env(**kwargs)
        self.env_id = kwargs["env_id"]
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        self.networks = ApproxContainer(**kwargs)
        self.render = kwargs["is_render"]
        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", False)

        self.print_time = 0
        self.print_iteration = -1

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, epsilon):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        reward_ctrl_list = []
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        print_step = 0
        print_step_list = []
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits, epsilon=epsilon)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            
            if self.env_id == "gym_sparsereachercontrol":
                reward -= next_info["reward_ctrl"]
            
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if self.render:
                self.env.render()
            reward_list.append(reward)
            print_step_list.append(print_step)
            reward_ctrl_list.append(info.get("reward_ctrl"))
        if self.eval_save and self.print_time == 9:
            if iteration == 0:
                os.makedirs(self.save_folder + "/evaluator", exist_ok=True)
                with open(self.save_folder + "/evaluator/eval_dict.csv", "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["iteration", "action0", "action1", "reward"])
            with open(
                self.save_folder + "/evaluator/eval_dict.csv", "a"
            ) as f:
                writer = csv.writer(f)
                for action, print_step, reward in zip(action_list, print_step_list, reward_list):
                    writer.writerow([iteration+print_step, action[0], action[1], reward])

        tb_tags["reward_ctrl"] = np.mean(reward_ctrl_list).item()
        episode_return = sum(reward_list)
        return episode_return

    def run_n_episodes(self, n, iteration, epsilon):
        episode_return_list = []
        for _ in range(n):
            episode_return_list.append(self.run_an_episode(iteration, epsilon))
        return np.mean(episode_return_list)

    def run_evaluation(self, iteration, epsilon):
        return self.run_n_episodes(self.num_eval_episode, iteration, epsilon)


def create_evaluator(**kwargs):
    evaluator = Evaluator(**kwargs)
    print("Create evaluator successfully!")
    return evaluator
