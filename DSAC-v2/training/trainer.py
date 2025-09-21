__all__ = ["OffSerialTrainer"]

from cmath import inf
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.tensorboard_setup import add_scalars
from utils.tensorboard_setup import tb_tags
from utils.common_utils import ModuleOnDevice
from dotenv import load_dotenv
from utils.wandb_setup import add_scalar as wandb_add_scalar
from utils.wandb_setup import wandb_init, wandb_finish
from tqdm import tqdm, trange


class OffSerialTrainer:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.start_e = kwargs.get("start_e", 0)
        self.end_e = kwargs.get("end_e", 0)
        self.exploration_fraction = kwargs.get("exploration_fraction", 0)
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0
        self.epsilon = 0
        
        self.track = kwargs.get("track", False)
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        if self.track:
            dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            load_dotenv(dotenv_path=dotenv_path)
            wandb_project = os.getenv("WANDB_PROJECT_NAME")
            wandb_entity = os.getenv("WANDB_ENTITY")
            wandb_init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{self.alg.__class__.__name__}-{kwargs['env_id']}-SEED={kwargs['seed']}-{time.strftime('%Y-%m-%d-%H-%M-%S')}",
                group=kwargs.get("wandb_group", None),
                config=kwargs
            )
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        # pre sampling
        while self.buffer.size < kwargs["buffer_warm_size"]:
            samples, _ = self.sampler.sample(1.0)
            self.buffer.add_batch(samples)

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            torch.device("cuda")
            print(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            self.networks.cuda()

        self.start_time = time.time()

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        try:
            slope = (end_e - start_e) / duration
        except ZeroDivisionError:
            slope = 0
        return max(slope * t + start_e, end_e)

    def step(self):
        self.epsilon = self.linear_schedule(
            self.start_e,
            self.end_e,
            self.exploration_fraction * self.max_iteration,
            self.iteration,
        )
        # sampling
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                sampler_samples, sampler_tb_dict = self.sampler.sample(self.epsilon)
            self.buffer.add_batch(sampler_samples)

        # replay
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()

        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, self.iteration, 0.0
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(replay_samples, self.iteration, 0.0)

        # log
        if self.iteration % self.log_save_interval == 0:
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration % self.eval_interval == 0 or self.iteration == self.max_iteration - 1:
            with ModuleOnDevice(self.networks, "cpu"):
                total_avg_return = self.evaluator.run_evaluation(self.iteration, 0.0)

            if (
                total_avg_return >= self.best_tar
                and self.iteration >= self.max_iteration / 5
            ):
                self.best_tar = total_avg_return
                tqdm.write(f"New best TAR: {self.best_tar} at iteration {self.iteration}")
                for filename in os.listdir(self.save_folder + "/apprfunc/"):
                    if filename.endswith("_opt.pkl"):
                        os.remove(self.save_folder + "/apprfunc/" + filename)

                torch.save(
                    self.networks.state_dict(),
                    self.save_folder
                    + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                )

            self.writer.add_scalar(
                tb_tags["Buffer RAM of RL iteration"],
                self.buffer.__get_RAM__(),
                self.iteration,
            )
            self.writer.add_scalar(
                tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
            )
            self.writer.add_scalar(
                tb_tags["TAR of replay samples"],
                total_avg_return,
                self.iteration * self.replay_batch_size,
            )
            self.writer.add_scalar(
                tb_tags["TAR of total time"],
                total_avg_return,
                int(time.time() - self.start_time),
            )
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                self.sampler.get_total_sample_number(),
            )
            if self.track:
                wandb_add_scalar({"Total Average Return": total_avg_return, "global_step": self.iteration})
                wandb_add_scalar({"Reward Control": tb_tags["reward_ctrl"], "global_step": self.iteration})

            if self.iteration % 30000 == 0:
                tqdm.write(f"Iteration {self.iteration}: TAR = {total_avg_return}")
        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

    def train(self):
        for _ in trange(self.max_iteration):
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()
        if self.track:
            wandb_finish()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )


def create_trainer(alg, sampler, buffer, evaluator, **kwargs):
    trainer = OffSerialTrainer(alg, sampler,buffer, evaluator, **kwargs)
    print("Create trainer successfully!")
    return trainer
