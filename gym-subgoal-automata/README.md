## OfficeWorld environment

In this repository you can find the code to run the C51 algorithm and its heuristic-guided variants on the OfficeWorld environment. The code is adapted from CleanRL to work with OfficeWorld. In `env_README.md` you can find the original README.md of the OfficeWorld environment and in `gym-subgoal-automata` you can find the code for all the environments.

### Repository Structure

- **baselines/**:  
  This folder contains the implementation of the normal C51 algorithm without any additional rules and the reward machine variant where non suggested actions are penalized.

- **Heuristic-Guided variants**:
  `h_c51_product.py`, `h_c51_shift.py` are the heuristic-guided variants of C51.

- **config.py**:  
  This file is used to set the different parameters for each run (learning rate, batch size, environment settings, etc.).

### How to run

1. **Create a Python environment**  
   Create a virtual environment with Python 3.7.9 and install the packages. For example, using conda:
   ```bash
   conda create -n xai-project-officeworld python=3.7.9
   conda activate xai-project-officeworld
   pip install -r requirements.txt
   ```
   Or you can install `uv` and do `uv sync`.

2. **Configure wandb (optional)** \
   If you want to use wandb tracking, create a `.env` file in this folder and set the proper environment variables (WANDB_KEY, WANDB_PROJECT_NAME, WANDB_ENTITY).

3. **Configure the config.py file** \
   You can find the hyperparameters to configure the `config.py` file to replicate the experiments of the paper in each map looking at the Appendix in the paper and the seeds tested in `seeds.txt`

4. **Run the scripts** \
   To run the different scripts just do e.g.: ```python h_c51_product.py``` or ```uv run h_c51_product.py``` if you are using `uv` 
