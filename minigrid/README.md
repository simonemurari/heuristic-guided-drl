## MiniGrid environment

In this repository you can find the code to run the C51 algorithm and its heuristic-guided variants on the MiniGrid environment from Gymnasium. The code is adapted from CleanRL to work with MiniGrid.

### Repository Structure

- **baselines/**:  
  This folder contains the implementation of the normal C51 algorithm without any additional rules and the reward machine variant where non suggested actions are penalized.

- **Heuristic-Guided variants**:
  `h_c51_product.py`, `h_c51_product_ablation_exploration.py`, `h_c51_product_ablation_exploitation.py`, `h_c51_shift.py` are the heuristic-guided variants of C51.

- **config.py**:  
  This file is used to set the different parameters for each run (learning rate, batch size, environment settings, etc.).

### How to run

1. **Create a Python environment**  
   Create a virtual environment with Python 3.12.6 and install the packages. For example, using conda:
   ```bash
   conda create -n xai-project-minigrid python=3.12.6
   conda activate xai-project-minigrid
   pip install -r requirements.txt
   ```
   Or you can install `uv` and do `uv sync`.

2. **Modify the default doorkey.py file** \
   In order to use MiniGrid-DoorKey maps with more than 1 key you need to substitute the `doorkey.py` file in `.venv/lib/python3.12/site-packages/minigrid/envs` with the `doorkey.py` in the main folder of this repository

3. **Configure wandb (optional)** \
   If you want to use wandb tracking, create a `.env` file in the repository root and set the proper environment variables (WANDB_KEY, WANDB_PROJECT_NAME, WANDB_ENTITY).

4. **Configure the config.py file** \
   You can find the hyperparameters to configure the `config.py` file to replicate the experiments of the paper in each map looking at the Appendix in the paper and the seeds tested in `seeds.txt`

5. **Run the scripts** \
   To run the different scripts just do e.g.: ```python baselines/c51.py``` or ```uv run baselines/c51.py``` if you are using `uv` 
