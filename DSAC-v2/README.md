## H-DSAC-T algorithm

In this repository you can find the code to run the H-DSAC-T algorithm and the reward machine baselines on the SparseReacher-v2 environment from Gym. In `original_README.md` you can find the original `README.md` file of the DSAC-T repository.

### Repository Structure

  In the `example_train/` folder you can find the files `dsacv2_mlp_mujocoSR_RMMIN_offserial.py`, `dsacv2_mlp_mujocoSR_RMH_offserial.py` and `dsacv2_mlp_mujocoSRH_offserial.py` with all the hyperparameters already set and ready to be run. The heuristics are integrated in the action distribution in `utils/act_distribution_cls.py`, in the `dsac_v2.py` file and in the files in the `training/` folder.

### How to run

   **Requires**
   1. Windows 7 or greater or Linux.
   2. Python 3.8.
   3. The installation path must be in English.

   **Installation**
   ```bash
   # Please make sure not to include Chinese characters in the installation path, as it may result in a failed execution.
   cd DSAC-T
   # create conda environment
   conda env create -f DSAC2.0_environment.yml
   conda activate DSAC2.0
   # install DSAC2.0
   pip install -e.
   ```

   **Configure wandb (optional)** \
   If you want to use wandb tracking, create a `.env` file in this folder and set the proper environment variables (WANDB_KEY, WANDB_PROJECT_NAME, WANDB_ENTITY).

   **Hyperparameters** \
   All the hyperparameters are preconfigured in the scripts in `example_train/`. To reproduce the results of the paper, set the random seeds to the values listed in `seeds.txt` (you can edit the seed in the script or pass it as an argument).

   **Run the scripts** \
   To run the different scripts just do e.g.: `python example_train/dsacv2_mlp_mujocoSR_RMH_offserial` after activating the conda environment.
