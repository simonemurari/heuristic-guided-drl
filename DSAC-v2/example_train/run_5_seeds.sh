#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DSAC2.0
for seed in 31241 89 47235 31241 6
do
    python dsacv2_mlp_mujocoSRH_offserial.py --seed $seed --wandb_group DSAC-v2_SparseReacherHeuristic-v2-TEST
    echo "Finished training with seed $seed"
done
for seed in 31241 89 47235 31241 6
do
    python dsacv2_mlp_mujocoSRH_offserial.py --seed $seed --wandb_group DSAC-v2_SparseReacherHeuristic-v2-TEST_ef005 --end_e 0.05
    echo "Finished training with seed $seed"
done
for seed in 31241 89 47235 31241 6
do
    python dsacv2_mlp_mujocoSRH_offserial.py --seed $seed --wandb_group DSAC-v2_SparseReacherHeuristic-v2-TEST_ef04 --end_e 0.4
    echo "Finished training with seed $seed"
done
# for seed in 47235 31241 6
# do
#     python dsacv2_mlp_mujocoSRC_offserial.py --seed $seed --wandb_group DSAC-v2_SparseReacherControl-v2-VARIANT_WEIGHTED_01 --reward_control_weight 0.1
#     echo "Finished training with seed $seed"
# done
# for seed in 47235 31241 6
# do
#     python dsacv2_mlp_mujocoSRC_offserial.py --seed $seed --wandb_group DSAC-v2_SparseReacherControl-v2-VARIANT_WEIGHTED_005 --reward_control_weight 0.05
#     echo "Finished training with seed $seed"
# done
# for seed in 47235 31241 6
# do
#     python dsacv2_mlp_mujocoSRC_offserial.py --seed $seed --wandb_group DSAC-v2_SparseReacherControl-v2-VARIANT_WEIGHTED_002 --reward_control_weight 0.02
#     echo "Finished training with seed $seed"
# done