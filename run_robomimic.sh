REWARD_TYPE='C'

for SEED in 150 100 50; do
for ENV in 'lift' 'can'; do
for DATASET in 'mg'; do

echo $ENV
echo $DATASET

ALPHA=2
DETERMINISTIC=1

echo $LR_ACTOR $DETERMINISTIC $ALPHA
python run_rgm.py --env_name $ENV --dataset $DATASET --actor_deterministic $DETERMINISTIC --alpha $ALPHA --seed $SEED --num_expert_traj 0 --reward_type $REWARD_TYPE
done
done
done