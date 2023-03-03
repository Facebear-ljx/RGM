REWARD_TYPE='P'

for SEED in 150 100 50 0; do
for ENV in 'walker2d' 'halfcheetah' 'hopper'; do
#ENV='halfcheetah'
for DATASET in 'medium-expert' 'medium-replay' 'medium' 'random'; do
ENV_NAME=$ENV'-'$DATASET'-v2'
echo $ENV_NAME

LR_ACTOR=3e-4
ALPHA=0.5
DETERMINISTIC=0

if [[ "$ENV_NAME" =~ "hopper" ]]; then
DETERMINISTIC=1
if [[ "$ENV_NAME" =~ "hopper-medium-v2" ]]; then
LR_ACTOR=3e-4
fi
fi

if [[ "$ENV_NAME" =~ "walker2d-medium-replay-v2" ]]; then
ALPHA=4
fi

echo $LR_ACTOR $DETERMINISTIC $ALPHA
python run_rgm.py --env_name $ENV --dataset $DATASET --actor_deterministic $DETERMINISTIC --actor_lr $LR_ACTOR --alpha $ALPHA --seed $SEED --num_expert_traj 0 --reward_type $REWARD_TYPE
done
done
done