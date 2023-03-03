from ast import arg, parse
import wandb
import argparse
from RL_algos.RGM import rgm
import datetime
import random
import os


ROBOMIMIC = ['lift', 'can', 'square', 'tool_hang', 'transport']

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    wandb.init(project="MG_release")

    seed = random.randint(0, 1000)
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='hopper', type=str)
    parser.add_argument('--dataset', default='medium-replay', type=str)
    parser.add_argument('--state', default=False, type=boolean)
    parser.add_argument('--mismatch', default=False, type=boolean)
    parser.add_argument('--algo_type', default='smodice', type=str)
    parser.add_argument('--disc_type', default='learned', type=str)
    parser.add_argument('--gamma', default=0.99, type=float)

    parser.add_argument('--num_expert_traj', default=0, type=int)
    parser.add_argument('--num_offline_traj', default=2000, type=int)
    parser.add_argument('--total_iterations', default=int(1e6), type=int)
    parser.add_argument('--disc_iterations', default=int(1e3), type=int)
    parser.add_argument('--log_iterations', default=int(5e3), type=int)
    parser.add_argument('--episodes', default=10, type=int)

    parser.add_argument('--bc_only', default=False, type=boolean)
    parser.add_argument('--actor_deterministic', default=True, type=boolean)
    parser.add_argument('--absorbing_state', default=True, type=boolean)
    parser.add_argument('--standardize_reward', default=True, type=boolean)
    parser.add_argument('--standardize_obs', default=True, type=boolean)
    parser.add_argument('--reward_type', default='P', type=str, help='choose from T/P/C')
    parser.add_argument('--reward_scale', default=1, type=float)
    parser.add_argument('--res_scale', default=3, type=float)
    parser.add_argument('--mean_range', default=(-7.24, 7.24))
    parser.add_argument('--logstd_range', default=(-5., 2.))

    parser.add_argument('--hidden_sizes', default=(256, 256))
    parser.add_argument('--num_hidden', default=256, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--f', default='kl', type=str)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--lr_ratio', default=0.001, type=float)
    parser.add_argument('--v_l2_reg', default=0.0001, type=float)
    parser.add_argument('--r_l2_reg', default=0.0001, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--use_policy_entropy_constraint', default=True, type=boolean)
    parser.add_argument('--target_entropy', default=None, type=float)

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--wandb', default=True, type=boolean)
    parser.add_argument('--render', default=False, type=boolean)
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--robomimic_path', default='/home/airtrans01/LJX/robomimic/datasets', type=str, help='please change to your robomimic dataset path')
    args = parser.parse_args()

    if args.env_name == 'antmaze':
        args.num_expert_traj = 0
        args.absorbing_state = False
        args.actor_deterministic = True
        args.alpha = 0.5
        args.episodes = 20
        # args.gamma = 0.995

    if args.env_name == 'kitchen':
        args.num_expert_traj = 0
        args.num_offline_traj = 2000
        args.absorbing_state = False
        args.f = 'chi'
        args.dataset = 'mixed'
        args.actor_deterministic = False

    if args.env_name in ROBOMIMIC:
        args.total_iterations = int(2e5)
        args.f = 'chi'
        args.num_expert_traj = 0
        args.episodes = 50
        args.disc_iterations = 1000
        args.batch_size = 100
        args.alpha = 2
        args.absorbing_state = False
        args.standardize_reward = True
        args.standardize_obs = True
        args.num_hidden = 256
        args.log_iterations = 5000
        
    wandb.config.update(args)

    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{env_name}_{args.dataset}_{args.alpha}_{args.f}"

    agent_rgm = rgm(env_name=env_name,
                  dataset=args.dataset,
                  seed=args.seed,
                  args=vars(args)
                  )

    if args.bc_only:
        agent_rgm.train_bc_only(total_time_step=args.total_iterations)
    else:
        agent_rgm.learn(total_time_step=args.total_iterations)


if __name__ == '__main__':
    main()
