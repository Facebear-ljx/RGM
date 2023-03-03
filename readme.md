# Mind the Gap: Offline Policy Optimization for Imperfect Rewards (ICLR 2023)
This is the official implementation of RGM (Reward Gap Minimization) (https://openreview.net/forum?id=WumysvcMvV6). RGM can be perceived as a hybrid offline RL
and offline IL method that can handle diverse types of imperfect rewards include but not limited to
**partially correct reward, sparse reward, multi-task datasharing setting and completely incorrect rewards.**

![avatar](documents/diverse_rewards)

RGM formalizes offline policy optimization for imperfect rewards as a bilevel optimization problem, where the upper layer optimizes a reward correction
term that performs visitation distribution matching w.r.t. some expert data and the lower layer solves a pessimistic RL problem with the corrected rewards.

![avatar](documents/framework)
#### Usage
To install the dependencies, use 
```python
    pip install -r requirements.txt
```
If you want conduct experiments on Robomimic datasets. You need to install Robomimic according to the
instructions in [Robomimic](https://robomimic.github.io/)

#### Benchmark experiments
You can reproduce the  Mujoco tasks and Robomimic tasks like so:
```python
    bash run d4rl.sh
```
```python
    bash run_robomimic.sh
```
For the experiments on multi-task datasharing setting, we'll release soon.

#### Visulization of Learning curves
You can resort to [wandb](https://wandb.ai/site) to login your personal account via export your own wandb api key.
```
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```
and run 
```
wandb online
```
to turn on the online syncronization.


If you find our code and paper can help, please cite our paper as:
#### Bibtex

```
@inproceedings{
li2023mind,
title={Mind the Gap: Offline Policy Optimization for Imperfect Rewards},
author={Jianxiong Li and Xiao Hu and Haoran Xu and Jingjing Liu and Xianyuan Zhan and Qing-Shan Jia and Ya-Qin Zhang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=WumysvcMvV6}
}
```

