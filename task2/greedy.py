import csv
from collections import defaultdict
from tqdm import tqdm
# 从CSV文件中读取数据
def read_txt(filename):
    ret = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            ret.append([int(line[0]), int(line[1]), int(line[2])])
    return ret

import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)  
sys.path.append(parent_path) 

import random 
import math
import gymnasium as gym
import torch
from tianshou.utils.net.common import ActorCritic
import model
import arguments
import numpy as np
from tools import *
from mycollector import PackCollector
from masked_ppo import MaskedPPOPolicy
import warnings
warnings.filterwarnings("ignore")

def build_net(args, device):
    feature_net = model.ShareNet(
        k_placement=args.env.k_placement, 
        box_max_size=args.env.box_big, 
        container_size=args.env.container_size, 
        embed_size=args.model.embed_dim, 
        num_layers=args.model.num_layers,
        forward_expansion=args.model.forward_expansion,
        heads=args.model.heads,
        dropout=args.model.dropout,
        device=device,
        place_gen=args.env.scheme
    )

    actor = model.ActorHead(
        preprocess_net=feature_net, 
        embed_size=args.model.embed_dim, 
        padding_mask=args.model.padding_mask,
        device=device, 
    ).to(device)

    critic = model.CriticHead(
        preprocess_net=feature_net, 
        k_placement=args.env.k_placement,
        embed_size=args.model.embed_dim,
        padding_mask=args.model.padding_mask,
        device=device, 
    ).to(device)

    return actor, critic

def test(args):

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")
        
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # environment
    
    test_env = gym.make(
        args.env.id, 
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type=args.env.box_type,
        item_set=args.env.box_size_set, 
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=args.render
    )
    # print('观测空间 = {}'.format(test_env.observation_space))
    # print('动作空间 = {}'.format(test_env.action_space))
    # print('动作数 = {}'.format(test_env.action_space.n))

    # network
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    
    # RL agent 
    dist = CategoricalMasked

    policy = MaskedPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.train.gamma,
        eps_clip=args.train.clip_param,
        advantage_normalization=False,
        vf_coef=args.loss.value,
        ent_coef=args.loss.entropy,
        gae_lambda=args.train.gae_lambda,
        action_space=test_env.action_space,
    )
    
    policy.eval()
    try:
        policy.load_state_dict(torch.load(args.ckp, map_location=device))
        # print(policy)
    except FileNotFoundError:
        print("No model found")
        exit()

    test_collector = PackCollector(policy, test_env)

    # Evaluation
    result = test_collector.collect(n_episode=args.test_episode, render=args.render)
    if args.render:
        for i in range(args.test_episode):
            print(f"episode {i+1}\t => \tratio: {result['ratios'][i]:.4f} \t| total: {result['nums'][i]}")
        print('All cases have been done!')

    return result['num'], result['ratio']


# 测试数据

def one_thread(args, thread_dataset):
    registration_envs()
    best_ratio = 0
    best_list = []
    for item_list in tqdm(thread_dataset):
        args.env.box_size_set = item_list
        args.seed = 80
        args.test_episode = 1
        ok_num, ok_ratio = test(args)
        if ok_ratio > best_ratio:
            best_ratio = ok_ratio
            best_list = item_list
    print(f"best_ratio: {best_ratio} best_list: {best_list}")

if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps 
    if args.tqdm:
        args.render = False
    # get data
    import copy
    test_dataset = read_txt("dataset.txt")
    for i in range(len(test_dataset)):
        best_ratio = 0
        best_list = []
        best_num = 0
        args.env.box_size_set = test_dataset
        args.seed = 80
        args.test_episode = 1
        ok_num, ok_ratio = test(args)
        best_ratio = ok_ratio
        best_num = ok_num
        best_list = copy.deepcopy(test_dataset)
        temp = copy.deepcopy(test_dataset)
        temp[i][0], temp[i][2] = temp[i][2], temp[i][0]
        args.env.box_size_set = temp
        args.seed = 80
        args.test_episode = 1
        ok_num, ok_ratio = test(args)
        if ok_ratio > best_ratio:
            best_ratio = ok_ratio
            best_num = ok_num
            best_list = copy.deepcopy(temp)
        temp[i][2], temp[i][1] = temp[i][1], temp[i][2]
        args.env.box_size_set = temp
        args.seed = 80
        args.test_episode = 1
        ok_num, ok_ratio = test(args)
        if ok_ratio > best_ratio:
            best_ratio = ok_ratio
            best_num = ok_num
            best_list = copy.deepcopy(temp)
        if i >= best_num:
            break
        test_dataset = copy.deepcopy(best_list)
    print(f"best_ratio: {best_ratio} best_list: {best_list}")    
    args.render = True
    args.env.box_size_set = best_list
    args.seed = 80
    args.test_episode = 1
    ok_num, ok_ratio = test(args)
