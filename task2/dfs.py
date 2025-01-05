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

def choose_container_size(container_set, box_size_set):
    container_choice = container_set.copy()
    # if container can't hold any of the box_set, remove it from container_choice
    for container in container_set:
        for box_size in box_size_set:
            if (container[0] < box_size[0] or container[1] < box_size[1] or container[2] < box_size[2]) and (container[0] < box_size[1] or container[1] < box_size[0] or container[2] < box_size[2]):
                container_choice.remove(container)
                break
    if len(container_choice) == 0:
        container_choice = container_set.copy()
        # swap container[0] and container[2] of each container
        for container in container_choice:
            # 'tuple' object does not support item assignment
            container[0], container[2] = container[2], container[0]
            # if container can't hold any of the box_set, remove it from container_choice
            for box_size in box_size_set:
                if (container[0] < box_size[0] or container[1] < box_size[1] or container[2] < box_size[2]) and (container[0] < box_size[1] or container[1] < box_size[0] or container[2] < box_size[2]):
                    container_choice.remove(container)
                    break
    # calculate the sum of the volume of the boxes
    box_volume = 0
    for box_size in box_size_set:
        box_volume += box_size[0] * box_size[1] * box_size[2]
    # choose the smallest container that can hold the boxes
    for container in container_choice:
        if box_volume < container[0] * container[1] * container[2]:
            return container
    # if no container can hold the boxes, choose the largest container
    return container_choice[-1]

from itertools import product

def generate_swapped_combinations(data):
    # 替换每个子列表的第三个值，与第一个和第二个值交换或保持原值
    swap_options = []

    _data = data[::-1]

    for sublist in _data:
        first, second, third = sublist
        swap_options.append([
            [third, second, first],  # 与第一个值交换
            [first, third, second],  # 与第二个值交换
            [first, second, third],  # 保持不变
        ])
    # 使用笛卡尔积生成所有组合
    result = list(product(*swap_options))
    return [list(combination) for combination in result]


# 测试数据

def one_thread(args, thread_dataset):
    registration_envs()
    best_ratio = 0
    best_list = []
    for item_list in tqdm(thread_dataset):
        args.env.box_size_set = item_list
        args.seed = 80
        args.test_episode = 1
        ok_num, ok_ratio, ok_num = test(args)
        if ok_ratio > best_ratio:
            best_ratio = ok_ratio
            best_list = item_list
    print(f"best_ratio: {best_ratio} best_list: {best_list}")
def cut_branch(item_list, cut_set):
    for cut in cut_set:
        if item_list[:len(cut)] == cut:
            return True
    return False

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
    print(f"test_dataset: {test_dataset}")
    rotate_dataset = generate_swapped_combinations(test_dataset)
    true_rotate_dataset = []
    for item in rotate_dataset:
        true_rotate_dataset.append(item[::-1])

    del_set = ()
    best_ratio = 0
    best_list = []
    now_cnt = 0
    for item_list in tqdm(true_rotate_dataset):
        now_cnt += 1
        if cut_branch(item_list, del_set):
            continue
        args.env.box_size_set = item_list
        args.seed = 80
        args.test_episode = 1
        ok_num, ok_ratio = test(args)
        ok_num = int(ok_num)
        if now_cnt > 3**ok_num:
            
            del_set += tuple([item_list[:ok_num+1]])
            # print(f"ccut: del_set: {del_set}")
        if ok_ratio > best_ratio:
            best_ratio = ok_ratio
            best_list = item_list
    print(f"best_ratio: {best_ratio} best_list: {best_list}")
    args.render = True
    args.env.box_size_set = best_list
    args.seed = 80
    args.test_episode = 1
    ok_num, ok_ratio = test(args)
