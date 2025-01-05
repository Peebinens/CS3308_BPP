import csv
from collections import defaultdict
from tqdm import tqdm
# 从CSV文件中读取数据
def read_csv(filename):
    grouped_data = defaultdict(list)

    with open(filename, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # print(row)
            sta_code = row['sta_code']
            sku_info = {
                "sku_code": row['sku_code'],
                "dimensions": {
                    "length": float(row['长(CM)']),
                    "width": float(row['宽(CM)']),
                    "height": float(row['高(CM)'])
                },
            }
            qty = int(row['qty'])
            for _ in range(qty):    
                grouped_data[sta_code].append(sku_info)
    return grouped_data

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
    # for i in range(args.test_episode):
    #     print(f"episode {i+1}\t => \tratio: {result['ratios'][i]:.4f} \t| total: {result['nums'][i]}")
    # print('All cases have been done!')
    # print('----------------------------------------------')
    # print('average space utilization: %.4f'%(result['ratio']))
    # print('average put item number: %.4f'%(result['num']))
    # print("standard variance: %.4f"%(result['ratio_std']))
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
if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps 
    if args.tqdm:
        args.render = False
    # get data
    test_dataset = read_csv('data_clean.csv')
    container_set = [(35, 23, 13), (37, 36, 13), (38, 26, 13), (40, 28 ,16), (42, 30, 18), (42, 30, 40), (52, 40, 17), (54, 45, 36)]
    sumsumbox = 0
    sumsumcontainer = 0 
    max_ratio = 0
    container_set.sort(key=lambda x: x[0] * x[1] * x[2])
    max_order = ''
    if args.tqdm:
        test_dataset = tqdm(test_dataset.items())
    else:
        test_dataset = test_dataset.items()
    for sta_code, sku_list in test_dataset:
        if not args.tqdm:
            print(f"sta_code: {sta_code}")
        box_size_set = []
        all_box_volume = 0
        # print(f"sta_code: {sta_code}")
        for sku in sku_list:
            all_box_volume += sku['dimensions']['length'] * sku['dimensions']['width'] * sku['dimensions']['height']
            # # put length, width, height into box_size_set and ceil them
            if args.tqdm:
                box_size_set.append((math.ceil(sku['dimensions']['length']), math.ceil(sku['dimensions']['width']), math.ceil(sku['dimensions']['height'])))
            else:
                box_size_set.append((math.ceil(sku['dimensions']['length']), math.ceil(sku['dimensions']['width']), math.ceil(sku['dimensions']['height']), sku))

        box_size_set.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
        all_container_volume = 0
        while len(box_size_set) > 0:
            args.env.box_size_set = box_size_set
            container_size = choose_container_size(container_set, box_size_set)
            if not args.tqdm:
                print(f"    container_size: {container_size}")
            args.env.container_size = container_size
            args.seed = 80
            args.test_episode = 1
            ok_num, ok_ratio = test(args)
            all_container_volume += container_size[0] * container_size[1] * container_size[2]

            for i in range(int(ok_num)):
                box_size_set.pop(0)

            if int(ok_num) == 0:
                exit()   
        # sta_code ratio
        now_ratio = all_box_volume / all_container_volume
        if now_ratio > max_ratio:
            max_ratio = now_ratio
            max_order = sta_code
        sumsumbox += all_box_volume
        sumsumcontainer += all_container_volume
        if not args.tqdm:
            print(f"ratio: {all_box_volume / all_container_volume}")
            print('----------------------------------------------')
    print("All cases have been done!")
    print(f"average space utilization: {sumsumbox / sumsumcontainer}")
    # print(f"max_order: {max_order} max_ratio: {max_ratio}")

    
    
