import argparse
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from RNN import RNN_LSTM_onlyLastHidden, BRNN, RNN_LSTM_withAttention
from rl_dqn import (
    ReplayBuffer,
    FCN_QNet,
    train,
    device,
    gamma,
    batch_size,
    train_times,
)
from count_param_pytorch import count_parameters

import torch
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
import datetime

from time import time
from timer import secondsToStr, time_log

from plot_rewards import (
    plot_moving_avg,
    log_rewards_frequency,
    plot_rewards_histogram,
    plot_print_rewards_stats,
)

from annealing_schedule import (
    ExponentialDecay,
    Plot_Anneal_Schedule,
)

from early_stop import get_F_patterns, seq_parity_stats, early_stop_S_B

parser = argparse.ArgumentParser(
    usage="%(prog)s [seq] [seed] [algo] [num_episodes] [use_early_stop]...",
    description="DQN learning for Lattice 3D HP"
)
parser.add_argument(
    "seq",
)
parser.add_argument(
    "seed",
    type=int,
)
parser.add_argument(
    "algo",
)
parser.add_argument(
    "num_episodes",
    type=int,
)
parser.add_argument(
    "use_early_stop",
    type=int,
)
args = parser.parse_args()

seq = args.seq.upper()
seed = args.seed
algo = args.algo
num_episodes = args.num_episodes
use_early_stop = args.use_early_stop

base_dir = f"./{datetime.datetime.now().strftime('%m%d-%H%M')}-"
config_str = f"{seq[:6]}-{algo}-seed{seed}-{num_episodes}epi"
save_path = base_dir + config_str + "/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

#display mode
display_mode = "save"
if display_mode == "save":
    save_fig = True
else:
    save_fig = False

#redirecting printing to a text file as opposed to the terminal
orig_stdout = sys.stdout
f = open(save_path + 'out.txt', 'w')
sys.stdout = f

print = partial(print, flush = True)

#logging to the system
print("args parse seq = ", seq)
print("args parse seed = ", seed)
print("args parse algo = ", algo)
print("args parse save_path = ", save_path)
print("args parse num_episodes = ", num_episodes)
print("args parse use_early_stop = ", use_early_stop)

max_steps_per_episode = len(seq)
learning_rate = 0.0001
mem_start_train = max_steps_per_episode * 50
TARGET_UPDATE = 50
buffer_limit = int(min(50000, num_episodes // 10))

print("##### Summary of Hyperparameters #####")
print("learning_rate: ", learning_rate)
print("BATCH_SIZE: ", batch_size)
print("GAMMA: ", gamma)
print("mem_start_train: ", mem_start_train)
print("TARGET_UPDATE: ", TARGET_UPDATE)
print("buffer_limit: ", buffer_limit)
print("train_times: ", train_times)
print("##### End of Summary of Hyperparameters #####")

# Exploration parameters
max_exploration_rate = 1
min_exploration_rate = 0.05

show_every = num_episodes // 1000
pause_t = 0.0
rewards_all_episodes = np.zeros(
    (num_episodes,),
)
reward_max = 0
num_trapped = 0
num_early_stopped = 0
warmRestart = True
decay_mode = "exponential"
num_restarts = 1
exploration_decay_rate = 5
start_decay = 0
#visualizing epsilon curve
Plot_Anneal_Schedule(
    num_episodes,
    min_exploration_rate,
    max_exploration_rate,
    mode=display_mode,
    save_path=save_path,
    warmRestart=warmRestart,
    decay_mode=decay_mode,
    num_restarts=num_restarts,
    exploration_decay_rate=exploration_decay_rate,
    start_decay=start_decay,
)

#early stopped schemes

(
    N_half,
    F_half_pattern,
    F_half_minus_one_pattern,
) = get_F_patterns(seq)

print(f"N_half={N_half}\nF_half_pattern={F_half_pattern}\nF_half_minus_one_pattern={F_half_minus_one_pattern}")

condensed_F_pattern = F_half_minus_one_pattern.replace(", ", '')
print("condensed_F_pattern = ", condensed_F_pattern)

    Odd_H_indices,
    Even_H_indices,
    O_S,
    E_S,
    O_terminal_H,
    E_terminal_H,
    OPT_S,
) = seq_parity_stats(seq)

hp_depth = 2
action_depth = 6
energy_depth = 0
seq_bin_arr = np.asarray([1 if x == "H" else 0 for x in seq])
seq_one_hot = F.one_hot(torch.from_numpy(seq_bin_arr), num_classes=hp_depth)
seq_one_hot = seq_one_hot.cpu().numpy()

init_HP_len = 2
first_two_actions = np.zeros((init_HP_len, ), dtype=int)

def one_hot_state(state_arr, seq_one_hot, action_depth):
    state_arr = np.asarray(state_arr).flatten()  # Convert state_arr to a flat numpy array
    state_arr = np.concatenate((first_two_actions, state_arr))
    state_arr = F.one_hot(torch.from_numpy(state_arr), num_classes=action_depth)
    state_arr = state_arr.numpy()
    state_arr = np.concatenate((state_arr, seq_one_hot), axis=1)
    return state_arr

env_id = "gym_lattice:Lattice3D-5actionStateEnv-v0"
obs_output_mode = "tuple"

env = gym.make(
    id=env_id,
    seq=seq,
    obs_output_mode=obs_output_mode,
)

#to ensure reproducibility
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

initial_state = env.reset()
print("initial state/obs:", initial_state)

n_actions = env.action_space.n
print("n_actions:", n_actions)

network_choice = "test"
row_width = action_depth + hp_depth + energy_depth
col_length = env.observation_space.shape[0] + init_HP_len

if network_choice == "FCN_Reservoir_QNet":
    # FCN_QNet() takes two params: insize and outsize
    # insize ==> input size == size of the observation space
    # insize is flattened obs
    insize = col_length * row_width
    print("FCN_QNet insize = ", insize)
    # outsize ==> output size == number of actions
    print("FCN_QNet outsize = ", n_actions)
    q = FCN_QNet(insize, n_actions).to(device)
    q_target = FCN_QNet(insize, n_actions).to(device)
elif network_choice == "RNN_LSTM_onlyLastHidden":
    # config for RNN
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 512
    num_layers = 5

    print("RNN_LSTM_onlyLastHidden with:")
    print(f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}")
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = RNN_LSTM_onlyLastHidden(input_size, hidden_size, num_layers, n_actions).to(device)
    q_target = RNN_LSTM_onlyLastHidden(input_size, hidden_size, num_layers, n_actions).to(device)
elif network_choice == "BRNN":
    input_size = row_width
    hidden_size = 512
    num_layers = 5
    print("Bidirectional RNN with:")
    print(f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}")
    q = BRNN(input_size, hidden_size, num_layers, n_actions).to(device)
    q_target = BRNN(input_size, hidden_size, num_layers, n_actions).to(device)
elif network_choice == "test":
    # Config for RNN with Attention
    input_size = row_width
    hidden_size = 512
    num_layers = 5
    num_heads = 4  # Number of attention heads

    print("RNN_LSTM_withAttention with:")
    print(f"input_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_heads={num_heads} num_classes={n_actions}")

    # Initialize the networks using RNN_LSTM_withAttention class
    q = RNN_LSTM_withAttention(input_size, hidden_size, num_layers, n_actions, num_heads).to(device)
    q_target = RNN_LSTM_withAttention(input_size, hidden_size, num_layers, n_actions, num_heads).to(device)

q_target.load_state_dict(q.state_dict())
count_parameters(q)
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
memory = ReplayBuffer(buffer_limit)

loss_values = []
print("torch.cuda.is_available() = ", torch.cuda.is_available())
print("device = ", device)

if device.type == "cuda":
     print(torch.cuda.get_device_name(torch.cuda.current_device()))

print("Model's state_dict:")
for param_tensor in q.state_dict():
    print(param_tensor, "\t", q.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

#timing RL experiment
start_time = time()
time_log("Start RL Program")

for n_episode in range(num_episodes):
    if n_episode == 0:
        torch.save(q.state_dict(), f'{save_path}{config_str}-{n_episode}-state_dict.pth')
    if n_episode != num_episodes-1 and ((n_episode + 1) % 40000) == 0:
            torch.save(q.state_dict(), f'{save_path}{config_str}-{n_episode}-state_dict.pth')
    if (n_episode == 0) or ((n_episode+1) % show_every == 0):
        if display_mode == "show":
            render = True
        elif display_mode == "save":
            render = False
    else:
        render = False
    epsilon = ExponentialDecay(
            n_episode,
            num_episodes,
            min_exploration_rate,
            max_exploration_rate,
            exploration_decay_rate=exploration_decay_rate,
            start_decay=start_decay,
        )

    #resetting the environment
    s = env.reset()
    s = one_hot_state(s, seq_one_hot, action_depth)
    done = False
    score = 0.0
    early_stopped = False
    avoid_F = False

    for step in range(max_steps_per_episode):
        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsilon)
        if use_early_stop:
            if a == 1 and avoid_F:
                a = np.random.choice([0, 4])
            avoid_F = False

        s_prime, r, done, info = env.step(a)

        while s_prime is None:
            a = ((a + 1) % 5)
            s_prime, r, done, info = env.step(a)

        a = env.last_action
        (state_E, step_E, reward) = r
        s_prime = one_hot_state(s_prime, seq_one_hot, action_depth)

        if info["is_trapped"]:
            reward = state_E

        if use_early_stop and not done:
            info_actions_str = ''.join(info["actions"])
            info_actions_str_with_F = info_actions_str + 'F'
            if condensed_F_pattern in info_actions_str_with_F:
                avoid_F = True
            max_delta = early_stop_S_B(seq, step+1, O_S, E_S,
                            Odd_H_indices, Even_H_indices,
                            O_terminal_H, E_terminal_H, OPT_S)
            if (state_E + max_delta) < reward_max:
                reward = state_E
                done = True
                early_stopped = True

        r = reward
                done_mask = 0.0 if done else 1.0
        memory.put((s, a, r, s_prime, done_mask))
        s = s_prime
        score += r

        if render:
            env.render(
                mode=display_mode,
                pause_t=pause_t,
                save_fig=save_fig,
                score=score,
            )
            print("step-{} render done\n".format(step))
        if done:
            if len(info['actions']) == (len(seq) - 2):
                pass
            else:
                if use_early_stop and early_stopped:
                    num_early_stopped += 1
                else:
                    num_trapped += 1
            break

        if memory.size()>mem_start_train:
            episode_loss_values = train(q, q_target, memory, optimizer)
            loss_values.extend(episode_loss_values)

        if n_episode % TARGET_UPDATE == 0:
            q_target.load_state_dict(q.state_dict())

        rewards_all_episodes[n_episode] = score
        if score > reward_max:
            print("found new highest reward = ", score)
            reward_max = score
            env.render(
            mode=display_mode,
            save_fig=save_fig,
            save_path=save_path,
            score=score,
            )
        if (n_episode == 0) or ((n_episode+1) % show_every == 0):
            print("Episode {}, score: {:.1f}, epsilon: {:.1f}, reward_max: {}".format(
            n_episode,
            score,
            epsilon,
            reward_max,
            ))
            print(f"\ts_prime: {s_prime[:3], s_prime.shape}, reward: {r}, done: {done}, info: {info}")
print("complete")
end_time = time()
elapsed = end_time - start_time
time_log("End Program", secondsToStr(elapsed))

with open(f'{save_path}{config_str}-rewards_all_episodes.npy', 'wb') as f:
    np.save(f, rewards_all_episodes)

torch.save(q.state_dict(), f'{save_path}{config_str}-state_dict.pth')

# ***** plot the stats and save in save_path *****
def plot_loss_curve(loss_values, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_values)), loss_values)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig(f"{save_path}loss_curve.png")
    plt.close()

plot_moving_avg(rewards_all_episodes, mode=display_mode, save_path=save_path)
log_rewards_frequency(rewards_all_episodes)
plot_rewards_histogram(
    rewards_all_episodes,
    mode=display_mode,
    save_path=save_path,
    config_str=config_str,
)
plot_print_rewards_stats(
    rewards_all_episodes,
    show_every,
    args,
    mode=display_mode,
    save_path=save_path,
)

plot_loss_curve(loss_values, save_path)

env.close()

print("\nnum_trapped = ", num_trapped)
if use_early_stop:
    print("num_early_stopped = ", num_early_stopped)

print("\nreward_max = ", reward_max)

sys.stdout = orig_stdout
f.close()
