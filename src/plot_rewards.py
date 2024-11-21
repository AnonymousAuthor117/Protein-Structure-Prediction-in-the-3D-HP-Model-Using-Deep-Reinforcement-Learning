import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import MaxNLocator, MultipleLocator

def plot_moving_avg(scores, n=300, mode="show", save_path=""):
    print("Mean score:", scores.mean())

    moving_avg = np.convolve(scores, np.ones(n) / n, mode='valid')

    # Save moving average data to CSV file
    csv_filename = save_path + "moving_avg-" + str(n) + ".csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Episode', 'Moving Average Reward'])
        for i, avg in enumerate(moving_avg, start=n):
            csv_writer.writerow([i, avg])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(n, len(moving_avg) + n), moving_avg, color='#1f77b4', linewidth=2)
    ax.set(xlabel='Episode', ylabel='Moving Average Reward',
           title=f'Moving Average of Rewards (n={n})')
    ax.grid(True, linewidth=0.8, linestyle='--', alpha=0.7)
    fig.tight_layout()

    if mode == "show":
        plt.show()
    elif mode == "save":
        plt.savefig(save_path + "moving_avg-" + str(n) + ".png", dpi=300)
    plt.close()

def log_rewards_frequency(rewards_all_episodes):
    print("$$$ rewards_all_episodes: ", rewards_all_episodes)
    print("$$$ rewards_all_episodes last 10 rewards = ",
          rewards_all_episodes[-10:])

    unique_elements, counts_elements = np.unique(
                                            rewards_all_episodes,
                                            return_counts=True)
    print("Frequency of unique rewards of rewards_all_episodes:")
    with np.printoptions(suppress=True):
        print(np.asarray((unique_elements, counts_elements)))

def plot_rewards_histogram(rewards_all_episodes, mode="show", save_path="", config_str=""):
    data = rewards_all_episodes
    d = np.diff(np.unique(data)).min()
    left_of_first_bin = data.min() - float(d) / 2
    right_of_last_bin = data.max() + float(d) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d), density=True,
            color='#2ca02c', edgecolor='black', linewidth=0.8, alpha=0.7)

    if right_of_last_bin - left_of_first_bin > 20:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set(xlabel='Rewards', ylabel='Frequency (Number of Episodes)',
           title=f'Histogram of Rewards: {config_str}')
    ax.grid(True, linewidth=0.8, linestyle='--', alpha=0.7)
    fig.tight_layout()

    if mode == "show":
        plt.show()
    elif mode == "save":
        plt.savefig(save_path + "rewards_histogram.png", dpi=300)
    plt.close()

def plot_print_rewards_stats(rewards_all_episodes, show_every, args, mode="show", save_path=""):
    seq = args.seq
    seed = args.seed
    algo = args.algo
    num_episodes = args.num_episodes

    rewards_per_N_episodes = np.split(np.array(rewards_all_episodes), num_episodes/show_every)
    count = show_every
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    print(f"\n*******Stats per {show_every} episodes*******\n")
    for r in rewards_per_N_episodes:
        aggr_ep_rewards['ep'].append(count)
        aggr_ep_rewards['avg'].append(sum(r/show_every))
        aggr_ep_rewards['min'].append(min(r))
        aggr_ep_rewards['max'].append(max(r))
        count += show_every

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="Average Rewards", color=colors[0], marker='o', markersize=5, linewidth=2)
    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Max Rewards", color=colors[1], marker='^', markersize=5, linewidth=2)
    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Min Rewards", color=colors[2], marker='v', markersize=5, linewidth=2)

    ax.set(xlabel='Episode Index', ylabel='Episode Reward')
    ax.legend(loc='lower right', fontsize=12)

    chunks, chunk_size = len(seq), 10
    seq_title_str = ''.join([seq[i:i+chunk_size]+"\n" for i in range(0, chunks, chunk_size)])
    title = f"Protein Folding Rewards - {seq_title_str}Algorithm: {algo}, Episodes: {num_episodes}, Seed: {seed}"
    plt.title(title, fontsize=16)

    plt.grid(True, which="major", linewidth=1.2, linestyle='-', alpha=0.7)
    plt.grid(True, which="minor", linewidth=0.8, linestyle='--', alpha=0.4)
    fig.tight_layout()

    if mode == "show":
        plt.show()
    elif mode == "save":
        plt.savefig(f"{save_path}Seq_{seq}-{algo}-Eps{num_episodes}-Seed{seed}.png", dpi=300)
    plt.close()

def extract_max_per_chunk(rewards_all_episodes, num_episodes, show_every):
    """Extract the max per chunk"""
    rewards_per_N_episodes = np.split(rewards_all_episodes, num_episodes // show_every)
    aggr_max = np.max(rewards_per_N_episodes, axis=1)
    return aggr_max

def avg_std_of_max(seed_42_data, seed_1984_data, seed_1991_data, seed_2021_data, num_episodes, N_chunks, show_every):
    max_data = np.stack([
        extract_max_per_chunk(seed_42_data, num_episodes, show_every),
        extract_max_per_chunk(seed_1984_data, num_episodes, show_every),
        extract_max_per_chunk(seed_1991_data, num_episodes, show_every),
        extract_max_per_chunk(seed_2021_data, num_episodes, show_every)
    ])

    print("max_data shape =", max_data.shape)
    avg = np.mean(max_data, axis=0)
    std = np.std(max_data, axis=0)

    return avg, std
