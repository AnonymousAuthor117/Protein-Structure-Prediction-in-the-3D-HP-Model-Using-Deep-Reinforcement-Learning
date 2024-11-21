import numpy as np
import math
import matplotlib.pyplot as plt

def ExponentialDecay(episode, num_episodes, min_exploration_rate, max_exploration_rate, exploration_decay_rate = 5, start_decay = 0):
    decay_duration = num_episodes - start_decay
    exploration_rate = max_exploration_rate
    if episode > start_decay:
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * (episode - start_decay)/decay_duration)
    return exploration_rate

def Plot_Anneal_Schedule(num_episodes, eta_min = 0.01, eta_max = 1.0, mode="save", save_path="", warmRestart=True, decay_mode="cosine", num_restarts=10, exploration_decay_rate=5, start_decay=0):
    print("warmRestart = ", warmRestart)
    max_exploration_rate = eta_max
    min_exploration_rate = eta_min

    y = np.zeros((num_episodes, ))

    for n_episode in range(num_episodes):
        y[n_episode] = ExponentialDecay(
            n_episode,
            num_episodes,
            min_exploration_rate,
            max_exploration_rate,
            exploration_decay_rate=exploration_decay_rate,
            start_decay=start_decay,
        )

    fig, ax = plt.subplots()
    plt.yticks(np.arange(-.1, 1.1, 0.1))

    ax.plot(np.arange(num_episodes), y)
    ax.set(xlabel='Episode', ylabel='epsilon',
        title=f'Epsilon Decay with {decay_mode}Decay')
    ax.grid()

    if mode == "show":
        plt.show()
    elif mode == "save":
        plt.savefig(save_path + "Anneal_Schedule_" + str(num_episodes) + "_episodes.png")
    plt.close()
