import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def label_conf_seq(conf_seq, conf_title):

    str_seq = conf_title[:-5]
    assert len(str_seq) == len(conf_seq)
    labelled_conf = []
    for index, letter in enumerate(str_seq):
        labelled_conf.append((conf_seq[index], letter))

    return labelled_conf

#action map
LFR_map = {
    "left": 0,
    "forward": 1,
    "right": 2,
    "up": 3,
    "down": 4,
    "error": -1,
    "0": "left",
    "1": "forward",
    "2": "right",
    "3": "up",
    "4": "down",
    "-1": "error",
}

def move_LFR_direction(p1, p2, move_direction):
    x1, y1, z1 = p1
    x2, y2, z2 = p2


    p3_candidates = [
        (x2 - 1, y2, z2),  # left
        (x2, y2 + 1, z2),  # forward
        (x2 + 1, y2, z2),  # right
        (x2, y2, z2 + 1),  # up
        (x2, y2, z2 - 1),  # down
    ]

    p3_candidates = [p for p in p3_candidates if p != p1]

    if move_direction == 0:  # left
        p3 = next((p for p in p3_candidates if p[0] < p2[0]), p2)
    elif move_direction == 1:  # forward
        p3 = next((p for p in p3_candidates if p[1] > p2[1]), p2)
    elif move_direction == 2:  # right
        p3 = next((p for p in p3_candidates if p[0] > p2[0]), p2)
    elif move_direction == 3:  # up
        p3 = next((p for p in p3_candidates if p[2] > p2[2]), p2)
    elif move_direction == 4:  # down
        p3 = next((p for p in p3_candidates if p[2] < p2[2]), p2)
    else:
        raise ValueError(f"Invalid move_direction: {move_direction}")

    return p3

def derive_LFR_direction(p1, p2, p3):
    if p3 == p1:
        print("ILLEGAL: folded back to p1")
        return -1

    if p3 == p2:
        print("ILLEGAL: cannot stay put at p2")
        return -1

    if p3[2] > p2[2]:
        print("Moving up")
        return LFR_map["up"]
    elif p3[2] < p2[2]:
        print("Moving down")
        return LFR_map["down"]

    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    print("v1: ", v1)
    print("v2: ", v2)

    cross_product = np.cross(v1, v2)
    print("cross v1xv2: ", cross_product)

    if np.array_equal(cross_product, [0, 0, 1]) or np.array_equal(cross_product, [0, -1, 0]):
        print("Left")
        return LFR_map["left"]
    elif np.array_equal(cross_product, [0, 0, -1]) or np.array_equal(cross_product, [0, 1, 0]):
        print("Right")
        return LFR_map["right"]
    elif np.array_equal(cross_product, [0, 0, 0]) or np.array_equal(cross_product, [1, 0, 0]) or np.array_equal(cross_product, [-1, 0, 0]):
        print("Forward")
        return LFR_map["forward"]
    else:
        print("Out of range moves or unrecognized direction")
        return LFR_map["error"]

def plot_HPSandbox_conf(labelled_conf, mode="draw", pause_t=0.5, save_fig=False, save_path="", score=2022, optima_idx=0, info={}):
    sns.set(style="white", font_scale=1.5)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = [t[0][0] for t in labelled_conf]
    y = [t[0][1] for t in labelled_conf]
    z = [t[0][2] for t in labelled_conf]

    str_seq = ''.join([t[1] for t in labelled_conf])
    assert len(str_seq) == info["chain_length"]

    # Plot backbone
    ax.plot(x, y, z, color='#1f77b4', linewidth=1, alpha=0.7, label="Backbone")

    # Plot amino acids
    H_seq = [t[0] for t in labelled_conf if t[1] == 'H']
    P_seq = [t[0] for t in labelled_conf if t[1] == 'P']
    ax.scatter([h[0] for h in H_seq], [h[1] for h in H_seq], [h[2] for h in H_seq], color='#ff7f0e', s=100, label="H", edgecolors='black', linewidths=0.5)
    ax.scatter([p[0] for p in P_seq], [p[1] for p in P_seq], [p[2] for p in P_seq], color='#2ca02c', s=100, label="P", edgecolors='black', linewidths=0.5)

    # Plot H-H bonds (only adjacent H-H bonds not connected by the backbone)
    plotted_hh_bond = False
    for i, (coord_i, ti) in enumerate(labelled_conf):
        if ti == 'H':
            for j, (coord_j, tj) in enumerate(labelled_conf):
                if tj == 'H' and i != j and abs(i - j) > 1:
                    dist = ((coord_i[0] - coord_j[0])**2 + (coord_i[1] - coord_j[1])**2 + (coord_i[2] - coord_j[2])**2)**0.5
                    if dist == 1:
                        if not plotted_hh_bond:
                            ax.plot([coord_i[0], coord_j[0]], [coord_i[1], coord_j[1]], [coord_i[2], coord_j[2]], color='red', linestyle='--', linewidth=1, label="H-H Bond")
                            plotted_hh_bond = True
                        else:
                            ax.plot([coord_i[0], coord_j[0]], [coord_i[1], coord_j[1]], [coord_i[2], coord_j[2]], color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('X', fontsize=18, labelpad=15)
    ax.set_ylabel('Y', fontsize=18, labelpad=15)
    ax.set_zlabel('Z', fontsize=18, labelpad=15)

    # Remove tick labels and tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0)

    pad_len = info['seq_length'] - info['chain_length']
    str_seq = str_seq + '_' * pad_len
    action_str = ''.join(info["actions"])
    padded_action_str = action_str + '_' * pad_len

    title = f"Sequence: {str_seq}\nActions: {action_str}"
    ax.set_title(title, fontsize=20, pad=20)
    ax.legend(loc='upper right', fontsize=16, frameon=False, bbox_to_anchor=(1.1, 1.1))

    # Set aspect ratio and adjust subplot parameters
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    # Adjust camera position and scaling for better visualization
    ax.view_init(elev=20, azim=45)
    ax.dist = 10

    if save_fig:
        plt.savefig(f"{save_path}/{str_seq[:6]}_{action_str[:6]}_{info['seq_length']}mer_E{int(score)}_ID{optima_idx}.png",
                    bbox_inches='tight', dpi=300)

    if mode == "show":
        plt.show()

    plt.close()
