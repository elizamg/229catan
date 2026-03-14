import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformer_winner_crossattn_concat import CatanWinnerCrossAttnConcat
from transformer_winner_raw_crossattn import CatanWinnerRawCrossAttn
from features.generate_features import build_joint_winner_dataset, HAND_FEATS_PER_ROUND
from config import RANDOM_SEED, TEST_SPLIT, PROJECT_ROOT

RESOURCE_NAMES = ["BRICK", "WOOD", "SHEEP", "WHEAT", "ORE", "DESERT"]
PORT_RESOURCE_NAMES = ["BRICK", "WOOD", "SHEEP", "WHEAT", "ORE", "ANY"]
PLAYER_NAMES = ["P1", "P2", "P3", "P4"]
STRUCT_TYPES = ["settle", "city"]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize cross-attention weights")
    parser.add_argument("model", help="path to model checkpoint")
    parser.add_argument(
        "--game-idx", type=int, default=0, help="index into test set (default: 0)"
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def build_token_labels(sample):
    labels = []
    token_types = []

    # 19 tile tokens
    for i in range(19):
        res = RESOURCE_NAMES[sample["tile_resource"][i]]
        dice = sample["tile_dicenum"][i]
        if dice == 0:
            labels.append(f"{res}")
        else:
            labels.append(f"{res}-{dice + 1}")
        token_types.append("tile")

    # 9 port tokens
    for i in range(9):
        res = PORT_RESOURCE_NAMES[sample["port_resource"][i]]
        labels.append(f"Port:{res}")
        token_types.append("port")

    # structure tokens 
    n_structs = (~sample["struct_mask"]).sum().item()
    for i in range(n_structs):
        owner = PLAYER_NAMES[sample["struct_owner"][i]]
        stype = STRUCT_TYPES[sample["struct_type"][i]]
        pos = sample["struct_pos"][i].item()
        labels.append(f"{owner} {stype}@{pos}")
        token_types.append("struct")

    # road tokens
    n_roads = (~sample["road_mask"]).sum().item()
    for i in range(n_roads):
        owner = PLAYER_NAMES[sample["road_owner"][i]]
        a = sample["road_a"][i].item()
        b = sample["road_b"][i].item()
        labels.append(f"{owner} road {a}-{b}")
        token_types.append("road")

    return labels, token_types


def get_test_sample(data, winners, game_ids, idx):
    unique_games = np.unique(game_ids)
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_games)

    n_test = int(len(unique_games) * TEST_SPLIT)
    test_games = set(unique_games[:n_test])
    test_mask = np.array([gid in test_games for gid in game_ids])
    test_indices = np.where(test_mask)[0]

    real_idx = test_indices[idx]
    mask_keys = {"struct_mask", "road_mask"}
    sample = {}
    for key, val in data.items():
        if key == "hand_features":
            continue
        elif key in mask_keys:
            sample[key] = torch.BoolTensor(val[real_idx])
        else:
            sample[key] = torch.LongTensor(val[real_idx])
    sample["hand_features"] = torch.FloatTensor(data["hand_features"][real_idx])
    sample["winner"] = int(winners[real_idx])
    return sample


def main():
    args = parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    d_embed = checkpoint.get("d_embed", 64)
    stem = os.path.splitext(os.path.basename(args.model))[0]
    is_raw = "raw" in stem
    if is_raw:
        n_heads = checkpoint.get("n_heads", 2)
        n_layers = checkpoint.get("n_layers", 1)
        model = CatanWinnerRawCrossAttn(
            d_embed=d_embed, n_heads=n_heads, n_layers=n_layers
        )
    else:
        n_hand = checkpoint["n_hand_feats_per_player"]
        model = CatanWinnerCrossAttnConcat(n_hand, d_embed=d_embed)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data, winners, game_ids = build_joint_winner_dataset()
    sample = get_test_sample(data, winners, game_ids, args.game_idx)

    labels, token_types = build_token_labels(sample)
    n_active = len(labels)

    fwd_args = [
        sample["tile_resource"].unsqueeze(0),
        sample["tile_dicenum"].unsqueeze(0),
        sample["tile_pos"].unsqueeze(0),
        sample["port_resource"].unsqueeze(0),
        sample["port_pos"].unsqueeze(0),
        sample["struct_owner"].unsqueeze(0),
        sample["struct_type"].unsqueeze(0),
        sample["struct_pos"].unsqueeze(0),
        sample["road_owner"].unsqueeze(0),
        sample["road_a"].unsqueeze(0),
        sample["road_b"].unsqueeze(0),
    ]
    if not is_raw:
        fwd_args.append(sample["hand_features"].unsqueeze(0))
    fwd_args.extend(
        [
            sample["struct_mask"].unsqueeze(0),
            sample["road_mask"].unsqueeze(0),
        ]
    )
    with torch.no_grad():
        logits, attn_weights = model(*fwd_args, return_attn_weights=True)

    attn = attn_weights[0].numpy()
    probs = torch.softmax(logits[0], dim=0).numpy()
    pred = np.argmax(probs)
    winner = sample["winner"]

    attn = attn[:, :n_active]

    fig, ax = plt.subplots(figsize=(max(14, n_active * 0.4), 5))

    im = ax.imshow(attn, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="Attention weight", shrink=0.8)

    ax.set_yticks(range(4))
    ax.set_yticklabels(
        [f"{PLAYER_NAMES[i]} ({probs[i]:.1%})" for i in range(4)],
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xticks(range(n_active))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)

    type_colors = {
        "tile": "#2196F3",
        "port": "#4CAF50",
        "struct": "#FF9800",
        "road": "#9C27B0",
    }
    for i, tt in enumerate(token_types):
        ax.get_xticklabels()[i].set_color(type_colors[tt])

    # highlight winner row
    winner_rect = mpatches.FancyBboxPatch(
        (-0.5, winner - 0.5),
        n_active,
        1,
        boxstyle="round,pad=0",
        linewidth=2,
        edgecolor="green",
        facecolor="none",
    )
    ax.add_patch(winner_rect)

    sep_positions = [19, 28]  # after tiles, after ports
    n_structs = (~sample["struct_mask"]).sum().item()
    if n_structs > 0:
        sep_positions.append(28 + n_structs)
    for sep in sep_positions:
        if sep < n_active:
            ax.axvline(sep - 0.5, color="white", linewidth=2)

    legend_patches = [
        mpatches.Patch(color=c, label=t.capitalize()) for t, c in type_colors.items()
    ]
    legend_patches.append(
        mpatches.Patch(edgecolor="green", facecolor="none", linewidth=2, label="Winner")
    )
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    ax.set_title(
        f"Cross-Attention Weights — Game {args.game_idx} "
        f"(pred: {PLAYER_NAMES[pred]}, actual: {PLAYER_NAMES[winner]})",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    model_tag = stem.replace("winner_", "")
    out_path = os.path.join(
        PROJECT_ROOT, f"attention_{model_tag}_game{args.game_idx}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
