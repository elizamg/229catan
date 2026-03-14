import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from transformer_winner_global import CatanWinnerTransformerGlobal
from transformer_winner_crossattn import CatanWinnerTransformerCrossAttn
from transformer_winner_crossattn_concat import CatanWinnerCrossAttnConcat
from transformer_winner_raw_crossattn import CatanWinnerRawCrossAttn
from config import PROJECT_ROOT

RESOURCE_NAMES = ["BRICK", "WOOD", "SHEEP", "WHEAT", "ORE", "DESERT"]
DICE_LABELS = ["NONE"] + [str(v) for v in range(2, 13)]
DICE_PROBS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize learned embeddings")
    parser.add_argument(
        "model",
        nargs="?",
        default=os.path.join(PROJECT_ROOT, "winner_transformer_global_model.pt"),
        help="path to model checkpoint",
    )
    return parser.parse_args()


def plot_pca(
    weights, labels, title, out_path, colors=None, cmap=None, colorbar_label=None
):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(weights)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_kwargs = dict(s=120, zorder=5)
    if colors is not None:
        scatter_kwargs.update(c=colors, cmap=cmap, edgecolors="black", linewidths=0.5)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)
    if colors is not None and colorbar_label:
        plt.colorbar(scatter, ax=ax, label=colorbar_label)

    for i, name in enumerate(labels):
        ax.annotate(
            name,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    max_abs = max(np.abs(coords).max() * 1.3, 0.1)
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


def plot_cosine_heatmap(weights, labels, title, out_path):
    sim = cosine_similarity(weights)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{sim[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(sim[i, j]) > 0.5 else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


def main():
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    d_embed = checkpoint.get("d_embed", 64)

    stem = os.path.splitext(os.path.basename(args.model))[0]
    prefix = stem.replace("winner_", "")
    if "raw_crossattn" in stem:
        n_heads = checkpoint.get("n_heads", 2)
        n_layers = checkpoint.get("n_layers", 1)
        model = CatanWinnerRawCrossAttn(
            d_embed=d_embed, n_heads=n_heads, n_layers=n_layers
        )
    elif "crossattn_concat" in stem:
        n_hand = checkpoint["n_hand_feats_per_player"]
        model = CatanWinnerCrossAttnConcat(n_hand, d_embed=d_embed)
    elif "crossattn" in stem:
        n_hand = checkpoint["n_hand_feats_per_player"]
        model = CatanWinnerTransformerCrossAttn(n_hand, d_embed=d_embed)
    else:
        n_hand = checkpoint["n_hand_feats_per_player"]
        model = CatanWinnerTransformerGlobal(n_hand, d_embed=d_embed)
    model.load_state_dict(checkpoint["model_state_dict"])

    resource_weights = model.boardEmbedding.resource_embed.weight.detach().numpy()

    plot_pca(
        resource_weights,
        RESOURCE_NAMES,
        f"Resource Embeddings — {prefix} (PCA projection)",
        os.path.join(PROJECT_ROOT, f"resource_embeddings_{prefix}_pca.png"),
    )
    plot_cosine_heatmap(
        resource_weights,
        RESOURCE_NAMES,
        f"Resource Embedding Cosine Similarity — {prefix}",
        os.path.join(PROJECT_ROOT, f"resource_embeddings_{prefix}_cosine.png"),
    )

    dice_weights = model.boardEmbedding.dicenum_embed.weight.detach().numpy()
    dice_probs = [0] + [DICE_PROBS[v] / 36 for v in range(2, 13)]

    plot_pca(
        dice_weights,
        DICE_LABELS,
        f"Dice Number Embeddings — {prefix} (PCA projection)",
        os.path.join(PROJECT_ROOT, f"dicenum_embeddings_{prefix}_pca.png"),
        colors=dice_probs,
        cmap="YlOrRd",
        colorbar_label="Roll probability",
    )
    plot_cosine_heatmap(
        dice_weights,
        DICE_LABELS,
        f"Dice Number Embedding Cosine Similarity — {prefix}",
        os.path.join(PROJECT_ROOT, f"dicenum_embeddings_{prefix}_cosine.png"),
    )


if __name__ == "__main__":
    main()
