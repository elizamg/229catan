# reframe problem as winner likelihood prediction
import argparse
import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from scipy.stats import spearmanr

from features.generate_features import build_winner_classification_dataset_feature_testing as build_winner_classification_dataset
from board_constants import RESOURCES
from config import RANDOM_SEED, TEST_SPLIT, PROJECT_ROOT

# define hyperparameters for testing the MLP model

MLP_HIDDEN_LAYERS = (512, 256, 128)
MLP_MAX_ITER = 3000
MLP_ALPHA = 1e-2

print(f"MLP Hyperparameters:")
print(f"  Hidden Layers: {MLP_HIDDEN_LAYERS}")
print(f"  Max Iterations: {MLP_MAX_ITER}")
print(f"  Alpha: {MLP_ALPHA}")

def split_by_game(X, y, game_ids, test_frac=TEST_SPLIT, seed=RANDOM_SEED):
    """
    Note that all samples from the same game stay together.

    Returns:
        X_train, X_test, y_train, y_test, game_ids_train, game_ids_test
    """
    unique_games = np.unique(game_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_games)

    n_test = int(len(unique_games) * test_frac)
    test_games = set(unique_games[:n_test])

    test_mask = np.array([gid in test_games for gid in game_ids])
    train_mask = ~test_mask

    return (
        X[train_mask],
        X[test_mask],
        y[train_mask],
        y[test_mask],
        game_ids[train_mask],
        game_ids[test_mask],
    )

# evaluate for classification task

def evaluate_winner_accuracy_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    game_ids: np.ndarray,
) -> float:
    """Winner accuracy when each game has multiple rows (one per player).

    Despite the name, `y_prob` can be any per-player score (probability, logit, heuristic score).
    Picks argmax score per game and checks if that player is among the true winners.
    Tie-aware on ground truth: if multiple winners exist (same max VP), any is correct.
    """
    unique_games = np.unique(game_ids)
    correct = 0
    for gid in unique_games:
        mask = game_ids == gid
        y_g = y_true[mask]
        p_g = y_prob[mask]
        if y_g.size == 0:
            continue
        actual_winners = set(np.where(y_g == y_g.max())[0])
        if int(np.argmax(p_g)) in actual_winners:
            correct += 1
    return correct / len(unique_games) if len(unique_games) > 0 else 0.0


def sampled_random_winner_accuracy(y_true: np.ndarray, game_ids: np.ndarray, seed: int) -> float:
    """Tie-aware random-guess accuracy by sampling one random player per game."""
    unique_games = np.unique(game_ids)
    if len(unique_games) == 0:
        return 0.0
    rng = np.random.RandomState(seed)
    correct = 0
    for gid in unique_games:
        mask = game_ids == gid
        y_g = y_true[mask]
        if y_g.size == 0:
            continue
        pick = int(rng.randint(0, y_g.size))
        actual_winners = set(np.where(y_g == y_g.max())[0])
        if pick in actual_winners:
            correct += 1
    return correct / float(len(unique_games))

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "mlp_model.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / evaluate MLP model"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        metavar="PATH",
        help="load a saved model instead of training",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"trained model save path (default: {DEFAULT_MODEL_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    res_lower = [r.lower() for r in RESOURCES]
    feature_names = (
        # Placement features (21)
        [f"{r}_count" for r in res_lower]
        + ["total_prod"]
        + [f"{r}_prod" for r in res_lower]
        + ["resource_diversity", "number_diversity", "has_port"]
        + [f"port_{r}" for r in res_lower]
        + ["port_any"]
        + ["expansion"]
        # Opponent features (7)
        + ["shared_tile_opps"]
        + [f"comp_{r}" for r in res_lower]
        + ["blocked_neighbors"]
        # Board aggregate features (11)
        + [f"board_{r}_prod" for r in res_lower]
        + [f"board_{r}_high" for r in res_lower]
        + ["desert_ring"]
        # Raw board features (19 tiles x 8)
        + [
            f"tile{t}_{attr}"
            for t in range(19)
            for attr in [*res_lower, "desert", "prob", "number"]
        ]
        # Turn order features (6)
        + [f"player_pos_{i}" for i in range(4)]
        + ["placement_round", "placement_order"]
    )

    print("Loading dataset...")
    X, y, game_ids = build_winner_classification_dataset()
    # 
    print(f"Loaded {len(np.unique(game_ids))} games")

    X_train, X_test, y_train, y_test, gid_train, gid_test = split_by_game(
        X,
        y,
        game_ids,
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    if args.load:
        print(f"\nLoading model from {args.load}")
        saved = joblib.load(args.load)
        model = saved["model"]
        scaler = saved["scaler"]
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            max_iter=MLP_MAX_ITER,
            random_state=RANDOM_SEED,
            early_stopping=True,
            alpha=MLP_ALPHA,
        )
        
        model.fit(X_train_scaled, y_train)

        joblib.dump({"model": model, "scaler": scaler}, args.output)
        print("\nTraining completed")
        print(f"Model saved to {args.output}")

    # Per-game winner accuracy from predicted win probabilities.
    if hasattr(model, "predict_proba"):
        p_train = model.predict_proba(X_train_scaled)[:, 1]
        p_test = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # Fallback: treat raw predictions as scores
        p_train = model.predict(X_train_scaled)
        p_test = model.predict(X_test_scaled)


    print("\nWinner prediction (full board context):")
    train_winner_acc = evaluate_winner_accuracy_from_probs(y_train, p_train, gid_train)
    test_winner_acc = evaluate_winner_accuracy_from_probs(y_test, p_test, gid_test)
    print("\nSummary:")
    print(
        f"WinnerAcc:        {test_winner_acc:.4f} "
        #f"(model={args.model}, n_test_games={len(np.unique(gid_test))})"
    )
    #print(f"RandomPickAcc:    {rand_sampled:.4f} (uniform random player per game)")

    # feature importance - based on the first layer weights
    """
    coefs = model.coefs_[0]
    feature_importance = np.abs(coefs).sum(axis=1)
    top_idx = np.argsort(feature_importance)[::-1][:10]
    print("\nTop 10 features by feature importance")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[i]:30s}  {feature_importance[i]:+.4f}")"""


if __name__ == "__main__":
    main()
