import argparse
import os
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

from features.generate_features import build_dataset
from board_constants import RESOURCES
from config import RIDGE_ALPHA, RANDOM_SEED, TEST_SPLIT, PROJECT_ROOT


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


def evaluate(model, X, y, game_ids):
    """
    Returns the following:
    - MSE, RMSE, MAE (overall prediction quality)
    - Spearman rank correlation per board:
      For each game, rank the 8 placements by predicted VP and by actual VP.
      Compute Spearman correlation. Report mean across games.
      This measures how well the model ranks placements within a board,
      which is what matters for the actual use case.
    - Precision@1: fraction of games where the model's top-ranked
      placement is also the actual highest-VP placement.
    """
    y_hat = model.predict(X)

    mse = mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_hat)

    spearman_scores = []
    precision_at_1_hits = 0
    unique_games = np.unique(game_ids)

    for gid in unique_games:
        mask = game_ids == gid
        y_game = y[mask]
        y_hat_game = y_hat[mask]

        if len(y_game) < 2:
            continue

        corr, _ = spearmanr(y_game, y_hat_game)
        if not np.isnan(corr):
            spearman_scores.append(corr)

        if np.argmax(y_hat_game) == np.argmax(y_game):
            precision_at_1_hits += 1

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Spearman (mean)": np.mean(spearman_scores) if spearman_scores else 0.0,
        "Precision@1": precision_at_1_hits / len(unique_games),
    }


DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ridge_model.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / evaluate linear model with ridge regression"
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
    X, y, game_ids = build_dataset()
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

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_train_scaled, y_train)

        joblib.dump({"model": model, "scaler": scaler}, args.output)
        print("\nTraining completed")
        print(f"Model saved to {args.output}")

    train_metrics = evaluate(model, X_train_scaled, y_train, gid_train)
    test_metrics = evaluate(model, X_test_scaled, y_test, gid_test)

    print("\nTrain:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nTest:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    coefs = model.coef_
    top_idx = np.argsort(np.abs(coefs))[::-1][:10]
    print("\nTop 10 features by |coefficient|")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[i]:30s}  {coefs[i]:+.4f}")


if __name__ == "__main__":
    main()
