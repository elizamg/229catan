import argparse
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from features.generate_features import build_winner_classification_dataset_feature_testing as build_winner_classification_dataset
from board_constants import RESOURCES
from config import RANDOM_SEED, TEST_SPLIT, PROJECT_ROOT


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


DEFAULT_WINNER_LOGREG_PATH = os.path.join(PROJECT_ROOT, "winner_logreg.pkl")
DEFAULT_WINNER_XGB_PATH = os.path.join(PROJECT_ROOT, "winner_xgb.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / evaluate models for Catan opening strength"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="winner",
        choices=["winner"],
        help="task to optimize (winner classification; optimized for WinnerAcc)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        metavar="PATH",
        help="load a saved model instead of training",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["ridge", "xgb"],
        help="which model to train/evaluate (default: ridge)",
    )
    return parser.parse_args()


def _winner_feature_names(res_lower: list[str]) -> list[str]:
    """Feature names for build_winner_classification_dataset(): 237 features."""
    placement = (
        ["total_prod"]
        + [f"{r}_prod" for r in res_lower]
        + ["resource_diversity", "num_6_or_8", "has_port"]
        + [f"port_{r}" for r in res_lower]
        + ["port_any", "port_prod_match", "expansion"]
        + [f"share_{r}" for r in res_lower]
    )
    opponent = ["shared_tile_opps"] + [f"comp_{r}" for r in res_lower] + ["blocked_neighbors"]
    turn = [f"player_pos_{i}" for i in range(4)] + ["placement_round", "placement_order"]

    def pref(prefix: str, names: list[str]) -> list[str]:
        return [f"{prefix}{n}" for n in names]

    board = (
        [f"board_{r}_prod" for r in res_lower]
        + [f"board_{r}_high" for r in res_lower]
        + [f"scarcity_{r}" for r in res_lower]
    )
    raw = [
        f"tile{t}_{attr}"
        for t in range(19)
        for attr in [*res_lower, "desert", "prob", "number"]
    ]

    return (
        pref("r0_", placement + opponent + turn)
        + pref("r1_", placement + opponent + turn)
        + board
        + raw
    )


def main():
    args = parse_args()

    output_path = DEFAULT_WINNER_LOGREG_PATH if args.model == "ridge" else DEFAULT_WINNER_XGB_PATH

    res_lower = [r.lower() for r in RESOURCES]
    feature_names = _winner_feature_names(res_lower)

    print("Loading dataset...")
    X, y, game_ids = build_winner_classification_dataset()
    print(f"Loaded {len(np.unique(game_ids))} games")

    # Single deterministic split.
    seed = RANDOM_SEED
    X_train, X_test, y_train, y_test, gid_train, gid_test = split_by_game(
        X,
        y,
        game_ids,
        seed=seed,
    )

    # Model selection
    if args.load:
        print(f"\nLoading model from {args.load}")
        saved = joblib.load(args.load)
        model = saved["model"]
        scaler = saved.get("scaler")

        if scaler is None:
            from models.xgboost_model import IdentityTransformer

            scaler = IdentityTransformer()

        X_train_used = scaler.transform(X_train)
        X_test_used = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_train_used = scaler.fit_transform(X_train)
        X_test_used = scaler.transform(X_test)

        if args.model == "ridge":
            model = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
            )
            model.fit(X_train_used, y_train)
        else:
            try:
                from xgboost import XGBClassifier
            except Exception as e:
                raise RuntimeError("xgboost is required for --model xgb") from e

            model = XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.0,
                min_child_weight=1.0,
                gamma=0.0,
                objective="binary:logistic",
                random_state=seed,
                n_jobs=-1,
                eval_metric="logloss",
            )
            model.fit(X_train_used, y_train)

    # Per-game winner accuracy from predicted win probabilities.
    if hasattr(model, "predict_proba"):
        p_train = model.predict_proba(X_train_used)[:, 1]
        p_test = model.predict_proba(X_test_used)[:, 1]
    else:
        # Fallback: treat raw predictions as scores
        p_train = model.predict(X_train_used)
        p_test = model.predict(X_test_used)

    train_winner_acc = evaluate_winner_accuracy_from_probs(y_train, p_train, gid_train)
    test_winner_acc = evaluate_winner_accuracy_from_probs(y_test, p_test, gid_test)

    # Random baselines on the same split (tie-aware).
    rand_sampled = sampled_random_winner_accuracy(y_test, gid_test, seed=seed)

    # Save trained model.
    if not args.load:
        model_type = "winner_logreg" if args.model == "ridge" else "winner_xgb_classifier"
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "model_type": model_type,
                "task": "winner",
            },
            output_path,
        )

        print(f"\nModel saved to {output_path}")

    print("\nSummary:")
    print(
        f"WinnerAcc:        {test_winner_acc:.4f} "
        f"(model={args.model}, n_test_games={len(np.unique(gid_test))})"
    )
    print(f"RandomPickAcc:    {rand_sampled:.4f} (uniform random player per game)")

    # Interpretability only on last trained model, and only when the feature-name list matches.
    try:
        if hasattr(model, "coef_"):
            coefs = np.asarray(model.coef_).ravel()
            top_idx = np.argsort(np.abs(coefs))[::-1][:20]
            print("\nTop 20 features by |coefficient|")
            for rank, i in enumerate(top_idx, 1):
                if i < len(feature_names):
                    print(f"  {rank:2d}. {feature_names[i]:30s}  {coefs[i]:+.4f}")
        else:
            from models.xgboost_model import top_feature_importances

            top_idx = top_feature_importances(model, k=20)
            importances = getattr(model, "feature_importances_", None)
            if importances is not None and len(top_idx) > 0:
                print("\nTop 20 features by importance")
                for rank, i in enumerate(top_idx, 1):
                    if i < len(feature_names):
                        print(f"  {rank:2d}. {feature_names[i]:30s}  {importances[i]:.6f}")
    except Exception:
        # If xgboost isn't installed or importances unavailable, skip.
        pass


if __name__ == "__main__":
    main()