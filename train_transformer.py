import os
import argparse

# fallback because the tensor mask operation didn't work through MPS on a mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

from transformer import CatanTransformer

from features.generate_features_transformer import build_transformer_dataset, build_transformer_winner_dataset
from config import RANDOM_SEED, TEST_SPLIT, PROJECT_ROOT

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


class CatanDataset(Dataset):
    def __init__(self, data, y):
        self.data = data
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "tile_resource": torch.LongTensor(self.data["tile_resource"][idx]),
            "tile_dicenum": torch.LongTensor(self.data["tile_dicenum"][idx]),
            "tile_pos": torch.LongTensor(self.data["tile_pos"][idx]),
            "port_resource": torch.LongTensor(self.data["port_resource"][idx]),
            "port_pos": torch.LongTensor(self.data["port_pos"][idx]),
            "struct_owner": torch.LongTensor(self.data["struct_owner"][idx]),
            "struct_type": torch.LongTensor(self.data["struct_type"][idx]),
            "struct_pos": torch.LongTensor(self.data["struct_pos"][idx]),
            "road_owner": torch.LongTensor(self.data["road_owner"][idx]),
            "road_a": torch.LongTensor(self.data["road_a"][idx]),
            "road_b": torch.LongTensor(self.data["road_b"][idx]),
            "hand_features": torch.FloatTensor(self.data["hand_features"][idx]),
            "struct_mask": torch.BoolTensor(self.data["struct_mask"][idx]),
            "road_mask": torch.BoolTensor(self.data["road_mask"][idx]),
            "y": torch.FloatTensor([self.y[idx]]),
        }


def split_by_game(data, y, game_ids, test_frac=TEST_SPLIT, seed=RANDOM_SEED):
    """
    Split data dict, y, and game_ids by game. All samples from the same
    game stay together.
    """
    unique_games = np.unique(game_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_games)

    n_test = int(len(unique_games) * test_frac)
    test_games = set(unique_games[:n_test])

    test_mask = np.array([gid in test_games for gid in game_ids])
    train_mask = ~test_mask

    train_data = {key: val[train_mask] for key, val in data.items()}
    test_data = {key: val[test_mask] for key, val in data.items()}

    return (
        train_data,
        test_data,
        y[train_mask],
        y[test_mask],
        game_ids[train_mask],
        game_ids[test_mask],
    )


def predict_all(model, loader):
    """Run model on all batches and return (predictions, labels)."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            pred = model(
                batch["tile_resource"],
                batch["tile_dicenum"],
                batch["tile_pos"],
                batch["port_resource"],
                batch["port_pos"],
                batch["struct_owner"],
                batch["struct_type"],
                batch["struct_pos"],
                batch["road_owner"],
                batch["road_a"],
                batch["road_b"],
                batch["hand_features"],
                batch["struct_mask"],
                batch["road_mask"],
            ).squeeze()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch["y"].squeeze().cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate(y, y_hat, game_ids):
    """
    Returns:
    - MSE, RMSE, MAE (overall prediction quality)
    - Spearman rank correlation per board (mean across games)
    - Precision@1: fraction of games where the model's top-ranked
      placement is also the actual highest-VP placement.
    """
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


def evaluate_winner_prediction(model, split_game_ids):
    """
    Predict which player wins each game based on initial placements.

    Uses round-1 placement with all 8 placements visible. For each game,
    check if the player with the highest predicted VP is the actual winner.

    Returns accuracy as a float.
    """
    w_data, w_y, w_gids = build_transformer_winner_dataset()

    split_set = set(split_game_ids)
    mask = np.array([g in split_set for g in w_gids])
    w_data = {key: val[mask] for key, val in w_data.items()}
    w_y = w_y[mask]
    w_gids = w_gids[mask]

    w_dataset = CatanDataset(w_data, w_y)
    w_loader = DataLoader(w_dataset, batch_size=256, shuffle=False)

    w_preds, _ = predict_all(model, w_loader)

    correct = 0
    unique_games = np.unique(w_gids)
    for gid in unique_games:
        gmask = w_gids == gid
        actual = w_y[gmask]
        predicted = w_preds[gmask]
        actual_winners = set(np.where(actual == actual.max())[0])
        if np.argmax(predicted) in actual_winners:
            correct += 1

    return correct / len(unique_games) if len(unique_games) > 0 else 0.0


DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "transformer_model.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / evaluate Catan transformer model"
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
    print(f"Using device: {DEVICE}")

    print("Loading dataset...")
    data, y, game_ids = build_transformer_dataset()
    print(f"Loaded {len(np.unique(game_ids))} games, {len(y)} samples")

    train_data, test_data, y_train, y_test, gid_train, gid_test = split_by_game(
        data,
        y,
        game_ids,
    )
    print(f"Train: {len(y_train)}  Test: {len(y_test)}")

    train_dataset = CatanDataset(train_data, y_train)
    test_dataset = CatanDataset(test_data, y_test)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    n_hand_feats = data["hand_features"].shape[1]
    model = CatanTransformer(n_hand_feats).to(DEVICE)

    if args.load:
        print(f"\nLoading model from {args.load}")
        checkpoint = torch.load(args.load, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        print("\nTraining...")
        for epoch in range(20):
            model.train()
            total_loss = 0
            n_samples = 0
            for batch in train_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                pred = model(
                    batch["tile_resource"],
                    batch["tile_dicenum"],
                    batch["tile_pos"],
                    batch["port_resource"],
                    batch["port_pos"],
                    batch["struct_owner"],
                    batch["struct_type"],
                    batch["struct_pos"],
                    batch["road_owner"],
                    batch["road_a"],
                    batch["road_b"],
                    batch["hand_features"],
                    batch["struct_mask"],
                    batch["road_mask"],
                ).squeeze()

                loss = loss_fn(pred, batch["y"].squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch["y"])
                n_samples += len(batch["y"])

            avg_train_loss = total_loss / n_samples

            y_hat_test, _ = predict_all(model, test_loader)
            test_mse = mean_squared_error(y_test, y_hat_test)
            print(
                f"Epoch {epoch+1:2d}: Train MSE = {avg_train_loss:.4f}  Test MSE = {test_mse:.4f}"
            )

        torch.save({
            "model_state_dict": model.state_dict(),
            "n_hand_feats": n_hand_feats,
        }, args.output)
        print(f"\nModel saved to {args.output}")

    print("\n--- Final Evaluation ---")
    y_hat_train, _ = predict_all(model, train_eval_loader)
    y_hat_test, _ = predict_all(model, test_loader)

    train_metrics = evaluate(y_train, y_hat_train, gid_train)
    test_metrics = evaluate(y_test, y_hat_test, gid_test)

    print("\nTrain:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nTest:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nWinner prediction (full board context):")
    train_winner_acc = evaluate_winner_prediction(model, gid_train)
    test_winner_acc = evaluate_winner_prediction(model, gid_test)
    print(f"  Train: {train_winner_acc:.4f}")
    print(f"  Test:  {test_winner_acc:.4f}")


if __name__ == "__main__":
    main()
