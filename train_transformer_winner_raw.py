import os
import argparse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer_winner_raw import CatanWinnerTransformerRaw
from features.generate_features import build_joint_winner_dataset
from config import RANDOM_SEED, TEST_SPLIT, PROJECT_ROOT

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


class WinnerDatasetRaw(Dataset):
    """Same data as WinnerDataset but drops hand_features from the batch."""

    def __init__(self, data, winners):
        self.data = data
        self.winners = winners

    def __len__(self):
        return len(self.winners)

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
            "struct_mask": torch.BoolTensor(self.data["struct_mask"][idx]),
            "road_mask": torch.BoolTensor(self.data["road_mask"][idx]),
            "winner": torch.LongTensor([self.winners[idx]]).squeeze(),
        }


def split_by_game(data, winners, game_ids, test_frac=TEST_SPLIT, seed=RANDOM_SEED):
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
        winners[train_mask],
        winners[test_mask],
        game_ids[train_mask],
        game_ids[test_mask],
    )


def _forward(model, batch):
    return model(
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
        batch["struct_mask"],
        batch["road_mask"],
    )


def predict_all(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = _forward(model, batch)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["winner"].cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def evaluate(logits, winners):
    preds = np.argmax(logits, axis=1)
    return (preds == winners).mean()


DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "winner_transformer_raw_model.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / evaluate Catan winner transformer (raw tokens only, no hand features)"
    )
    parser.add_argument(
        "--load", type=str, default=None, metavar="PATH",
        help="load a saved model instead of training",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=DEFAULT_MODEL_PATH, metavar="PATH",
        help=f"trained model save path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Using device: {DEVICE}")

    print("Loading dataset...")
    data, winners, game_ids = build_joint_winner_dataset()
    print(f"Loaded {len(winners)} games")

    random_acc = 0.25
    print(f"Random baseline: {random_acc:.4f}")

    train_data, test_data, w_train, w_test, gid_train, gid_test = split_by_game(
        data, winners, game_ids,
    )
    print(f"Train: {len(w_train)}  Test: {len(w_test)}")

    train_dataset = WinnerDatasetRaw(train_data, w_train)
    test_dataset = WinnerDatasetRaw(test_data, w_test)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = CatanWinnerTransformerRaw(dropout=args.dropout).to(DEVICE)

    if args.load:
        print(f"\nLoading model from {args.load}")
        checkpoint = torch.load(args.load, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        print("\nTraining...")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            n_samples = 0
            for batch in train_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = _forward(model, batch)
                loss = loss_fn(logits, batch["winner"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch["winner"])
                n_samples += len(batch["winner"])

            avg_train_loss = total_loss / n_samples

            test_logits, test_labels = predict_all(model, test_loader)
            test_acc = evaluate(test_logits, test_labels)
            print(
                f"Epoch {epoch+1:2d}: Train Loss = {avg_train_loss:.4f}  Test Acc = {test_acc:.4f}"
            )

        torch.save({
            "model_state_dict": model.state_dict(),
        }, args.output)
        print(f"\nModel saved to {args.output}")

    print("\n--- Final Evaluation ---")
    train_logits, train_labels = predict_all(model, train_eval_loader)
    test_logits, test_labels = predict_all(model, test_loader)

    train_acc = evaluate(train_logits, train_labels)
    test_acc = evaluate(test_logits, test_labels)

    print(f"  Random baseline: {random_acc:.4f}")
    print(f"  Train accuracy:  {train_acc:.4f}")
    print(f"  Test accuracy:   {test_acc:.4f}")


if __name__ == "__main__":
    main()
