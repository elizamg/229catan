
# reframe problem as winner likelihood prediction
import argparse
import os
import joblib
import numpy as np

from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from features.generate_features_deep import build_pytorch_dataset
from board_constants import RESOURCES
from config import RANDOM_SEED, TEST_SPLIT, PROJECT_ROOT

# define hyperparameters for testing the MLP model

HIDDEN_DIMS = [64, 128]
OUTPUT_DIM = [32, 64]
NUM_EPOCHS = 20
BATCH_SIZE = [256, 1024]
LEARNING_RATE = [1e-2, 1e-3, 1e-4]

print(f"Pytorch Hyperparameters:")
print(f"  Hidden Dims: {HIDDEN_DIMS}")
print(f"  Output Dim: {OUTPUT_DIM}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Number of Epochs: {NUM_EPOCHS}")

def get_probabilities(model, X_s1, X_s2, X_global, batch_size, device="cpu"):
    """
    outputs the probabilities of each player of winning the game for each game in the dataset, based on the model
    """
    model.eval()
    tensor_eval_data = TensorDataset(
        torch.tensor(X_s1, dtype=torch.float32),
        torch.tensor(X_s2, dtype=torch.float32),
        torch.tensor(X_global, dtype=torch.float32),
    )
    eval_loader = DataLoader(tensor_eval_data, batch_size = batch_size, shuffle=False)

    probs_list = []
    with torch.no_grad():
        for s1, s2, globe  in eval_loader:
            s1 = s1.to(device)
            s2 = s2.to(device)
            globe = globe.to(device)
            logits = model(s1, s2, globe)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs_list)

def split_by_game(X_s1, X_s2, X_global, y, game_ids, test_frac=TEST_SPLIT, seed=RANDOM_SEED):
    """
    Note that all samples from the same game stay together.

    Returns:
        x_s1_train, x_s2_train, x_global_train, x_s1_test, x_s2_test, x_global_test, y_train, y_test, game_ids_train, game_ids_test
    """
    unique_games = np.unique(game_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_games)

    n_test = int(len(unique_games) * test_frac)
    test_games = set(unique_games[:n_test])

    test_mask = np.array([gid in test_games for gid in game_ids])
    train_mask = ~test_mask

    return (
        X_s1[train_mask],
        X_s2[train_mask],
        X_global[train_mask],
        X_s1[test_mask],
        X_s2[test_mask],
        X_global[test_mask],
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading dataset...")
    X_s1, X_s2, X_global, y, game_ids = build_pytorch_dataset()

    # 
    print(f"Loaded {len(np.unique(game_ids))} games")

    X_train_s1, X_train_s2, X_train_global, X_test_s1, X_test_s2, X_test_global, y_train, y_test, gid_train, gid_test = split_by_game(
        X_s1,
        X_s2,
        X_global,
        y,
        game_ids,
    )
    print(f"Train: {X_train_s1.shape[0]}  Test: {X_test_s1.shape[0]}")

    scaler_s1 = StandardScaler()
    scaler_s2 = StandardScaler()
    scaler_global = StandardScaler()
    sclr_s1_fitted = scaler_s1.fit_transform(X_train_s1)
    sclr_s2_fitted = scaler_s2.fit_transform(X_train_s2)
    sclr_global_fitted = scaler_global.fit_transform(X_train_global)
    sclr_s1_transformed = scaler_s1.transform(X_test_s1)
    sclr_s2_transformed = scaler_s2.transform(X_test_s2)
    sclr_global_transformed = scaler_global.transform(X_test_global)

    class WinnerPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, global_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
            )
            total_dim = output_dim * 4 + global_dim
            self.classifier = nn.Sequential(
                nn.Linear(total_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward_encoder(self, x):
            return self.encoder(x)
        def forward(self, x_s1, x_s2, x_global):
            x_s1_encoded = self.forward_encoder(x_s1)
            x_s2_encoded = self.forward_encoder(x_s2)
            combined_tensor = torch.cat([x_s1_encoded, x_s2_encoded, x_s2_encoded * x_s1_encoded, torch.abs(x_s1_encoded - x_s2_encoded), x_global], dim=1)
            return self.classifier(combined_tensor).squeeze(1)

    # grid search hyperparameters

    for hidden_dim in HIDDEN_DIMS:
        for output_dim in OUTPUT_DIM:
            for learning_rate in LEARNING_RATE:
                for batch_size in BATCH_SIZE:
                    print(f"model params: hidden_dim={hidden_dim}, output_dim={output_dim}, learning_rate={learning_rate}, batch_size={batch_size}")

                    model = WinnerPredictor(
                        input_dim=X_train_s1.shape[1], 
                        hidden_dim=hidden_dim, 
                        output_dim=output_dim, 
                        global_dim=X_train_global.shape[1], 
                    ).to(device)
                    
                    tensor_train_data = TensorDataset(
                        torch.tensor(sclr_s1_fitted, dtype=torch.float32),
                        torch.tensor(sclr_s2_fitted, dtype=torch.float32),
                        torch.tensor(sclr_global_fitted, dtype=torch.float32),
                        torch.tensor(y_train, dtype=torch.float32),
                    )
                    tensor_test_data = TensorDataset(
                        torch.tensor(sclr_s1_transformed, dtype=torch.float32),
                        torch.tensor(sclr_s2_transformed, dtype=torch.float32),
                        torch.tensor(sclr_global_transformed, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.float32),
                    )
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = nn.BCEWithLogitsLoss()
                    train_loader = DataLoader(tensor_train_data, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(tensor_test_data, batch_size = batch_size, shuffle = False)
                    
                    for epoch in range(NUM_EPOCHS):
                        model.train()

                        for xs1, xs2, xglobal, yb in train_loader:
                            xs1 = xs1.to(device)
                            xs2 = xs2.to(device)
                            xglobal = xglobal.to(device)
                            yb = yb.to(device)

                            optimizer.zero_grad()
                            y_pred = model(xs1, xs2, xglobal)
                            loss = criterion(y_pred, yb)
                            loss.backward()
                            optimizer.step()

                    p_train = get_probabilities(model, sclr_s1_fitted, sclr_s2_fitted, sclr_global_fitted, batch_size, device=device)
                    p_test = get_probabilities(model, sclr_s1_transformed, sclr_s2_transformed, sclr_global_transformed, batch_size, device=device)

    # Per-game winner accuracy from predicted win probabilities.

                    print("\nWinner prediction (full board context):")
                    train_winner_acc = evaluate_winner_accuracy_from_probs(y_train, p_train, gid_train)
                    test_winner_acc = evaluate_winner_accuracy_from_probs(y_test, p_test, gid_test)
                    print("\nSummary:")
                    print(
                        f"WinnerAcc:        {test_winner_acc:.4f} "
                        #f"(model={args.model}, n_test_games={len(np.unique(gid_test))})"
                    )

if __name__ == "__main__":
    main()
