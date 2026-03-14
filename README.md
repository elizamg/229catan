# 229catan

## Data Download

Note: This data is of 50,000 4-player Catan games. `games.csv` holds game-level info (`board_layout`, `winner`). `players.csv` holds player-level info (`player`, `initial_placements`, `vps`). Both files share `game_id`.

Note: The data has already been parsed with `data_parser.py`. You can also generate more data with [Catanatron](https://github.com/bcollazo/catanatron) and parse it yourself.

## Steps

1. Download the data from [Google Drive](https://drive.google.com/file/d/1pjTVI4o3klDr0ahS7pJY5J9ooI2Lbg8h/view?usp=sharing).
2. Place `games.csv` and `players.csv` in `data_generation/`.

## Project Setup

This project uses `uv` to manage dependencies.

```bash
uv sync
```

Run scripts inside the project environment with:

```bash
uv run mypythonfile.py
```

## Models

`train.py` supports:

- `ridge` — ridge regression for placement-to-VP prediction
- `xgb` — XGBoost regressor for placement-to-VP prediction

`train_winner.py` supports:

- `logreg` — logistic regression for winner classification
- `xgb` — XGBoost classifier for winner classification

MLP and PyTorch (deep / winner prediction):

- `train-mlp.py` — sklearn MLP regressor for placement-to-VP prediction; also reports winner-prediction accuracy.
- `train-mlp-wp.py` — sklearn MLP classifier for winner classification (winner prediction).
- `train-pytorch.py` — PyTorch neural network for winner prediction (single hyperparameter set).
- `train-pytorch-grid.py` — PyTorch winner model with grid search over hidden dims, output dim, batch size, and learning rate.

Each of these supports `--load PATH` (evaluate a saved model) and `-o PATH` / `--output PATH` (default: `mlp_model.pkl`).

## Training

Placement model, default ridge:

```bash
uv run train.py
```

Placement model, explicit ridge:

```bash
uv run train.py --model ridge
```

Placement model, XGBoost:

```bash
uv run train.py --model xgb
```

Winner model, default logistic regression:

```bash
uv run train_winner.py
```

Winner model, explicit logistic regression:

```bash
uv run train_winner.py --model logreg
```

Winner model, XGBoost:

```bash
uv run train_winner.py --model xgb
```

MLP placement (VP prediction):

```bash
uv run train-mlp.py
```

MLP winner prediction:

```bash
uv run train-mlp-wp.py
```

PyTorch winner prediction:

```bash
uv run train-pytorch.py
```

PyTorch winner prediction (grid search over hyperparameters):

```bash
uv run train-pytorch-grid.py
```

## Running A Saved Model

```bash
uv run train.py --load path/to/model.pkl
uv run train_winner.py --load path/to/model.pkl
uv run train-mlp.py --load path/to/mlp_model.pkl
uv run train-mlp-wp.py --load path/to/mlp_model.pkl
uv run train-pytorch.py --load path/to/mlp_model.pkl
uv run train-pytorch-grid.py --load path/to/mlp_model.pkl
```

## Default Outputs

- `ridge_model.pkl`
- `xgb_model.pkl`
- `winner_logreg.pkl`
- `winner_xgb.pkl`
- `mlp_model.pkl` (train-mlp, train-mlp-wp, train-pytorch, train-pytorch-grid with default `-o`)
