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

## Running A Saved Model

```bash
uv run train.py --load path/to/model.pkl
uv run train_winner.py --load path/to/model.pkl
```

## Default Outputs

- `ridge_model.pkl`
- `xgb_model.pkl`
- `winner_logreg.pkl`
- `winner_xgb.pkl`
