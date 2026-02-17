import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_generation")
GAMES_CSV = os.path.join(DATA_DIR, "games.csv")
PLAYERS_CSV = os.path.join(DATA_DIR, "players.csv")

RANDOM_SEED = 229
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RIDGE_ALPHA = 1.0
