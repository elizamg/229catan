# Data Download

**Note:** This data is of 50,000 4-player Catan games. The games.csv holds the game-level info ('board_layout', 'winner'). The players.csv holds the player info ('player', 'initial_placements', 'vps'). Both CSVs have a 'game_id', which can be used to map between the two.

**Note:** This data has already been parsed using `data_parser.py`. You can also generate more data using [Catanatron](https://github.com/bcollazo/catanatron) and parse it by running the parser.

## Steps

1. **Download** the data from [Google Drive](https://drive.google.com/file/d/1pjTVI4o3klDr0ahS7pJY5J9ooI2Lbg8h/view?usp=sharing).
2. **Place** the files in the `data_generation` folder — `games.csv` and `players.csv` should both be in the `data_generation` folder.

## Project Setup

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies. Install `uv` on your computer, then run `uv sync` to install dependencies in the environment. `uv run mypythonfile.py` will run that python file in the uv environment with the project dependencies.
