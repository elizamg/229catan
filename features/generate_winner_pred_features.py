"""Feature generation for winner prediction.

This module builds (X, y, game_ids) numpy arrays for winner classification.

- One row per player per game (4 rows/game).
- Labels are binary and tie-aware: 1 if the player is a winner (max VP in that game), else 0.

Usage:
    X, y, game_ids = build_winner_classification_dataset()
"""

import csv
import json
import numpy as np

from board_constants import (
    NODE_TO_TILE_IDS,
    NODE_ADJACENCY,
    PORT_COORD_TO_NODES,
    NUMBER_PROBABILITIES,
    RESOURCES,
)
from config import GAMES_CSV, PLAYERS_CSV


def _compute_board_stats(tile_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-board aggregates once.

    Returns:
        board_resource_production: (5,) total production per resource
        num_high_value: (5,) count of 6/8 tiles per resource
        scarcity: (5,) 1 / (# tiles of that resource)
    """
    board_resource_production = np.zeros(5)
    num_high_value = np.zeros(5)
    board_resource_counts = np.zeros(5)

    for tile in tile_info.values():
        res = tile["resource"]
        num = tile["number"]
        if res is None:
            continue
        idx = RESOURCES.index(res)
        board_resource_counts[idx] += 1
        if num is not None:
            board_resource_production[idx] += NUMBER_PROBABILITIES.get(num, 0.0)
            num_high_value[idx] += float(num in (6, 8))

    scarcity = 1.0 / np.maximum(board_resource_counts, 1.0)
    return board_resource_production, num_high_value, scarcity


def parse_board(board_layout_json: str) -> dict:
    """
    Takes in the board_layout string from games.csv

    Returns:
        tile_info: dict mapping tile_id (int) -> {
            "resource": str or None (None for desert),
            "number": int or None (None for desert),
        }
        port_map: dict mapping node_id (int) -> port resource str
            ("BRICK", "WOOD", etc. or "ANY" for 3:1 ports)
    """
    board_layout = json.loads(board_layout_json)
    resource_tiles = []
    ports = []
    desert = None

    for location in board_layout:
        if location["tile"]["type"] == "RESOURCE_TILE":
            resource_tiles.append(location)
        elif location["tile"]["type"] == "PORT":
            ports.append(location)
        elif location["tile"]["type"] == "DESERT":
            desert = location

    tile_info = {
        loc["tile"]["id"]: {
            "resource": loc["tile"].get("resource"),
            "number": loc["tile"].get("number"),
        }
        for loc in resource_tiles
    }
    tile_info[desert["tile"]["id"]] = {"resource": None, "number": None}
    port_map = {}
    for loc in ports:
        loc_1, loc_2 = PORT_COORD_TO_NODES[tuple(loc["coordinate"])]
        resource = loc["tile"].get("resource")
        port_map[loc_1] = resource
        port_map[loc_2] = resource

    return tile_info, port_map


def parse_placements(initial_placements_json: str) -> list[dict]:
    """
    Returns list of 2 dicts, one per round:
        [
            {"settlement_node": int, "road_edge": (int, int)},  # round 1
            {"settlement_node": int, "road_edge": (int, int)},  # round 2
        ]
    """
    initial_placements = json.loads(initial_placements_json)
    return [
        {
            "settlement_node": initial_placements[0][0][2],
            "road_edge": (
                initial_placements[1][0][2][0],
                initial_placements[1][0][2][1],
            ),
        },
        {
            "settlement_node": initial_placements[2][0][2],
            "road_edge": (
                initial_placements[3][0][2][0],
                initial_placements[3][0][2][1],
            ),
        },
    ]


def extract_placement_features(
    settlement_node: int,
    road_edge: tuple[int, int],
    tile_info: dict,
    port_map: dict,
    board_resource_production: np.ndarray,
    occupied_nodes: list[int] | None = None,
) -> np.ndarray:
    """
    Features for a single settlement + road placement.

    Returns:
        1D numpy array of features

        - Total production value: sum of NUMBER_PROBABILITIES for adjacent tiles.
        - Per-resource production: [brick_prod, wood_prod, ...]
        - Resource diversity: number of distinct resource types adjacent
        - High value numbers: number of adjacent tiles with 6 or 8
        - Has port: whether this node has any port access
        - Port type: one-hot for specific resource port or 3:1 "ANY" port
        - Port-resource match: dot(resource_port_onehot, resource_production)
        - Expansion potential: best production potential among candidate settlement nodes reachable from the road's other endpoint
        - Resource share: fraction of total board production for each resource captured by this settlement
    """
    adj_tiles = NODE_TO_TILE_IDS[settlement_node]

    resource_production = np.zeros(5)
    total_production = 0.0
    distinct_resources = set()
    num_6_or_8 = 0.0

    for tid in adj_tiles:
        tile = tile_info[tid]
        res = tile["resource"]
        num = tile["number"]
        if res is None:
            continue
        idx = RESOURCES.index(res)
        prob = NUMBER_PROBABILITIES.get(num, 0)
        resource_production[idx] += prob
        total_production += prob
        distinct_resources.add(res)
        if num in (6, 8):
            num_6_or_8 += 1.0

    resource_diversity = len(distinct_resources)

    has_port = float(settlement_node in port_map)
    port_onehot = np.zeros(6)  # BRICK, WOOD, SHEEP, WHEAT, ORE, ANY
    if settlement_node in port_map:
        port_res = port_map[settlement_node]
        if port_res is None:
            port_onehot[5] = 1.0  # 3:1 ANY port
        else:
            port_onehot[RESOURCES.index(port_res)] = 1.0

    # Synergy between having a specific 2:1 port and producing that resource.
    # (3:1 ANY port is not included here.)
    port_prod_match = float(np.dot(port_onehot[:5], resource_production))

    other_node = road_edge[0] if road_edge[1] == settlement_node else road_edge[1]

    best_expansion_prod = 0.0
    occupied = set(occupied_nodes or [])
    for cand in NODE_ADJACENCY.get(other_node, []):
        if cand == settlement_node:
            continue
        if cand in occupied:
            continue
        if occupied and any(nb in occupied for nb in NODE_ADJACENCY.get(cand, [])):
            continue
        cand_prod = 0.0
        for tid in NODE_TO_TILE_IDS.get(cand, []):
            num = tile_info[tid]["number"]
            if num is None:
                continue
            cand_prod += NUMBER_PROBABILITIES.get(num, 0.0)
        if cand_prod > best_expansion_prod:
            best_expansion_prod = cand_prod

    # Share-of-board production: (this settlement's production per resource) / (total board production per resource).
    prod_share = resource_production / (board_resource_production + 1e-9)

    return np.concatenate(
        [
            [total_production],  # 1
            resource_production,  # 5
            [resource_diversity],  # 1
            [num_6_or_8],  # 1
            [has_port],  # 1
            port_onehot,  # 6
            [port_prod_match],  # 1
            [best_expansion_prod],  # 1
            prod_share,  # 5
        ]
    )  # total: 22


def extract_opponent_features(
    settlement_node: int,
    tile_info: dict,
    opponent_placements: list[int],
) -> np.ndarray:
    """
    Features about opponents' existing placements for a given potential settlement node

    Args:
        settlement_node: the candidate node
        tile_info: parsed board tile info
        opponent_placements: list of node IDs already placed by opponents (empty if placing first)

    Returns:
        1D numpy array of features

        - Shared tile opponents: number of opponent settlements on tiles touching the canditate node
        - Resource competition: for each resource adjacent to this node, how many opponent settlements also touch that resource [num_brick_settlements, num_wood_settlements, ...]
        - Blocked neighbors: how many of this node's NODE_ADJACENCY neighbors are occupied (affects future expansion)
    """
    my_tiles = set(NODE_TO_TILE_IDS[settlement_node])

    resource_competition = np.zeros(5)
    shared_tile_opponents = 0
    for opp_node in opponent_placements:
        opp_tiles = set(NODE_TO_TILE_IDS[opp_node])
        if opp_tiles & my_tiles:
            shared_tile_opponents += 1
        for tid in opp_tiles & my_tiles:
            res = tile_info[tid]["resource"]
            if res is not None:
                resource_competition[RESOURCES.index(res)] += 1

    # Blocked neighbors
    neighbors = NODE_ADJACENCY[settlement_node]
    opp_set = set(opponent_placements)
    blocked = sum(1 for n in neighbors if n in opp_set)

    return np.concatenate(
        [
            [shared_tile_opponents],  # 1
            resource_competition,  # 5
            [blocked],  # 1
        ]
    )  # total: 7


def extract_board_features(tile_info: dict) -> np.ndarray:
    """Global features about the board

    Returns:
        1D numpy array of features
        - Total production per resource: [brick_prod, wood_prod, sheep_prod, wheat_prod, ore_prod]
        - Number of high-value tiles (6 or 8) per resource: [brick_num_high, wood_num_high, ...]
        - Resource scarcity: 1 / (# tiles of that resource)
    """
    resource_production, num_high_value, scarcity = _compute_board_stats(tile_info)
    return np.concatenate([resource_production, num_high_value, scarcity])


def extract_raw_board_features(tile_info: dict) -> np.ndarray:
    """
    Raw per-tile encoding of the full board layout.

    For each of the 19 tiles in fixed ID order:
    - Resource one-hot: [BRICK, WOOD, SHEEP, WHEAT, ORE, DESERT]
    - Dice probability from NUMBER_PROBABILITIES (0 for desert)
    - Normalized raw dice number: (number - 2) / 10 (0 for desert)

    Returns:
        1D numpy array of 19 * 8 = 152 features
    """
    NUM_TILES = 19
    FEATS_PER_TILE = 8
    out = np.zeros(NUM_TILES * FEATS_PER_TILE)
    for tid in range(NUM_TILES):
        tile = tile_info[tid]
        offset = tid * FEATS_PER_TILE
        res = tile["resource"]
        num = tile["number"]
        if res is None:
            out[offset + 5] = 1.0  # desert
        else:
            out[offset + RESOURCES.index(res)] = 1.0
            out[offset + 6] = NUMBER_PROBABILITIES[num]
            out[offset + 7] = (num - 2) / 10.0
    return out


def extract_turn_order_features(
    player_index: int,
    placement_round: int,
) -> np.ndarray:
    """
    Args:
        player_index: 0-3 (order in which players take turns)
        placement_round: 0 or 1 (first or second settlement)

    Returns:
        1D numpy array of concatenated features
        - Player position one-hot
        - Placement round, 0 or 1
        - Combined placement order: the actual sequential position (0-7) in the
            snake draft (round 1: 0,1,2,3; round 2: 3,2,1,0).
    """
    position = np.eye(4)[player_index]
    placement_order = player_index
    if placement_round == 1:
        placement_order = 7 - player_index
    return np.concatenate([position, [placement_round], [placement_order]])


def build_winner_classification_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dataset for winner prediction.

    One row per player per game (4 rows/game).
    Features are built from BOTH of the player's placements and the full opening state.
    Labels are binary: 1 if the player is a winner (tie-aware), else 0.

    Returns:
        X: (n_games * 4, n_features)
        y: (n_games * 4,) in {0.0, 1.0}
        game_ids: (n_games * 4,)
    """
    game_boards = {}
    with open(GAMES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            game_boards[row["game_id"]] = row["board_layout"]

    game_players: dict[str, list[dict]] = {}
    with open(PLAYERS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            gid = row["game_id"]
            game_players.setdefault(gid, []).append(
                {
                    "player": row["player"],
                    "placements_json": row["initial_placements"],
                    "vps": int(row["vps"]),
                }
            )

    all_features = []
    all_labels = []
    all_game_ids = []

    for gid, players in game_players.items():
        if gid not in game_boards:
            continue
        if len(players) == 0:
            continue

        tile_info, port_map = parse_board(game_boards[gid])
        board_resource_production, _, _ = _compute_board_stats(tile_info)
        parsed = [parse_placements(p["placements_json"]) for p in players]

        max_vp = max(p["vps"] for p in players)

        # All settlements placed in the opening (8 total for 4 players).
        all_settlements: list[int] = []
        for pi in range(len(players)):
            all_settlements.append(parsed[pi][0]["settlement_node"])  # round 0
            all_settlements.append(parsed[pi][1]["settlement_node"])  # round 1

        board_feats = extract_board_features(tile_info)
        raw_board_feats = extract_raw_board_features(tile_info)

        for pi in range(len(players)):
            label = float(players[pi]["vps"] == max_vp)

            per_player_parts = []
            for rnd in (0, 1):
                settlement = parsed[pi][rnd]["settlement_node"]
                road = parsed[pi][rnd]["road_edge"]

                occupied = [s for s in all_settlements if s != settlement]

                placement_feats = extract_placement_features(
                    settlement,
                    road,
                    tile_info,
                    port_map,
                    board_resource_production=board_resource_production,
                    occupied_nodes=occupied,
                )
                opp_feats = extract_opponent_features(
                    settlement,
                    tile_info,
                    opponent_placements=occupied,
                )
                turn_feats = extract_turn_order_features(pi, rnd)

                per_player_parts.extend([placement_feats, opp_feats, turn_feats])

            x = np.concatenate([*per_player_parts, board_feats, raw_board_feats])
            all_features.append(x)
            all_labels.append(label)
            all_game_ids.append(gid)

    X = np.array(all_features)
    y = np.array(all_labels, dtype=float)
    game_ids = np.array(all_game_ids)
    return X, y, game_ids
