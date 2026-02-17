"""
Produces (X, y, game_ids) numpy arrays.

Each training sample represents one player's placement context in one game.
Note that each game produces 8 samples (4 players x 2 rounds).

Usage:
    X, y, game_ids = build_dataset()
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
) -> np.ndarray:
    """
    Features for a single settlement + road placement.

    Returns:
        1D numpy array of features

        - Number of adjacent resources: [brick_count, wood_count, sheep_count, wheat_count, ore_count]
        - Total production value: sum of NUMBER_PROBABILITIES for adjacent tiles.
        - Per-resource production: [brick_prod, wood_prod, ...]
        - Resource diversity: number of distinct resource types adjacent
        - Number diversity: number of distinct dice numbers adjacent
        - Has port: whether this node has any port access
        - Port type: one-hot for specific resource port or 3:1 "ANY" port
        - Expansion potential: number of land-adjacent tiles reachable from the road's other endpoint
    """
    adj_tiles = NODE_TO_TILE_IDS[settlement_node]

    resource_counts = np.zeros(5)
    resource_production = np.zeros(5)
    total_production = 0.0
    distinct_resources = set()
    distinct_numbers = set()

    for tid in adj_tiles:
        tile = tile_info[tid]
        res = tile["resource"]
        num = tile["number"]
        if res is None:
            continue
        idx = RESOURCES.index(res)
        prob = NUMBER_PROBABILITIES.get(num, 0)
        resource_counts[idx] += 1
        resource_production[idx] += prob
        total_production += prob
        distinct_resources.add(res)
        distinct_numbers.add(num)

    resource_diversity = len(distinct_resources)
    number_diversity = len(distinct_numbers)

    has_port = float(settlement_node in port_map)
    port_onehot = np.zeros(6)  # BRICK, WOOD, SHEEP, WHEAT, ORE, ANY
    if settlement_node in port_map:
        port_res = port_map[settlement_node]
        if port_res is None:
            port_onehot[5] = 1.0  # 3:1 ANY port
        else:
            port_onehot[RESOURCES.index(port_res)] = 1.0

    other_node = road_edge[0] if road_edge[1] == settlement_node else road_edge[1]
    expansion = len(NODE_TO_TILE_IDS.get(other_node, []))

    return np.concat(
        [
            resource_counts,  # 5
            [total_production],  # 1
            resource_production,  # 5
            [resource_diversity],  # 1
            [number_diversity],  # 1
            [has_port],  # 1
            port_onehot,  # 6
            [expansion],  # 1
        ]
    )  # total: 21


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

    # for each resource touched, how many opponents also touch it
    my_resources = {}
    for tid in my_tiles:
        res = tile_info[tid]["resource"]
        if res is not None:
            my_resources.setdefault(res, set()).add(tid)

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

    return np.concat(
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
        - Desert ring: which ring the desert is on: 0 (center), 1 (middle), 2 (outer)

    """
    resource_production = np.zeros(5)
    num_high_value = np.zeros(5)
    desert_ring = 0
    for tid, tile in tile_info.items():
        res = tile["resource"]
        num = tile["number"]
        if res is None:  # desert
            if tid >= 7:
                desert_ring = 2
            elif tid >= 1:
                desert_ring = 1
            else:
                desert_ring = 0
        else:
            idx = RESOURCES.index(res)
            resource_production[idx] += NUMBER_PROBABILITIES[num]
            num_high_value[idx] += int(num in (6, 8))
    return np.concat([resource_production, num_high_value, [desert_ring]])


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
    return np.concat([position, [placement_round], [placement_order]])


def build_feature_vector(
    settlement_node: int,
    road_edge: tuple[int, int],
    tile_info: dict,
    port_map: dict,
    opponent_placements: list[int],
    player_index: int,
    placement_round: int,
) -> np.ndarray:
    """Combine all feature groups into a single feature vector.

    Returns:
        1D numpy array: concatenation of all feature groups
    """
    placement_feats = extract_placement_features(
        settlement_node,
        road_edge,
        tile_info,
        port_map,
    )
    opponent_feats = extract_opponent_features(
        settlement_node,
        tile_info,
        opponent_placements,
    )
    board_feats = extract_board_features(tile_info)
    raw_board_feats = extract_raw_board_features(tile_info)
    turn_feats = extract_turn_order_features(player_index, placement_round)
    return np.concat(
        [placement_feats, opponent_feats, board_feats, raw_board_feats, turn_feats]
    )


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CSVs and build the full dataset.

    Each game gives 8 training samples (4 players x 2 placements)

    Returns:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) VP labels
        game_ids: (n_samples,) game ID strings (for group-aware splitting)
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
    all_vps = []
    all_game_ids = []

    for gid, players in game_players.items():
        tile_info, port_map = parse_board(game_boards[gid])
        parsed = [parse_placements(p["placements_json"]) for p in players]
        n = len(players)  # should be 4

        for rnd in range(2):
            if rnd == 0:
                order = list(range(n))
            else:
                order = list(range(n - 1, -1, -1))

            placed_so_far = []
            if rnd == 1:
                placed_so_far = [parsed[i][0]["settlement_node"] for i in range(n)]

            round1_placed = []
            for pi in order:
                own_r0 = parsed[pi][0]["settlement_node"] if rnd == 1 else None
                opponent_settlements = [
                    s
                    for s in (placed_so_far + round1_placed)
                    if s != own_r0 or rnd == 0
                ]
                if rnd == 0:
                    opponent_settlements = list(placed_so_far)

                settlement = parsed[pi][rnd]["settlement_node"]
                road = parsed[pi][rnd]["road_edge"]

                feat = build_feature_vector(
                    settlement,
                    road,
                    tile_info,
                    port_map,
                    opponent_settlements,
                    pi,
                    rnd,
                )
                all_features.append(feat)
                all_vps.append(players[pi]["vps"])
                all_game_ids.append(gid)

                if rnd == 0:
                    placed_so_far.append(settlement)
                else:
                    round1_placed.append(settlement)

    X = np.array(all_features)
    y = np.array(all_vps, dtype=float)
    game_ids = np.array(all_game_ids)

    return X, y, game_ids
