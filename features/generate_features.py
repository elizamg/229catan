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


def build_transformer_dataset():
    """
    Build dataset for the transformer model.

    Each game gives 8 training samples (4 players x 2 rounds).
    Each sample includes raw token IDs for the transformer plus
    hand-engineered features for the hybrid MLP head.

    Only placements visible at decision time are included (matching
    what the model would see during inference). Structure and road
    arrays are zero-padded to max length (8) with boolean masks
    indicating which positions are padding (True = ignore).

    Owner IDs are rotated so the current player is always 0.

    Returns:
        data: dict of numpy arrays:
            tile_resource:  (n, 19)  resource type IDs (0-5)
            tile_dicenum:   (n, 19)  dice number IDs (0=NONE, 1-11 for 2-12)
            tile_pos:       (n, 19)  always [0..18]
            port_resource:  (n, 9)   port type IDs (0-4 resources, 5=ANY)
            port_pos:       (n, 9)   always [0..8]
            struct_owner:   (n, 8)   owner (0=current player, 1-3=opponents)
            struct_type:    (n, 8)   0=settlement, 1=city
            struct_pos:     (n, 8)   node IDs (0-53)
            struct_mask:    (n, 8)   True=padding, False=real token
            road_owner:     (n, 8)   owner (0=current player, 1-3=opponents)
            road_a:         (n, 8)   endpoint node IDs (0-53)
            road_b:         (n, 8)   endpoint node IDs (0-53)
            road_mask:      (n, 8)   True=padding, False=real token
            hand_features:  (n, F)   hand-engineered feature vector
        y: (n_samples,) VP labels
        game_ids: (n_samples,) game ID strings
    """
    RESOURCE_TO_ID = {"BRICK": 0, "WOOD": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
    PORT_TYPE_TO_ID = {"BRICK": 0, "WOOD": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}

    # Max placement tokens: 8 settlements + 8 roads in a full game
    # Samples with fewer visible placements are zero-padded
    # a boolean mask marks which positions are real (False) vs padding (True)
    MAX_STRUCTS = 8
    MAX_ROADS = 8

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

    tile_pos_fixed = np.arange(19, dtype=np.int64)
    port_pos_fixed = np.arange(9, dtype=np.int64)

    acc_tile_resource = []
    acc_tile_dicenum = []
    acc_tile_pos = []
    acc_port_resource = []
    acc_port_pos = []
    acc_struct_owner = []
    acc_struct_type = []
    acc_struct_pos = []
    acc_struct_mask = []  # True = padding (ignore)
    acc_road_owner = []
    acc_road_a = []
    acc_road_b = []
    acc_road_mask = []  # True = padding (ignore)
    acc_hand_features = []
    all_vps = []
    all_game_ids = []

    for gid, players in game_players.items():
        board_layout_json = game_boards[gid]
        tile_info, port_map = parse_board(board_layout_json)
        board_layout = json.loads(board_layout_json)

        # tile tokens
        tile_resource = np.zeros(19, dtype=np.int64)
        tile_dicenum = np.zeros(19, dtype=np.int64)
        for tid in range(19):
            tile = tile_info[tid]
            res = tile["resource"]
            num = tile["number"]
            if res is None:  # desert
                tile_resource[tid] = 5
                tile_dicenum[tid] = 0  # NONE
            else:
                tile_resource[tid] = RESOURCE_TO_ID[res]
                tile_dicenum[tid] = num - 1  # 2->1, 3->2, ..., 12->11

        # port tokens
        port_resources = []
        for loc in board_layout:
            if loc["tile"]["type"] == "PORT":
                res = loc["tile"].get("resource")
                if res is None:
                    port_resources.append(5)  # ANY / 3:1
                else:
                    port_resources.append(PORT_TYPE_TO_ID[res])
        port_resource = np.array(port_resources, dtype=np.int64)

        n = len(players)  # should be 4
        parsed = [parse_placements(p["placements_json"]) for p in players]

        # track which placements are visible at each decision point
        # visible_placements grows as each player places in draft order
        visible_placements = []

        for rnd in range(2):
            if rnd == 0:
                order = list(range(n))
            else:
                order = list(range(n - 1, -1, -1))

            if rnd == 1 and not visible_placements:
                for i in range(n):
                    snode = parsed[i][0]["settlement_node"]
                    redge = parsed[i][0]["road_edge"]
                    visible_placements.append((i, snode, redge))

            for pi in order:
                # hand-engineered features
                if rnd == 0:
                    opponent_settlements = [vp[1] for vp in visible_placements]
                else:
                    own_r0 = parsed[pi][0]["settlement_node"]
                    opponent_settlements = [
                        vp[1]
                        for vp in visible_placements
                        if not (vp[0] == pi and vp[1] == own_r0)
                    ]

                settlement = parsed[pi][rnd]["settlement_node"]
                road = parsed[pi][rnd]["road_edge"]

                hand_feat = build_feature_vector(
                    settlement,
                    road,
                    tile_info,
                    port_map,
                    opponent_settlements,
                    pi,
                    rnd,
                )

                # structure tokens
                current_placement = (pi, settlement, road)
                all_visible = visible_placements + [current_placement]

                struct_owner = np.zeros(MAX_STRUCTS, dtype=np.int64)
                struct_type = np.zeros(MAX_STRUCTS, dtype=np.int64)
                struct_pos = np.zeros(MAX_STRUCTS, dtype=np.int64)
                struct_mask = np.ones(MAX_STRUCTS, dtype=bool)  # True = pad

                road_owner = np.zeros(MAX_ROADS, dtype=np.int64)
                road_a_arr = np.zeros(MAX_ROADS, dtype=np.int64)
                road_b_arr = np.zeros(MAX_ROADS, dtype=np.int64)
                road_mask = np.ones(MAX_ROADS, dtype=bool)  # True = pad

                for idx, (owner, snode, redge) in enumerate(all_visible):
                    struct_owner[idx] = (owner - pi) % n
                    struct_pos[idx] = snode
                    struct_mask[idx] = False  # real token

                    road_owner[idx] = (owner - pi) % n
                    road_a_arr[idx] = redge[0]
                    road_b_arr[idx] = redge[1]
                    road_mask[idx] = False  # real token

                acc_tile_resource.append(tile_resource)
                acc_tile_dicenum.append(tile_dicenum)
                acc_tile_pos.append(tile_pos_fixed)
                acc_port_resource.append(port_resource)
                acc_port_pos.append(port_pos_fixed)
                acc_struct_owner.append(struct_owner)
                acc_struct_type.append(struct_type)
                acc_struct_pos.append(struct_pos)
                acc_struct_mask.append(struct_mask)
                acc_road_owner.append(road_owner)
                acc_road_a.append(road_a_arr)
                acc_road_b.append(road_b_arr)
                acc_road_mask.append(road_mask)
                acc_hand_features.append(hand_feat)
                all_vps.append(players[pi]["vps"])
                all_game_ids.append(gid)

                # After player places, their placement becomes visible
                visible_placements.append((pi, settlement, road))

    data = {
        "tile_resource": np.array(acc_tile_resource),
        "tile_dicenum": np.array(acc_tile_dicenum),
        "tile_pos": np.array(acc_tile_pos),
        "port_resource": np.array(acc_port_resource),
        "port_pos": np.array(acc_port_pos),
        "struct_owner": np.array(acc_struct_owner),
        "struct_type": np.array(acc_struct_type),
        "struct_pos": np.array(acc_struct_pos),
        "struct_mask": np.array(acc_struct_mask),
        "road_owner": np.array(acc_road_owner),
        "road_a": np.array(acc_road_a),
        "road_b": np.array(acc_road_b),
        "road_mask": np.array(acc_road_mask),
        "hand_features": np.array(acc_hand_features),
    }
    y = np.array(all_vps, dtype=np.float32)
    game_ids = np.array(all_game_ids)

    return data, y, game_ids


def build_winner_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    To predict the winner of the game, we evaluate the position of the player's
    second settlement/road with all of the other players placements.

    Returns:
        X: (n_games * 4, n_features) feature matrix
        y: (n_games * 4,) VP labels
        game_ids: (n_games * 4,) game ID strings
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
        n = len(players)

        all_settlements = []
        for pi in range(n):
            all_settlements.append(parsed[pi][0]["settlement_node"])  # round 0
            all_settlements.append(parsed[pi][1]["settlement_node"])  # round 1

        for pi in range(n):
            settlement = parsed[pi][1]["settlement_node"]
            road = parsed[pi][1]["road_edge"]

            # all 7 other opponent settlements (excluds this player's round 1)
            opponent_settlements = [
                s for idx, s in enumerate(all_settlements) if idx != pi * 2 + 1
            ]

            feat = build_feature_vector(
                settlement,
                road,
                tile_info,
                port_map,
                opponent_settlements,
                pi,
                1,  # placement_round=1
            )
            all_features.append(feat)
            all_vps.append(players[pi]["vps"])
            all_game_ids.append(gid)

    X = np.array(all_features)
    y = np.array(all_vps, dtype=float)
    game_ids = np.array(all_game_ids)

    return X, y, game_ids


def build_transformer_winner_dataset():
    """
    Transformer version of build_winner_dataset.

    For each game, evaluate each player's round-1 placement with all 8
    placements (all players, both rounds) visible. Returns 4 samples per game.

    Returns:
        data: dict of numpy arrays (same format as build_transformer_dataset)
        y: (n_games * 4,) VP labels
        game_ids: (n_games * 4,) game ID strings
    """
    RESOURCE_TO_ID = {"BRICK": 0, "WOOD": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
    PORT_TYPE_TO_ID = {"BRICK": 0, "WOOD": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
    MAX_STRUCTS = 8
    MAX_ROADS = 8

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

    tile_pos_fixed = np.arange(19, dtype=np.int64)
    port_pos_fixed = np.arange(9, dtype=np.int64)

    acc_tile_resource = []
    acc_tile_dicenum = []
    acc_tile_pos = []
    acc_port_resource = []
    acc_port_pos = []
    acc_struct_owner = []
    acc_struct_type = []
    acc_struct_pos = []
    acc_struct_mask = []
    acc_road_owner = []
    acc_road_a = []
    acc_road_b = []
    acc_road_mask = []
    acc_hand_features = []
    all_vps = []
    all_game_ids = []

    for gid, players in game_players.items():
        board_layout_json = game_boards[gid]
        tile_info, port_map = parse_board(board_layout_json)
        board_layout = json.loads(board_layout_json)

        tile_resource = np.zeros(19, dtype=np.int64)
        tile_dicenum = np.zeros(19, dtype=np.int64)
        for tid in range(19):
            tile = tile_info[tid]
            res = tile["resource"]
            num = tile["number"]
            if res is None:
                tile_resource[tid] = 5
                tile_dicenum[tid] = 0
            else:
                tile_resource[tid] = RESOURCE_TO_ID[res]
                tile_dicenum[tid] = num - 1

        port_resources = []
        for loc in board_layout:
            if loc["tile"]["type"] == "PORT":
                res = loc["tile"].get("resource")
                if res is None:
                    port_resources.append(5)
                else:
                    port_resources.append(PORT_TYPE_TO_ID[res])
        port_resource = np.array(port_resources, dtype=np.int64)

        n = len(players)
        parsed = [parse_placements(p["placements_json"]) for p in players]

        # all 8 placements (both rounds, all players)
        all_placements = []
        for i in range(n):
            for rnd in range(2):
                snode = parsed[i][rnd]["settlement_node"]
                redge = parsed[i][rnd]["road_edge"]
                all_placements.append((i, snode, redge))

        all_settlements = [snode for (_, snode, _) in all_placements]

        for pi in range(n):
            settlement = parsed[pi][1]["settlement_node"]
            road = parsed[pi][1]["road_edge"]

            # all 7 other settlements (exclude this player's round 1)
            opponent_settlements = [
                s for idx, s in enumerate(all_settlements) if idx != pi * 2 + 1
            ]

            hand_feat = build_feature_vector(
                settlement,
                road,
                tile_info,
                port_map,
                opponent_settlements,
                pi,
                1,
            )

            struct_owner = np.zeros(MAX_STRUCTS, dtype=np.int64)
            struct_type = np.zeros(MAX_STRUCTS, dtype=np.int64)
            struct_pos = np.zeros(MAX_STRUCTS, dtype=np.int64)
            struct_mask = np.ones(MAX_STRUCTS, dtype=bool)

            road_owner = np.zeros(MAX_ROADS, dtype=np.int64)
            road_a_arr = np.zeros(MAX_ROADS, dtype=np.int64)
            road_b_arr = np.zeros(MAX_ROADS, dtype=np.int64)
            road_mask = np.ones(MAX_ROADS, dtype=bool)

            for idx, (owner, snode, redge) in enumerate(all_placements):
                struct_owner[idx] = (owner - pi) % n
                struct_pos[idx] = snode
                struct_mask[idx] = False

                road_owner[idx] = (owner - pi) % n
                road_a_arr[idx] = redge[0]
                road_b_arr[idx] = redge[1]
                road_mask[idx] = False

            acc_tile_resource.append(tile_resource)
            acc_tile_dicenum.append(tile_dicenum)
            acc_tile_pos.append(tile_pos_fixed)
            acc_port_resource.append(port_resource)
            acc_port_pos.append(port_pos_fixed)
            acc_struct_owner.append(struct_owner)
            acc_struct_type.append(struct_type)
            acc_struct_pos.append(struct_pos)
            acc_struct_mask.append(struct_mask)
            acc_road_owner.append(road_owner)
            acc_road_a.append(road_a_arr)
            acc_road_b.append(road_b_arr)
            acc_road_mask.append(road_mask)
            acc_hand_features.append(hand_feat)
            all_vps.append(players[pi]["vps"])
            all_game_ids.append(gid)

    data = {
        "tile_resource": np.array(acc_tile_resource),
        "tile_dicenum": np.array(acc_tile_dicenum),
        "tile_pos": np.array(acc_tile_pos),
        "port_resource": np.array(acc_port_resource),
        "port_pos": np.array(acc_port_pos),
        "struct_owner": np.array(acc_struct_owner),
        "struct_type": np.array(acc_struct_type),
        "struct_pos": np.array(acc_struct_pos),
        "struct_mask": np.array(acc_struct_mask),
        "road_owner": np.array(acc_road_owner),
        "road_a": np.array(acc_road_a),
        "road_b": np.array(acc_road_b),
        "road_mask": np.array(acc_road_mask),
        "hand_features": np.array(acc_hand_features),
    }
    y_out = np.array(all_vps, dtype=np.float32)
    game_ids_out = np.array(all_game_ids)

    return data, y_out, game_ids_out


def _hand_feats_for_placement(
    settlement, road, tile_info, port_map, opponent_settlements, player_index, rnd
):
    """Hand-engineered features for a single placement (placement + opponent + turn)."""
    placement_feats = extract_placement_features(settlement, road, tile_info, port_map)
    opponent_feats = extract_opponent_features(
        settlement, tile_info, opponent_settlements
    )
    turn_feats = extract_turn_order_features(player_index, rnd)
    return np.concatenate([placement_feats, opponent_feats, turn_feats])


# Number of hand features per placement round (21 + 7 + 6)
HAND_FEATS_PER_ROUND = 34


def build_joint_winner_dataset():
    """
    Build dataset for the joint 4-player winner prediction transformer.

    One sample per game. Includes all 8 placements (full visibility) for
    training/benchmarking. The model architecture supports variable numbers
    of placements via masking, so at inference time fewer placements can be
    provided to evaluate candidate positions.

    Token ordering for structs/roads follows the snake draft:
        round 0: player 0, 1, 2, 3
        round 1: player 3, 2, 1, 0

    Owner IDs are absolute (0-3).

    Hand features are (4, 2*F) per game: for each player, round-0 and round-1
    features concatenated. F = HAND_FEATS_PER_ROUND = 34.

    Returns:
        data: dict of numpy arrays:
            tile_resource:  (n_games, 19)
            tile_dicenum:   (n_games, 19)
            tile_pos:       (n_games, 19)
            port_resource:  (n_games, 9)
            port_pos:       (n_games, 9)
            struct_owner:   (n_games, 8)   absolute player ID 0-3
            struct_type:    (n_games, 8)   all 0 (settlements)
            struct_pos:     (n_games, 8)   node IDs (0-53)
            struct_mask:    (n_games, 8)   True=padding (all False here)
            road_owner:     (n_games, 8)
            road_a:         (n_games, 8)
            road_b:         (n_games, 8)
            road_mask:      (n_games, 8)   True=padding (all False here)
            hand_features:  (n_games, 4, 2*F) per-player features
        winner: (n_games,) int64, winner player index (0-3)
        game_ids: (n_games,) game ID strings
    """
    RESOURCE_TO_ID = {"BRICK": 0, "WOOD": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
    PORT_TYPE_TO_ID = {"BRICK": 0, "WOOD": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
    MAX_STRUCTS = 8
    MAX_ROADS = 8
    F = HAND_FEATS_PER_ROUND

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

    tile_pos_fixed = np.arange(19, dtype=np.int64)
    port_pos_fixed = np.arange(9, dtype=np.int64)

    acc_tile_resource = []
    acc_tile_dicenum = []
    acc_tile_pos = []
    acc_port_resource = []
    acc_port_pos = []
    acc_struct_owner = []
    acc_struct_type = []
    acc_struct_pos = []
    acc_struct_mask = []
    acc_road_owner = []
    acc_road_a = []
    acc_road_b = []
    acc_road_mask = []
    acc_hand_features = []
    all_winners = []
    all_game_ids = []

    for gid, players in game_players.items():
        if gid not in game_boards:
            continue
        n = len(players)
        if n != 4:
            continue

        board_layout_json = game_boards[gid]
        tile_info, port_map = parse_board(board_layout_json)
        board_layout = json.loads(board_layout_json)

        # tile tokens
        tile_resource = np.zeros(19, dtype=np.int64)
        tile_dicenum = np.zeros(19, dtype=np.int64)
        for tid in range(19):
            tile = tile_info[tid]
            res = tile["resource"]
            num = tile["number"]
            if res is None:
                tile_resource[tid] = 5
                tile_dicenum[tid] = 0
            else:
                tile_resource[tid] = RESOURCE_TO_ID[res]
                tile_dicenum[tid] = num - 1

        # port tokens
        port_resources = []
        for loc in board_layout:
            if loc["tile"]["type"] == "PORT":
                res = loc["tile"].get("resource")
                if res is None:
                    port_resources.append(5)
                else:
                    port_resources.append(PORT_TYPE_TO_ID[res])
        port_resource = np.array(port_resources, dtype=np.int64)

        parsed = [parse_placements(p["placements_json"]) for p in players]

        # snake draft order: r0 = [0,1,2,3], r1 = [3,2,1,0]
        draft_order = list(range(n)) + list(range(n - 1, -1, -1))

        # collect all settlement nodes for opponent feature computation
        all_settlements = []
        for pi in range(n):
            all_settlements.append(parsed[pi][0]["settlement_node"])
            all_settlements.append(parsed[pi][1]["settlement_node"])

        # structure and road tokens in draft order
        struct_owner = np.zeros(MAX_STRUCTS, dtype=np.int64)
        struct_type = np.zeros(MAX_STRUCTS, dtype=np.int64)
        struct_pos = np.zeros(MAX_STRUCTS, dtype=np.int64)
        struct_mask = np.zeros(MAX_STRUCTS, dtype=bool)  # all real
        road_owner = np.zeros(MAX_ROADS, dtype=np.int64)
        road_a_arr = np.zeros(MAX_ROADS, dtype=np.int64)
        road_b_arr = np.zeros(MAX_ROADS, dtype=np.int64)
        road_mask = np.zeros(MAX_ROADS, dtype=bool)  # all real

        for idx, pi in enumerate(draft_order):
            rnd = 0 if idx < n else 1
            struct_owner[idx] = pi
            struct_pos[idx] = parsed[pi][rnd]["settlement_node"]
            road_owner[idx] = pi
            road_a_arr[idx] = parsed[pi][rnd]["road_edge"][0]
            road_b_arr[idx] = parsed[pi][rnd]["road_edge"][1]

        # per-player hand features: both rounds concatenated, (4, 2*F)
        hand_feats = np.zeros((4, 2 * F))
        for pi in range(n):
            for rnd in range(2):
                settlement = parsed[pi][rnd]["settlement_node"]
                road = parsed[pi][rnd]["road_edge"]
                settle_idx = 2 * pi + rnd
                opp = [s for k, s in enumerate(all_settlements) if k != settle_idx]

                feats = _hand_feats_for_placement(
                    settlement,
                    road,
                    tile_info,
                    port_map,
                    opp,
                    pi,
                    rnd,
                )
                hand_feats[pi, rnd * F : (rnd + 1) * F] = feats

        # winner = player with max VPs
        vps = [p["vps"] for p in players]
        winner = int(np.argmax(vps))

        acc_tile_resource.append(tile_resource)
        acc_tile_dicenum.append(tile_dicenum)
        acc_tile_pos.append(tile_pos_fixed)
        acc_port_resource.append(port_resource)
        acc_port_pos.append(port_pos_fixed)
        acc_struct_owner.append(struct_owner)
        acc_struct_type.append(struct_type)
        acc_struct_pos.append(struct_pos)
        acc_struct_mask.append(struct_mask)
        acc_road_owner.append(road_owner)
        acc_road_a.append(road_a_arr)
        acc_road_b.append(road_b_arr)
        acc_road_mask.append(road_mask)
        acc_hand_features.append(hand_feats)
        all_winners.append(winner)
        all_game_ids.append(gid)

    data = {
        "tile_resource": np.array(acc_tile_resource),
        "tile_dicenum": np.array(acc_tile_dicenum),
        "tile_pos": np.array(acc_tile_pos),
        "port_resource": np.array(acc_port_resource),
        "port_pos": np.array(acc_port_pos),
        "struct_owner": np.array(acc_struct_owner),
        "struct_type": np.array(acc_struct_type),
        "struct_pos": np.array(acc_struct_pos),
        "struct_mask": np.array(acc_struct_mask),
        "road_owner": np.array(acc_road_owner),
        "road_a": np.array(acc_road_a),
        "road_b": np.array(acc_road_b),
        "road_mask": np.array(acc_road_mask),
        "hand_features": np.array(acc_hand_features),  # (n_games, 4, 2*F)
    }
    winner = np.array(all_winners, dtype=np.int64)
    game_ids = np.array(all_game_ids)

    return data, winner, game_ids
