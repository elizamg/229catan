
import json
import os
import csv

def parse_initial_placements(actions):
    player_placements = {"WHITE": [], "RED": [], "BLUE": [], "ORANGE": []}
    for action in actions:
        if action[0][0] == "WHITE":
            player_placements["WHITE"].append(action)
        elif action[0][0] == "BLUE":
            player_placements["BLUE"].append(action)
        elif action[0][0] == "RED":
            player_placements["RED"].append(action)
        else:
            player_placements["ORANGE"].append(action)
    return player_placements

def parse_vps(vps, colors):
    player_vps = {}
    player_vps = {colors[0]: vps['P0_ACTUAL_VICTORY_POINTS'], colors[1]: vps['P1_ACTUAL_VICTORY_POINTS'], colors[2]: vps['P2_ACTUAL_VICTORY_POINTS'], colors[3]: vps['P3_ACTUAL_VICTORY_POINTS']}
    return player_vps

def main():
    game_data = []
    player_data = []

    for file in os.listdir('data'):

        with open(f'data/{file}', 'r') as f:
            if not file.endswith('.json'):
                continue
            data = json.load(f)

        game_id = file.split('.')[0]

        colors = data['colors']

        # parse the necessary data from json and save as a csv
        # board layout

        board_layout = data['tiles']

        winner = data['winning_color']

        game_data.append({
            'game_id': game_id,
            'board_layout': json.dumps(board_layout),
            'winner': winner
        })

        initial_placements = []

        for action in data['action_records']:
            if action[0][1] != 'ROLL':
                initial_placements.append(action)
            else:
                break


        player_placements = parse_initial_placements(initial_placements)

        
        player_vps = parse_vps(data['player_state'], colors)

        # add entries to csvs
        for player in colors:
            player_data.append({
                'game_id': game_id,
                'player': player,
                'initial_placements': json.dumps(player_placements[player]),
                'vps': player_vps[player]
            })
    with open('games.csv', 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['game_id', 'board_layout', 'winner']
        )
        writer.writeheader()
        writer.writerows(game_data)
    with open('players.csv', 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['game_id', 'player', 'initial_placements', 'vps']
        )
        writer.writeheader()
        writer.writerows(player_data)


if __name__ == '__main__':
    main()