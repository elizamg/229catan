import torch
import torch.nn as nn

from transformer import BoardEmbedding


class CatanWinnerTransformerRaw(nn.Module):
    """Joint 4-player winner prediction transformer without hand-engineered features.

    Same BoardEmbedding and encoder as CatanWinnerTransformer, but the head
    uses per-player token pooling instead of hand features to differentiate
    players.

    Head design:
        1. Encode all tokens (tiles, ports, structs, roads) with the transformer.
        2. For each player i, pool that player's struct/road tokens → player_pool.
        3. Also pool tiles+ports → board_pool (shared across players).
        4. concat [player_pool, board_pool] → shared MLP → 1 logit per player.
        5. Stack 4 logits → softmax CE.

    Supports variable visibility via struct_mask / road_mask. If a player has
    no visible tokens, their player_pool is a zero vector.
    """

    def __init__(self, d_embed=64, dropout=0.1):
        super().__init__()
        self.d_embed = d_embed
        self.boardEmbedding = BoardEmbedding(d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.head_dropout = nn.Dropout(dropout)
        self.player_mlp = nn.Sequential(
            nn.Linear(2 * d_embed, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        tile_resource,
        tile_dicenum,
        tile_pos,
        port_resource,
        port_pos,
        struct_owner,
        struct_type,
        struct_pos,
        road_owner,
        road_a,
        road_b,
        struct_mask,
        road_mask,
    ):
        batch_size = tile_resource.shape[0]

        tile_port_mask = torch.zeros(batch_size, 28, dtype=torch.bool, device=tile_resource.device)
        padding_mask = torch.cat([tile_port_mask, struct_mask, road_mask], dim=1)

        embedded = self.boardEmbedding(
            tile_resource,
            tile_dicenum,
            tile_pos,
            port_resource,
            port_pos,
            struct_owner,
            struct_type,
            struct_pos,
            road_owner,
            road_a,
            road_b,
        )  # (B, 44, d_embed)

        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)

        # zero out padding positions
        mask_expanded = padding_mask.unsqueeze(-1)
        encoded = encoded.masked_fill(mask_expanded, 0.0)

        # board pool: tiles (0-18) + ports (19-27), always present
        board_pool = encoded[:, :28, :].mean(dim=1)  # (B, d_embed)

        # per-player pool from their struct/road tokens
        # struct tokens are at indices 28-35, road tokens at 36-43
        # struct_owner[i] and road_owner[i] give the absolute player ID (0-3)
        logits = []
        for pi in range(4):
            # find this player's unmasked struct tokens
            struct_belongs = (struct_owner == pi) & (~struct_mask)  # (B, 8)
            road_belongs = (road_owner == pi) & (~road_mask)  # (B, 8)

            # extract and pool struct tokens for this player
            struct_encoded = encoded[:, 28:36, :]  # (B, 8, d_embed)
            road_encoded = encoded[:, 36:44, :]  # (B, 8, d_embed)

            # mask: (B, 8, 1) for broadcasting
            s_mask = struct_belongs.unsqueeze(-1).float()
            r_mask = road_belongs.unsqueeze(-1).float()

            player_sum = (struct_encoded * s_mask).sum(dim=1) + (road_encoded * r_mask).sum(dim=1)
            player_count = s_mask.sum(dim=1) + r_mask.sum(dim=1)  # (B, 1)

            # avoid division by zero for players with no visible tokens
            player_pool = player_sum / player_count.clamp(min=1.0)  # (B, d_embed)

            combined = torch.cat([player_pool, board_pool], dim=1)  # (B, 2*d_embed)
            combined = self.head_dropout(combined)
            logits.append(self.player_mlp(combined))

        return torch.cat(logits, dim=1)  # (B, 4)
