import torch
import torch.nn as nn

from transformer import BoardEmbedding


class CatanWinnerTransformer(nn.Module):
    """Joint 4-player winner prediction transformer.

    Reuses BoardEmbedding and TransformerEncoder from the VP regression model.
    Supports variable numbers of placement tokens via masking.

    Head design:
        1. Encode all tokens with the transformer.
        2. Pool tiles+ports → board_pool (shared across players).
        3. For each player i, pool their struct/road tokens → player_pool.
        4. concat [player_pool, board_pool, hand_features_i] → shared MLP → 1 logit.
        5. Stack 4 logits → softmax CE.

    The shared MLP ensures the model is equivariant across player seats:
    player identity comes only from the tokens and hand features, not the head weights.
    """

    def __init__(self, n_hand_feats_per_player, d_embed=64):
        super().__init__()
        self.boardEmbedding = BoardEmbedding(d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.player_mlp = nn.Sequential(
            nn.Linear(2 * d_embed + n_hand_feats_per_player, 64),
            nn.ReLU(),
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
        hand_features,  # (B, 4, F)
        struct_mask,
        road_mask,
    ):
        batch_size = tile_resource.shape[0]

        # tiles and ports are always present (no masking)
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
        struct_encoded = encoded[:, 28:36, :]  # (B, 8, d_embed)
        road_encoded = encoded[:, 36:44, :]  # (B, 8, d_embed)

        logits = []
        for pi in range(4):
            struct_belongs = (struct_owner == pi) & (~struct_mask)  # (B, 8)
            road_belongs = (road_owner == pi) & (~road_mask)  # (B, 8)

            s_mask = struct_belongs.unsqueeze(-1).float()
            r_mask = road_belongs.unsqueeze(-1).float()

            player_sum = (struct_encoded * s_mask).sum(dim=1) + (road_encoded * r_mask).sum(dim=1)
            player_count = s_mask.sum(dim=1) + r_mask.sum(dim=1)
            player_pool = player_sum / player_count.clamp(min=1.0)  # (B, d_embed)

            combined = torch.cat([player_pool, board_pool, hand_features[:, pi, :]], dim=1)
            logits.append(self.player_mlp(combined))

        return torch.cat(logits, dim=1)  # (B, 4)
