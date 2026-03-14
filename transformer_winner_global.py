import torch
import torch.nn as nn

from transformer import BoardEmbedding


class CatanWinnerTransformerGlobal(nn.Module):
    """Joint 4-player winner prediction transformer with global pooling + hand features.

    Head design:
        1. Encode all tokens with the transformer.
        2. Global mean-pool all unmasked tokens → board state vector (d_embed).
        3. For each player i, concat [board_state, hand_features_i] → shared MLP → 1 logit.
        4. Stack 4 logits → softmax CE.

    The global pool captures board-level context; hand features provide per-player
    differentiation. This separation of concerns gives the best generalization.
    """

    def __init__(self, n_hand_feats_per_player, d_embed=64, dropout=0.2):
        super().__init__()
        self.boardEmbedding = BoardEmbedding(d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.pool_dropout = nn.Dropout(dropout)
        self.player_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_embed + n_hand_feats_per_player, 64),
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
        hand_features,  # (B, 4, F)
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

        # global mean pool over unmasked tokens
        mask_expanded = padding_mask.unsqueeze(-1)
        encoded = encoded.masked_fill(mask_expanded, 0.0)
        token_counts = (~padding_mask).sum(dim=1, keepdim=True)
        board_state = encoded.sum(dim=1) / token_counts  # (B, d_embed)
        board_state = self.pool_dropout(board_state)

        # per-player logits via shared MLP
        logits = []
        for pi in range(4):
            combined = torch.cat([board_state, hand_features[:, pi, :]], dim=1)
            logits.append(self.player_mlp(combined))

        return torch.cat(logits, dim=1)  # (B, 4)
