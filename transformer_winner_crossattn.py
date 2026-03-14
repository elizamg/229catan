import torch
import torch.nn as nn

from transformer import BoardEmbedding


class CatanWinnerTransformerCrossAttn(nn.Module):
    """Joint 4-player winner prediction transformer with cross-attention + hand features.

    Head design:
        1. Encode all tokens with the transformer encoder.
        2. Project each player's hand features into a query vector.
        3. Cross-attend: player query attends over encoded board tokens →
           player-specific board summary (d_embed).
        4. Concat [cross_attn_output, hand_features] → shared MLP → 1 logit.
        5. Stack 4 logits → softmax CE.
    """

    def __init__(self, n_hand_feats_per_player, d_embed=64, dropout=0.2):
        super().__init__()
        self.boardEmbedding = BoardEmbedding(d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # project hand features → query for cross-attention
        self.query_proj = nn.Linear(n_hand_feats_per_player, d_embed)

        # cross-attention: player query attends over board tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_embed, num_heads=4, dropout=dropout, batch_first=True
        )

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
        )

        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)

        # per-player logits via cross-attention + MLP
        logits = []
        for pi in range(4):
            # project hand features to query: (B, F) → (B, 1, d_embed)
            query = self.query_proj(hand_features[:, pi, :]).unsqueeze(1)

            # cross-attend over board tokens
            attn_out, _ = self.cross_attn(
                query, encoded, encoded, key_padding_mask=padding_mask
            )  # (B, 1, d_embed)
            attn_out = attn_out.squeeze(1)  # (B, d_embed)

            combined = torch.cat([attn_out, hand_features[:, pi, :]], dim=1)
            logits.append(self.player_mlp(combined))

        return torch.cat(logits, dim=1)  # (B, 4)
