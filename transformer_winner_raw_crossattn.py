import torch
import torch.nn as nn

from transformer_concat import BoardEmbeddingConcat


class CatanWinnerRawCrossAttn(nn.Module):
    """Winner prediction from raw board tokens only (no hand features).

    Uses learnable player query embeddings for cross-attention:
    each player gets a learned query vector that attends over the
    encoded board tokens to produce a player-specific board summary.
    The model must learn all player-differentiating information from
    the token embeddings (especially owner_embed in structure/road tokens).
    """

    def __init__(self, d_embed=16, n_heads=2, n_layers=1, dropout=0.0):
        super().__init__()
        self.boardEmbedding = BoardEmbeddingConcat(d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=n_heads, dim_feedforward=4 * d_embed,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        # learnable query per player
        self.player_queries = nn.Embedding(4, d_embed)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_embed, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.player_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 1),
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
        return_attn_weights=False,
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

        player_ids = torch.arange(4, device=tile_resource.device)

        logits = []
        attn_weights_list = []
        for pi in range(4):
            # learnable query for this player: (1, d_embed) → (B, 1, d_embed)
            query = self.player_queries(player_ids[pi]).unsqueeze(0).unsqueeze(0)
            query = query.expand(batch_size, -1, -1)

            attn_out, attn_w = self.cross_attn(
                query, encoded, encoded, key_padding_mask=padding_mask
            )
            attn_out = attn_out.squeeze(1)  # (B, d_embed)
            logits.append(self.player_mlp(attn_out))
            if return_attn_weights:
                attn_weights_list.append(attn_w.squeeze(1))  # (B, seq_len)

        logits = torch.cat(logits, dim=1)  # (B, 4)
        if return_attn_weights:
            return logits, torch.stack(attn_weights_list, dim=1)
        return logits
