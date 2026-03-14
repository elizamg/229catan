import torch
import torch.nn as nn

from transformer_concat import BoardEmbeddingConcat


class CatanWinnerCrossAttnConcat(nn.Module):
    """Cross-attention winner prediction with concatenated (not summed) embeddings.

    Same architecture as CatanWinnerTransformerCrossAttn but uses BoardEmbeddingConcat
    which concatenates sub-embeddings then projects, keeping each embedding table's
    dimensions isolated for interpretability.
    """

    def __init__(self, n_hand_feats_per_player, d_embed=64, dropout=0.2):
        super().__init__()
        self.boardEmbedding = BoardEmbeddingConcat(d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.query_proj = nn.Linear(n_hand_feats_per_player, d_embed)

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

        logits = []
        attn_weights_list = []
        for pi in range(4):
            query = self.query_proj(hand_features[:, pi, :]).unsqueeze(1)
            attn_out, attn_w = self.cross_attn(
                query, encoded, encoded, key_padding_mask=padding_mask
            )
            attn_out = attn_out.squeeze(1)
            combined = torch.cat([attn_out, hand_features[:, pi, :]], dim=1)
            logits.append(self.player_mlp(combined))
            if return_attn_weights:
                attn_weights_list.append(attn_w.squeeze(1))  # (B, seq_len)

        logits = torch.cat(logits, dim=1)  # (B, 4)
        if return_attn_weights:
            # (B, 4, seq_len)
            return logits, torch.stack(attn_weights_list, dim=1)
        return logits
