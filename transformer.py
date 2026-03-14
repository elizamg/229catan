import torch
import torch.nn as nn


class BoardEmbedding(nn.Module):
    def __init__(self, d_embed=64):
        super().__init__()
        self.tiletype_embed = nn.Embedding(4, d_embed)  # tile, port, settlement, road

        # for tile tokens
        self.resource_embed = nn.Embedding(6, d_embed)  # 5 tile resources + desert
        self.dicenum_embed = nn.Embedding(12, d_embed)  # 2-12 + NONE
        self.position_embed = nn.Embedding(19, d_embed)  # 19 tile postions

        # for port tokens
        self.port_resource_embed = nn.Embedding(6, d_embed)  # 5 port resources + ANY
        self.port_position_embed = nn.Embedding(9, d_embed)  # 9 port positions

        # for structure (settlement/city) tokens
        self.owner_embed = nn.Embedding(4, d_embed)  # 4 players
        self.structure_type_embed = nn.Embedding(2, d_embed)  # settlement, city
        self.node_pos_embed = nn.Embedding(
            54, d_embed
        )  # 54 potential placement locations

        # for road tokens
        # share owner_embed
        # shared node_pos_embed to represent road endpoints

    def tile_forward(self, resource, dicenum, tile_pos_id):
        """
        resource: brick=0, wood=1, sheep=2, wheat=3, ore=4, desert=5
        position: tile id (0-18)
        dicenum: the dice number, 0 for NONE, or raw dice number minuse one
        """
        return (
            self.resource_embed(resource)
            + self.dicenum_embed(dicenum)
            + self.position_embed(tile_pos_id)
            + self.tiletype_embed(torch.zeros_like(resource))
        )

    def port_forward(self, resource, position):
        """
        resource: brick=0, wood=1, sheep=2, wheat=3, ore=4, desert=5
        position: port id (0-8)
        """
        return (
            self.port_resource_embed(resource)
            + self.port_position_embed(position)
            + self.tiletype_embed(torch.ones_like(resource))
        )

    def structure_forward(self, owner, structure_type, position):
        """
        owner: current player is 0, opponents are 1-3
        type: settlement=0, city=1
        position: node id (0-53)
        """
        return (
            self.owner_embed(owner)
            + self.structure_type_embed(structure_type)
            + self.node_pos_embed(position)
            + self.tiletype_embed(torch.ones_like(owner) * 2)
        )

    def road_forward(self, owner, endpoint_a, endpoint_b):
        """
        owner: current player is 0, opponents are 1-3
        endpoint_a: node at one end of the road, node id (0-53)
        endpoint_b: node at the other end of the road, node id (0-53)
        """
        return (
            self.owner_embed(owner)
            + self.node_pos_embed(endpoint_a)
            + self.node_pos_embed(endpoint_b)
            + self.tiletype_embed(torch.ones_like(owner) * 3)
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
    ):

        tiles = self.tile_forward(tile_resource, tile_dicenum, tile_pos)
        ports = self.port_forward(port_resource, port_pos)
        structures = self.structure_forward(struct_owner, struct_type, struct_pos)
        roads = self.road_forward(road_owner, road_a, road_b)

        return torch.cat(
            [tiles, ports, structures, roads], dim=1
        )  # (batch, 44, d_embed)


class CatanTransformer(nn.Module):
    def __init__(self, n_hand_feats, d_embed=64):
        super().__init__()
        self.boardEmbedding = BoardEmbedding(d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_embed + n_hand_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
        hand_features,
        struct_mask,
        road_mask
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
        mask_expanded = padding_mask.unsqueeze(-1)
        encoded = encoded.masked_fill(mask_expanded, 0.0)
        token_counts = (~padding_mask).sum(dim=1, keepdim=True)
        pooled = encoded.sum(dim=1) / token_counts
        with_hand_feats = torch.cat([pooled, hand_features], dim=1)
        return self.mlp_head(with_hand_feats)
