import torch
import torch.nn as nn


class BoardEmbeddingConcat(nn.Module):
    """Like BoardEmbedding, but concatenates sub-embeddings then projects to d_embed.

    This keeps each embedding table's dimensions isolated, making individual
    tables more interpretable (e.g. resource_embed only encodes resource identity).
    """

    def __init__(self, d_embed=64):
        super().__init__()
        # each sub-embedding has its own dimension d_sub
        # tiles: 4 sub-embeddings (resource, dicenum, position, tiletype)
        # ports: 3 sub-embeddings (resource, position, tiletype)
        # structures: 4 sub-embeddings (owner, type, position, tiletype)
        # roads: 4 sub-embeddings (owner, endpoint_a, endpoint_b, tiletype)
        d_sub = d_embed  # each sub-embedding is d_embed, then projected after concat

        self.tiletype_embed = nn.Embedding(4, d_sub)

        # tile
        self.resource_embed = nn.Embedding(6, d_sub)
        self.dicenum_embed = nn.Embedding(12, d_sub)
        self.position_embed = nn.Embedding(19, d_sub)
        self.tile_proj = nn.Linear(4 * d_sub, d_embed)

        # port
        self.port_resource_embed = nn.Embedding(6, d_sub)
        self.port_position_embed = nn.Embedding(9, d_sub)
        self.port_proj = nn.Linear(3 * d_sub, d_embed)

        # structure
        self.owner_embed = nn.Embedding(4, d_sub)
        self.structure_type_embed = nn.Embedding(2, d_sub)
        self.node_pos_embed = nn.Embedding(54, d_sub)
        self.struct_proj = nn.Linear(4 * d_sub, d_embed)

        # road (shares owner_embed and node_pos_embed)
        self.road_proj = nn.Linear(4 * d_sub, d_embed)

    def tile_forward(self, resource, dicenum, tile_pos_id):
        return self.tile_proj(torch.cat([
            self.resource_embed(resource),
            self.dicenum_embed(dicenum),
            self.position_embed(tile_pos_id),
            self.tiletype_embed(torch.zeros_like(resource)),
        ], dim=-1))

    def port_forward(self, resource, position):
        return self.port_proj(torch.cat([
            self.port_resource_embed(resource),
            self.port_position_embed(position),
            self.tiletype_embed(torch.ones_like(resource)),
        ], dim=-1))

    def structure_forward(self, owner, structure_type, position):
        return self.struct_proj(torch.cat([
            self.owner_embed(owner),
            self.structure_type_embed(structure_type),
            self.node_pos_embed(position),
            self.tiletype_embed(torch.ones_like(owner) * 2),
        ], dim=-1))

    def road_forward(self, owner, endpoint_a, endpoint_b):
        return self.road_proj(torch.cat([
            self.owner_embed(owner),
            self.node_pos_embed(endpoint_a),
            self.node_pos_embed(endpoint_b),
            self.tiletype_embed(torch.ones_like(owner) * 3),
        ], dim=-1))

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

        return torch.cat([tiles, ports, structures, roads], dim=1)  # (batch, 44, d_embed)
