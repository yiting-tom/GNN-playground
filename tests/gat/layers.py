import torch
from models.gat.layers import GATLayer

def test_gat_layers(c_gat_layer, N):
    c = c_gat_layer

    model = GATLayer(c)

    h = torch.randn(N, c.in_features)
    adj = torch.randn(N, N, 1)

    assert model(h, adj).shape == (N, c.out_features)