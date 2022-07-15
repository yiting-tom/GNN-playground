from typing import Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, root_validator

class ConfGATLayer(BaseModel):
    """ConfGATLayer class

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        n_heads (int): number of heads
        dropout (float): dropout probability
        alpha (float): the negative slope of the leaky relu
        concat (bool): whether to concatenate the output of different heads
        original_attn (bool): use the original method of GAT or not (default: False),
            if not, use the broadcast addition method.

    Raises:
        ValueError: When `concat` is True and `out_features` cannot divide `n_head`
    """
    in_features: int
    out_features: int
    n_head: int
    dropout_rate: float
    alpha: float
    concat: bool
    original_attn: bool = False
    shared_weight: bool = False

    @root_validator
    def check_n_head_divide_by_out_features(cls, v: dict) -> dict:
        if v['concat'] and v['out_features'] % v['n_head'] != 0:
            raise ValueError(
                f'n_head ({v["n_head"]}) should be a multiple of out_features ({v["out_features"]}).'
            )
        return v

    @root_validator
    def check_shared_weight(cls, v: dict) -> dict:
        if v['shared_weight'] and v['original_attn']:
            raise ValueError(
                f'original_attn and share_weight cannot be both True.'
            )
        return v


class GATLayer(pl.LightningModule):
    """GATLayer class

    paper: https://arxiv.org/pdf/1710.10903v3.pdf

    Will do following:
    1. embedding (h -> emb)
    2. repeat, repeat_interleave then concatenate (emb -> concatenated)
    3. attention + activation (concatenated -> e)
    4. masked fill (e, adj -> e)
    5. softmax + dropout (e -> attn)
    6. attention * emb (attn, emb -> out)
    8. concatenate or average (out -> out)

    Args:
        c (ConfGATLayer): The configuration of the GATLayer.
    """

    def __init__(self, c: ConfGATLayer):
        super().__init__()
        self.c = c

        # concat or average
        if c.concat:
            assert c.out_features % c.n_head == 0
            self.n_hidden = c.out_features // c.n_head
        else:
            self.n_hidden = c.out_features

        # the attention method
        if c.original_attn:
            self.attn_process = self.__original_attn_process
            self.attn_in_dim = self.n_hidden * 2
            # Embedding
            self.W = nn.Linear(
                in_features=c.in_features,
                out_features=self.n_hidden * c.n_head,
                bias=False,
            )
            nn.init.xavier_normal_(self.W.weight)

        else:
            self.attn_process = self.__broadcast_addition_attn_process
            self.attn_in_dim = self.n_hidden
            self.Wl = nn.Linear(
                in_features=c.in_features,
                out_features=self.n_hidden * c.n_head,
                bias=False,
            )
            nn.init.xavier_normal_(self.Wl.weight)

            # would be the embedding
            self.Wr = nn.Linear(
                in_features=c.in_features,
                out_features=self.n_hidden * c.n_head,
                bias=False,
            ) if not c.shared_weight else self.Wl
            nn.init.xavier_normal_(self.Wr.weight)

        # Attention
        self.attn = nn.Linear(
            in_features=self.attn_in_dim,
            out_features=1,
            bias=False,
        )
        nn.init.xavier_normal_(self.attn.weight)

        # Activation
        self.activate = nn.LeakyReLU(
            negative_slope=c.alpha,
            inplace=True,
        )

        # Dropout
        self.dropout = nn.Dropout(
            p=c.dropout_rate,
            inplace=True,
        )


    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """forward

        Parameters
        ----------
        h : torch.Tensor
            The hidden state of the previous layer.
        adj : torch.Tensor
            The adjacency matrix must with shape (N, N, n_heads) or (N, N, 1).
        """
        # The number of nodes.
        N = h.shape[0]

        # The adjacency matrix should have shape
        # [n_nodes, n_nodes, n_heads] or [n_nodes, n_nodes, 1]
        assert adj.shape[0] == 1 or adj.shape[0] == N
        assert adj.shape[1] == 1 or adj.shape[1] == N
        assert adj.shape[2] == 1 or adj.shape[2] == self.c.n_heads

        # emb = [N, n_head, n_hidden] (where n_hidden = out_features // n_head)
        # e (attn score) = [N, N, out_features, 1]
        emb, e = self.attn_process(h)

        # shape = [N, N, out_features]
        e.squeeze_(-1)

        # masking
        e.masked_fill_(
            mask=adj == 0,
            value=-torch.inf,
        )

        # shape = [N, N, n_head]
        attn = self.dropout(F.softmax(e, dim=1))

        # attn = [N, N, n_head]
        # emb = [N, n_head, n_hidden] (where n_hidden = out_features // n_head)
        # out = [N, n_head, n_hidden]
        out = torch.einsum('ijk, jkl -> ikl', attn, emb)

        if self.c.concat:
            # shape = [N, n_head * n_hidden]
            return out.reshape(N, self.c.n_head * self.n_hidden)

        # shape = [N, n_head * n_hidden]
        return out.mean(dim=1)
    
    def __original_attn_process(
        self,
        h: torch.Tensor     # x = [N, n_head, n_hidden]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """__original_attn_process method

        Args:
            h (torch.Tensor): The hidden state with shape (N, n_head, n_hidden).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                First is the embedding with shape (N, n_head, n_hidden),
                second is the attention score with shape (N, N, n_head, 1).
        """
        N = h.shape[0]

        # emb = [N, n_head, n_hidden]
        emb = self.W(h).view(N, self.c.n_head, self.n_hidden)

        # Repeat the matrix N times
        # Assume the features = {f1, f2, ..., fn}, each fi with shape [n_head, n_hidden]
        # input: [f1, f2, f3, ..., fk]
        # output: [f1, f2, ..., fk, f1, f2, ..., fk, ..., fk]
        # shape = [N*N, n_head, n_hidden]
        repeated = emb.repeat(N, 1, 1)

        # Interleave repeat the matrix N times
        # Assume the features = {f1, f2, ..., fn}, each fi with shape [n_head, n_hidden]
        # input: [f1, f2, f3, ..., fk]
        # output: [f1, f1, ...,f1, f2, f2, ..., f2, f3, f3, ..., f3, ..., fk, fk, ..., fk]
        # shape = [N*N, n_head, n_hidden]
        interleaved = emb.repeat_interleave(N, dim=0)

        # Concatenate the matrix
        # shape = [N*N, n_head, 2*n_hidden]
        concat = torch.cat(
            tensors=[repeated, interleaved],
            dim=-1,
        )

        # reshape to [N, N, n_head, 2*n_hidden]
        # N, N: is for adjacency matrix
        # n_head: is for the number of heads
        # 2*n_hidden: is for g_i || g_j
        concatenated = concat.view(N, N, self.c.n_head, 2 * self.n_hidden)

        # self.attn: 2*n_hidden -> 1
        e = self.activate(self.attn(concatenated))

        # emb = [N, n_head, n_hidden] (where n_hidden = out_features // n_head)
        # e (attention score) = [N, N, n_head, 1]
        return emb, e

    def __broadcast_addition_attn_process(
        self,
        h: torch.Tensor     # x = [N, n_head, n_hidden]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """__broadcast_addition_attn_process method

        Args:
            h (torch.Tensor): The hidden state with shape (N, n_head, n_hidden).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                First is the embedding with shape (N, n_head, n_hidden),
                second is the attention score with shape (N, N, n_head, 1).
        """
        N = h.shape[0]

        gl = self.Wl(h).view(N, self.c.n_head, self.n_hidden)
        gr = self.Wr(h).view(N, self.c.n_head, self.n_hidden)
        emb = gr

        # [N, n_head, n_hidden]
        repeated = gl.repeat(N, 1, 1)

        # [N, n_head, n_hidden]
        interleaved = gr.repeat_interleave(N, dim=0)

        # Broadcast the matrix
        # |-repeated--|
        # . . . . . . .  -
        # . . . . . . .  |
        # . . . . . . .  |
        # . . . . . . .  | interleaved
        # . . . . . . .  |
        # . . . . . . .  |
        # . . . . . . .  -
        # g11 + g11, g11 + g12, ..., g11 + g1n, g12 + g12, ..., g12 + g1n, ..., g1n + g1n
        # shape = [N, N, n_head, n_hidden]
        g_sum = repeated + interleaved

        # reshape to [N, N, n_head, 2*n_hidden]
        g_sum = g_sum.view(N, N, self.c.n_head, self.n_hidden)

        # self.attn: n_hidden -> 1
        # attention score = [N, N, n_head, 1]
        e = self.attn(self.activate(g_sum))

        # emb = [N, n_head, n_hidden] (where n_hidden = out_features // n_head)
        # e (attention score) = [N, N, n_head, 1]
        return emb, e