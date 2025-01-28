import torch

class FruitFly(torch.nn.Module):
    def __init__(self, K, k, N_vocab, synapses=None):
        super(FruitFly, self).__init__()
        self.K = K
        self.k = k
        self.W = torch.nn.Parameter(torch.randn(K,
                                                N_vocab,
                                                dtype=torch.float32),
                                    requires_grad=True)
        self.N_vocab = N_vocab
        if synapses is not None:
            self.W = torch.nn.Parameter(synapses, requires_grad=True)

    def forward(self, ids, Ps, pos, top_k):
        batch_size = len(Ps)
        window_size = len(ids)//batch_size

        indices = ids.reshape(-1, window_size).T.long()
        W_indices = self.W.T[indices]
        W_indices = torch.permute(W_indices, (1, 2, 0))
        mu = W_indices.sum(dim=-1).argmax(dim=-1)
        W_mu_nonzero = torch.gather(self.W[mu], 1, indices.T)

        out = None
        numerator = (Ps*W_mu_nonzero).sum(dim=-1)
        denominator = torch.linalg.norm(self.W, dim=1)[mu]
        E = -(numerator/denominator).sum()
        return out, E
