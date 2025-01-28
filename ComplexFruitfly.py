import torch


class ComplexFruitFly(torch.nn.Module):
    def __init__(self, K, k, N_vocab, window_size, synapses=None):
        super(ComplexFruitFly, self).__init__()
        self.K = K
        self.k = k
        weights = torch.randn(K, N_vocab, 2, dtype=torch.float32)
        torch.view_as_complex(weights).imag = -torch.view_as_complex(weights).imag.abs()
        self.W = torch.nn.Parameter(weights, requires_grad=True)
        self.N_vocab = N_vocab
        if synapses is not None:
            self.W = torch.nn.Parameter(synapses, requires_grad=True)
            self.K = torch.view_as_complex(self.W).shape[0]
            self.N_vocab = torch.view_as_complex(self.W).shape[1]

    def forward(self, ids, Ps, pos, top_k=1):

        with torch.no_grad():
            window_size = pos.shape[-1]
            re = torch.cos(torch.pi * pos)
            im = torch.sin(torch.pi * pos)
            one_roots = torch.complex(re, im)

            indices = ids.reshape(-1, window_size).T.long()
            W_indices = torch.view_as_complex(self.W).T[indices]
            W_indices = torch.permute(W_indices, (2, 1, 0))
            W_abs = W_indices.abs()
            phis = (W_indices * torch.conj(one_roots)).angle().abs()
            phi_factor = W_abs + phis
            phi_factor = torch.permute(phi_factor, (1, 0, 2))
            mu = phi_factor.sum(dim=-1).argsort(dim=-1, descending=True)[:, :top_k]
            mu = mu.reshape(-1)

        out = mu
        indices = torch.repeat_interleave(indices, top_k, dim=1)
        pos = torch.repeat_interleave(pos, top_k, dim=0)
        one_roots = torch.repeat_interleave(one_roots, top_k, dim=0)
        Ps = torch.repeat_interleave(Ps, top_k, dim=0)

        W_mu_nonzero = torch.gather(torch.view_as_complex(self.W)[mu], 1, indices.T)
        alpha = W_mu_nonzero * torch.conj(Ps * one_roots)
        del W_mu_nonzero
        # alpha.angle() might return nans if the denominator for
        # atan is close to 0, since the weights of some tokens
        # are shrinking we should do something to thresshold them
        r_alpha = torch.view_as_real(alpha)
        treshold = 1e-10
        r_alpha[(r_alpha < treshold) & (r_alpha > -1 * treshold)] = treshold
        alpha = torch.view_as_complex(r_alpha)

        alpha_phi_factor = alpha.angle().abs() * Ps
        alpha_abs = alpha.abs()

        numerator = (alpha_abs).sum(dim=-1)
        denominator = torch.linalg.norm(torch.view_as_complex(self.W), dim=1)[mu]
        E = -(numerator / denominator).sum() - alpha_phi_factor.sum()
        del alpha
        del denominator
        return out, E
