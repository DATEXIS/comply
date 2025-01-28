from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from typing import Dict, Any, List
import torch
import numpy as np


class IMPALAComplexFruitfly(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        K,
        k,
        tokenizer_name="bert-base-uncased",
        model_path=None,
        fs_path=None,
        added_factors=True,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        nn.Module.__init__(self)

        config = AutoConfig.from_pretrained(tokenizer_name)
        self.sample_count = 0
        self.config = model_config
        self.K = K
        self.k = k
        self.added_factors = added_factors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is not None:
            W = torch.load(model_path, map_location=self.device)
            self.W = torch.nn.Parameter(torch.view_as_complex(W))
            self.vocab_size = self.W.shape[-1]
        else:
            self.vocab_size = config.vocab_size
            self.W = torch.nn.Parameter(
                torch.view_as_complex(
                    torch.randn(
                        K,
                        self.vocab_size,
                        2,
                        dtype=torch.float32,
                        device=self.device,
                    )
                ),
                requires_grad=True,
            )

        self.observation_space = obs_space
        self.action_space = action_space
        self.window_size = 11
        self.e_weight = 0.5  # Weight of energy custom loss
        self.activation = nn.GELU()
        self.action_outputs = nn.Linear(
            K, num_outputs, dtype=torch.float32, device=self.device, bias=True
        )
        self.fs_path = fs_path

        self.value_network = nn.Linear(K, 1, dtype=torch.float32, device=self.device)
        if fs_path is not None:
            self.fs = torch.load(fs_path, map_location=self.W.device)
            self.fs = torch.nn.Parameter(self.fs, requires_grad=False)
        else:
            # Acumulated frequencies of tokens
            self.fs = torch.nn.Parameter(
                torch.ones(self.vocab_size), requires_grad=False
            )
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict, state: List, seq_lens: Any):
        inputs = input_dict["obs"]
        ids = inputs.squeeze(1).int()
        window_size = ids.shape[-1]
        non_zeros = ids != 0
        position_ids = non_zeros.cumsum(dim=-1)
        sentence_lengths = non_zeros.sum(-1)
        # rays init dummy batch is filled with zeros so the division will nan
        if non_zeros.sum() == 0:
            pos = torch.pi * position_ids / ids.shape[-1]
        else:
            pos = torch.pi * position_ids / sentence_lengths.unsqueeze(-1)
        re = torch.cos(pos)
        im = torch.sin(pos)
        one_roots = torch.complex(re, im)

        # Compute HASH
        with torch.no_grad():
            indices = ids.reshape(-1, window_size).T.long()
            W_indices = self.W.T[indices]
            W_indices = torch.permute(W_indices, (2, 1, 0))
            W_abs = W_indices.abs()
            phis = W_indices * torch.conj(one_roots)
            r_phis = torch.view_as_real(phis)
            treshold = 1e-10
            r_phis[(r_phis < treshold) & (r_phis > -1 * treshold)] = treshold
            phis = torch.view_as_complex(r_phis)

            phis = phis.angle().abs()
            if self.added_factors:
                phi_factor = (W_abs + phis) * non_zeros
            else:
                phi_factor = (W_abs * phis) * non_zeros

            phi_factor = torch.permute(phi_factor, (1, 0, 2))
            phi_factor = phi_factor.sum(dim=-1)
            binary_hash = torch.zeros_like(
                phi_factor, dtype=torch.bool, device=self.device
            )
            order = phi_factor.argsort(dim=-1, descending=True)
            trues = order[:, : self.k]
            binary_hash = binary_hash.scatter_(
                dim=1,
                index=trues,
                src=torch.ones_like(trues, dtype=torch.bool),
            )
        self._features = binary_hash.float()
        logits = self.action_outputs(self._features)
        return logits, state

    def batch_ids_and_one_roots(self, ids, device):
        non_zeros = ids != 0
        position_ids = non_zeros.cumsum(dim=-1)
        sentence_lengths = non_zeros.sum(-1)
        # rays init dummy batch is filled with zeros so the division will nan
        if non_zeros.sum() == 0:
            pos = torch.pi * position_ids / ids.shape[-1]
        else:
            pos = torch.pi * position_ids / sentence_lengths.unsqueeze(-1)
        re = torch.cos(pos)
        im = torch.sin(pos)
        one_roots = torch.complex(re, im)

        input_ids = ids[ids != 0]
        one_roots = one_roots[ids != 0]

        def unfold_single_sequence(values, window_size, original_ids):
            result = values.unfold(0, window_size, 1).clone().reshape(-1, window_size)
            lens = (original_ids != 0).sum(
                axis=1
            )  # sequence length for every sample in the batch
            offsets = lens.cumsum(
                dim=0
            )  # end of each example when filtering out the [PAD] tokens
            offsets_start = (
                offsets - window_size + 1
            )  # start of the part of the sequence that is
            # incomplete (or merges 2 sequences) after unfold

            all_rows = torch.arange(len(result) + window_size).to(device)
            # Now we'll find the rows of the unfolded matrix that are
            # not needed because they merge 2 samples when ignoring the [PAD]
            greater = all_rows.unsqueeze(0) >= offsets_start.unsqueeze(0).T
            lower = all_rows.unsqueeze(0) < offsets.unsqueeze(0).T
            between = torch.logical_and(lower, greater)
            to_keep = torch.logical_not(between.any(dim=0))
            result = result[to_keep[:-window_size]]
            return result

        input_ids = unfold_single_sequence(input_ids, self.window_size, ids)
        one_roots = unfold_single_sequence(one_roots, self.window_size, ids)
        return input_ids, one_roots.T

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        inputs = loss_inputs["obs"]
        device = inputs.device
        ids = inputs.squeeze(1).int()
        input_ids, one_roots = self.batch_ids_and_one_roots(ids, device)

        # Updating the frequency of the tokens
        if self.sample_count < 1e9 and self.fs_path is not None:
            full_sequence_batch_indices = (
                torch.arange(0, len(ids), dtype=torch.int32)
                .repeat_interleave(loss_inputs["obs"].shape[1])
                .to(loss_inputs["obs"].device)
            )

            full_sequence_coordinates = torch.stack(
                (full_sequence_batch_indices, ids.reshape(-1))
            ).T
            out_V_A_s = torch.sparse_coo_tensor(
                full_sequence_coordinates.T,
                torch.ones_like(full_sequence_coordinates.T[0]),
                (ids.shape[0], self.vocab_size),
                dtype=torch.float32,
            )
            fs = torch.sparse.sum(out_V_A_s, dim=0)
            self.fs[fs.indices().squeeze()] += fs.values()
            self.sample_count += 1

        Ps = 1 / self.fs[ids.reshape(-1)].reshape(ids.shape[0], ids.shape[1])

        indices_batch = (
            torch.arange(0, len(input_ids), dtype=torch.int32)
            .repeat_interleave(self.window_size)
            .to(input_ids.device)
        )
        coordinates = torch.stack((indices_batch, input_ids.reshape(-1))).T

        # Get inverse frequencies for sliding window data
        Ps = (
            1
            / self.fs[coordinates.T[1]]
            .reshape(input_ids.shape[0], input_ids.shape[1])
            .T
        )

        with torch.no_grad():
            indices = input_ids
            W_indices = self.W.T[indices]
            W_indices = torch.permute(W_indices, (2, 1, 0))
            W_abs = W_indices.abs()
            phis = (W_indices * torch.conj(one_roots)).angle().abs()
            phi_factor = W_abs + phis
            phi_factor = torch.permute(phi_factor, (1, 0, 2))
            mu = phi_factor.sum(dim=-1).argmax(dim=-1)

        W_mu_nonzero = torch.gather(self.W[mu], 1, indices.long().T)

        alpha = W_mu_nonzero * torch.conj(Ps * one_roots)
        del W_mu_nonzero
        r_alpha = torch.view_as_real(alpha)
        treshold = 1e-10
        r_alpha[(r_alpha < treshold) & (r_alpha > -1 * treshold)] = treshold
        alpha = torch.view_as_complex(r_alpha)

        alpha_phi_factor = alpha.angle().abs() * Ps
        alpha_abs = alpha.abs()
        numerator = alpha_abs
        denominator = torch.linalg.norm(self.W, dim=1)[mu]
        E = -(numerator / denominator.unsqueeze(-1) + alpha_phi_factor).sum()

        self.fruit_fly_energy_metric = E.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

        return [
            (1 - self.e_weight) * loss_ + self.e_weight * E for loss_ in policy_loss
        ]

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "Must call forward first"
        return self.activation(self.value_network(self._features)).squeeze(-1)

    def metrics(self):
        return {
            "policy_loss": self.policy_loss_metric,
            "fruit_fly_energy": self.fruit_fly_energy_metric,
        }
