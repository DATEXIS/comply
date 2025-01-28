import torch
import ray
from mteb import MTEB
import argparse
from flyvec import FlyVec
from transformers import AutoTokenizer
import logging
import pickle
import os
import pandas as pd
import datasets
import numpy as np
from sklearn.model_selection import KFold
from datasets import ReadInstruction
from datasets import Sequence

logging.basicConfig(level=logging.INFO)


@ray.remote
def og_ff_encode_chunk(W, K, k, sentences):
    Nvocab = 20_000
    fly = FlyVec.load(force_redownload=False)
    W = torch.tensor(W)
    hashes = []
    activation_orders = []
    for s in sentences:
        binary_hash = torch.zeros(K, dtype=torch.bool)
        activation_order = torch.ones_like(binary_hash) * -1
        if s != "":
            s = s.replace(".", "").replace("?", "").strip()
            ids = fly.tokenizer.encode(s)
            ids = torch.tensor(ids)
            if len(ids) == 0:
                hashes.append(binary_hash)
                activation_orders.append(activation_order)
                continue  # unknown characters
            ids = torch.cat((ids, ids + Nvocab))  # Context hash
            window_size = len(ids)
            indices = ids.reshape(-1, window_size).T.long()
            W_indices = W.T[indices]
            W_indices = torch.permute(W_indices, (1, 2, 0))
            activation_order = (
                W_indices.sum(dim=-1)
                .argsort(dim=-1, descending=True)
                .squeeze(0)
            )
            trues = activation_order[:k]
            trues = trues
            binary_hash = binary_hash.scatter_(
                dim=0,
                index=trues,
                src=torch.ones_like(trues, dtype=torch.bool),
            )  # noqa
        hashes.append(binary_hash.float())
        activation_orders.append(activation_order)

    hashes = torch.stack(hashes)
    activation_orders = torch.stack(activation_orders)
    return hashes, activation_orders


class FruitFlyEncoder:
    def __init__(
        self,
        task,
        task_splits,
        force_redownload=False,
        n_chunks=40,
    ):
        self.k = None
        self.fly = FlyVec.load(force_redownload=force_redownload)
        self.W = np.load(self.fly.synapse_file)
        self.K = self.W.shape[0]
        self.rayW = ray.put(self.W)
        self.task = task
        self.task_splits = task_splits
        self.cache_path = None
        self.sentence_group = 0
        self.cached_activation_orders = []
        self.n_chunks = n_chunks
        try:
            with open(
                os.path.join(cache_path, "activation_order.dat"), "rb"
            ) as f:
                self.cached_activation_orders = pickle.load(f)
        except Exception as e:
            pass

    def tokenize(self, sentence):
        sentence = sentence.replace(".", "").replace("?", "").strip()
        ids = self.fly.tokenizer.encode(sentence)
        if len(ids) == 0:
            breakpoint()
        return torch.tensor(ids)

    def reset(self):
        self.sentence_group = 0

    def reset_cache(self):
        self.cached_activation_orders = []

    def set_n_chunks(self, n_chunks):
        self.n_chunks = n_chunks

    def set_cache_path(self, cache_path):
        self.cache_path = cache_path

    def set_k(self, k):
        self.k = k

    def load_cache(self):
        with open(
            os.path.join(self.cache_path, "activation_order.dat"), "rb"
        ) as f:
            self.cached_activation_orders = pickle.load(f)

    def encode(self, sentences, batch_size=17600, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences # noqa
        """
        if len(self.cached_activation_orders) == self.task_splits:
            activation_order = self.cached_activation_orders[
                self.sentence_group
            ]
            activation_order = torch.tensor(activation_order)
            hashes = torch.zeros_like(activation_order, dtype=torch.bool)
            trues = activation_order[:, : self.k]
            to_fix = (trues < 0).any(dim=1)  # couldn't tokenize these
            trues[to_fix, :] = 0
            hashes = hashes.scatter_(
                dim=1,
                index=trues,
                src=torch.ones_like(trues, dtype=torch.bool),
            )
            hashes[to_fix, :] = 0

            self.sentence_group += 1
            assert len(hashes) == len(sentences)
            return hashes.float().numpy()

        n_chunks = self.n_chunks
        chunk_size = len(sentences) // n_chunks
        chunks = [
            sentences[c * chunk_size : (c + 1) * chunk_size]
            for c in range(n_chunks - 1)
        ]
        # Last chunk
        chunks.append(sentences[(n_chunks - 1) * chunk_size :])
        futures = [
            og_ff_encode_chunk.remote(self.rayW, self.K, self.k, c) for c in chunks
        ]
        results = ray.get(futures)
        hashes, activation_orders = [], []
        for h, a in results:
            hashes.append(h)
            activation_orders.append(a)
        assert sum([len(h) for h in hashes]) == len(sentences)
        all_hashes = torch.concat(hashes).numpy()
        all_activation_orders = torch.concat(activation_orders).numpy()
        self.sentence_group += 1
        self.cached_activation_orders.append(all_activation_orders)

        if self.sentence_group == self.task_splits:
            os.makedirs(self.cache_path, exist_ok=True)
            with open(
                os.path.join(self.cache_path, "activation_order.dat"), "wb"
            ) as f:
                pickle.dump(self.cached_activation_orders, f)
        return all_hashes
