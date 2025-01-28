import torch
import ray
from mteb import MTEB
import argparse
from flyvec import FlyVec
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader
import logging
import pickle
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)


class BERT20k:
    def __init__(self):
        self.model = AutoModel.from_pretrained("bert-base-uncased").to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(self, sentence):
        sentence = sentence.replace(".", "").replace("?", "").strip()
        ids = self.fly.tokenizer.encode(sentence)
        if len(ids) == 0:
            breakpoint()
        return torch.tensor(ids)

    def reset(self):
        self.sentence_group = 0

    def reset_cache(self):
        pass

    def set_k(self, k):
        pass

    def set_n_chunks(self, n_chunks):
        pass

    def set_cache_path(self, cache_path):
        pass

    def load_cache(self):
        pass

    def encode(self, sentences, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences # noqa
        """
        batch = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )
        ids = batch["input_ids"].to('cuda')
        tt_ids = batch["token_type_ids"].to('cuda')
        a_mask = batch["attention_mask"].to('cuda')
        # Just like in FlyVec prune the ids > 20000
        ids[ids > 20000] = 0
        a_mask[ids > 20000] = 0
        dataset = TensorDataset(ids, tt_ids, a_mask)
        train_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=False)
        mean_embeddings = []
        for b_ids, b_tt_ids, b_a_mask in train_loader:
            with torch.no_grad():
                outs = self.model(
                    input_ids=b_ids, token_type_ids=b_tt_ids, attention_mask=b_a_mask
                )
            last_hidden_state = outs["last_hidden_state"]
            mean_embeddings.append(last_hidden_state.mean(dim=1).cpu())

        all_embeddings = torch.cat(mean_embeddings, dim=0)
        return all_embeddings
