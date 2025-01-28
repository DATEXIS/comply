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
from run_mteb_og_ff import FruitFlyEncoder
from run_mteb_bert import BERT20k

logging.basicConfig(level=logging.INFO)


@ray.remote
def encode_chunk(Wcmplx, K, k, sentences, added_phi_factor):
    fly = FlyVec.load(force_redownload=False)
    hashes = []
    activation_orders = []
    Wcmplx = torch.tensor(Wcmplx)
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
            window_size = len(ids)
            ks = torch.arange(0, window_size)
            re = torch.cos(torch.pi * ks / window_size)
            im = torch.sin(torch.pi * ks / window_size)
            one_roots = torch.complex(re, im)
            indices = ids.reshape(-1, window_size).T.long()
            W_indices = Wcmplx.T[indices]
            W_indices = torch.permute(W_indices, (1, 2, 0))
            W_abs = W_indices.abs()
            phis = (W_indices * torch.conj(one_roots)).angle().abs()
            if added_phi_factor:
                activation_order = (
                    (W_abs + phis)
                    .sum(dim=-1)
                    .squeeze()
                    .argsort(descending=True)
                )  # noqa
            else:
                activation_order = (
                    (W_abs * phis)
                    .sum(dim=-1)
                    .squeeze()
                    .argsort(descending=True)
                )  # noqa

            trues = activation_order[:k]
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


class ComplexFruitFlyEncoder:
    def __init__(
        self,
        weights_path,
        task,
        task_splits,
        added_hash,
        force_redownload=False,
        n_chunks=40,
    ):
        self.k = None
        self.fly = FlyVec.load(force_redownload=force_redownload)
        self.Wcmplx = torch.load(weights_path, map_location="cpu").detach()
        self.Wcmplx = torch.view_as_complex(self.Wcmplx)
        self.K = self.Wcmplx.shape[0]
        self.ttok = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.rayWcmplx = ray.put(self.Wcmplx.numpy())
        self.task = task
        self.task_splits = task_splits
        self.cache_path = None
        self.sentence_group = 0
        self.cached_activation_orders = []
        self.added_hash = added_hash
        self.n_chunks = n_chunks

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

    def set_k(self, k):
        self.k = k

    def set_n_chunks(self, n_chunks):
        self.n_chunks = n_chunks

    def set_cache_path(self, cache_path):
        self.cache_path = cache_path

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
            encode_chunk.remote(
                self.rayWcmplx, self.K, self.k, c, self.added_hash
            )
            for c in chunks
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

        print(self.sentence_group, f" Storing in {self.cache_path}")
        if self.sentence_group == self.task_splits:
            os.makedirs(self.cache_path, exist_ok=True)
            with open(
                os.path.join(self.cache_path, "activation_order.dat"), "wb"
            ) as f:
                pickle.dump(self.cached_activation_orders, f)
        return all_hashes


def main():

    parser = argparse.ArgumentParser(description="Train the FruitFly")
    # Data parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/2023_08_08__082624983601_c563f6aa/checkpoint_2.pth",  # noqa
        help="Path to the model",
    )
    parser.add_argument(
        "--added_hash",
        action="store_true",
        help="Use additive version of hash phi_factor computation",
    )  # noqa
    parser.add_argument(
        "--k_min", type=int, default=1, help="min k to try"
    )  # noqa
    parser.add_argument(
        "--k_max", type=int, default=400, help="max k to try"
    )  # noqa

    parser.add_argument(
        "--flyvec",
        action="store_true",
        help="Evaluate the orginal FlyVec model",
    )  # noqa

    parser.add_argument(
        "--bert20k",
        action="store_true",
        help="Evaluate BERT with a top 20k tokens from tokenizer",
    )  # noqa

    parser.add_argument(
        "--only_benchmark",
        action="store_true",
        help="Run only the 100k hash to benchmark the time",
    )  # noqa

    args = parser.parse_args()

    if args.bert20k and args.flyvec:
        raise Exception(
            "Cannot use flyvec and bert20k arguments at the same time"
        )

    num_gpus = 1 if args.bert20k else None
    ray.init(num_gpus=num_gpus)

    tasks_splits = {
        # S2S
        "BIOSSES": 2,
        "SICK-R": 2,
        "STS12": 2,
        "STS13": 2,
        "STS14": 2,
        "STS15": 2,
        "STS16": 2,
        "STS17": 2,
        "STSBenchmark": 2,
        # Pair classification
        "SprintDuplicateQuestions": 1,  # see attention at the end of the file
        "TwitterSemEval2015": 1,
        "TwitterURLCorpus": 1,
    }

    def col(task, column):
        if task in ["STS17", "STS22"]:
            return column
        else:
            return [column]

    def target_col(task):
        if task in ["STS17", "STS22"]:
            return "score"
        else:
            return "labels"

    def build_splits(task, task_object):
        if task in [
            "SprintDuplicateQuestions",
            "TwitterSemEval2015",
            "TwitterURLCorpus",
            "STS17",
            "STS22",
        ]:
            full_dataset = datasets.load_dataset(
                split="test", **task_object.metadata_dict["dataset"]
            )

            shuffle = False
            random_state = None
            if task == "SprintDuplicateQuestions":
                shuffle = True
                random_state = 42

            if task in ["STS17", "STS22"]:
                lan_filter = {"STS17": "en-en", "STS22": "en"}
                full_dataset = full_dataset.filter(
                    lambda e: e["lang"] == lan_filter[task]
                )
                sentence1 = np.array(full_dataset["sentence1"])
                sentence2 = np.array(full_dataset["sentence2"])
                labels = np.array(full_dataset[target_col(task)])
            else:
                sentence1 = np.array(full_dataset[0]["sent1"])
                sentence2 = np.array(full_dataset[0]["sent2"])
                labels = np.array(full_dataset[0]["labels"])

            trains, tests = [], []
            for tr_i, tst_i in KFold(
                n_splits=5, shuffle=shuffle, random_state=random_state
            ).split(labels):
                tests.append(
                    datasets.Dataset.from_dict(
                        {
                            "sentence1": col(task, sentence1[tst_i]),
                            "sentence2": col(task, sentence2[tst_i]),
                            target_col(task): col(task, labels[tst_i]),
                        },
                    )
                )
                trains.append(
                    datasets.Dataset.from_dict(
                        {
                            "sentence1": col(task, sentence1[tr_i]),
                            "sentence2": col(task, sentence2[tr_i]),
                            target_col(task): col(task, labels[tr_i]),
                        },
                    )
                )
        else:
            tests = datasets.load_dataset(
                split=[
                    datasets.ReadInstruction(
                        "test", from_=k, to=k + 20, unit="%"
                    )
                    for k in range(0, 100, 20)
                ],
                **task_object.metadata_dict["dataset"],
            )
            trains = datasets.load_dataset(
                split=[
                    datasets.ReadInstruction("test", to=k, unit="%")
                    + datasets.ReadInstruction("test", from_=k + 20, unit="%")
                    for k in range(0, 100, 20)
                ],
                **task_object.metadata_dict["dataset"],
            )
        if True:  # in FlyVec they use one fold to tune and 5 to test
            return tests, trains
        return trains, tests

    if args.flyvec:
        args.model_path = "FlyVec"

    if args.bert20k:
        args.model_path = "bert20k"

    def build_task_dataset(task, current_index):
        if task == "STS17":
            return {"en-en": {"test": current_index}}
        elif task == "STS22":
            return {"en": {"test": current_index}}
        else:
            return {"test": current_index}

    def evaluate_k(
        model, k, all_evaluation_results, create_cache, cache_k, task, bert
    ):
        evaluation = MTEB(task_langs=["en"], tasks=[task])
        task_object = evaluation.tasks[0]
        model.set_k(k)
        trains, tests = build_splits(task, task_object)

        for i, (train_index, test_index) in enumerate(zip(trains, tests)):
            for current_index_name, current_index in {
                "train": train_index,
                "test": test_index,
            }.items():
                return_results = {}
                print(
                    f"sentences in fold {i} {current_index_name}",
                    current_index.num_rows,
                )
                if task in ["BIOSSES", "STS17", "STS22"]:  # very small dataset
                    model.set_n_chunks(10)
                else:
                    model.set_n_chunks(40)

                task_object.dataset = build_task_dataset(task, current_index)
                task_object.data_loaded = True
                model.set_cache_path(
                    f"mteb_results/{args.model_path}/{task}/fold{i}/{current_index_name}_{cache_k}_{added_hash}"
                )
                output_folder = f"mteb_results/{args.model_path}/{task}/fold{i}/{current_index_name}_{k}_{added_hash}"

                if not create_cache:
                    model.load_cache()
                    output_folder = None

                encode_kwargs = {}
                if bert:
                    encode_kwargs = {"batch_size": 32}
                evaluation_results = evaluation.run(
                    model,
                    output_folder=output_folder,
                    verbosity=0,
                    eval_splits=["test"],
                    encode_kwargs=encode_kwargs,
                )
                return_results["k"] = k
                return_results["model_path"] = args.model_path
                return_results["evaluation_results"] = evaluation_results
                return_results["fold"] = i
                return_results["fold_split"] = current_index_name
                all_evaluation_results.append(return_results)
                model.reset()
                model.reset_cache()

    added_hash = ""
    if args.added_hash:
        added_hash = "added_hash"

    def build_model(model_path, task, task_splits, added_hash, flyvec, bert):
        if flyvec:
            model = FruitFlyEncoder(
                task=task,
                task_splits=task_splits,
            )

        elif bert:
            model = BERT20k()
        else:
            model = ComplexFruitFlyEncoder(
                model_path,
                task=task,
                task_splits=task_splits,
                added_hash=added_hash,
            )
        return model

    # Compute the cache once
    all_evaluation_results = {}
    for task, task_splits in tasks_splits.items():
        all_evaluation_results[task] = []
        cache_k = 100
        cache_path = None
        model = build_model(
            args.model_path,
            task,
            task_splits,
            args.added_hash,
            args.flyvec,
            args.bert20k,
        )
        evaluate_k(
            model,
            cache_k,
            all_evaluation_results[task],
            True,
            cache_k,
            task,
            args.bert20k,
        )

        if args.bert20k:
            with open(
                os.path.join(
                    f"mteb_results/{args.model_path}/{task}/{added_hash}_all_results.dat"
                ),
                "wb",
            ) as f:
                pickle.dump(all_evaluation_results[task], f)

    # End the script here
    if args.bert20k or args.only_benchmark:
        return None

    # Paralellize the hash building and evaluation
    n_chunks = 40
    chunk_size = args.k_max // n_chunks
    ks = list(range(args.k_min, args.k_max, 1))
    chunks = [
        ks[c * chunk_size : (c + 1) * chunk_size] for c in range(n_chunks - 1)
    ]
    chunks.append(ks[(n_chunks - 1) * chunk_size :])

    @ray.remote
    def evaluate_model(chunk, task, args, task_splits):
        added_hash = ""
        if args.added_hash:
            added_hash = "added_hash"
        chunk_evaluation_results = []
        model = build_model(
            args.model_path,
            task,
            task_splits,
            args.added_hash,
            args.flyvec,
            args.bert20k,
        )
        for eval_k in chunk:
            evaluate_k(
                model,
                eval_k,
                chunk_evaluation_results,
                False,
                cache_k,
                task,
                args.bert20k,
            )

        return chunk_evaluation_results

    for task, task_splits in tasks_splits.items():
        futures = [
            evaluate_model.remote(c, task, args, task_splits) for c in chunks
        ]
        results = ray.get(futures)
        for chunk_results in results:
            for r in chunk_results:
                all_evaluation_results[task].append(r)

        with open(
            os.path.join(
                f"mteb_results/{args.model_path}/{task}/{added_hash}_all_results.dat"
            ),
            "wb",
        ) as f:
            pickle.dump(all_evaluation_results[task], f)


# 🤮🤮🤮🤮🤮🤮🤮
# ATTENTION!!!!!
# /usr/local/lib/python3.10/dist-packages/mteb/evaluation/evaluators/PairClassificationEvaluator.py
# has an unintended effect for caching when the input data contains duplicates:
# sentences = list(set(self.sentences1 + self.sentences2)), the set changes the order hence
# our cache implementation is rendered useless... so change that line for this:
# sentences = list({s:None for s in self.sentences1 + self.sentences2}.keys())
# 🤮🤮🤮🤮🤮🤮🤮


if __name__ == "__main__":
    main()
