import pandas as pd
import numpy as np
from flyvec import FlyVec
import torch
from functools import partial
from ray import tune
from copy import deepcopy

from hyperopt import hp
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.config import RunConfig


from sklearn.model_selection import KFold, StratifiedKFold
import ray
import argparse
import os
import json


def read_data(wic_path):
    train = pd.read_csv(wic_path, delimiter="\t", header=None)
    train.columns = ["target", "label", "positions", "context 1", "context 2"]
    return train




def all_word_hashes(W, k):
    Nvocab = W.shape[1] // 2
    syn_order = W.T[Nvocab:].argsort(descending=True)
    word_hashes = torch.zeros_like(
        syn_order, dtype=torch.bool, device=W.device
    )
    trues = syn_order[:, :k]
    word_hashes = word_hashes.scatter(
        dim=1, index=trues, src=torch.ones_like(trues, dtype=torch.bool)
    )

    return word_hashes


def all_complex_word_hashes(W, k):
    Nvocab = W.shape[1] // 2
    syn_order = (
        torch.view_as_complex(W).T[Nvocab:].abs().argsort(descending=True)
    )
    word_hashes = torch.zeros_like(
        syn_order, dtype=torch.bool, device=W.device
    )
    trues = syn_order[:, :k]
    word_hashes = word_hashes.scatter(
        dim=1, index=trues, src=torch.ones_like(trues, dtype=torch.bool)
    )
    return word_hashes




def context_hash(W, k, ids):
    Nvocab = W.shape[1] // 2
    ids = torch.tensor(ids, device=W.device)
    ids = torch.cat((ids, ids + Nvocab))
    coordinates = torch.stack((torch.zeros_like(ids), ids)).T
    V_A_s = torch.sparse_coo_tensor(
        coordinates.T,
        torch.ones_like(ids),
        (1, 2 * Nvocab),
        dtype=torch.float,
        device=W.device,
    )
    V_AxWT = torch.sparse.mm(V_A_s, W.T)

    activation_order = V_AxWT.squeeze().argsort(descending=True)
    sample_hash = torch.zeros_like(activation_order, dtype=torch.bool)
    sample_hash[activation_order[:k]] = True
    return sample_hash


def batched_context_hash(W, k, batched_ids):
    Nvocab = W.shape[1] // 2
    batched_ids = np.stack(batched_ids)
    batched_ids = torch.tensor(batched_ids, device=W.device)
    ids = torch.cat((batched_ids, batched_ids + Nvocab), axis=1)
    batch_indices = (
        torch.arange(0, len(ids), dtype=torch.int32)
        .repeat_interleave(ids.shape[-1])
        .to(W.device)
    )
    values = (ids >= 0).long()
    coordinates = torch.stack((batch_indices, ids.reshape(-1))).T
    coordinates = torch.max(torch.tensor(0), coordinates)
    V_A_s = torch.sparse_coo_tensor(
        coordinates.T,
        values.reshape(-1),
        (values.shape[0], 2 * Nvocab),
        dtype=torch.float,
        device=W.device,
    )
    V_AxWT = torch.sparse.mm(V_A_s, W.T)

    activation_order = V_AxWT.squeeze().argsort(descending=True)
    hashes = torch.zeros_like(activation_order, dtype=torch.bool)
    trues = activation_order[:, :k]
    hashes = hashes.scatter(
        dim=1, index=trues, src=torch.ones_like(trues, dtype=torch.bool)
    )
    return hashes


def batched_complex_context_hash(W, k, batched_ids, window_size, no_target):
    if no_target:
        Nvocab = W.shape[1]
    else:
        Nvocab = W.shape[1] // 2
    batched_ids = np.stack(batched_ids)
    batched_ids = torch.tensor(batched_ids, device=W.device)
    if no_target:
        ids = batched_ids
    else:
        ids = torch.cat((batched_ids, batched_ids + Nvocab), axis=1)
    sample_lengths = (batched_ids > 0).sum(axis=1)
    resampled_one_roots = torch.zeros(
        (ids.shape[0], ids.shape[1]), dtype=torch.cfloat, device=W.device
    )
    for isl, sl in enumerate(sample_lengths):
        ks = torch.arange(0, sl, device=W.device)
        re = torch.cos(torch.pi * ks / sl)
        im = torch.sin(torch.pi * ks / sl)
        resampled_one_roots[isl, :sl] = torch.complex(re, im)

    indices = ids
    W_indices = torch.view_as_complex(W).T[indices]
    W_indices = torch.permute(W_indices, (2, 0, 1))
    W_abs = W_indices.abs()
    phis = (W_indices * torch.conj(resampled_one_roots)).angle().abs()
    phis = torch.permute(phis, (0, 2, 1)) / (sample_lengths * torch.pi)
    phis = torch.permute(phis, (0, 2, 1))
    activation_order = (W_abs + phis).sum(dim=-1).argsort(dim=0, descending=True).T
    hashes = torch.zeros_like(activation_order, dtype=torch.bool)
    trues = activation_order[:, :k]
    hashes = hashes.scatter(
        dim=1, index=trues, src=torch.ones_like(trues, dtype=torch.bool)
    )
    return hashes


def all_J_dot(c1_hashes, c2_hashes):
    return (c1_hashes * c2_hashes).sum(axis=1)




def top_q(word_hashes, q, sample_hash):
    try:
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            sample_hash.to(mps_device)
    except:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            sample_hash.to(device)
    context_word_hash_distance = torch.mm(
        word_hashes.float(), sample_hash.unsqueeze(0).T.float()
    )
    top_words = context_word_hash_distance.squeeze(1).argsort(descending=True)[
        :q
    ]
    return top_words




def top_q_m(q, word_hashes, sample_hashes):
    try:
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            sample_hashes = sample_hashes.contiguous().to(torch.device("cpu"))
    except:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            sample_hashes = sample_hashes.to(device)
            word_hashes = word_hashes.to(device)
    context_word_hash_distance = torch.mm(
        word_hashes.float(), sample_hashes.T.float()
    )
    top_words = context_word_hash_distance.argsort(descending=True, axis=0)[:q]
    return top_words


def J_nn(top_q_left, top_q_right):
    c1set = set(top_q_left.tolist())
    c2set = set(top_q_right.tolist())
    return len(c1set.intersection(c2set)) / len(c1set)




def WiC_acc(
    device,
    W,
    X,
    truth,
    alpha,
    theta,
    q,
    k,
    w,
    no_target=False,
    complexFF=False,
):
    W = torch.tensor(ray.get(W), device=device)
    X = X.copy()

    def wPositions(w, p, l):
        return [max(p - w // 2, 0), min(l - 1, p + w // 2)]

    wp = partial(wPositions, w)

    def filter_context_lengths_w21(positions, context_ids):
        return context_ids[positions[0] : positions[1]]

    X["wc1"] = X.apply(lambda x: wp(x["p1"], x["lenc1"]), axis=1)
    X["wc2"] = X.apply(lambda x: wp(x["p2"], x["lenc2"]), axis=1)

    X["c1ids"] = X.apply(
        lambda x: filter_context_lengths_w21(x["wc1"], x["c1ids"]), axis=1
    )
    X["c2ids"] = X.apply(
        lambda x: filter_context_lengths_w21(x["wc2"], x["c2ids"]), axis=1
    )

    max_len_c1 = X.lenc1.max()
    max_len_c2 = X.lenc2.max()

    def pad_c1(x):
        return np.pad(
            np.array(x),
            pad_width=(0, max_len_c1 - len(x)),
            mode="constant",
            constant_values=-1 - W.shape[1] // 2,
        )

    def pad_c2(x):
        return np.pad(
            np.array(x),
            pad_width=(0, max_len_c2 - len(x)),
            mode="constant",
            constant_values=-1 - W.shape[1] // 2,
        )

    X["padded_c1"] = X.c1ids.apply(pad_c1)
    X["padded_c2"] = X.c2ids.apply(pad_c2)

    if complexFF:
        c1_hashes = batched_complex_context_hash(
            W, k, X.padded_c1.values, 11, no_target
        )  
        c2_hashes = batched_complex_context_hash(
            W, k, X.padded_c2.values, 11, no_target
        ) 
    else:
        c1_hashes = batched_context_hash(W, k, X.padded_c1.values)
        c2_hashes = batched_context_hash(W, k, X.padded_c2.values)
    J_dots = all_J_dot(c1_hashes, c2_hashes)

    if complexFF:
        all_hashes = all_complex_word_hashes(W, k)
    else:
        all_hashes = all_word_hashes(W, k)
    c1_top_q = top_q_m(q, all_hashes, c1_hashes).T
    c2_top_q = top_q_m(q, all_hashes, c2_hashes).T
    all_J_nns = torch.tensor(list(map(J_nn, c1_top_q, c2_top_q))).to(W.device)

    J = alpha * J_dots / k + (1 - alpha) * all_J_nns
    return (
        ((J > theta) == torch.tensor(truth, device=W.device)).sum() / len(X)
    ).item()


def main(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if not args.debug:
        wic_path = "WiC_dataset/train/train.data.txt"
        n_splits = 5
    else:
        wic_path = "WiC_dataset/train/train.data.txt"
        n_splits = 5

    train = read_data(wic_path)
    if args.force_redownload:
        og_ff = FlyVec.load(force_redownload=True)
    else:
        og_ff = FlyVec.load()  # force_redownload=True)

    if args.model_path:
        W = torch.load(args.model_path, map_location="cpu").detach()
        W = torch.nan_to_num(W).numpy()
        W = ray.put(W)
    else:
        W = ray.put(np.load(og_ff.synapse_file))

    def tokenlist2ids(x):
        return [og_ff.tokenizer.token2id(e) for e in x]

    def clean(x):
        return x[:-1].lower().split(" ")[:-1]

    def label(x):
        return True if x == "V" else False

    train["c1"] = train["context 1"].apply(clean)
    train["c2"] = train["context 2"].apply(clean)

    train["p1"] = train["positions"].apply(lambda x: int(x.split("-")[0]))
    train["p2"] = train["positions"].apply(lambda x: int(x.split("-")[1]))
    train["l"] = train["label"].apply(label)
    train["lenc1"] = train.c1.apply(lambda x: len(x))
    train["lenc2"] = train.c2.apply(lambda x: len(x))
    train["c1ids"] = train.c1.apply(tokenlist2ids)
    train["c2ids"] = train.c2.apply(tokenlist2ids)
    truth = train["l"].values

    train = train[["p1", "p2", "lenc1", "lenc2", "c1ids", "c2ids"]]

    kf = StratifiedKFold(n_splits=n_splits)
    splits = kf.split(train, truth)

    all_results = []
    if not args.debug:
        for ifold, (test1, train1) in enumerate(splits):

            def training_function(config):
                alpha = config["alpha"]
                theta = config["theta"]
                q = config["q"]
                k = config["k"]
                w = config["w"]
                tune.report(
                    accuracy=WiC_acc(
                        device,
                        W,
                        train.iloc[train1],
                        truth[train1],
                        alpha,
                        theta,
                        q,
                        k,
                        w,
                        args.no_target,
                        args.complex,
                    )
                )

            config_space = {
                "alpha": hp.uniform("alpha", args.min_alpha, args.max_alpha),
                "theta": hp.quniform(
                    "theta", args.min_theta, args.max_theta, 1 / args.max_k
                ),
                "q": hp.randint("q", 7, 30),
                "k": hp.choice(
                    "k", np.arange(args.min_k, args.max_k, 1, dtype=int)
                ),
                "w": hp.randint("w", args.min_w, args.max_w),
            }
            algo = HyperOptSearch(
                space=config_space, metric="accuracy", mode="max"
            )
            algo = ConcurrencyLimiter(
                algo, max_concurrent=args.max_concurrent_workers
            )
            scheduler = AsyncHyperBandScheduler()
            training_function_with_resources = tune.with_resources(
                training_function,
                {"cpu": args.cpu_per_worker, "gpu": args.gpu_per_worker},
            )
            tuner = tune.Tuner(
                training_function_with_resources,
                tune_config=tune.TuneConfig(
                    metric="accuracy",
                    mode="max",
                    search_alg=algo,
                    scheduler=scheduler,
                    num_samples=args.num_samples,
                    reuse_actors=True,
                ),
                run_config=RunConfig(local_dir=args.log_dir),
            )
            results = tuner.fit()
            print(
                results.get_dataframe().sort_values(
                    "accuracy", ascending=False
                )
            )
            print(
                "Best hyperparameters found were: ",
                results.get_best_result(metric="accuracy", mode="max"),
            )
            results_df = results.get_dataframe()
            experiment_path = results_df.logdir[0].split("/")[6]
            with open(
                os.path.join(args.log_dir, experiment_path, "args.json"), "w"
            ) as f:
                json.dump(vars(args), f)
            results_df.to_csv(
                os.path.join(
                    args.log_dir, experiment_path, f"fold_{ifold}_results.csv"
                )
            )
            top1 = (
                results.get_dataframe()
                .sort_values("accuracy", ascending=False)
                .head(1)
            )
            alpha = top1["config/alpha"].values[0]
            k = top1["config/k"].values[0]
            q = top1["config/q"].values[0]
            theta = top1["config/theta"].values[0]
            w = top1["config/w"].values[0]
            train_accuracy = top1["accuracy"].values[0]
            test_accuracy = WiC_acc(
                device,
                W,
                train.iloc[test1],
                truth[test1],
                alpha,
                theta,
                q,
                k,
                w,
                args.no_target,
                args.complex,
            )

            train_record = {
                "fold": ifold,
                "alpha": alpha,
                "k": k,
                "q": q,
                "theta": theta,
                "w": w,
                "split": "train",
                "score": train_accuracy,
            }
            all_results.append(train_record)
            test_record = deepcopy(train_record)
            test_record['score'] = test_accuracy
            test_record['split'] = 'test'
            all_results.append(test_record)

        pd.DataFrame.from_records(all_results).to_csv(
            os.path.join(
                args.log_dir, "all_fold_results.csv"
            )
        )

    else:
        alpha = 0.5
        theta = 0.8
        q = 10
        k = 50
        w = 21
        accuracy = WiC_acc(
            device,
            W,
            train.iloc[train1],
            truth[train1],
            alpha,
            theta,
            q,
            k,
            w,
            args.no_target,
            args.complex,
        )
        print(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FruitFly")
    parser.add_argument("--debug", action="store_true", help="Use debug data")

    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="Force fruitfly redownload of weights and data",
    )

    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples "
    )
    parser.add_argument(
        "--max_concurrent_workers",
        type=int,
        default=4,
        help="Number of concurrent workers ",
    )
    parser.add_argument(
        "--gpu_per_worker",
        type=float,
        default=0.49,  # Per A100
        help="Fractional GPU per worker",
    )
    parser.add_argument(
        "--cpu_per_worker",
        type=int,
        default=20,
        help="Number of CPU per worker",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/pvc/privatefs/data/ray_results/WiC/",
        help="Directory for Ray logs",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/2022_10_06__193831569877_67238134/checkpoint_19.pth",
        help="Path to custom model",
    )
    parser.add_argument(
        "--complex", action="store_true", help="complex FF model"
    )
    parser.add_argument(
        "--no_target",
        action="store_true",
        help="No additional target portion in W",
    )

    parser.add_argument(
        "--min_k", type=int, default=10, help="Minimum size of the hash"
    )

    parser.add_argument(
        "--max_k", type=int, default=400, help="Maximum size of the hash"
    )

    parser.add_argument(
        "--min_q", type=int, default=0, help="Minimum number of top words"
    )

    parser.add_argument(
        "--max_q", type=int, default=20, help="Maximum number of top words"
    )

    parser.add_argument(
        "--min_w", type=int, default=0, help="Minimum size of context window"
    )

    parser.add_argument(
        "--max_w", type=int, default=30, help="Maximum size of context window"
    )

    parser.add_argument(
        "--min_theta", type=float, default=0.0, help="Minimum theta"
    )

    parser.add_argument(
        "--max_theta", type=float, default=1.0, help="Maximum theta"
    )

    parser.add_argument(
        "--min_alpha", type=float, default=0.0, help="Minimum alpha"
    )

    parser.add_argument(
        "--max_alpha", type=float, default=1.0, help="Maximum alpha"
    )

    args = parser.parse_args()
    main(args)
