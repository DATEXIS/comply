from FruitFly import FruitFly
from ComplexFruitfly import ComplexFruitFly
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from dataset_dp import OWTDataset
import ray
import math
import json
import datetime
import uuid
import os
import shutil
from tensorboardX import SummaryWriter


def build_output_paths(args):
    short_uuid = str(uuid.uuid4())[:8]
    model_name = (
        datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S%f") + f"_{short_uuid}"
    )
    model_output_path = os.path.join(args.model_path, model_name)
    os.makedirs(model_output_path, exist_ok=True)
    with open(os.path.join(model_output_path, "args.json"), "w") as f:
        json.dump(vars(args), f)
    return model_output_path


def main(args):

    torch.manual_seed(args.seed)
    ray.init(
        object_store_memory=args.ray_gb * 1024 * 1024 * 1024, num_cpus=args.num_cpus
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        args.batch_size *= n_gpus
        args.max_batch_size *= n_gpus

    model_output_path = build_output_paths(args)
    writer = SummaryWriter(model_output_path)
    shutil.copy2(
        "ComplexFruitfly.py", os.path.join(model_output_path, "ComplexFruitfly.py")
    )
    shutil.copy2("FruitFly.py", os.path.join(model_output_path, "Fruitfly_dp.py"))
    shutil.copy2("dataset_dp.py", os.path.join(model_output_path, "dataset_dp.py"))
    shutil.copy2("main_dp.py", os.path.join(model_output_path, "main_dp.py"))

    chunk_size = args.max_batch_size

    dataset = OWTDataset(
        window_size=args.window_size,
        device=device,
        encoded_path=args.encoded_path,
        offsets_path=args.offsets_path,
        vocab_size=args.vocab_size,
        num_cpus=args.num_cpus,
        chunk_size=chunk_size,
        seed=args.seed,
        max_batching_workers=args.num_cpus // 2,
        num_ddp_workers=torch.cuda.device_count(),
        ddp_rank=args.local_rank,
        mock_data=args.mock_data,
        stride=args.stride_length,
        mock_n_sentences=args.mock_n_sentences,
        mock_sentence_length=args.mock_sentence_length,
        mock_increment=args.mock_increment,
        data_subset=args.data_subset,
    )

    data_loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    device = torch.device(device)
    if not args.complex:
        model = FruitFly(args.K, args.k, 2 * args.vocab_size)
    else:
        if args.no_target:
            model = ComplexFruitFly(args.K, args.k, args.vocab_size, args.window_size)
        else:
            model = ComplexFruitFly(
                args.K, args.k, 2 * args.vocab_size, args.window_size
            )
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    n_devices = max(1, n_gpus)
    model.to(device)
    batches = math.ceil(dataset.samples / args.max_batch_size)  # Approximate

    params = model.parameters()
    optimizer = torch.optim.AdamW(
        params, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0, total_iters=args.epochs, verbose=True
    )  # noqa
    batch_count = 0
    if args.gaba:
        top_k = args.start_k_gaba
        gaba_step = (top_k - 1) / (batches * args.anealing_gaba_batches)
        dataset.set_batch_size(args.batch_size)
    else:
        top_k = 1

    for epoch in tqdm(range(args.epochs), desc="epoch"):
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
        dataset.shuffle_chunks()
        for batch in tqdm(data_loader, total=batches):
            if args.gaba and (top_k - 1) != 0:
                top_k = args.start_k_gaba - int(gaba_step * batch_count)
                new_batch_size = int(
                    (args.max_batch_size / (50 * n_gpus)) * (1 - top_k / args.K)
                )
                dataset.set_batch_size(new_batch_size)
                torch.cuda.empty_cache()
                print(f"top k set to :{top_k} current batch_size:{new_batch_size}")
            if top_k == 1:
                dataset.set_batch_size(args.max_batch_size)
            b = batch.squeeze(0)

            if len(b) != args.batch_size:
                b = b[: n_devices * (len(b) // n_devices)]

            ids = b[:, 0].type(torch.int32).to(device)
            Ps = b[:, 1].type(torch.float32).to(device)
            pos = b[:, 2].type(torch.float32).to(device)

            # Add the target in the second half of the 2*vocab index
            if not args.no_target:
                ids.T[args.window_size // 2] += args.vocab_size
            ids = ids.reshape(-1).int()
            optimizer.zero_grad(set_to_none=True)
            out, E = model(ids, Ps, pos, top_k)

            if n_devices > 1:
                E = E.sum()

            E.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % 1000 == 0:
                writer.add_scalar("top_k", top_k, batch_count)
                grad_norm = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.grad.detach(), 2)
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                parameter_norm = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.detach(), 2)
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                writer.add_scalar("train_loss", E.item(), batch_count)
                writer.add_scalar("grad_l2_norm", grad_norm.item(), batch_count)
                writer.add_scalar("param_l2_norm", parameter_norm.item(), batch_count)
        scheduler.step()

        if n_devices > 1:
            W = model.module.W
        else:
            W = model.W
        torch.save(W, f"{model_output_path}/checkpoint_{epoch}.pth")


if __name__ == "__main__":
    ray.shutdown()
    parser = argparse.ArgumentParser(description="Train the FruitFly")
    # Data parameters
    parser.add_argument(
        "--encoded_path",
        type=str,
        default="/pvc/privatefs/data/openwebtext/all_encoded.npy",  # noqa
        help="Path to the tokenized ids",
    )

    parser.add_argument(
        "--offsets_path",
        type=str,
        default="/pvc/privatefs/data/openwebtext/all_offsets.npy",  # noqa
        help="Path to the sentence offsets",
    )
    # MOCK DATA ARGUMENTS
    # The data is just random ramps with a given increment and possible maximum length # noqa
    parser.add_argument(
        "--mock_data",
        action="store_true",
        help="Use randomly generated integer sequences instead of text data",
    )  # noqa

    parser.add_argument(
        "--mock_n_sentences",
        type=int,
        default=10_000_000,
        help="Number of mock sentences to create",
    )

    parser.add_argument(
        "--mock_sentence_length",
        type=int,
        default=200,
        help="Maximum length of a mock sentence.",
    )

    parser.add_argument(
        "--mock_increment",
        type=int,
        default=50,
        help="Size of increment for mock sequences",
    )

    parser.add_argument(
        "--data_subset",
        type=int,
        default=None,
        help="Number of samples to take from the data",
    )

    parser.add_argument(
        "--no_target",
        action="store_true",
        help="if using complex model, skip adding the vocab to the id of the target word",
    )  # noqa

    parser.add_argument(
        "--gaba", action="store_true", help="use top k anealling of the activations"
    )  # noqa

    parser.add_argument(
        "--anealing_gaba_batches",
        type=float,
        default=0.5,
        help="proportion of batches to anneal when gaba switch is enabled",
    )

    parser.add_argument(
        "--start_k_gaba", type=int, default=5, help="Initial k for the top_k"
    )

    parser.add_argument(
        "--complex",
        action="store_true",
        help="Use the complex model with encoding of position in the phase",
    )  # noqa

    parser.add_argument(
        "--window_size",
        type=int,
        default=11,
        help="Window size of the context for the embeddings",
    )

    parser.add_argument(
        "--stride_length",
        type=int,
        default=1,
        help="Length of the stride when sliding the window at training",
    )  # noqa

    parser.add_argument(
        "--ray_gb", type=int, default=100, help="GB of ray shared memory"
    )

    parser.add_argument(
        "--num_cpus",
        type=int,
        default=250,
        help="Number of concurrent processes for computing the inverse frequencies",
    )  # noqa
    # Model parameters
    parser.add_argument("--K", type=int, default=400, help="Number of Kenyon Cells")
    parser.add_argument("--k", type=int, default=51, help="Hash length")
    parser.add_argument(
        "--vocab_size", type=int, default=20_000, help="vocab_size of the tokenizer"
    )

    # Training parameters
    parser.add_argument("--seed", type=int, default=2020, help="seed")

    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")

    parser.add_argument(
        "--epochs_per_checkpoint", type=int, default=1, help="Number of epochs"
    )

    parser.add_argument("--batch_size", type=int, default=35_000, help="Batch size")
    parser.add_argument("--max_batch_size", type=int, default=35_000, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=4e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Adam weight decay"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/pvc/privatefs/afigueroa/fruitfly/models",
        help="Folder to output models",
    )

    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="Automatically set by torch.distributed.launch",
    )

    args = parser.parse_args()
    main(args)

    ray.shutdown()
