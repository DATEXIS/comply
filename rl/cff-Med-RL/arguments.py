import argparse
from argparse import Namespace


def parseArguments() -> Namespace:
    parser = argparse.ArgumentParser("Perform Medical Reinforcement Learning")
    parser.add_argument(
        "--fruitfly", action="store_true", help="Use the fruitfly model"
    )
    parser.add_argument(
        "--complex", action="store_true", help="Use the complex fruitfly model"
    )
    parser.add_argument(
        "--predict_disease",
        action="store_true",
        help="use disease predictin auxiliary loss",
    )
    parser.add_argument("--freeze_encoder", action="store_true", help="freeze encoder")
    parser.add_argument(
        "--is_csv",
        action="store_true",
        help="Use labeled csv data instead of phospital",
    )
    parser.add_argument(
        "--use_action_embeddings",
        action="store_true",
        help="Add Embedding Layer for actions that is multiplied with observation encoding",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=100, help="Number of Episodes to run."
    )
    parser.add_argument(
        "--num_gpu", type=int, default=1, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--num_cpus", type=int, default=32, help="Number of CPUs to use for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker agents to use for training",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="length of observation sequences",
    )
    parser.add_argument(
        "--rollout_fragment_length",
        type=int,
        default=16,
        help="number of rollouts/size of replay buffer",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=8,
        help="Number of environments per worker",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Train batch size sent to GPU"
    )
    parser.add_argument(
        "--max_diseases",
        type=int,
        default=256,
        help="Limit number of diseases to simplify problem",
    )
    parser.add_argument(
        "--num_gpu_per_worker",
        type=float,
        default=1,
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--replay_proportion",
        type=float,
        default=0.1,
        help="Set >0 to enable experience replay with p:1 ratio",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, help="Learning rate"
    )
    parser.add_argument(
        "--adam_decay", type=float, default=0.99, help="Adam decay for IMPALA"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Transformer Encoder to use.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../Medical-Gym/data/project_hospital",
        help="Path to Environment data",
    )
    parser.add_argument(
        "--log_level", type=str, default="WARN", help="Log Level to use"
    )
    parser.add_argument(
        "--log_dir", type=str, default="~/ray_results", help="Directory for Ray logs"
    )
    parser.add_argument(
        "--fruitfly_model_path", type=str, default=None, help="fruitfly checkpoint"
    )
    parser.add_argument(
        "--fruitfly_tokenizer",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="fruitfly tokenizer",
    )
    parser.add_argument(
        "--checkpoint_frequency", type=int, default=50, help="Frequency of checkpoints"
    )
    parser.add_argument(
        "--transformer_mlm", action="store_true", help="Use mlm custom objective"
    )
    parser.add_argument(
        "--fruitfly_model_fs",
        type=str,
        default=None,
        help="frequencies of words of the environment to load",
    )
    parser.add_argument(
        "--fruitfly_added_factors",
        action="store_true",
        help="Use either added or multiplicative factors for complex ff",
    )

    return parser.parse_known_args()[0]
