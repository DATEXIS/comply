import ray
from argparse import Namespace
from typing import Dict
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models import ModelCatalog
from tqdm import tqdm
from ray import air, tune
from arguments import parseArguments
from model import (
    IMPALATransformer,
    IMPALATransformerDiseasePrediction,
    IMPALAFruitfly,
)
from complex_fruitfly import IMPALAComplexFruitfly
from utils import actionCallbacks, srcLoggerCallback
from gym_medical.envs.doctor_sim import DoctorSim

#TODO: ATTENTION !!!!
# gym compatibility stuff
# don't forget to copy serialization into /opt/conda/lib/python3.10/site-packages/ray/rllib/utils/serialization.py

def build_config(args: Namespace) -> Dict:
    config = {}
    config["framework"] = "torch"
    config["num_gpus"] = args.num_gpu
    config["num_multi_gpu_tower_stacks"] = 2
    config["num_workers"] = args.num_workers
    config["num_envs_per_worker"] = args.inference_batch_size
    config["num_cpus_per_worker"] = (
        args.inference_batch_size + 1
    )  
    config["num_gpus_per_worker"] = args.num_gpu_per_worker
    config["rollout_fragment_length"] = args.rollout_fragment_length
    config["train_batch_size"] = args.batch_size
    config["replay_proportion"] = args.replay_proportion
    config["replay_buffer_num_slots"] = 128
    config["env"] = "doctorsim_csv"
    config["disable_env_checking"] = True

    config["log_level"] = args.log_level
    config["env_config"] = {}
    config["env_config"]["observation_length"] = args.sequence_length
    config["env_config"]["tokenizer"] = args.model_name_or_path
    if args.fruitfly:
        config["env_config"]["tokenizer"] = args.fruitfly_tokenizer

    config["env_config"]["is_csv"] = args.is_csv
    config["env_config"]["data_path"] = args.data_path
    config["env_config"]["max_diseases"] = args.max_diseases
    config["env_config"]["fruitfly"] = args.fruitfly
    config["lr"] = args.learning_rate
    config["decay"] = args.adam_decay
    config["learner_queue_size"] = 1
    config["exploration_config"] = {
        "type": "EpsilonGreedy",
        "warmup_timesteps": 10000,
        "epsilon_timesteps": 5e5,
    }
    model_config = MODEL_DEFAULTS.copy()

    model_config["custom_model"] = "IMPALATransformer"
    if args.predict_disease:
        model_config["custom_model"] = "IMPALATransformerDiseasePrediction"
    if args.transformer_mlm:
        model_config["custom_model"] = "IMPALATransformerLM"
    model_config["custom_model_config"] = {
        "model_name_or_path": args.model_name_or_path,
        "num_diseases": args.max_diseases,
    }

    if args.fruitfly:
        model_config["custom_model"] = "IMPALAFruitfly"
        if args.complex:
            model_config["custom_model"] = "IMPALAComplexFruitfly"
        model_config["custom_model_config"] = {
            "K": 400,
            "k": 50,
            "tokenizer_name": args.fruitfly_tokenizer,
            "model_path": args.fruitfly_model_path,
            "fs_path": args.fruitfly_model_fs,
            "added_factors": args.fruitfly_added_factors
        }
            

    config["model"] = model_config

    return config


if __name__ == "__main__":

    args = parseArguments()
    config = build_config(args)

    ray.init(num_cpus=args.num_cpus or None, _temp_dir=args.log_dir)
    ray.tune.register_env("doctorsim_csv", lambda cfg: DoctorSim(**cfg))
    ModelCatalog.register_custom_model("IMPALATransformer", IMPALATransformer)
    ModelCatalog.register_custom_model(
        "IMPALATransformerDiseasePrediction",
        IMPALATransformerDiseasePrediction,
    )
    ModelCatalog.register_custom_model("IMPALAFruitfly", IMPALAFruitfly)
    ModelCatalog.register_custom_model(
        "IMPALAComplexFruitfly", IMPALAComplexFruitfly
    )

    stop = {"episodes_total": args.max_episodes}

    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=args.checkpoint_frequency
    )

    src_files = [
               "/pvc/privatefs/afigueroa/cff-Med-RL/complex_fruitfly.py",
               "/pvc/privatefs/afigueroa/cff-Med-RL/main.py",
               "/pvc/privatefs/afigueroa/cff-Med-RL/utils.py",
            ]

    copy_src = srcLoggerCallback(file_paths=src_files)
    run_config = air.RunConfig(
        stop=stop,
        verbose=3,
        storage_path=args.log_dir,
        name=f"{args.model_name_or_path.replace('/','')}-{args.max_diseases}",
        checkpoint_config=checkpoint_config,
        callbacks=[copy_src]
    )
    tuner = tune.Tuner(
        "IMPALA",
        param_space=config,
        run_config=run_config,
    )
    tuner.fit()
    ray.shutdown()
