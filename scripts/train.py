#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))  # for plibs, third_party

DEBUG_ROOT = REPO_ROOT / "debug"
DEBUG_ROOT.mkdir(parents=True, exist_ok=True)

# enable torchsprase for TRELLIS.
# NOTE: we specifically put it here instead of making it as default for all scripts.
# The reason is that pre-trained TRELLIS uses `spconv`,
# which produces different results if we call the pre-trained model with `torchsparse`.
# Since `train.py` (this script) is only used when training a new model,
# putting it here enforce all newly-trained models use `torchsparse`.
# os.environ["SPARSE_BACKEND"] = "torchsparse"

import argparse
import copy
import datetime
import os
import pprint
import resource
import sys
import tempfile
import typing as T
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore all types of warnings

import lightning.pytorch as pl
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
import yaml

import torch
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity

from lito.script_utils import pl_utils
from lito.script_utils.config_utils import instantiate_from_config

# Custom resolver to get current time (simplified example)
OmegaConf.register_new_resolver(
    "now",
    lambda fmt="%Y%m%d_%H%M%S": datetime.datetime.now().strftime(fmt),
    use_cache=False,  # important: evaluate each time it's accessed
)

# Ensure determinism
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)


def main(config: T.Dict[str, T.Any]):
    # copy the input config dict
    input_config = copy.deepcopy(config)

    # get environment info
    env_info = pl_utils.get_environment_info(config)
    print(f"Environment info:\n{yaml.dump(env_info, indent=2)}")

    # save input_config to checkpoint_dir
    filename = os.path.join(env_info["checkpoint_dir"], "config.yaml")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(input_config),
        filename,
    )

    # Set all random seeds, we set `workers` to True to make sure each dataloader
    # threads on different nodes have different random seed
    if config["seed"] is not None:
        pl.seed_everything(seed=config["seed"], workers=True)

    # Create PytorchLightningModule (PLM)
    plm: pl.LightningModule = instantiate_from_config(config["plm_config"])
    plm.config = input_config  # save the input config dict for future reference

    # add our custom checkpoint callbacks
    logger_list = pl_utils.setup_loggers(
        config,
        env_info=env_info,
        is_interactive=env_info["is_interactive"],
    )
    # plugin_list = pl_utils.setup_plugins(config, env_info=env_info)
    callback_list = pl_utils.setup_callbacks(config, env_info=env_info)

    # control ddp timeout
    if env_info["is_interactive"] and (torch.cuda.is_available() and torch.cuda.device_count() <= 1):
        print(
            f"Setting strategy to auto: cuda: {torch.cuda.is_available()}, num gpus: {torch.cuda.device_count()} <= 1"
        )
        strategy = "auto"
        plugin_list = []
    elif config["strategy"] == "ddp":
        strategy = DDPStrategy(
            find_unused_parameters=config["find_unused_parameters"],
            timeout=datetime.timedelta(minutes=config["ddp_timeout_min"]),
            cluster_environment=pl_utils.CustomClusterEnvironment(
                node_rank=env_info["node_rank"],
                main_address=env_info["master_ip"],
                main_port=env_info["master_port"],
            ),
        )
        plugin_list = []
    else:
        strategy = config["strategy"]
        plugin_list = pl_utils.setup_plugins(config, env_info=env_info)

    # build the trainer
    # NOTE: The global_step in PyTorch Lightning is incremented each time optimizer.step() is called.
    # So we need to change the val_check_interval accordingly.
    # https://github.com/Lightning-AI/pytorch-lightning/discussions/8007#discussioncomment-887212
    val_check_interval = (
        config["plm_config"]["params"]["optim_config"]["val_check_interval"] * config["accumulate_grad_batches"]
    )

    # profiler
    if config.get("profiler", None) == "pytorch":
        profiler = PyTorchProfiler(
            dirpath=logger_list[0].log_dir,  # where traces will be written
            filename="pl_profile",  # (optional) text table saved as pl_profile.txt
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logger_list[0].log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        config["plm_config"]["params"]["optim_config"]["max_steps"] = max(
            config["plm_config"]["params"]["optim_config"]["max_steps"],
            30,
        )
    else:
        profiler = config.get("profiler", None)

    trainer = pl.Trainer(
        devices=env_info["devices"],  # "auto",
        num_nodes=env_info["num_nodes"],
        strategy=strategy,
        precision=config["precision"],
        logger=logger_list,
        plugins=plugin_list,
        callbacks=callback_list,
        use_distributed_sampler=False,  # we maintain our own sampler using datamodule
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,  # we use number of iteration to control validation frequency
        num_sanity_val_steps=config["plm_config"]["params"]["optim_config"]["num_sanity_val_steps"],
        max_epochs=config["plm_config"]["params"]["optim_config"]["max_epochs"],
        max_steps=config["plm_config"]["params"]["optim_config"]["max_steps"],
        gradient_clip_val=config["plm_config"]["params"]["optim_config"]["gradient_clip_val"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        fast_dev_run=config["fast_dev_run"],
        limit_val_batches=config["limit_val_batches"],
        overfit_batches=config["overfit_batches"],
        detect_anomaly=config["debug"],
        profiler=profiler,
        log_every_n_steps=config.get("log_every_n_steps", 50),  # default is 50
        # deterministic=config.get("deterministic", "warn"),
    )

    # get dataset
    datamodule = pl_utils.get_datamodule(config)

    # download existing checkpoint
    ckpt_path = None
    recovery_dir = env_info["recovery_dir"]

    print(f"\n\n{recovery_dir=}\n\n")

    checkpoint_filename = os.path.join(recovery_dir, "last.ckpt")
    # if checkpoint exists, it means the job had run and stop and resume from bolt
    # so skip download resume_checkpoint_url
    if (not os.path.exists(checkpoint_filename)) and (config.get("resume_checkpoint_url", None) is not None):
        assert os.path.exists(config["resume_checkpoint_url"]), f"{config['resume_checkpoint_url']} does not exist"
        ckpt_path = config["resume_checkpoint_url"]

    if (ckpt_path is None) and os.path.exists(checkpoint_filename):
        ckpt_path = checkpoint_filename

    if config["fast_dev_run"]:
        ckpt_path = None

    if ckpt_path is not None:
        print(f"Restoring existing model from {ckpt_path}.")

    # load only the model parameters
    if (ckpt_path is None) and (config.get("finetune_checkpoint_url", None) is not None):
        assert os.path.exists(config["finetune_checkpoint_url"]), f"{config['finetune_checkpoint_url']} does not exist"
        checkpoint = torch.load(config["finetune_checkpoint_url"], map_location="cpu")
        load_result = plm.load_state_dict(checkpoint["state_dict"], strict=False)
        epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        print(
            f"Loaded model parameters from {config.get('finetune_checkpoint_url')}, "
            f"epoch = {epoch}, global step = {global_step}"
        )
        if len(load_result.missing_keys) > 0:
            print(f"  Missing keys in state_dict: {load_result.missing_keys}")
        if len(load_result.unexpected_keys) > 0:
            print(f"  Unexpected keys in state_dict: {load_result.unexpected_keys}")
        del checkpoint

    # load specific module's state_dict
    # - key: the specific module name in the PyTorchLighting module, e.g., voxel_decoder;
    # - value: the corresponding checkpoint S3 path that contains pre-trained model for that specific module.
    if config.get("module_checkpoint_url_dict", None) is not None:
        module_checkpoint_url_dict = config["module_checkpoint_url_dict"]
        for tmp_module_name, tmp_ckpt_url in module_checkpoint_url_dict.items():
            assert hasattr(plm, tmp_module_name), f"{tmp_module_name=}"

            with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
                assert os.path.exists(tmp_ckpt_url), f"{tmp_ckpt_url=}"
                tmp_ckpt = torch.load(tmp_ckpt_url, map_location="cpu", weights_only=False)

                # We assume the specific module is saved with the whole model.
                # Thus, the module's corresponding keys in state_dict contains the module's prefix,
                # we need to remove this.
                # For example, voxel_decoder.XXX -> XXX
                tmp_prefix = f"{tmp_module_name}."
                tmp_state_dict = {
                    k[len(tmp_prefix) :]: v for k, v in tmp_ckpt["state_dict"].items() if k.startswith(tmp_prefix)
                }
                getattr(plm, tmp_module_name, None).load_state_dict(tmp_state_dict, strict=True)

    # start training
    trainer.fit(
        plm,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    # Set the soft and hard limit for open file descriptors
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Original max number of open files: soft={soft}, hard={hard}")
    # Set new soft limit (must not exceed hard limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard - 1, hard), hard))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"New max number of open files: soft={soft}, hard={hard}")

    mp.set_start_method("spawn", force=True)  # for stability over the default fork
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Filename to the yaml config file",
    )
    args = parser.parse_args()
    config_filename = args.config
    assert os.path.exists(config_filename), f"{config_filename} not exists"

    # print(f"\nhas now resolver? {OmegaConf.has_resolver('now')}\n")  # should be True

    # read the default config and replace the values with the given config yaml
    default_config_filename = str(REPO_ROOT / "configs" / "default_config.yaml")
    default_config = OmegaConf.load(default_config_filename)
    config = OmegaConf.load(config_filename)
    config = OmegaConf.merge(default_config, config)

    # resolve all interpolations in-place
    OmegaConf.resolve(config)

    # all reference need to be able to be resolved
    config = OmegaConf.to_container(config, resolve=False)

    # print config
    yaml_string = OmegaConf.to_yaml(config, resolve=False)
    print(yaml_string)

    # set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")

    main(config)
