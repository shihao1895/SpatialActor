import os
import tqdm
import random
import yaml
import argparse
from collections import defaultdict
from contextlib import redirect_stdout

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import configs.config as cfg_mod
from spatial_actor.models.agent import Agent
from spatial_actor.models.network import Network
from spatial_actor.models.agent import print_loss_log
from spatial_actor.datasets.get_dataset import get_dataset
import spatial_actor.utils.ddp_utils as ddp_utils
from spatial_actor.utils.agent_utils import (
    TensorboardManager,
    short_name,
    load_agent_state,
    RLBENCH_TASKS,
)
from spatial_actor.utils.constant import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)


def train(
        agent,
        dataset,
        training_iterations,
        rank,
        start_iter=None,
        iters_per_epoch=None,
        log_dir=None,
        tb=None,
        ckpt_freq=None
):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)

    start = start_iter if start_iter is not None else 0
    iter_command = range(start, training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):
        raw_batch = next(data_iter)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "step": iteration,
        }
        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == start_iter),
                "eval_log": False,
            }
        )
        agent.update(**update_args)

        if start_iter is not None:
            completed = iteration + 1
            if (completed % iters_per_epoch == 0) or (completed == training_iterations):
                fake_epoch = completed // iters_per_epoch

                if rank == 0:
                    assert tb is not None
                    save_agent(agent, f"{log_dir}/model_last.pth", fake_epoch)
                    if (fake_epoch % ckpt_freq == 0) or (completed == training_iterations):
                        save_agent(agent, f"{log_dir}/model_{fake_epoch}.pth", fake_epoch)
                    log = print_loss_log(agent)
                    tb.update("train", fake_epoch, log)

                dist.barrier()

            if completed % 100 == 0 and rank == 0:
                assert tb is not None
                fake_epoch = completed // iters_per_epoch
                log = print_loss_log(agent)
                tb.update("train", fake_epoch, log)

    if start_iter is None and rank == 0:
        log = print_loss_log(agent)

    return log


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )

def get_tasks(cfg):
    parsed_tasks = cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks


def get_logdir(cmd_args, cfg):
    log_dir = os.path.join(cmd_args.log_dir, cfg.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(cfg, cmd_args, log_dir):
    with open(f"{log_dir}/cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


def main(rank, cmd_args, devices, port):
    """
    :param rank:
    :param cmd_args:
    :param devices: list or int. if list, we use ddp else not
    """
    device = devices[rank]
    device = f"cuda:{device}"
    ddp = len(devices) > 1
    ddp_utils.setup(rank, world_size=len(devices), port=port)

    cfg = cfg_mod.get_cfg_defaults()
    if cmd_args.cfg_path != "":
        cfg.merge_from_file(cmd_args.cfg_path)
    if cmd_args.cfg_opts != "":
        cfg.merge_from_list(cmd_args.cfg_opts.split(" "))

    if ddp:
        print(f"Running DDP on rank {rank}.")

    lr_per_batch = cfg.lr
    old_exp_id = cfg.exp_id

    cfg.lr *= len(devices) * cfg.bs
    if cmd_args.cfg_opts != "":
        cfg.exp_id += f"_{short_name(cmd_args.cfg_opts)}"

    cfg.model.feat_dim = cfg.num_rotation_classes * 3 + 2 + 2

    if rank == 0:
        print(f"dict(cfg)={dict(cfg)}")
    cfg.freeze()

    # Things to change
    BATCH_SIZE_TRAIN = cfg.bs
    # iterations per epoch
    TRAINING_ITERATIONS = int(cfg.train_iter // (cfg.bs * len(devices)))
    EPOCHS = cfg.epochs
    log_dir = get_logdir(cmd_args, cfg)
    tasks = get_tasks(cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))

    # for maintaining backward compatibility
    assert cfg.model.num_rot == cfg.num_rotation_classes, print(
        cfg.model.num_rot, cfg.num_rotation_classes
    )

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    network = Network(
        renderer_device=device,
        **cfg.model,
    ).to(device)

    if ddp:
        network = DDP(network, device_ids=[device], find_unused_parameters=True)

    DATA_FOLDER = cmd_args.data_folder

    agent = Agent(
        network=network,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
        **cfg,
    )
    agent.build(training=True, device=device)

    TRAIN_REPLAY_STORAGE_DIR = cmd_args.train_replay_dir
    if cfg.model.sem_enc_type.startswith("CLIP-"):
        clip_text_type = cfg.model.sem_enc_type.replace("CLIP-", "")
    else:
        clip_text_type = 'RN50'
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        DATA_FOLDER,
        cmd_args.num_train,
        None,
        cmd_args.refresh_replay,
        device,
        num_workers=cfg.num_workers,
        only_train=True,
        sample_distribution_mode=cfg.sample_distribution_mode,
        clip_type=clip_text_type,
    )
    train_dataset, _ = get_dataset_func()

    if cmd_args.iter_based:
        iters_per_epoch = TRAINING_ITERATIONS
        TRAINING_ITERATIONS = EPOCHS * iters_per_epoch
        EPOCHS = 1

    start_epoch = 0
    end_epoch = EPOCHS

    agent_path = f"{log_dir}/model_last.pth" if cfg.resume == "last" else cfg.resume
    if os.path.exists(agent_path):
        start_epoch = load_agent_state(agent_path, agent, only_epoch=False)
        print(f"Recovering model and checkpoint from {agent_path}")
    elif cfg.resume != "last":
        print(f"Error: {agent_path} does not exist")

    dist.barrier()

    if cmd_args.iter_based:
        start_iter = start_epoch * iters_per_epoch
    else:
        start_iter = None

    tb = None
    if rank == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = cfg.lr
        temp2 = cfg.exp_id
        cfg.defrost()
        cfg.lr = lr_per_batch
        cfg.exp_id = old_exp_id
        dump_log(cfg, cmd_args, log_dir)
        cfg.lr = temp1
        cfg.exp_id = temp2
        cfg.freeze()
        tb = TensorboardManager(log_dir)

    print("Start training ...", flush=True)
    if cmd_args.iter_based:
        _ = train(
            agent=agent,
            dataset=train_dataset,
            training_iterations=TRAINING_ITERATIONS,
            rank=rank,
            start_iter=start_iter,
            iters_per_epoch=iters_per_epoch,
            log_dir=log_dir,
            tb=tb if rank==0 else None,
            ckpt_freq=cmd_args.ckpt_freq,
        )
    else:
        i = start_epoch
        while True:
            if i == end_epoch:
                break

            i += 1

            print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")

            out = train(
                agent=agent,
                dataset=train_dataset,
                training_iterations=TRAINING_ITERATIONS,
                rank=rank,
            )

            if rank == 0:
                tb.update("train", i, out)

            if (i % cmd_args.ckpt_freq == 0) or (i == end_epoch):
                if rank == 0:
                    save_agent(agent, f"{log_dir}/model_{i}.pth", i)

            if rank == 0:
                save_agent(agent, f"{log_dir}/model_last.pth", i)

            dist.barrier()

    if rank == 0:
        tb.close()
        print("[Finish]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--cfg_path", type=str, default="")
    parser.add_argument("--cfg_opts", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="log")
    parser.add_argument("--ckpt_freq", type=int, default=5)
    parser.add_argument("--iter-based", action="store_true", default=False)
    parser.add_argument("--data-folder", type=str, default="rlbench")
    parser.add_argument("--train-replay-dir", type=str, default="replay_train")
    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--num-train", type=int, default=100)

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]

    port = (random.randint(0, 3000) % 3000) + 27000
    mp.spawn(main, args=(cmd_args, devices, port), nprocs=len(devices), join=True)
