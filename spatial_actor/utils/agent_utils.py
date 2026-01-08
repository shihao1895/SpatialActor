import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


def get_pc_img_feat(obs, pcd, bounds=None):
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1
    )

    img_feat = (img_feat + 1) / 2

    return pc, img_feat


def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def preprocess_inputs(replay_sample, cameras):
    obs, pcds = [], []
    for n in cameras:
        rgb = stack_on_channel(replay_sample["%s_rgb" % n])
        pcd = stack_on_channel(replay_sample["%s_point_cloud" % n])

        rgb = _norm_rgb(rgb)

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds


class TensorboardManager:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            if "image" in k:
                for i, x in enumerate(v):
                    self.writer.add_image(f"{split}_{step}", x, i)
            elif "hist" in k:
                if isinstance(v, list):
                    self.writer.add_histogram(k, v, step)
                elif isinstance(v, dict):
                    hist_id = {}
                    for i, idx in enumerate(sorted(v.keys())):
                        self.writer.add_histogram(f"{split}_{k}_{step}", v[idx], i)
                        hist_id[i] = idx
                    self.writer.add_text(f"{split}_{k}_{step}_id", f"{hist_id}")
                else:
                    assert False
            else:
                self.writer.add_scalar("%s_%s" % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


def short_name(cfg_opts):
    SHORT_FORMS = {
        "exp_id": "EXP",
        "tasks": "TSK",
        "bs": "BS",
        "epochs": "E",
        "train_iter": "ITER",
        "num_workers": "WKR",
        "resume": "RES",

        "lr": "LR",
        "optimizer_type": "OPT",
        "warmup_steps": "WARM",
        "lr_cos_dec": "LCD",
        "amp": "AMP",
        "bnb": "BNB",
        "lambda_weight_l2": "L2",

        "sample_distribution_mode": "SDM",
        "noise_type": "NOI",

        "gt_hm_sigma": "GHS",

        "model.sem_enc_type": "SE",
        "model.geo_enc_type": "GE",
        "model.dep_exp_type": "DE",
        "model.dep_exp_path": "DE_PATH",

        "model.add_proprio": "PROP",
        "model.proprio_cat_dim": "PROP_D",

        "model.im_channels": "IM_C",
        "model.spt_view_layers": "VIEW_L",
        "model.spt_scene_layers": "SC_L",
        "model.attn_dim": "ATTN_D",
        "model.attn_heads": "ATTN_H",
        "model.attn_dim_head": "ATTN_DH",
        "model.attn_dropout": "ATTN_DROP",
        "model.activation": "ACT",

        "model.img_size": "IMG_SIZE",
        "model.final_dim": "FINAL_D",

        "model.xops": "M_XOP",
        "model.reenc_text": "RE_TEXT",
        "model.align_loss": "ALIGN",

        "True": "T",
        "False": "F",
    }

    if "resume" in cfg_opts:
        cfg_opts = cfg_opts.split(" ")
        res_idx = cfg_opts.index("resume")
        cfg_opts.pop(res_idx + 1)
        cfg_opts = " ".join(cfg_opts)

    cfg_opts = cfg_opts.replace(" ", "_")
    cfg_opts = cfg_opts.replace("/", "_")
    cfg_opts = cfg_opts.replace("[", "")
    cfg_opts = cfg_opts.replace("]", "")
    cfg_opts = cfg_opts.replace("..", "")
    for a, b in SHORT_FORMS.items():
        cfg_opts = cfg_opts.replace(a, b)

    return cfg_opts


def get_num_feat(cfg):
    num_feat = cfg.num_rotation_classes * 3
    # 2 for grip, 2 for collision
    num_feat += 4
    return num_feat


RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]


def load_agent_state(agent_path, agent=None, only_epoch=False):
    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

        if isinstance(model, DDP):
            model = model.module

        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            try:
                print(
                    "WARNING: loading states in spacial_actor. "
                    "Be cautious if you are using a two stage network."
                )
                model.spacial_actor.load_state_dict(checkpoint["model_state"])
            except RuntimeError:
                print(
                    "WARNING: loading states with strick=False! "
                    "KNOW WHAT YOU ARE DOING!!"
                )
                model.load_state_dict(checkpoint["model_state"], strict=False)

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print(
                "WARNING: No optimizer_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

        if "lr_sched_state" in checkpoint:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        else:
            print(
                "WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

    return epoch
