from yacs.config import CfgNode as CN

_C = CN()
_C.exp_id = "spact"
_C.tasks = "all"
_C.bs = 8
_C.epochs = 50
_C.num_workers = 8
_C.train_iter = 16 * 10000
_C.resume = ''

# lr should be thought on per sample basis
# effective lr is multiplied by bs * num_devices
_C.lr = 1.25e-5
_C.optimizer_type = "lamb"
_C.warmup_steps = 2000
_C.lr_cos_dec = True
_C.add_rgc_loss = True
_C.lambda_weight_l2 = 1e-4
_C.amp = True
_C.bnb = True

# 'transition_uniform' or 'task_uniform'
_C.sample_distribution_mode = 'task_uniform'
_C.img_aug = 0.0
_C.transform_augmentation = True
_C.transform_augmentation_xyz = [0.125, 0.125, 0.125]
_C.transform_augmentation_rpy = [0.0, 0.0, 45.0]

_C.num_rotation_classes = 72
_C.gt_hm_sigma = 1.5
_C.place_with_mean = False
_C.move_pc_in_bound = True
_C.noise_type = 'none'

_C.model = CN()
_C.model.sem_enc_type = 'CLIP-RN50'
_C.model.geo_enc_type = 'RN50'
_C.model.dep_exp_type = 'DA-vitb'
_C.model.dep_exp_path = 'Depth-Anything-V2-Base'

_C.model.lang_dim = 512
_C.model.lang_len = 77
_C.model.add_proprio = True
_C.model.proprio_dim = 4
_C.model.proprio_cat_dim = 64
_C.model.im_channels = 128

_C.model.spt_view_layers = 4
_C.model.spt_scene_layers = 4
_C.model.attn_dim = 512
_C.model.attn_heads = 8
_C.model.attn_dim_head = 64
_C.model.activation = "lrelu"
_C.model.weight_tie_layers = False
_C.model.attn_dropout = 0.1
_C.model.img_patch_size = 8
_C.model.final_dim = 64

_C.model.img_feat_dim = 3
_C.model.num_rot = 72
_C.model.feat_dim = (72 * 3) + 2 + 2

_C.model.img_size = 224
_C.model.add_corr = True
_C.model.norm_corr = True
_C.model.add_pixel_loc = True
_C.model.add_depth = True
_C.model.rend_three_views = True
_C.model.xops = True
_C.model.st_sca = 4
_C.model.st_wpt_loc_aug = 0.05
_C.model.st_wpt_loc_inp_no_noise = True

_C.model.reenc_text = False
_C.model.align_loss = 0.0

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
