import copy
import clip

import torch
from torch import nn
from torch.cuda.amp import autocast

from spatial_actor.models.modules.depth_expert.depth_anything_v2 import DepthAnythingV2
import spatial_actor.utils.model_utils as model_utils
from spatial_actor.models.model import SpatialActor


def encoder_text(clip_model, dtype, texts=None, tokens=None, return_cls=False):
    assert texts is not None or tokens is not None

    if tokens is None:
        tokens = clip.tokenize(texts).to(clip_model.token_embedding.weight.device)

    x = clip_model.token_embedding(tokens).type(dtype)
    x = x + clip_model.positional_embedding.type(dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).float()
    if return_cls:
        cls = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ clip_model.text_projection
        return x, cls
    return x


class Network(nn.Module):
    def __init__(
        self,
        img_size,
        proprio_dim,
        lang_dim,
        lang_len,
        add_depth,
        rend_three_views,
        st_sca,
        st_wpt_loc_aug,
        st_wpt_loc_inp_no_noise,
        renderer_device="cuda:0",
        reenc_text=False,
        sem_enc_type=None,
        dep_exp_type=None,
        dep_exp_path=None,
        align_loss=0.0,
        **kwargs,
    ):
        """
        :param st_sca: scaling of the pc in the second stage
        :param st_wpt_loc_aug: how much noise is to be added to wpt_local when
            transforming the pc in the second stage while training. This is
            expressed as a percentage of total pc size which is 2.
        :param st_wpt_loc_inp_no_noise: whether or not to add any noise to the
            wpt_local location which is fed to stage_two. This wpt_local
            location is used to extract features for rotation prediction
            currently. Other use cases might also arise later on. Even if
            st_wpt_loc_aug is True, this will compensate for that if set to
            True.
        """
        super().__init__()

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        del args['kwargs']
        args.update(kwargs)

        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise
        self.proprio_dim = proprio_dim
        self.img_size = img_size
        self.sem_enc_type = sem_enc_type
        self.reenc_text = reenc_text
        self.dep_exp_type = dep_exp_type
        self.align_loss = align_loss
        self.lang_dim = lang_dim
        self.lang_len = lang_len

        from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer
        global BoxRenderer

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )

        self.num_img = self.renderer.num_img

        if self.reenc_text:
            self.clip_model, _ = clip.load(self.sem_enc_type.replace('CLIP-', ''))
            self.clip_model_dtype = self.clip_model.dtype
            self.clip_model.visual = None
            self.add_module("clip_model", self.clip_model)
            self.clip_model.eval()
        else:
            self.clip_model = None

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        model_type = self.dep_exp_type.replace('DA-', '')
        self.depth_expert = DepthAnythingV2(**model_configs[model_type])
        self.depth_expert.load_state_dict(
            torch.load(f'{dep_exp_path}/depth_anything_v2_{model_type}.pth')
        )
        self.depth_expert = self.depth_expert.pretrained
        self.add_module("depth_expert", self.depth_expert)
        for p in self.depth_expert.parameters():
            p.requires_grad = False

        self.spatial_actor1 = SpatialActor(
            **args,
            renderer=self.renderer,
            no_feat=True)

        self.spatial_actor2 = SpatialActor(
            **args,
            renderer=self.renderer,
            no_feat=False)


    def get_pt_loc_on_img(self, pt, spact_1_or_2, dyn_cam_info, out=None):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for spacial_actor2
        :param out: output from spacial_actor, when using spacial_actor2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, _np, x = pt.shape
        assert x == 3

        assert isinstance(spact_1_or_2, bool)
        if spact_1_or_2:
            assert out is None
            out = self.spatial_actor1.get_pt_loc_on_img(pt, dyn_cam_info)
        else:
            assert out is not None
            assert out['wpt_local1'].shape == (bs, 3)
            pt, _ = model_utils.trans_pc(pt, loc=out["wpt_local1"], sca=self.st_sca)
            pt = pt.view(bs, _np, 3)
            out = self.spatial_actor2.get_pt_loc_on_img(pt, dyn_cam_info)

        return out

    def get_wpt(self, out, spact_1_or_2, dyn_cam_info, y_q=None,):
        """
        Estimate the q-values given output from model
        :param out: output from model
        :param y_q: refer to the definition in modeel.get_wpt
        """
        assert isinstance(spact_1_or_2, bool)

        if spact_1_or_2:
            wpt = self.spatial_actor1.get_wpt(
                out, dyn_cam_info, y_q,
            )
        else:
            wpt = self.spatial_actor2.get_wpt(
                out["spacial_actor2"], dyn_cam_info, y_q,
            )
            wpt = out["rev_trans"](wpt)

        return wpt

    def render(self, pc, img_feat, img_aug, spact_1_or_2, dyn_cam_info):
        assert isinstance(spact_1_or_2, bool)
        if spact_1_or_2:
            spatial_actor = self.spatial_actor1
        else:
            spatial_actor = self.spatial_actor2

        with torch.no_grad():
            with autocast(enabled=False):
                if dyn_cam_info is None:
                    dyn_cam_info_itr = (None,) * len(pc)
                else:
                    dyn_cam_info_itr = dyn_cam_info

                if spatial_actor.add_corr:
                    if spatial_actor.norm_corr:
                        img = []
                        for _pc, _img_feat, _dyn_cam_info in zip(
                            pc, img_feat, dyn_cam_info_itr
                        ):
                            # fix when the pc is empty
                            max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                            img.append(
                                self.renderer(
                                    _pc,
                                    torch.cat((_pc / max_pc, _img_feat), dim=-1),
                                    fix_cam=True,
                                    dyn_cam_info=(_dyn_cam_info,)
                                    if not (_dyn_cam_info is None)
                                    else None,
                                ).unsqueeze(0)
                            )
                    else:
                        img = [
                            self.renderer(
                                _pc,
                                torch.cat((_pc, _img_feat), dim=-1),
                                fix_cam=True,
                                dyn_cam_info=(_dyn_cam_info,)
                                if not (_dyn_cam_info is None)
                                else None,
                            ).unsqueeze(0)
                            for (_pc, _img_feat, _dyn_cam_info) in zip(
                                pc, img_feat, dyn_cam_info_itr
                            )
                        ]
                else:
                    img = [
                        self.renderer(
                            _pc,
                            _img_feat,
                            fix_cam=True,
                            dyn_cam_info=(_dyn_cam_info,)
                            if not (_dyn_cam_info is None)
                            else None,
                        ).unsqueeze(0)
                        for (_pc, _img_feat, _dyn_cam_info) in zip(
                            pc, img_feat, dyn_cam_info_itr
                        )
                    ]

        img = torch.cat(img, 0)
        img = img.permute(0, 1, 4, 2, 3)

        # for visualization purposes
        if spatial_actor.add_corr:
            spatial_actor.img = img[:, :, 3:].clone().detach()
        else:
            spatial_actor.img = img.clone().detach()

        # image augmentation
        if img_aug != 0:
            stdv = img_aug * torch.rand(1, device=img.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)

        if spatial_actor.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = spatial_actor.pixel_loc.to(img.device)
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )

        return img

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        lang_token=None,
        lang=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param wpt_local: gt location of the wpt in 3D, tensor of shape
            (bs, 3)
        :param rot_x_y: (bs, 2) rotation in x and y direction
        """

        with torch.no_grad():
            if not self.training and self.clip_model is None:
                self.clip_model, _ = clip.load(self.sem_enc_type.replace('CLIP-', ''))
                self.clip_model_dtype = self.clip_model.dtype
                self.clip_model.visual = None
                self.clip_model.eval()
                self.add_module("clip_model", self.clip_model)

            lang_list = [item[0][0] for item in lang]
            if self.reenc_text or not self.training:
                lang_emb = encoder_text(self.clip_model, self.clip_model_dtype,
                                        texts=lang_list, tokens=lang_token)
            else:
                assert lang_emb is not None
                _, _lang_emb_len, _lang_dim = lang_emb.shape
                assert (
                    _lang_emb_len == self.lang_len
                ), "Does not support lang_emb of shape {lang_emb.shape}"
                assert (
                    _lang_dim == self.lang_dim
                ), "Does not support lang_emb of shape {lang_emb.shape}"

            img = self.render(
                pc=pc,
                img_feat=img_feat,
                img_aug=img_aug,
                spact_1_or_2=True,
                dyn_cam_info=None,
            )

        if self.training:
            wpt_local_stage_one = wpt_local
            wpt_local_stage_one = wpt_local_stage_one.clone().detach()
        else:
            wpt_local_stage_one = wpt_local

        out = self.spatial_actor1(
            img=img,
            proprio=proprio,
            lang_emb=lang_emb,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
            depth_expert=self.depth_expert,
        )

        with torch.no_grad():
            # adding then noisy location for training
            if self.training:
                # noise is added so that the wpt_local2 is not exactly at
                # the center of the pc
                wpt_local_stage_one_noisy = model_utils.add_uni_noi(
                    wpt_local_stage_one.clone().detach(), 2 * self.st_wpt_loc_aug
                )
                pc, rev_trans = model_utils.trans_pc(
                    pc, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                )

                if self.st_wpt_loc_inp_no_noise:
                    wpt_local2, _ = model_utils.trans_pc(
                        wpt_local, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                    )
                else:
                    wpt_local2, _ = model_utils.trans_pc(
                        wpt_local, loc=wpt_local_stage_one, sca=self.st_sca
                    )

            else:
                # bs, 3
                wpt_local = self.get_wpt(
                    out, y_q=None, spact_1_or_2=True,
                    dyn_cam_info=None,
                )
                pc, rev_trans = model_utils.trans_pc(
                    pc, loc=wpt_local, sca=self.st_sca
                )
                # bad name!
                wpt_local_stage_one_noisy = wpt_local

                # must pass None to spacial_actor2 while in eval
                wpt_local2 = None

            img = self.render(
                pc=pc,
                img_feat=img_feat,
                img_aug=img_aug,
                spact_1_or_2=False,
                dyn_cam_info=None,
            )

        out_spacial_actor2 = self.spatial_actor2(
            img=img,
            proprio=proprio,
            lang_emb=lang_emb,
            wpt_local=wpt_local2,
            rot_x_y=rot_x_y,
            depth_expert=self.depth_expert,
        )

        out["wpt_local1"] = wpt_local_stage_one_noisy
        out["rev_trans"] = rev_trans
        out["spacial_actor2"] = out_spacial_actor2

        return out
